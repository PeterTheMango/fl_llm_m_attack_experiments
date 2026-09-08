import json
from pathlib import Path

import pytest

from master_script.core import queue, runner
from master_script.core.yaml_config import ConfigError
from master_script.perform_experiments import main


def configs(tmp_path):
    paths = []
    for name in ("zlib", "min_k", "min_k_plus_plus"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"defaults: {{use_hf_models: false, attack_trials: 2}}\nattacks:\n  {name}: {{}}\n")
        paths.append(path)
    return paths


def test_three_distinct_files_run_fifo_with_full_results(tmp_path):
    paths = configs(tmp_path)
    before = [p.read_bytes() for p in paths]
    batch = queue.load_batch(paths)
    active = []
    observed = []

    def start(key, name, config):
        assert not active
        active.append(key)
        observed.append(name)

    def end(key, *args):
        assert active.pop() == key

    results, directory = queue.run_batch(batch, output_root=tmp_path / "out",
                                          use_firestore=False,
                                          on_run_start=start, on_run_end=end)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert observed == ["zlib", "min_k", "min_k_plus_plus"]
    assert manifest["status"] == "complete"
    assert [p.read_bytes() for p in paths] == before
    for index, entry in enumerate(manifest["entries"]):
        saved = json.loads((directory / entry["result_file"]).read_text())
        assert saved["metrics"]["num_trials"] == 2
        assert len(saved["attack_trials"]) == 2
        assert Path(saved["artifacts"]["artifact_dir"]).is_relative_to(directory)
        assert entry["source"] == str(paths[index].resolve())
        if index:
            assert entry["started_unix"] >= manifest["entries"][index - 1]["ended_unix"]


def test_late_invalid_file_blocks_all_compute_and_output(tmp_path, monkeypatch):
    paths = configs(tmp_path)
    paths[-1].write_text("attacks: {no_such_attack: {}}")
    monkeypatch.setattr(runner, "run_single_experiment", lambda *a, **k: pytest.fail("compute started"))
    output = tmp_path / "out"
    assert main(["--queue", *map(str, paths), "--queue-output", str(output), "--no-firestore"]) == 2
    assert not output.exists()


def test_failed_middle_run_is_persisted_and_next_runs(tmp_path, monkeypatch):
    batch = queue.load_batch(configs(tmp_path))
    original = runner.run_single_experiment

    def run(cfg, spec, **kwargs):
        if spec.name == "min_k":
            raise RuntimeError("injected simulation failure")
        return original(cfg, spec, **kwargs)

    monkeypatch.setattr(runner, "run_single_experiment", run)
    monkeypatch.setattr(runner, "reset_ray_after_failure", lambda: None)
    results, directory = queue.run_batch(batch, use_firestore=False, output_root=tmp_path / "out")
    assert [r["status"] for r in results] == ["complete", "failed", "complete"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    failure = json.loads((directory / manifest["entries"][1]["result_file"]).read_text())
    assert "injected simulation failure" in failure["error"]


def test_dry_run_never_creates_queue_directory(tmp_path):
    output = tmp_path / "out"
    assert main(["--queue", *map(str, configs(tmp_path)), "--queue-output", str(output),
                 "--dry-run", "--no-firestore"]) == 0
    assert not output.exists()


def test_batch_uses_validated_snapshot_not_later_file_edit(tmp_path):
    paths = configs(tmp_path)
    batch = queue.load_batch(paths)
    paths[0].write_text("invalid replacement")
    results, _ = queue.run_batch(batch, use_firestore=False, output_root=tmp_path / "out")
    assert len(results) == 3 and all(r["status"] == "complete" for r in results)


def test_interrupt_records_terminal_state_without_starting_next(tmp_path, monkeypatch):
    batch = queue.load_batch(configs(tmp_path))
    monkeypatch.setattr(runner, "run_single_experiment", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        queue.run_batch(batch, use_firestore=False, output_root=tmp_path / "out")
    manifest = json.loads(next((tmp_path / "out").glob("*/manifest.json")).read_text())
    assert manifest["status"] == "interrupted"
    assert [e["status"] for e in manifest["entries"]] == ["interrupted", "not_run", "not_run"]


def test_dashboard_queue_rejects_path_escape(tmp_path):
    from master_script.webui.launch import start_queue
    assert not start_queue(["../smoke.yaml"], use_firestore=False)["ok"]


def test_queue_disallows_parallel_mode(tmp_path):
    assert main(["--queue", *map(str, configs(tmp_path)), "--max-parallel", "2"]) == 2


def test_optional_remote_errors_preserve_completed_local_computation(tmp_path, monkeypatch):
    batch = queue.load_batch(configs(tmp_path))

    def unavailable(*args):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(runner.firestore, "load_cached_result", unavailable)
    monkeypatch.setattr(runner.firestore, "save_result", unavailable)
    results, directory = queue.run_batch(batch, use_firestore=True, output_root=tmp_path / "out")
    assert all(r["status"] == "complete" and not r["firestore_saved"] for r in results)
    for r in results:
        assert "remote unavailable" in r["cache_error"]
        assert "remote unavailable" in r["persistence_error"]
        local = json.loads((Path(r["artifacts"]["artifact_dir"]) / "result.json").read_text())
        assert local == r
