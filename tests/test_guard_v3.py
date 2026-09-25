from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from master_script.core import guard_runtime
from master_script.core.guard_runtime import prepare_guard, read_guard_events, parse_guard
from master_script.core.checkpoint_retention import retire_checkpoints
from master_script.core.guard_protocol import utility_report
from master_script.tools import collect_guard_traces
from master_script.tools.benchmark_guard_workers import benchmark
from tests.test_guard_integration import settings


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_checks_preserve_features_hashes_and_owned_nonzero_updates(tmp_path, workers):
    rng = np.random.default_rng(55)
    prior = [rng.normal(size=65539).astype("float32") for _ in range(5)]
    value = [p + np.float32(.0002) for p in prior]
    serial = prepare_guard(settings(tmp_path), "serial", prior, rounds=1)
    parallel = prepare_guard(replace(serial.settings, validation_workers=workers), "parallel", prior, rounds=1)
    a = serial.authorize(value, 0, 1)
    b = parallel.authorize(value, 0, 1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for x in value:
        x[:] = np.nan
    assert all(np.isfinite(x).all() for x in b)
    es = read_guard_events(tmp_path / "local")
    assert es[-1]["features"] == es[-2]["features"]
    assert es[-1]["request_sha256"] == es[-2]["request_sha256"]
    assert es[-1]["validation_workers"] == workers
    assert parallel.authorize(a, 0, 1) is None  # No duplicate-release bypass.


def test_reference_work_really_overlaps_and_reads_each_array_once(tmp_path, monkeypatch):
    guard = replace(settings(tmp_path), validation_workers=4)
    runtime = prepare_guard(guard, "parallel", [np.ones(8)] * 4, rounds=1)
    barrier = threading.Barrier(4)
    threads = set()
    original = guard_runtime._inspect_layer
    def inspect(*args):
        threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return original(*args)
    monkeypatch.setattr(guard_runtime, "_inspect_layer", inspect)
    reads = []
    get = np.lib.npyio.NpzFile.__getitem__
    def observed(self, key):
        reads.append(key)
        return get(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", observed)
    assert runtime.authorize([np.ones(8)] * 4, 0, 1) is not None
    assert len(threads) == 4 and sorted(reads) == [f"arr_{i}" for i in range(4)]


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_tampering_and_invalid_parameters_fail_before_reservation(tmp_path, workers):
    guard = replace(settings(tmp_path), validation_workers=workers)
    runtime = prepare_guard(guard, "parallel", [np.ones(8)] * 4, rounds=2)
    value = [np.ones(8)] * 3 + [np.full(8, np.nan)]
    assert runtime.authorize(value, 0, 1) is None
    event = read_guard_events(tmp_path / "local")[-1]
    assert event["features"] is None and event["reservation_count"] == 0
    np.savez(runtime.reference_path, *([np.zeros(8)] * 4))
    with pytest.raises(ValueError, match="integrity"):
        runtime.authorize([np.ones(8)] * 4, 0, 2)
    from master_script.core.release_ledger import ReleaseLedger
    assert ReleaseLedger(tmp_path / "local/release-ledger.sqlite").count("parallel:0") == 0


def test_worker_config_is_client_owned_and_strict(tmp_path):
    guard = settings(tmp_path)
    value = {"mode": "rules", "policy_file": guard.policy_file, "release_budget": 2}
    assert "validation_workers" not in parse_guard(value, "<config>").metadata()
    for invalid in [True, 0, 3, 8, "4"]:
        with pytest.raises(ValueError, match="validation_workers"):
            parse_guard({**value, "validation_workers": invalid}, "<config>")
    assert parse_guard({**value, "validation_workers": 4}, "<config>").metadata()["validation_workers"] == 4


def completed_fixture(tmp_path):
    path = tmp_path / "0000-example.json"
    artifact = tmp_path / "artifacts/0000-example"
    guard = replace(settings(tmp_path), runtime_directory=str(artifact / "client-guard"))
    runtime = prepare_guard(guard, "training:2000:True", [np.ones(8, dtype="float32")], rounds=1)
    runtime.authorize([np.ones(8, dtype="float32")], 0, 1)
    result = {"status": "complete", "run_id": "example", "implementation_fingerprint": "old",
              "pipeline": {"client_guard": {"mode": "rules"}},
              "guard_events": read_guard_events(artifact / "client-guard")}
    path.write_text(json.dumps(result))
    (artifact / "result.json").write_text(json.dumps(result))
    (tmp_path / "manifest.json").write_text(json.dumps({"entries": [
        {"status": "complete", "run_id": "example", "result_file": path.name}]}))
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / "answers.jsonl").write_text('{"answer":"keep"}\n')
    return path, artifact, runtime


def test_retirement_preserves_evidence_and_prevents_reinitialization(tmp_path):
    path, artifact, runtime = completed_fixture(tmp_path)
    kept = [path, artifact / "result.json", artifact / "answers.jsonl", artifact / "client-guard/release-ledger.sqlite"]
    hashes = {p: sha256(p.read_bytes()).hexdigest() for p in kept}
    report = retire_checkpoints(path)
    assert report["status"] == "retired" and report["reclaimed_bytes"] > 0
    assert len(report["files"]) == 2
    assert not Path(runtime.reference_path).exists()
    assert all(sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    with pytest.raises(ValueError, match="retired"):
        prepare_guard(runtime.settings, runtime.scope, [np.ones(8, dtype="float32")], rounds=1)


@pytest.mark.parametrize("problem", ["failed", "mismatch", "symlink"])
def test_retirement_refuses_unsafe_or_incomplete_artifacts(tmp_path, problem):
    path, artifact, runtime = completed_fixture(tmp_path)
    if problem == "failed":
        d = json.loads(path.read_text()); d["status"] = "failed"; path.write_text(json.dumps(d))
    elif problem == "mismatch":
        (artifact / "result.json").write_text('{}')
    else:
        (artifact / "cache.safetensors").symlink_to(path)
    with pytest.raises(ValueError):
        retire_checkpoints(path)
    assert Path(runtime.reference_path).exists() and (artifact / "model.safetensors").exists()
    assert not (artifact / "checkpoint-retirement.json").exists()


def test_worker_benchmark_never_changes_original_files(tmp_path):
    path, artifact, runtime = completed_fixture(tmp_path)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    report = benchmark(tmp_path, repeats=1)
    assert report["features_and_hashes_identical"]
    assert set(report["samples"]) == {"1", "2", "4"}
    assert all(p.read_bytes() == contents for p, contents in before.items())


def test_collection_plan_is_one_target_per_job_and_roles_are_frozen(tmp_path):
    out = tmp_path / "new"
    launch = collect_guard_traces.prepare(out, targets=8, workers=2)
    assert len(launch["jobs"]) == 24
    assert len({j["seed"] for j in launch["jobs"]}) == 8
    for seed in {j["seed"] for j in launch["jobs"]}:
        assert len({j["role"] for j in launch["jobs"] if j["seed"] == seed}) == 1
    from master_script.core.queue import load_batch
    for job in launch["jobs"]:
        batch = load_batch([out / job["config"]])
        cfg, spec = batch.pairs[0]
        assert spec.pipeline.client_guard.validation_workers == 2
        assert cfg.attack_trials in (2, 4)
        assert getattr(cfg, "attack_targets", 1) == 1


def test_collection_does_not_retry_an_interrupted_private_job(tmp_path):
    out = tmp_path / "new"
    collect_guard_traces.prepare(out, targets=3)
    (out / "progress.json").write_text(json.dumps({
        "launch_sha256": sha256((out / "launch.json").read_bytes()).hexdigest(),
        "active_job": {"index": 0}, "completed": []}))
    with pytest.raises(ValueError, match="interrupted"):
        collect_guard_traces.execute(out / "launch.json", gpu="0")
    assert not (out / ".collector.lock").exists()


def test_collection_lock_prevents_duplicate_private_jobs(tmp_path):
    lock = tmp_path / ".collector.lock"
    lock.write_text('active collector')
    with pytest.raises(ValueError, match="Collector is running"):
        collect_guard_traces.execute(tmp_path / "launch.json", gpu="0")
    assert lock.read_text() == 'active collector'


def test_collection_runs_bounded_jobs_keeps_traces_and_cleans_only_owned_scratch(tmp_path, monkeypatch):
    from master_script.core import datasets
    from master_script.core.queue import load_batch, write_json
    from master_script.core.config import implementation_fingerprint
    out = tmp_path / "collection"
    collect_guard_traces.prepare(out, targets=3)
    scratch = tmp_path / "scratch"; scratch.mkdir()
    unrelated = scratch / "other-job.log"; unrelated.write_text("keep")
    monkeypatch.setattr(datasets, "target_record_for", lambda cfg, fallback: f"target-{cfg.seed}")
    monkeypatch.setattr(collect_guard_traces, "require_headroom", lambda *args: None)
    calls = []
    def run(command, *, env, check):
        calls.append(command)
        ray = Path(env["RAY_TMPDIR"])
        (ray / "raylet.out").write_text("retained log")
        (ray / "object-store").write_bytes(b"discard temporary object")
        cfg, spec = load_batch([command[command.index("--queue") + 1]]).pairs[0]
        batch = Path(command[command.index("--queue-output") + 1]) / "batch"
        batch.mkdir(parents=True)
        filename = "0000-example.json"
        artifact = batch / "artifacts/0000-example"
        opts = replace(spec.pipeline.client_guard, runtime_directory=str(artifact / "client-guard"))
        runtime = prepare_guard(opts, f"training:{cfg.seed}:True", [np.ones(2)], rounds=1)
        runtime.authorize([np.ones(2)], 0, 1)
        result = {"run_id": "example", "status": "complete", "implementation_fingerprint": implementation_fingerprint(),
                  "pipeline": spec.pipeline.metadata(), "guard_events": read_guard_events(artifact / "client-guard"),
                  "attack_trials": [{"target_sha256": sha256(f"target-{cfg.seed}".encode()).hexdigest()}]}
        write_json(batch / filename, result); write_json(artifact / "result.json", result)
        write_json(batch / "manifest.json", {"status": "complete", "entries": [{"status": "complete", "result_file": filename,
            "source_sha256": sha256(Path(command[command.index("--queue") + 1]).read_bytes()).hexdigest()}]})
    monkeypatch.setattr(collect_guard_traces.subprocess, "run", run)
    for _ in range(2):
        collect_guard_traces.execute(out / "launch.json", gpu="0", scratch_root=scratch)
    state = json.loads((out / "progress.json").read_bytes())
    assert len(calls) == 2 and [r["job"] for r in state["completed"]] == [0, 1]
    assert state["active_job"] is None
    assert list(scratch.iterdir()) == [unrelated]
    assert len(list(out.rglob("ray-logs.tar.gz"))) == 2
    assert len(list(out.rglob("release-ledger.sqlite"))) == 2
    assert not list(out.rglob("*.npz"))
    assert len(json.loads((out / "splits.partial.json").read_bytes())["sources"]) == 2


@pytest.mark.parametrize("missing", [False, True])
def test_rag_passes_cannot_hide_no_context_failure_or_missing_data(missing):
    protocol = json.loads((collect_guard_traces.ROOT / "guard/study_protocol_v3.json").read_bytes())
    def row(condition):
        e = {"trial_id": 0, "no_retrieval_utility": {"token_f1": .02, "exact_match": 0.},
             "rag_conditions": {k: {"utility": {"token_f1": .4, "exact_match": .2}}
                                for k in ("public_ordinary", "private_ordinary")}}
        if missing and condition == "rules_only":
            e.pop("no_retrieval_utility")
        return {"run_id": condition, "pipeline": {"condition": condition}, "status": "complete",
                "config": {}, "attack_name": "amia", "pipeline_evaluations": [e]}
    report = utility_report([row("baseline"), row("rules_only")], protocol)["runs"][0]
    assert not report["all_measured_utility_gates_pass"]
    assert report["endpoints"]["no_retrieval"]["status"] == ("unmeasured" if missing else "fail")
    assert all(report["endpoints"][e]["status"] == "pass" for e in ("public_ordinary", "private_ordinary"))
