import errno
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from master_script.core import storage, queue, runner
from master_script.core.guard_runtime import prepare_guard


def settings(tmp_path):
    return SimpleNamespace(runtime_directory=str(tmp_path))


def test_space_check_accounts_for_payload_and_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=storage.MIN_FREE_BYTES + 10))
    storage.require_space(tmp_path, 10)
    with pytest.raises(storage.StorageCapacityError, match="Insufficient disk space"):
        storage.require_space(tmp_path, 11)


def test_partial_guard_write_leaves_no_published_snapshot(monkeypatch, tmp_path):
    def full(stream, *args):
        stream.write(b"partial zip")
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(np, "savez", full)
    with pytest.raises(OSError):
        prepare_guard(settings(tmp_path), "one", [np.ones(3)], rounds=1)
    assert list(tmp_path.iterdir()) == []


def test_existing_pin_is_validated_not_overwritten(tmp_path):
    runtime = prepare_guard(settings(tmp_path), "one", [np.ones(3)], rounds=1)
    original = Path(runtime.reference_path).read_bytes()
    with pytest.raises(ValueError, match="Invalid or changed"):
        prepare_guard(settings(tmp_path), "one", [np.zeros(3)], rounds=1)
    assert Path(runtime.reference_path).read_bytes() == original
    Path(runtime.reference_path).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="partial snapshot"):
        prepare_guard(settings(tmp_path), "one", [np.ones(3)], rounds=1)


def test_atomic_json_keeps_previous_file_on_disk_full(monkeypatch, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"status":"complete"}')
    monkeypatch.setattr(storage.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    with pytest.raises(OSError):
        queue.write_json(path, {"status": "failed"})
    assert json.loads(path.read_text())["status"] == "complete"
    assert list(tmp_path.iterdir()) == [path]


def test_storage_failure_stops_queue_and_marks_remaining_not_run(monkeypatch, tmp_path):
    from tests.test_queue import configs
    batch = queue.load_batch(configs(tmp_path))
    called = []
    def run(cfg, spec, **kwargs):
        called.append(spec.name)
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(runner, "run_single_experiment", run)
    monkeypatch.setattr(runner, "reset_ray_after_failure", lambda: None)
    with pytest.raises(OSError):
        queue.run_batch(batch, output_root=tmp_path / "queues", use_firestore=False)
    manifest = json.loads(next((tmp_path / "queues").glob("*/manifest.json")).read_text())
    assert called == ["zlib"]
    assert manifest["status"] == "storage_exhausted"
    assert [e["status"] for e in manifest["entries"]] == ["storage_exhausted", "not_run", "not_run"]
    assert not list((tmp_path / "queues").rglob(".manifest-reserve"))


def test_cleanup_preserves_results_and_ledger_and_failed_runs(tmp_path):
    from master_script.tools.cleanup_queue_models import cleanup, candidates
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "entries": [{"index": 0, "run_id": "ok", "status": "complete"}, {"index": 1, "run_id": "bad", "status": "failed"}]}))
    keep = []
    remove = []
    for run in ("0000-ok", "0001-bad"):
        root = tmp_path / "artifacts" / run
        for name in ("federated_model/model.safetensors", "client-guard/pin.npz", "client-guard/release-ledger.sqlite", "result.json", "observations/trial-0.json"):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("saved")
            (remove if run == "0000-ok" and path.suffix in (".safetensors", ".npz") else keep).append(path)
    cleanup(tmp_path)
    assert all(p.exists() for p in keep + remove)
    cleanup(tmp_path, apply=True)
    assert all(p.exists() for p in keep)
    assert not any(p.exists() for p in remove)
    manifest = json.loads((tmp_path / "manifest.json").read_text()); manifest["status"] = "running"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not finalized"):
        candidates(tmp_path)
