import time

from master_script.webui.launch import SweepWorker


def test_worker_calls_core_run_sweep_not_a_private_copy(monkeypatch):
    """The UI must reuse the CLI's code path."""
    called = {}

    import master_script.webui.launch as mod

    def _fake(pairs, **kw):
        called["pairs"] = list(pairs)
        return [{"run_id": "x", "status": "complete"}]

    monkeypatch.setattr(mod, "run_sweep", _fake)
    w = SweepWorker()
    w.start([("cfg", "spec")])
    for _ in range(100):
        if not w.is_running:
            break
        time.sleep(0.01)
    assert called["pairs"] == [("cfg", "spec")]


def test_worker_reports_not_running_before_start():
    assert SweepWorker().is_running is False


def test_worker_refuses_concurrent_sweeps(monkeypatch):
    import master_script.webui.launch as mod

    monkeypatch.setattr(mod, "run_sweep", lambda pairs, **kw: time.sleep(0.2) or [])
    w = SweepWorker()
    w.start([("a", "b")])
    try:
        assert w.start([("c", "d")]) is False
    finally:
        w.cancel()


def test_publish_running_writes_monitor_state(monkeypatch):
    import master_script.webui.launch as mod

    published = {}
    monkeypatch.setattr(mod, "publish_monitor_state", lambda s, **k: published.update(s) or True)
    mod.publish_running("abc", "zlib", {"model_id": "distilgpt2", "seed": 7})
    assert published["running"][0]["run_id"] == "abc"
    assert published["running"][0]["attack"] == "zlib"


def test_worker_records_error_without_crashing(monkeypatch):
    import master_script.webui.launch as mod

    def _boom(pairs, **kw):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(mod, "run_sweep", _boom)
    w = SweepWorker()
    w.start([("a", "b")])
    for _ in range(100):
        if not w.is_running:
            break
        time.sleep(0.01)
    assert "sweep exploded" in w.error
