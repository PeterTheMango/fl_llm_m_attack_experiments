import time

import pytest

from master_script.webui.launch import SweepWorker


class _spec:
    """Minimal stand-in for an AttackSpec.

    run_sweep hashes the config for the run_id, so key_fn has to be real
    enough to accept the placeholder configs these tests pass.
    """

    def __init__(self, name):
        self.name = name
        self.key_fn = lambda cfg: f"key-{cfg}"
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


def test_selecting_an_attack_the_config_lacks_is_refused_not_reported_as_started(monkeypatch):
    """A cheerful 'Started 0 run(s)' would hide the mistake."""
    import master_script.webui.launch as mod

    monkeypatch.setattr(mod, "load_config_file", lambda path, only=None: [])
    started = {}
    monkeypatch.setattr(mod.WORKER, "start", lambda *a, **k: started.setdefault("called", True))

    result = mod.start_sweep("smoke.yaml", ["not_an_attack"])
    assert result["ok"] is False
    assert "none of the selected attack" in result["message"]
    assert "called" not in started


def test_manual_start_publishes_a_manifest_and_starts_the_worker(monkeypatch):
    import master_script.webui.launch as mod

    published, started = {}, {}
    monkeypatch.setattr(mod, "publish_manifest", lambda pairs: published.setdefault("n", len(pairs)))
    monkeypatch.setattr(mod.WORKER, "start", lambda pairs, **kw: started.setdefault("n", len(pairs)) or True)
    monkeypatch.setattr(type(mod.WORKER), "is_running", property(lambda self: False))

    result = mod.start_manual({"attacks": [
        {"name": "zlib", "values": {"seed": "7, 11"}, "sweeps": ["seed"]},
    ]})
    assert result["ok"] is True and result["planned"] == 2
    assert published["n"] == 2 and started["n"] == 2


def test_manual_start_reports_a_bad_value_without_starting(monkeypatch):
    import master_script.webui.launch as mod

    started = {}
    monkeypatch.setattr(mod.WORKER, "start", lambda *a, **k: started.setdefault("called", True))

    result = mod.start_manual({"attacks": [
        {"name": "zlib", "values": {"federated_rounds": "three"}, "sweeps": []},
    ]})
    assert result["ok"] is False and "federated_rounds" in result["message"]
    assert "called" not in started


def test_manual_start_with_no_attacks_is_refused(monkeypatch):
    import master_script.webui.launch as mod

    started = {}
    monkeypatch.setattr(mod.WORKER, "start", lambda *a, **k: started.setdefault("called", True))
    result = mod.start_manual({"attacks": []})
    assert result["ok"] is False and "called" not in started


def test_a_second_start_publishes_no_manifest(monkeypatch):
    """A manifest published for a sweep that never starts would give the
    monitor a denominator for runs that will never arrive."""
    import master_script.webui.launch as mod

    published = {}
    monkeypatch.setattr(mod, "publish_manifest", lambda pairs: published.setdefault("called", True))
    monkeypatch.setattr(mod, "load_config_file", lambda path, only=None: [("cfg", "spec")])
    monkeypatch.setattr(type(mod.WORKER), "is_running", property(lambda self: True))

    result = mod.start_sweep("smoke.yaml")
    assert result["ok"] is False and "already running" in result["message"]
    assert "called" not in published
