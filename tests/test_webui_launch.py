import queue
import pickle

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


class _Messages:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


class _Process:
    def __init__(self, target, args, name):
        self.target, self.args, self.name = target, args, name
        self.pid = 4242
        self.exitcode = None
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.alive = False
        self.exitcode = -15

    kill = terminate


class _Context:
    def __init__(self):
        self.messages = _Messages()
        self.process = None

    def Queue(self):
        return self.messages

    def Process(self, **kwargs):
        self.process = _Process(**kwargs)
        return self.process


def test_child_calls_core_run_sweep_not_a_private_copy(monkeypatch):
    """The UI must reuse the CLI's code path."""
    called = {}

    import master_script.webui.launch as mod

    def _fake(pairs, **kw):
        called["pairs"] = list(pairs)
        return [{"run_id": "x", "status": "complete"}]

    monkeypatch.setattr(mod, "run_sweep", _fake)
    monkeypatch.setattr(mod, "setup_session_logging", lambda level: None)
    monkeypatch.setattr(mod.os, "setsid", lambda: None)
    messages = _Messages()
    mod._run_in_child([("cfg", "spec")], False, None, messages)

    assert called["pairs"] == [("cfg", "spec")]
    assert messages.items[0]["results"] == [{"run_id": "x", "status": "complete"}]


def test_worker_reports_not_running_before_start():
    assert SweepWorker().is_running is False


def test_worker_refuses_concurrent_sweeps(monkeypatch):
    w = SweepWorker(context=_Context())
    w.start([("a", "b")])
    try:
        assert w.start([("c", "d")]) is False
    finally:
        monkeypatch.setattr(w, "_terminate_process", lambda process: process.terminate())
        w.cancel()


def test_worker_records_error_without_crashing(monkeypatch):
    import master_script.webui.launch as mod

    def _boom(pairs, **kw):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(mod, "run_sweep", _boom)
    monkeypatch.setattr(mod, "setup_session_logging", lambda level: None)
    monkeypatch.setattr(mod.os, "setsid", lambda: None)
    messages = _Messages()
    with pytest.raises(RuntimeError, match="sweep exploded"):
        mod._run_in_child([("a", "b")], False, None, messages)
    assert "sweep exploded" in messages.items[0]["error"]


def test_worker_stop_terminates_the_owned_process(monkeypatch):
    context = _Context()
    worker = SweepWorker(context=context)
    assert worker.start([("a", "b")], use_firestore=False) is True
    monkeypatch.setattr(worker, "_terminate_process", lambda process: process.terminate())

    assert worker.stop() is True
    assert worker.status["running"] is False
    assert worker.status["stopped"] is True


def test_real_sweep_pairs_are_spawn_pickleable():
    """The stoppable worker uses multiprocessing spawn, not a fork-only trick."""
    from master_script.core.yaml_config import load_config_file
    from master_script.paths import CONFIGS_DIR

    pairs = load_config_file(CONFIGS_DIR / "smoke.yaml", only=["zlib"])

    assert pickle.loads(pickle.dumps(pairs))[0][1].name == "zlib"


def test_stop_sweep_clears_the_published_run_state(monkeypatch):
    import master_script.webui.launch as mod

    context = _Context()
    worker = SweepWorker(context=context)
    worker.start([("a", "b")], use_firestore=True)
    monkeypatch.setattr(worker, "_terminate_process", lambda process: process.terminate())
    published = {}
    monkeypatch.setattr(mod, "WORKER", worker)
    monkeypatch.setattr(mod, "publish_monitor_state", lambda state: published.update(state) or True)

    result = mod.stop_sweep()

    assert result["ok"] is True
    assert published == {"running": [], "manifest": []}


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


def test_a_spawn_failure_clears_the_manifest_it_published(monkeypatch):
    import master_script.webui.launch as mod

    published = []
    monkeypatch.setattr(type(mod.WORKER), "is_running", property(lambda self: False))
    monkeypatch.setattr(mod.WORKER, "start", lambda *a, **k: False)
    monkeypatch.setattr(mod.WORKER, "error", "spawn failed")
    monkeypatch.setattr(mod, "publish_manifest", lambda pairs: published.append("manifest") or True)
    monkeypatch.setattr(mod, "publish_monitor_state", lambda state: published.append(state) or True)

    result = mod._start([("cfg", _spec("zlib"))], True, "empty")

    assert result == {"ok": False, "message": "spawn failed"}
    assert published == ["manifest", {"running": [], "manifest": []}]
