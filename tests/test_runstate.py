# tests/test_runstate.py
"""The run-state report (§1.1): per-run, cleared on exit, heartbeated.

The report is what lets the dashboard say what is running. Its failure modes
are all "the dashboard believes something that isn't true", so these tests are
mostly about what it says *after* things go wrong.
"""
import time

import pytest

from master_script.core import runstate
from master_script.core.runstate import RunStateReporter


class _spec:
    """Minimal AttackSpec stand-in; run_sweep hashes the config for the run_id."""

    def __init__(self, name):
        self.name = name
        self.key_fn = lambda cfg: f"key-{cfg}"


@pytest.fixture
def published(monkeypatch):
    """Capture what would be written to monitor_state."""
    seen = []
    monkeypatch.setattr(runstate, "publish_monitor_state", lambda s, **k: seen.append(s) or True)
    monkeypatch.delenv(runstate.SUPPRESS_ENV, raising=False)
    return seen


def _running(published):
    return [p["running"] for p in published if "running" in p]


# ---------- the phantom-entry fix ----------

def test_reporter_publishes_on_start_and_clears_on_end(published):
    reporter = RunStateReporter()
    reporter.on_run_start("abc", "zlib", {"model_id": "distilgpt2", "seed": 7})
    assert published[0]["running"][0]["run_id"] == "abc"
    assert published[0]["running"][0]["config"]["seed"] == 7

    reporter.on_run_end("abc")
    assert published[-1]["running"] == []
    reporter.stop()


def test_run_sweep_brackets_every_run_not_just_the_first(monkeypatch, published):
    """The old code published run 1 and never updated it; bars showed run 1 forever."""
    from master_script.core import runner

    monkeypatch.setattr(runner, "run_single_experiment", lambda config, spec, **kw: {"run_id": "x"})
    reporter = RunStateReporter()
    runner.run_sweep(
        [("cfg-a", _spec("zlib")), ("cfg-b", _spec("min_k"))],
        use_firestore=False, **reporter.hooks,
    )
    reporter.stop()

    named = [r[0]["attack"] for r in _running(published) if r]
    assert named == ["zlib", "min_k"]
    assert _running(published)[-1] == []


def test_a_failing_run_still_clears_its_run_state(monkeypatch, published):
    """A failed run clears its marker while the fail-soft sweep completes."""
    from master_script.core import runner

    def _boom(config, spec, **kw):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner, "run_single_experiment", _boom)
    monkeypatch.setattr(runner, "reset_ray_after_failure", lambda: None)
    reporter = RunStateReporter()
    results = runner.run_sweep(
        [("cfg", _spec("zlib"))], use_firestore=False, **reporter.hooks
    )
    reporter.stop()
    assert results[0]["status"] == "failed"
    assert _running(published)[-1] == []


def test_stop_reports_an_empty_running_set(published):
    """A sweep that ends any other way must not leave the last run claimed."""
    reporter = RunStateReporter()
    reporter.on_run_start("abc", "zlib", {})
    reporter.stop()
    assert _running(published)[-1] == []


# ---------- heartbeat ----------

def test_entries_carry_a_heartbeat(published):
    reporter = RunStateReporter()
    reporter.on_run_start("abc", "zlib", {})
    assert published[0]["running"][0]["heartbeat_unix"] > 0
    reporter.stop()


def test_heartbeat_does_not_reset_the_elapsed_timer(published):
    """Re-stamping started_unix on every beat would peg elapsed at ~0."""
    reporter = RunStateReporter(interval=0.02)
    reporter.on_run_start("abc", "zlib", {})
    started = published[0]["running"][0]["started_unix"]
    time.sleep(0.12)
    beats = [r for r in _running(published) if r]
    reporter.stop()

    assert len(beats) > 1, "the reporter never beat after the initial publish"
    assert {b[0]["started_unix"] for b in beats} == {started}
    assert beats[-1][0]["heartbeat_unix"] >= beats[0][0]["heartbeat_unix"]


def test_heartbeat_survives_a_publish_failure(monkeypatch, published):
    """A dropped report degrades the reader to 'stale'; it must not kill the sweep."""
    def _explode(state, **kw):
        raise RuntimeError("firestore unavailable")

    monkeypatch.setattr(runstate, "publish_monitor_state", _explode)
    reporter = RunStateReporter(interval=0.02)
    reporter.on_run_start("abc", "zlib", {})  # must not raise
    time.sleep(0.06)
    reporter.stop()


# ---------- concurrency (--max-parallel 2) ----------

def test_two_concurrent_runs_are_both_reported(published):
    """One writer, one array: the second run must not clobber the first."""
    reporter = RunStateReporter()
    reporter.on_run_start("run-a", "zlib", {})
    reporter.on_run_start("run-b", "min_k", {})

    assert {e["run_id"] for e in published[-1]["running"]} == {"run-a", "run-b"}
    reporter.on_run_end("run-a")
    assert [e["run_id"] for e in published[-1]["running"]] == ["run-b"]
    reporter.stop()


def test_children_of_a_parallel_sweep_do_not_publish(monkeypatch, published):
    """GPU-pinned children are suppressed so the parent stays the only writer."""
    monkeypatch.setenv(runstate.SUPPRESS_ENV, "1")
    reporter = RunStateReporter()
    reporter.on_run_start("abc", "zlib", {})
    reporter.on_run_end("abc")
    assert published == []
    assert runstate.publish_manifest([("cfg", _spec("zlib"))]) is False


def test_manifest_publishes_one_entry_per_planned_run(published):
    """The manifest is what gives the monitor real sweep denominators (§2.2)."""
    runstate.publish_manifest([("a", _spec("zlib")), ("b", _spec("min_k"))])
    assert published[-1]["manifest"] == [
        {"run_id": "key-a", "attack": "zlib"},
        {"run_id": "key-b", "attack": "min_k"},
    ]
