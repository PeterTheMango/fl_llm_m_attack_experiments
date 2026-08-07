# master_script/core/runstate.py
"""The optional run-state report (§1.1): which runs are in flight, right now.

Coarse and transient. Never used to resume or reconstruct a run (§1.2) -- it
exists so the dashboard can say what is running instead of guessing.

Two properties make it trustworthy:

* **Per run, cleared on exit.** Runs announce themselves and clear on the way
  out, including when they raise, so the set is what is actually in flight.
* **Heartbeat.** A hard kill never gets to clear anything, so a live entry is
  re-stamped periodically. A reader ages out entries whose heartbeat stopped
  (see webui/monitor.py), which is the half that survives SIGKILL.

One writer per process. `--max-parallel 2` keeps the writer in the *parent*
and suppresses it in the GPU-pinned children (SUPPRESS_ENV), because the
report is a single array: two processes publishing would clobber each other.
"""
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional
import os
import threading
import time

from .firestore import publish_monitor_state

# How often a live entry is re-stamped. The reader's staleness threshold is a
# multiple of this (webui/monitor.STALE_AFTER_SECONDS).
HEARTBEAT_SECONDS = 30

# Set in GPU-pinned children so only the parent reports for a parallel sweep.
SUPPRESS_ENV = "MASTER_SCRIPT_SUPPRESS_RUN_STATE"

_STATE_CONFIG_KEYS = ("model_id", "federated_rounds", "seed", "num_clients")


def suppressed() -> bool:
    return bool(os.environ.get(SUPPRESS_ENV))


def as_dict(config) -> dict:
    """Config as a plain dict. Never raises: reporting must not break a sweep."""
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {}


def publish_manifest(pairs) -> bool:
    """Publish the planned sweep so the monitor has real denominators (§2.2)."""
    from .config import experiment_key

    if suppressed():
        return False
    return publish_monitor_state({
        "manifest": [
            {"run_id": experiment_key(cfg, spec), "attack": spec.name}
            for cfg, spec in pairs
        ]
    })


class RunStateReporter:
    """Owns monitor_state.running for this process.

    Holds every in-flight run (one for a sequential sweep, up to two for the
    parallel parent) and republishes the whole set on any change or heartbeat,
    so there is exactly one writer and no lost update.
    """

    def __init__(self, interval: int = HEARTBEAT_SECONDS) -> None:
        self.interval = interval
        self._runs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---- hooks for runner.run_sweep ----
    def on_run_start(self, run_id: str, attack: str, config, stage: str = "fine-tune") -> None:
        if suppressed():
            return
        now = int(time.time())
        payload = as_dict(config)
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "attack": attack,
                "stage": stage,
                "started_unix": now,
                "heartbeat_unix": now,
                "config": {k: payload.get(k) for k in _STATE_CONFIG_KEYS},
            }
        self._publish()
        self._ensure_beating()

    def on_run_end(self, run_id: str, attack: str = "", config=None) -> None:
        if suppressed():
            return
        with self._lock:
            self._runs.pop(run_id, None)
        self._publish()

    # ---- internals ----
    def _entries(self) -> List[dict]:
        now = int(time.time())
        with self._lock:
            # Re-stamp on the way out: the published heartbeat must be the time
            # of *this* write, not of whenever the entry was created.
            for entry in self._runs.values():
                entry["heartbeat_unix"] = now
            return [dict(e) for e in self._runs.values()]

    def _publish(self) -> bool:
        try:
            return publish_monitor_state({"running": self._entries()})
        except Exception:
            # A dropped report degrades the reader to "stale", which is the
            # honest reading. It must never take the sweep down with it.
            return False

    def _ensure_beating(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        # Idles between runs rather than exiting: a thread that exits exactly as
        # the next run starts would leave that run without a heartbeat, and the
        # reader would call a perfectly healthy long run stale.
        while not self._stop.wait(self.interval):
            with self._lock:
                idle = not self._runs
            if not idle:
                self._publish()

    def stop(self) -> None:
        """Stop beating and report an empty running set."""
        self._stop.set()
        with self._lock:
            self._runs.clear()
        if not suppressed():
            self._publish()

    @property
    def hooks(self) -> dict:
        """kwargs for run_sweep(**reporter.hooks)."""
        return {"on_run_start": self.on_run_start, "on_run_end": self.on_run_end}
