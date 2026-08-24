"""Configure and launch sweeps from the browser.

Calls core.runner.run_sweep -- the same function perform_experiments.py calls.
No experiment logic lives here, so the UI cannot drift from the CLI.
"""
import atexit
import multiprocessing
import os
import queue
import signal
import time
from typing import List, Optional

from ..core.firestore import publish_monitor_state
from ..core.registry import ATTACKS
from ..core.runner import run_sweep
from ..core.runstate import RunStateReporter, publish_manifest
from ..core.yaml_config import ConfigError, load_config_file
from ..logging_setup import setup_session_logging
from ..paths import CONFIGS_DIR


def _run_in_child(pairs, use_firestore: bool, keep_artifacts, messages) -> None:
    """Run the sweep in an isolated process that the dashboard can terminate.

    A new POSIX process group keeps descendants such as training workers inside
    the same stop boundary. The child owns run-state heartbeats because locks
    and Firestore clients must never be passed through multiprocessing spawn.
    """
    if os.name == "posix":
        os.setsid()

    setup_session_logging("INFO")
    reporter = RunStateReporter() if use_firestore else None
    try:
        completed = run_sweep(
            pairs,
            use_firestore=use_firestore,
            keep_artifacts=keep_artifacts,
            **(reporter.hooks if reporter else {}),
        )
        # Result documents can be large. The parent only needs session status;
        # Firestore remains the authoritative results layer.
        messages.put({
            "type": "done",
            "results": [
                {"run_id": result.get("run_id"), "status": result.get("status")}
                for result in completed if isinstance(result, dict)
            ],
        })
    except BaseException as exc:
        messages.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if reporter is not None:
            reporter.stop()


class SweepWorker:
    """Runs one sweep in a stoppable child process. At most one at a time."""

    def __init__(self, context=None) -> None:
        # spawn avoids forking a threaded web server or an initialized CUDA
        # runtime. Pairs contain module-level dataclasses/functions and are
        # intentionally spawn-pickleable.
        self._context = context or multiprocessing.get_context("spawn")
        self._process = None
        self._messages = None
        self._use_firestore = False
        self.results: List[dict] = []
        self.error: str = ""
        self.planned: int = 0
        self.started_unix: Optional[int] = None
        self.stopped_unix: Optional[int] = None
        self.was_stopped: bool = False

    def _drain_messages(self) -> None:
        if self._messages is None:
            return
        while True:
            try:
                message = self._messages.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            if message.get("type") == "done":
                self.results = message.get("results") or []
            elif message.get("type") == "error":
                self.error = message.get("error") or "Experiment process failed."

    def _sync(self) -> None:
        self._drain_messages()
        process = self._process
        if process is None or process.is_alive():
            return
        process.join(timeout=0)
        self._drain_messages()
        if process.exitcode not in (None, 0) and not self.was_stopped and not self.error:
            self.error = f"Experiment process exited with code {process.exitcode}."

    def _close_messages(self) -> None:
        if self._messages is not None and hasattr(self._messages, "close"):
            try:
                self._messages.close()
            except (OSError, ValueError):
                pass

    @property
    def is_running(self) -> bool:
        self._sync()
        return self._process is not None and self._process.is_alive()

    def start(self, pairs, *, use_firestore: bool = True, keep_artifacts=None) -> bool:
        if self.is_running:
            return False
        self._close_messages()
        self.error = ""
        self.results = []
        pairs = list(pairs)
        self.planned = len(pairs)
        self.started_unix = int(time.time())
        self.stopped_unix = None
        self.was_stopped = False
        self._use_firestore = use_firestore
        self._messages = self._context.Queue()
        self._process = self._context.Process(
            target=_run_in_child,
            args=(pairs, use_firestore, keep_artifacts, self._messages),
            name="canary-experiment-sweep",
        )
        try:
            self._process.start()
        except Exception as exc:
            self.error = f"Could not start experiment process: {exc}"
            self._process = None
            self._close_messages()
            return False
        return True

    @property
    def uses_firestore(self) -> bool:
        return self._use_firestore

    @staticmethod
    def _terminate_process(process, grace_seconds: float = 5.0) -> None:
        """Terminate the owned process group, escalating only if it ignores TERM."""
        sent_to_group = False
        if os.name == "posix" and process.pid:
            try:
                group_id = os.getpgid(process.pid)
                # The child calls setsid(). Never signal a group it has not yet
                # isolated, or the dashboard itself could share the target.
                if group_id == process.pid:
                    os.killpg(group_id, signal.SIGTERM)
                    sent_to_group = True
            except ProcessLookupError:
                return
            except OSError:
                pass
        if not sent_to_group:
            process.terminate()

        process.join(timeout=grace_seconds)
        if not process.is_alive():
            return

        if sent_to_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        elif hasattr(process, "kill"):
            process.kill()
        else:
            process.terminate()
        process.join(timeout=grace_seconds)

    def stop(self) -> bool:
        """Stop the active sweep and every child in its owned process group."""
        if not self.is_running:
            return False
        process = self._process
        self.was_stopped = True
        self.stopped_unix = int(time.time())
        self.error = ""
        self._terminate_process(process)
        self._sync()
        return not process.is_alive()

    def cancel(self) -> None:
        """Backward-compatible cleanup hook used by older callers/tests."""
        self.stop()

    @property
    def status(self) -> dict:
        running = self.is_running
        return {
            "running": running,
            "planned": self.planned,
            "finished": len(self.results),
            "error": self.error,
            "started_unix": self.started_unix,
            "stopped": self.was_stopped and not running,
            "stopped_unix": self.stopped_unix,
        }


WORKER = SweepWorker()
atexit.register(WORKER.stop)


def available_configs() -> List[str]:
    return sorted(p.name for p in CONFIGS_DIR.glob("*.yaml"))


def launch_payload() -> dict:
    from . import configs

    return {
        "configs": available_configs(),
        "attacks": sorted(ATTACKS),
        "worker": WORKER.status,
        "template": configs.TEMPLATE,
    }


def _start(pairs, use_firestore: bool, empty_message: str) -> dict:
    """Publish the plan and hand the pairs to the worker. Never raises."""
    if not pairs:
        # Starting nothing must not read as success.
        return {"ok": False, "message": empty_message}

    # Checked before publish_manifest: a manifest for a sweep that never starts
    # would give the monitor a denominator for runs that will never arrive.
    if WORKER.is_running:
        return {"ok": False, "message": "A sweep is already running."}

    # The plan goes out before the first run, so the sweep bars have a
    # denominator from the moment the first run appears.
    if use_firestore:
        publish_manifest(pairs)

    if not WORKER.start(pairs, use_firestore=use_firestore):
        if use_firestore:
            try:
                publish_monitor_state({"running": [], "manifest": []})
            except Exception:
                pass
        return {"ok": False, "message": WORKER.error or "A sweep is already running."}

    return {"ok": True, "message": f"Started {len(pairs)} run(s).", "planned": len(pairs)}


def start_sweep(config_file: str, attacks: Optional[List[str]] = None,
                use_firestore: bool = True) -> dict:
    """Load a config file, publish the plan, and start the sweep."""
    try:
        pairs = load_config_file(CONFIGS_DIR / config_file, only=attacks or None)
    except ConfigError as exc:
        return {"ok": False, "message": f"Config error: {exc}"}

    # Selecting an attack the config doesn't define must not read as success.
    return _start(pairs, use_firestore, (
        f"No runs to start: {config_file} defines none of the selected attack(s)."
    ))


def start_manual(payload: dict, use_firestore: bool = True) -> dict:
    """Start a sweep from the manual form, with no file in between."""
    from . import manual

    try:
        pairs = manual.pairs(payload)
    except (manual.ManualError, ConfigError) as exc:
        return {"ok": False, "message": f"Config error: {exc}"}

    return _start(pairs, use_firestore, "No runs to start: the manual configuration is empty.")


def stop_sweep() -> dict:
    """Stop the dashboard-owned sweep. External CLI processes are never guessed."""
    used_firestore = WORKER.uses_firestore
    if not WORKER.stop():
        return {"ok": False, "message": "No dashboard-launched sweep is running.",
                "worker": WORKER.status}

    if used_firestore:
        # A hard termination cannot run the child's finally block. Clear both
        # transient fields so the dashboard does not show a ghost or pending
        # work from a sweep the user explicitly abandoned.
        try:
            publish_monitor_state({"running": [], "manifest": []})
        except Exception:
            pass
    return {"ok": True, "message": "Stopped the active sweep and its child processes.",
            "worker": WORKER.status}
