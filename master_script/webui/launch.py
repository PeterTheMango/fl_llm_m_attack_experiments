"""Configure and launch sweeps from the browser.

Calls core.runner.run_sweep -- the same function perform_experiments.py calls.
No experiment logic lives here, so the UI cannot drift from the CLI.
"""
import threading
import time
from typing import List, Optional

from ..core.registry import ATTACKS
from ..core.runner import run_sweep
from ..core.runstate import RunStateReporter, publish_manifest
from ..core.yaml_config import ConfigError, load_config_file
from ..paths import CONFIGS_DIR

class SweepWorker:
    """Runs one sweep on a background thread. At most one at a time."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self.results: List[dict] = []
        self.error: str = ""
        self.planned: int = 0
        self.started_unix: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, pairs, **kwargs) -> bool:
        if self.is_running:
            return False
        self.error = ""
        self.results = []
        pairs = list(pairs)
        self.planned = len(pairs)
        self.started_unix = int(time.time())

        def _work():
            try:
                self.results = run_sweep(pairs, **kwargs)
            except Exception as exc:
                self.error = str(exc)

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        """In-progress runs are never resumed (§1.2); this just drops the handle."""
        self._thread = None

    @property
    def status(self) -> dict:
        return {
            "running": self.is_running,
            "planned": self.planned,
            "finished": len(self.results),
            "error": self.error,
            "started_unix": self.started_unix,
        }


WORKER = SweepWorker()
REPORTER = RunStateReporter()


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

    # Per-run reporting: every run announces itself and clears on the way out,
    # so the running set is the run actually in flight rather than run 1 forever.
    hooks = REPORTER.hooks if use_firestore else {}

    if not WORKER.start(pairs, use_firestore=use_firestore, **hooks):
        return {"ok": False, "message": "A sweep is already running."}

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
