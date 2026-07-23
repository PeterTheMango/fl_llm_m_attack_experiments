"""Configure and launch sweeps from the browser.

Calls core.runner.run_sweep -- the same function perform_experiments.py calls.
No experiment logic lives here, so the UI cannot drift from the CLI.
"""
import threading
import time
from typing import List, Optional

from nicegui import ui

from ..core.config import experiment_key
from ..core.firestore import publish_monitor_state
from ..core.registry import ATTACKS
from ..core.runner import run_sweep
from ..core.yaml_config import ConfigError, load_config_file
from ..paths import CONFIGS_DIR


def publish_running(run_id: str, attack: str, config: dict) -> bool:
    """Publish the optional run-state report (§1.1). Coarse and transient."""
    return publish_monitor_state({
        "running": [{
            "run_id": run_id,
            "attack": attack,
            "started_unix": int(time.time()),
            "config": {k: config.get(k) for k in ("model_id", "federated_rounds", "seed", "num_clients")},
        }]
    })


class SweepWorker:
    """Runs one sweep on a background thread. At most one at a time."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self.results: List[dict] = []
        self.error: str = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, pairs, **kwargs) -> bool:
        if self.is_running:
            return False
        self.error = ""
        self.results = []

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


WORKER = SweepWorker()


def render(state) -> None:
    """Launch panel: pick a config, pick attacks, start."""
    with ui.card().classes("w-full"):
        ui.label("Launch a sweep").classes("text-lg font-bold")
        config_files = sorted(p.name for p in CONFIGS_DIR.glob("*.yaml"))
        config_select = ui.select(config_files, value=config_files[0] if config_files else None,
                                  label="Config file")
        attack_select = ui.select(sorted(ATTACKS), multiple=True, label="Attacks (default: all in config)")
        firestore_switch = ui.switch("Persist to Firestore", value=True)
        status = ui.label("")

        def _start():
            try:
                pairs = load_config_file(
                    CONFIGS_DIR / config_select.value, only=attack_select.value or None
                )
            except ConfigError as exc:
                status.set_text(f"Config error: {exc}")
                return
            if not WORKER.start(pairs, use_firestore=firestore_switch.value):
                status.set_text("A sweep is already running.")
                return
            for cfg, spec in pairs[:1]:
                from dataclasses import asdict

                publish_running(experiment_key(cfg, spec), spec.name, asdict(cfg))
            status.set_text(f"Started {len(pairs)} run(s).")

        ui.button("Start sweep", on_click=_start)
