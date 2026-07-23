# master_script/webui/monitor.py
"""Live monitoring page: now-panel, sweep progress, recently-finished feed.

Thin by design (spec §2): all data logic lives in DashboardState. This module
only reads state and renders it.
"""
import time

from nicegui import ui

_DISTINGUISHING_KEYS = ("model_id", "federated_rounds", "seed")


def running_set_available(state) -> bool:
    """Without the script's run-state report, in-progress runs are unknowable (§2.4)."""
    return bool(state.running) or state.manifest is not None


def _format_elapsed(start_unix) -> str:
    if not start_unix:
        return "?"
    seconds = max(0, time.time() - start_unix)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _render_now_panel(state) -> None:
    ui.label("Currently running").classes("text-lg font-bold")
    if not running_set_available(state):
        ui.label(
            "Currently-running set unavailable — no run-state report. "
            "Showing completed/failed Firestore results only."
        ).classes("text-orange-700")
        return

    if not state.running:
        ui.label("No runs currently in progress.")
        return

    columns = [
        {"name": "attack_name", "label": "Attack", "field": "attack_name"},
        {"name": "run_id", "label": "Run ID", "field": "run_id"},
        {"name": "config", "label": "Config", "field": "config"},
        {"name": "start_time", "label": "Start time", "field": "start_time"},
        {"name": "elapsed", "label": "Elapsed", "field": "elapsed"},
        {"name": "stage", "label": "Stage", "field": "stage"},
    ]
    rows = []
    for run in state.running:
        cfg = run.get("config", {})
        distinguishing = ", ".join(
            f"{k}={cfg[k]}" for k in _DISTINGUISHING_KEYS if k in cfg
        )
        rows.append({
            "attack_name": cfg.get("attack_name", run.get("attack_name", "?")),
            "run_id": run.get("run_id", "?"),
            "config": distinguishing or "-",
            "start_time": run.get("start_time", "?"),
            "elapsed": _format_elapsed(run.get("start_time_unix")),
            "stage": run.get("stage", "-"),
        })
    ui.table(columns=columns, rows=rows, row_key="run_id").classes("w-full")


def _render_sweep_progress(state) -> None:
    ui.label("Sweep progress").classes("text-lg font-bold")
    progress = state.sweep_progress()
    if progress["total"] is None:
        ui.label(
            f"complete={progress['complete']} failed={progress['failed']} "
            f"running={progress['running']} (denominator unavailable)"
        )
    else:
        ui.label(
            f"complete={progress['complete']}/{progress['total']} "
            f"failed={progress['failed']} running={progress['running']} "
            f"pending={progress['pending']}"
        )


def _render_recently_finished(state) -> None:
    ui.label("Recently finished").classes("text-lg font-bold")
    recent = state.recently_finished(10)
    if not recent:
        ui.label("No finished runs yet.")
        return
    columns = [
        {"name": "run_id", "label": "Run ID", "field": "run_id"},
        {"name": "attack_name", "label": "Attack", "field": "attack_name"},
        {"name": "status", "label": "Status", "field": "status"},
        {"name": "adv", "label": "Adv", "field": "adv"},
    ]
    rows = []
    for run in recent:
        status = run.get("status", "?")
        metrics = run.get("metrics") or {}
        rows.append({
            "run_id": run.get("run_id", "?"),
            "attack_name": run.get("config", {}).get("attack_name", "?"),
            "status": status,
            "adv": f"{metrics['adv']:.3f}" if status == "complete" and "adv" in metrics else (
                "FAILED" if status == "failed" else "-"
            ),
        })
    ui.table(columns=columns, rows=rows, row_key="run_id").classes("w-full")


def render(state) -> None:
    """Render the live monitor page (§2): now-panel, sweep progress, recently finished."""
    ui.label("Monitor").classes("text-2xl font-bold")

    now_panel = ui.refreshable(_render_now_panel)
    sweep_panel = ui.refreshable(_render_sweep_progress)
    finished_panel = ui.refreshable(_render_recently_finished)

    now_panel(state)
    sweep_panel(state)
    finished_panel(state)
