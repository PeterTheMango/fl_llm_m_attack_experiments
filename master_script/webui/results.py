# master_script/webui/results.py
"""Results page: filterable grid, aggregates, comparison plots, run detail.

Thin by design (spec §3): all data logic lives in DashboardState / core.charts.
This module only reads state and renders it.
"""
from nicegui import ui

from ..core import charts

_TOLERANCE = 0.02

_CONFIG_FACTOR_KEYS = ("model_id", "federated_rounds", "num_clients", "local_epochs", "client_lr", "seed")


def privacy_direction(baseline_adv: float, current_adv: float) -> str:
    """§3.4: report the direction the runs imply. Makes no privacy claim of its own."""
    delta = current_adv - baseline_adv
    if abs(delta) <= _TOLERANCE:
        return f"Privacy unchanged (ΔAdv = {delta:+.3f})"
    if delta < 0:
        return f"Privacy improved — attack advantage fell (ΔAdv = {delta:+.3f})"
    return f"Privacy declined — attack advantage rose (ΔAdv = {delta:+.3f})"


def _render_grid(state, filters: dict) -> None:
    rows_data = state.filtered(**filters)
    columns = [
        {"name": "attack_name", "label": "Attack", "field": "attack_name", "sortable": True},
        {"name": "run_id", "label": "Run ID", "field": "run_id", "sortable": True},
        {"name": "status", "label": "Status", "field": "status", "sortable": True},
        {"name": "updated_at", "label": "Updated", "field": "updated_at", "sortable": True},
        {"name": "config", "label": "Config", "field": "config", "sortable": True},
        {"name": "adv", "label": "Adv", "field": "adv", "sortable": True},
        {"name": "tpr", "label": "TPR", "field": "tpr", "sortable": True},
        {"name": "tnr", "label": "TNR", "field": "tnr", "sortable": True},
        {"name": "num_trials", "label": "# Trials", "field": "num_trials", "sortable": True},
    ]
    rows = []
    for run in rows_data:
        cfg = run.get("config", {})
        metrics = run.get("metrics") or {}
        rows.append({
            "attack_name": cfg.get("attack_name", "?"),
            "run_id": run.get("run_id", "?"),
            "status": run.get("status", "?"),
            "updated_at": run.get("updated_at", "-"),
            "config": ", ".join(f"{k}={cfg[k]}" for k in _CONFIG_FACTOR_KEYS if k in cfg),
            "adv": metrics.get("adv", "-"),
            "tpr": metrics.get("tpr", "-"),
            "tnr": metrics.get("tnr", "-"),
            "num_trials": metrics.get("num_trials", "-"),
        })
    ui.table(columns=columns, rows=rows, row_key="run_id").classes("w-full")


def _render_aggregates(state, filters: dict) -> None:
    ui.label("Aggregates by attack").classes("text-lg font-bold")
    aggregates = state.aggregate_by("attack_name")
    if not aggregates:
        ui.label("No aggregate data yet.")
        return
    columns = [
        {"name": "attack_name", "label": "Attack", "field": "attack_name"},
        {"name": "mean_adv", "label": "Mean Adv", "field": "mean_adv"},
        {"name": "min_adv", "label": "Min Adv", "field": "min_adv"},
        {"name": "max_adv", "label": "Max Adv", "field": "max_adv"},
        {"name": "count", "label": "Count", "field": "count"},
    ]
    rows = [
        {
            "attack_name": key,
            "mean_adv": f"{vals['mean_adv']:.3f}",
            "min_adv": f"{vals['min_adv']:.3f}",
            "max_adv": f"{vals['max_adv']:.3f}",
            "count": vals["count"],
        }
        for key, vals in aggregates.items()
    ]
    ui.table(columns=columns, rows=rows, row_key="attack_name").classes("w-full")


def _render_comparison_plot(state, filters: dict, factor: str) -> None:
    ui.label(f"Adv vs. {factor}").classes("text-lg font-bold")
    filtered = state.filtered(**filters)
    if not filtered:
        ui.label("No data for the active filter.")
        return
    # Reuse core.charts rendering logic (writes a PNG), then display it.
    try:
        path = charts.render_adv_by_factor(filtered, factor)
        ui.image(str(path)).classes("w-full max-w-2xl")
    except Exception as exc:  # keep page usable even if plotting fails
        ui.label(f"Plot unavailable: {exc}")


def render(state) -> None:
    """Render the results page (§3.1-3.2): grid, filters, aggregates, comparison plot."""
    ui.label("Results").classes("text-2xl font-bold")

    filters: dict = {}

    grid_panel = ui.refreshable(_render_grid)
    aggregates_panel = ui.refreshable(_render_aggregates)
    plot_panel = ui.refreshable(_render_comparison_plot)

    def _refresh_all() -> None:
        grid_panel.refresh(state, filters)
        aggregates_panel.refresh(state, filters)
        plot_panel.refresh(state, filters, "federated_rounds")

    with ui.row():
        attack_select = ui.select(
            options=sorted({r.get("config", {}).get("attack_name") for r in state.runs.values()
                             if r.get("config", {}).get("attack_name")}),
            label="Attack", with_input=True, clearable=True,
        )
        status_select = ui.select(
            options=["complete", "failed", "running"], label="Status", clearable=True,
        )

        def _on_attack_change() -> None:
            filters["attack"] = attack_select.value or None
            if filters["attack"] is None:
                filters.pop("attack", None)
            _refresh_all()

        def _on_status_change() -> None:
            filters["status"] = status_select.value or None
            if filters["status"] is None:
                filters.pop("status", None)
            _refresh_all()

        attack_select.on_value_change(lambda _: _on_attack_change())
        status_select.on_value_change(lambda _: _on_status_change())

    grid_panel(state, filters)
    aggregates_panel(state, filters)
    plot_panel(state, filters, "federated_rounds")


def render_detail(state, run_id: str) -> None:
    """Render the per-run detail page (§3.3)."""
    run = state.runs.get(run_id)
    ui.label("Run detail").classes("text-2xl font-bold")

    if run is None:
        ui.label(f"No run found for run_id={run_id!r}.")
        return

    cfg = run.get("config", {})
    metrics = run.get("metrics") or {}

    # Header
    ui.label(f"{cfg.get('attack_name', '?')} — {run_id}").classes("text-xl font-bold")
    ui.label(f"Status: {run.get('status', '?')}")
    ui.label(f"Updated: {run.get('updated_at', '-')}")
    with ui.expansion("Full config"):
        ui.json_editor({"content": {"json": cfg}}) if hasattr(ui, "json_editor") else ui.label(str(cfg))

    # Methodology block
    methodology = run.get("methodology") or cfg.get("methodology")
    if methodology:
        with ui.expansion("Methodology"):
            ui.label(str(methodology))

    # Headline metrics
    with ui.row():
        ui.label(f"Adv: {metrics.get('adv', '-')}")
        ui.label(f"TPR: {metrics.get('tpr', '-')}")
        ui.label(f"TNR: {metrics.get('tnr', '-')}")
        ui.label(f"# Trials: {metrics.get('num_trials', '-')}")

    # federated_history per-round curve
    history = run.get("federated_history")
    if history:
        ui.label("Federated history").classes("text-lg font-bold")
        rows = [{"round": i, **h} if isinstance(h, dict) else {"round": i, "value": h}
                for i, h in enumerate(history)]
        ui.table(rows=rows).classes("w-full")

    # attack_trials table
    trials = run.get("attack_trials")
    if trials:
        ui.label("Attack trials").classes("text-lg font-bold")
        ui.table(rows=trials).classes("w-full")

        ui.label("Score distribution by true membership").classes("text-lg font-bold")
        try:
            path = charts.render_score_distribution(run)
            ui.image(str(path)).classes("w-full max-w-2xl")
        except Exception as exc:
            ui.label(f"Plot unavailable: {exc}")

    # ROC view (best-effort; relies on whatever ROC data the run carries)
    roc = run.get("roc") or metrics.get("roc")
    if roc:
        ui.label("ROC").classes("text-lg font-bold")
        ui.label(str(roc))

    # Artifact paths
    artifacts = run.get("artifacts")
    if artifacts:
        ui.label("Artifacts").classes("text-lg font-bold")
        ui.label("Paths may point at cleaned-up local paths or gs:// URIs.")
        for artifact in artifacts if isinstance(artifacts, list) else [artifacts]:
            ui.label(str(artifact))
