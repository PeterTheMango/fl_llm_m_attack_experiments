# master_script/webui/app.py
"""NiceGUI server. Mounts monitor, results, and tunnel on one process.

launch (Task 16) and tunnel (Task 17) are imported lazily inside build()/main()
so that importing this module (e.g. for the route wiring test) works before
those modules exist.
"""
from nicegui import ui

from .state import DashboardState
from . import monitor, results

ROUTES = ["/", "/results", "/tunnel"]

STATE = DashboardState()


def build(state: DashboardState) -> None:
    from . import launch, tunnel

    @ui.page("/")
    def _index():
        monitor.render(state)
        launch.render(state)

    @ui.page("/results")
    def _results():
        results.render(state)

    @ui.page("/results/{run_id}")
    def _detail(run_id: str):
        results.render_detail(state, run_id)

    @ui.page("/tunnel")
    def _tunnel():
        tunnel.render()


def main() -> None:
    from ..core.firestore import get_firestore_client  # noqa: F401

    build(STATE)
    try:
        STATE.attach_listener("ami_federated_llm_results", lambda: None)
    except Exception as exc:  # degrade gracefully: no credentials, no listener
        print(f"Firestore listener unavailable ({exc}); dashboard runs empty.")
    ui.run(host="0.0.0.0", port=8080, title="Research Monitor", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
