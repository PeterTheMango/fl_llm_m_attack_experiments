# tests/test_webui_pages.py
"""Wiring tests. Data logic is tested in tests/test_webui_state.py."""
from master_script.webui import app, monitor, results
from master_script.webui.state import DashboardState


def test_pages_expose_render_entrypoints():
    assert callable(monitor.render)
    assert callable(results.render)
    assert callable(results.render_detail)


def test_app_registers_three_routes():
    assert sorted(app.ROUTES) == ["/", "/results", "/tunnel"]


def test_privacy_direction_label_reads_lower_adv_as_improved():
    """§3.4: lower Adv within an environment => privacy improved."""
    assert "improved" in results.privacy_direction(baseline_adv=0.9, current_adv=0.6).lower()
    assert "declined" in results.privacy_direction(baseline_adv=0.6, current_adv=0.9).lower()


def test_privacy_direction_reports_unchanged_within_tolerance():
    assert "unchanged" in results.privacy_direction(0.7, 0.7).lower()


def test_monitor_reports_unavailable_running_set_without_manifest():
    """§2.4: say so rather than guessing."""
    s = DashboardState()
    s.ingest([{"run_id": "a", "status": "complete", "config": {}, "metrics": {"adv": 1.0}}])
    assert monitor.running_set_available(s) is False
