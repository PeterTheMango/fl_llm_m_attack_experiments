# tests/test_webui_state.py
from master_script.webui.state import DashboardState

DOCS = [
    {"run_id": "a", "status": "complete", "updated_at_unix": 100,
     "config": {"attack_name": "zlib", "federated_rounds": 1, "seed": 7},
     "metrics": {"adv": 0.5, "tpr": 0.5, "tnr": 0.5}},
    {"run_id": "b", "status": "complete", "updated_at_unix": 200,
     "config": {"attack_name": "zlib", "federated_rounds": 2, "seed": 7},
     "metrics": {"adv": 1.0, "tpr": 1.0, "tnr": 1.0}},
    {"run_id": "c", "status": "failed", "updated_at_unix": 300,
     "config": {"attack_name": "min_k", "federated_rounds": 1, "seed": 7}},
]


def test_ingest_indexes_runs_by_id():
    s = DashboardState()
    s.ingest(DOCS)
    assert set(s.runs) == {"a", "b", "c"}


def test_monitor_state_doc_is_not_treated_as_a_run():
    s = DashboardState()
    s.ingest(DOCS + [{"run_id": "monitor_state", "running": [], "manifest": []}])
    assert "monitor_state" not in s.runs


def test_filter_by_attack():
    s = DashboardState()
    s.ingest(DOCS)
    assert {r["run_id"] for r in s.filtered(attack="zlib")} == {"a", "b"}


def test_filter_by_status():
    s = DashboardState()
    s.ingest(DOCS)
    assert [r["run_id"] for r in s.filtered(status="failed")] == ["c"]


def test_filter_by_config_factor():
    s = DashboardState()
    s.ingest(DOCS)
    assert {r["run_id"] for r in s.filtered(federated_rounds=1)} == {"a", "c"}


def test_filters_compose():
    s = DashboardState()
    s.ingest(DOCS)
    assert [r["run_id"] for r in s.filtered(attack="zlib", federated_rounds=2)] == ["b"]


def test_aggregate_reports_mean_adv_per_attack():
    s = DashboardState()
    s.ingest(DOCS)
    agg = s.aggregate_by("attack_name")
    assert agg["zlib"]["mean_adv"] == 0.75
    assert agg["zlib"]["count"] == 2


def test_aggregate_ignores_failed_runs_without_metrics():
    s = DashboardState()
    s.ingest(DOCS)
    assert "min_k" not in s.aggregate_by("attack_name")


def test_recently_finished_is_newest_first():
    s = DashboardState()
    s.ingest(DOCS)
    assert [r["run_id"] for r in s.recently_finished()] == ["c", "b", "a"]


def test_running_set_empty_without_monitor_state():
    """Firestore alone cannot identify in-progress runs (spec §2.4)."""
    s = DashboardState()
    s.ingest(DOCS)
    assert s.running == []
    assert s.manifest is None


def test_running_set_read_from_monitor_state_doc():
    s = DashboardState()
    s.ingest(DOCS + [{
        "run_id": "monitor_state",
        "running": [{"run_id": "z", "attack": "zlib", "started_unix": 10}],
        "manifest": [{"run_id": "z"}, {"run_id": "a"}],
    }])
    assert s.running[0]["attack"] == "zlib"
    assert len(s.manifest) == 2


def test_sweep_progress_needs_manifest_for_denominator():
    s = DashboardState()
    s.ingest(DOCS)
    progress = s.sweep_progress()
    assert progress["complete"] == 2
    assert progress["failed"] == 1
    assert progress["total"] is None  # no manifest -> no denominator
