# tests/test_charts.py
from master_script.core import charts
from master_script.paths import CHARTS_DIR

RESULTS = [
    {"config": {"attack_name": "zlib", "federated_rounds": 1}, "metrics": {"adv": 0.5}},
    {"config": {"attack_name": "zlib", "federated_rounds": 2}, "metrics": {"adv": 0.75}},
    {"config": {"attack_name": "zlib", "federated_rounds": 4}, "metrics": {"adv": 1.0}},
]


def test_adv_by_factor_writes_into_charts_dir():
    out = charts.render_adv_by_factor(RESULTS, "federated_rounds")
    assert out.exists()
    assert out.parent == CHARTS_DIR
    assert out.suffix == ".png"
    out.unlink()


def test_chart_filenames_are_timestamped_so_runs_do_not_clobber():
    a = charts.render_adv_by_factor(RESULTS, "federated_rounds")
    b = charts.render_adv_by_factor(RESULTS, "federated_rounds")
    assert a != b
    a.unlink()
    b.unlink()


def test_score_distribution_splits_by_true_membership():
    result = {
        "config": {"attack_name": "zlib"},
        "run_id": "abc123",
        "attack_trials": [
            {"trial_id": 0, "truth_member": True, "score": 0.9, "pred_member": True},
            {"trial_id": 1, "truth_member": False, "score": 0.1, "pred_member": False},
        ],
    }
    out = charts.render_score_distribution(result)
    assert out.exists()
    out.unlink()


def test_empty_results_returns_no_charts():
    assert charts.render_sweep_summary([]) == []
