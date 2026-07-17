import math

import pytest

from master_script.core import metrics

PERFECT = [
    {"trial_id": 0, "truth_member": True, "score": 1.0, "pred_member": True},
    {"trial_id": 1, "truth_member": False, "score": 0.0, "pred_member": False},
    {"trial_id": 2, "truth_member": True, "score": 0.9, "pred_member": True},
    {"trial_id": 3, "truth_member": False, "score": 0.1, "pred_member": False},
]


def test_base_metrics_perfect_classifier():
    out = metrics.base_metrics(PERFECT)
    assert out["tpr"] == 1.0
    assert out["tnr"] == 1.0
    assert out["adv"] == 1.0
    assert out["num_trials"] == 4


def test_adv_is_mean_of_tpr_and_tnr():
    trials = [
        {"truth_member": True, "score": 1.0, "pred_member": True},
        {"truth_member": True, "score": 0.0, "pred_member": False},
        {"truth_member": False, "score": 0.0, "pred_member": False},
        {"truth_member": False, "score": 0.0, "pred_member": False},
    ]
    out = metrics.base_metrics(trials)
    assert out["tpr"] == 0.5
    assert out["tnr"] == 1.0
    assert out["adv"] == 0.75


def test_base_metrics_excludes_roc_auc():
    """reference_adaptations.ipynb has no roc_auc; the base set must match it."""
    assert "roc_auc" not in metrics.base_metrics(PERFECT)


def test_base_metrics_key_set_matches_reference_notebook():
    assert set(metrics.base_metrics(PERFECT)) == {
        "tp", "tn", "fp", "fn", "tpr", "tnr", "adv",
        "accuracy", "precision", "recall", "f1", "num_trials",
    }


def test_roc_auc_perfect_separation():
    assert metrics.roc_auc([True, True, False, False], [1.0, 0.9, 0.1, 0.0]) == 1.0


def test_roc_auc_ties_count_half():
    assert metrics.roc_auc([True, False], [0.5, 0.5]) == 0.5


def test_roc_auc_single_class_is_nan():
    assert math.isnan(metrics.roc_auc([True, True], [1.0, 0.5]))


def test_base_metrics_empty_trials_does_not_divide_by_zero():
    out = metrics.base_metrics([])
    assert out["adv"] == 0.0
    assert out["num_trials"] == 0
