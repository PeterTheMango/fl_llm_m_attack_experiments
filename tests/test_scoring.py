# tests/test_scoring.py
"""Every attack must separate members from non-members on the toy path.

This is the behavioral contract each notebook's smoke run asserts.
"""
import pytest

from master_script.core import federation
from master_script.core.registry import ATTACKS
from master_script.core.scoring import ScoreContext

TOY_ATTACKS = sorted(n for n, s in ATTACKS.items() if s.supports_toy and s.score_toy)


@pytest.mark.parametrize("attack", TOY_ATTACKS)
def test_member_scores_above_nonmember_on_toy_path(attack):
    spec = ATTACKS[attack]
    cfg = spec.config_cls()
    member, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    nonmember, _ = federation.run_toy_federated_finetune(cfg, truth_member=False)

    def _ctx(model):
        ref = None
        if spec.needs_reference:
            ref, _ = federation.run_toy_federated_finetune(cfg, truth_member=False)
        return ScoreContext(config=cfg, target=model, text=federation.TARGET_RECORD, reference=ref)

    assert spec.score_toy(_ctx(member)) > spec.score_toy(_ctx(nonmember))


@pytest.mark.parametrize("attack", TOY_ATTACKS)
def test_toy_scoring_is_deterministic(attack):
    spec = ATTACKS[attack]
    cfg = spec.config_cls()
    model, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    ctx = ScoreContext(config=cfg, target=model, text=federation.TARGET_RECORD)
    assert spec.score_toy(ctx) == spec.score_toy(ctx)


def test_zlib_score_matches_notebook_formula():
    """Guard the exact zlib formula: -(nll / zlib_entropy_bits)."""
    import zlib as _zlib

    from master_script.core.attacks.zlib import ZlibConfig, zlib_entropy_bits

    text = federation.TARGET_RECORD
    assert zlib_entropy_bits(text) == 8.0 * len(_zlib.compress(text.encode("utf-8")))


def test_samia_extra_metrics_include_tpr_at_fpr():
    trials = [
        {"truth_member": True, "score": 1.0, "pred_member": True},
        {"truth_member": False, "score": 0.0, "pred_member": False},
    ]
    out = ATTACKS["samia"].extra_metrics(trials)
    assert "roc_auc" in out and "tpr_at_fpr" in out


def test_reference_attack_has_no_extra_metrics():
    """reference_adaptations.ipynb emits no roc_auc; preserve that shape."""
    assert ATTACKS["reference"].extra_metrics is None
