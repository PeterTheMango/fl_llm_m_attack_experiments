import pytest

from master_script.core.registry import ATTACKS


@pytest.mark.parametrize("attack", ["amia", "loss"])
def test_legacy_attacks_declare_no_toy_support(attack):
    """Neither legacy notebook has a toy path.

    AMIA_adaptation.ipynb has no ToyFederatedLM at all; LOSS_adaptation.ipynb
    calls require_training_deps() and has no use_hf_models field to switch on.
    Both therefore require real HF models.
    """
    assert ATTACKS[attack].supports_toy is False


def test_legacy_configs_have_no_use_hf_models_field():
    """Guard the assumption that the toy/HF switch simply doesn't exist here."""
    from dataclasses import fields

    for attack in ("amia", "loss"):
        names = {f.name for f in fields(ATTACKS[attack].config_cls)}
        assert "use_hf_models" not in names


def test_amia_uses_strict_greater_than_threshold():
    """AMIA thresholds with `>` while every other attack uses `>=`. Preserve it."""
    from master_script.core.attacks.amia import predict_member

    cfg = ATTACKS["amia"].config_cls()
    assert predict_member(cfg.gradient_threshold, cfg) is False  # equal -> not member
    assert predict_member(cfg.gradient_threshold + 1e-9, cfg) is True


def test_loss_threshold_is_calibrated_not_constant():
    from master_script.core.attacks.loss import estimate_loss_threshold

    cfg = ATTACKS["loss"].config_cls()
    losses = [float(i) for i in range(100)]
    info = estimate_loss_threshold(losses, cfg)
    assert "threshold" in info
    assert info["threshold"] == pytest.approx(9.9, abs=0.5)  # 10th percentile


def test_loss_predicts_member_when_loss_below_threshold():
    from master_script.core.attacks.loss import predict_member_from_loss

    assert predict_member_from_loss(0.1, 1.0) is True
    assert predict_member_from_loss(2.0, 1.0) is False


def test_loss_uses_the_shared_firestore_collection():
    assert ATTACKS["loss"].config_cls().firestore_collection == "ami_federated_llm_results"


def test_loss_methodology_has_attacker_observation_key():
    """LOSS_adaptation.ipynb emits an extra methodology key; preserve the shape."""
    assert "attacker_observation" in ATTACKS["loss"].methodology


def test_both_legacy_attacks_supply_custom_trials():
    assert ATTACKS["amia"].custom_trials is not None
    assert ATTACKS["loss"].custom_trials is not None
