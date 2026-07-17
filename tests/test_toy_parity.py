"""Reproduce each notebook's smoke-run assertions through the consolidated core.

These are the notebooks' own acceptance criteria, re-asserted against core.
"""
import pytest

from master_script.core import runner
from master_script.core.registry import ATTACKS

TOY_ATTACKS = sorted(n for n, s in ATTACKS.items() if s.supports_toy)


@pytest.fixture(autouse=True)
def _no_firestore(monkeypatch):
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c: None)
    monkeypatch.setattr(runner.firestore, "save_result", lambda c, r: False)


@pytest.mark.parametrize("attack", TOY_ATTACKS)
def test_smoke_run_produces_both_worlds_and_core_metrics(attack):
    spec = ATTACKS[attack]
    result = runner.run_single_experiment(spec.config_cls(attack_trials=4, use_hf_models=False), spec)
    assert result["metrics"]["num_trials"] == 4
    assert any(r["truth_member"] for r in result["attack_trials"])
    assert any(not r["truth_member"] for r in result["attack_trials"])
    for key in ("tpr", "tnr", "adv"):
        assert key in result["metrics"]


def test_zlib_smoke_roc_auc_is_one():
    """zlib_adaptations.ipynb asserts perfect ranking on the toy path."""
    spec = ATTACKS["zlib"]
    result = runner.run_single_experiment(spec.config_cls(attack_trials=4), spec)
    assert result["metrics"]["roc_auc"] == 1.0
