"""Corrected notebooks delegate; method/source/revision changes isolate caches."""
from dataclasses import replace
import pytest
from master_script.core import config as identity
from master_script.core.registry import ATTACKS
from tests.notebook_configs import NOTEBOOKS, notebook_config_class

@pytest.mark.parametrize("attack", sorted(NOTEBOOKS))
def test_notebook_uses_canonical_config(attack):
    assert notebook_config_class(attack) is ATTACKS[attack].config_cls

@pytest.mark.parametrize("attack", sorted(NOTEBOOKS))
def test_corrected_identity_cannot_hit_historical_namespace(attack):
    spec = ATTACKS[attack]
    cfg = spec.config_cls()
    key = identity.experiment_key(cfg, spec)
    assert key.startswith(identity.METHOD_VERSION + "_")
    assert key != identity.legacy_experiment_key(cfg, spec)
    for field in ("model_revision", "reference_revision", "dataset_revision"):
        assert identity.experiment_key(replace(cfg, **{field: "a" * 40}), spec) != key

def test_source_change_invalidates_cache(monkeypatch):
    spec = ATTACKS["wbc"]
    before = identity.experiment_key(spec.config_cls(), spec)
    monkeypatch.setattr(identity, "implementation_fingerprint", lambda: "changed")
    assert identity.experiment_key(spec.config_cls(), spec) != before

def test_all_eleven_attacks_are_registered():
    assert set(ATTACKS) == set(NOTEBOOKS)
    assert len(ATTACKS) == 11
