import pytest

from master_script.core.yaml_config import ConfigError, load_config_file

SMOKE = """
defaults:
  seed: 11
attacks:
  zlib:
    base: {threshold: -0.006}
    sweep:
      federated_rounds: [1, 2]
"""


def _write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_expands_grid_into_one_pair_per_combination(tmp_path):
    pairs = load_config_file(_write(tmp_path, SMOKE))
    assert len(pairs) == 2
    assert {c.federated_rounds for c, _ in pairs} == {1, 2}


def test_defaults_apply_to_attack(tmp_path):
    pairs = load_config_file(_write(tmp_path, SMOKE))
    assert all(c.seed == 11 for c, _ in pairs)


def test_base_overrides_defaults(tmp_path):
    pairs = load_config_file(_write(tmp_path, SMOKE))
    assert all(c.threshold == -0.006 for c, _ in pairs)


def test_defaults_key_absent_on_attack_is_skipped_not_error(tmp_path):
    """min_k_percent exists on min_k but not zlib; defaults must tolerate that."""
    text = """
defaults:
  min_k_percent: 30
attacks:
  zlib: {}
  min_k: {}
"""
    pairs = load_config_file(_write(tmp_path, text))
    by_name = {spec.name: cfg for cfg, spec in pairs}
    assert by_name["min_k"].min_k_percent == 30
    assert not hasattr(by_name["zlib"], "min_k_percent")


def test_unknown_key_in_attack_base_is_hard_error(tmp_path):
    text = """
attacks:
  zlib:
    base: {epsilon: 8}
"""
    with pytest.raises(ConfigError, match="epsilon"):
        load_config_file(_write(tmp_path, text))


def test_unknown_attack_name_is_error(tmp_path):
    with pytest.raises(ConfigError, match="nonexistent"):
        load_config_file(_write(tmp_path, "attacks:\n  nonexistent: {}\n"))


def test_only_filter_selects_subset(tmp_path):
    text = "attacks:\n  zlib: {}\n  min_k: {}\n"
    pairs = load_config_file(_write(tmp_path, text), only=["zlib"])
    assert [spec.name for _, spec in pairs] == ["zlib"]


def test_amia_without_hf_models_is_rejected(tmp_path):
    text = "attacks:\n  amia:\n    base: {use_hf_models: false}\n"
    with pytest.raises(ConfigError, match="toy"):
        load_config_file(_write(tmp_path, text))


def test_shipped_smoke_config_loads(tmp_path):
    from master_script.paths import CONFIGS_DIR

    pairs = load_config_file(CONFIGS_DIR / "smoke.yaml")
    assert pairs
