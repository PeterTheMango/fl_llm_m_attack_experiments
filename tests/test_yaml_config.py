from dataclasses import asdict

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


def test_load_config_doc_expands_an_already_parsed_dict():
    """Manual mode has no file; it must reach the same expansion."""
    from master_script.core.yaml_config import load_config_doc

    pairs = load_config_doc({"attacks": {"zlib": {"sweep": {"seed": [7, 11]}}}})
    assert [cfg.seed for cfg, _spec in pairs] == [7, 11]


def test_load_config_doc_names_its_source_in_errors():
    """The caller decides what a config is called; there may be no path."""
    from master_script.core.yaml_config import ConfigError, load_config_doc

    with pytest.raises(ConfigError) as exc:
        load_config_doc({"attacks": {"nonexistent": {}}}, source="<manual>")
    assert "<manual>" in str(exc.value)


def test_firestore_collection_is_fixed_to_the_shared_collection(tmp_path):
    text = """
attacks:
  loss:
    base: {firestore_collection: loss_federated_llm_results}
"""
    with pytest.raises(ConfigError, match="fixed to.*ami_federated_llm_results"):
        load_config_file(_write(tmp_path, text))


def test_all_datasets_master_sweeps_every_registered_dataset_at_baseline_values():
    from master_script.core.datasets import available_dataset_names
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / "all_datasets_master.yaml")

    assert len(pairs) == 11 * len(available_dataset_names()) == 77
    assert {cfg.dataset_name for cfg, _spec in pairs} == set(available_dataset_names())
    for cfg, spec in pairs:
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        actual.pop("dataset_name")
        expected.pop("dataset_name")
        assert actual == expected


def test_gpt2_master_changes_only_the_model_from_baseline():
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / "gpt2_master.yaml")

    assert len(pairs) == 11
    for cfg, spec in pairs:
        assert cfg.model_id == "gpt2"
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        actual.pop("model_id")
        expected.pop("model_id")
        if spec.name == "wbc":
            assert actual.pop("reference_model_id") == "gpt2"
            assert expected.pop("reference_model_id") == "distilgpt2"
        assert actual == expected


def test_seed_sweep_master_changes_only_seed_from_baseline():
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / "seed_sweep_master.yaml")

    assert len(pairs) == 55
    assert {cfg.seed for cfg, _spec in pairs} == {7, 11, 23, 42, 101}
    for cfg, spec in pairs:
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        actual.pop("seed")
        expected.pop("seed")
        assert actual == expected


def test_high_trials_master_changes_only_trial_count_from_baseline():
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / "high_trials_master.yaml")

    assert len(pairs) == 11
    assert {cfg.attack_trials for cfg, _spec in pairs} == {100}
    for cfg, spec in pairs:
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        actual.pop("attack_trials")
        expected.pop("attack_trials")
        assert actual == expected


def test_high_clients_master_changes_only_client_counts_from_baseline():
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / "high_clients_master.yaml")

    assert len(pairs) == 11
    assert {(cfg.num_clients, cfg.clients_per_round) for cfg, _spec in pairs} == {(8, 8)}
    for cfg, spec in pairs:
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        for field in ("num_clients", "clients_per_round"):
            actual.pop(field)
            expected.pop(field)
        assert actual == expected


@pytest.mark.parametrize(
    "filename,field,values",
    [
        ("federated_rounds_sweep_master.yaml", "federated_rounds", {1, 2, 4}),
        ("local_epochs_sweep_master.yaml", "local_epochs", {1, 2, 4}),
        ("client_lr_sweep_master.yaml", "client_lr", {1e-5, 5e-5, 1e-4}),
        ("max_length_sweep_master.yaml", "max_length", {64, 128, 256}),
    ],
)
def test_single_factor_master_sweeps_change_only_the_named_baseline_field(
    filename, field, values
):
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / filename)

    assert len(pairs) == 33
    assert {getattr(cfg, field) for cfg, _spec in pairs} == values
    for cfg, spec in pairs:
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        actual.pop(field)
        expected.pop(field)
        assert actual == expected


def test_client_participation_sweep_fixes_eight_clients_and_changes_only_participation():
    from master_script.paths import CONFIGS_DIR

    baseline = {
        spec.name: cfg for cfg, spec in load_config_file(CONFIGS_DIR / "baseline_master.yaml")
    }
    pairs = load_config_file(CONFIGS_DIR / "client_participation_sweep_master.yaml")

    assert len(pairs) == 33
    assert {cfg.num_clients for cfg, _spec in pairs} == {8}
    assert {cfg.clients_per_round for cfg, _spec in pairs} == {2, 4, 8}
    for cfg, spec in pairs:
        actual = asdict(cfg)
        expected = asdict(baseline[spec.name])
        for field in ("num_clients", "clients_per_round"):
            actual.pop(field)
            expected.pop(field)
        assert actual == expected
