from dataclasses import replace

import pytest

from master_script.core import runner
from master_script.core.registry import ATTACKS


def test_trials_alternate_positive_and_negative(monkeypatch):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls(attack_trials=4)
    trials = runner.run_attack_trials(cfg, spec)
    assert [t["truth_member"] for t in trials] == [True, False, True, False]


def test_run_id_is_stable_despite_per_trial_reseed():
    """Per-trial seed bumps must not leak into the run's hash."""
    from master_script.core.config import experiment_key

    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    before = experiment_key(cfg)
    runner.run_attack_trials(cfg, spec)
    assert experiment_key(cfg) == before


def test_real_dataset_trials_pair_member_and_nonmember_seeds(monkeypatch):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls(dataset_name="squad", use_hf_models=True, attack_trials=4)
    seen = []

    def fake_finetune(trial_cfg, truth_member):
        seen.append((trial_cfg.seed, truth_member))
        return {
            "model": object(), "tokenizer": object(), "device": "cpu",
            "target_record": f"target-for-seed-{trial_cfg.seed}",
        }, []

    monkeypatch.setattr(runner.federation, "run_hf_federated_finetune", fake_finetune)
    paired_spec = replace(spec, score_hf=lambda ctx: 1.0 if ctx.config.seed else 0.0)
    runner.run_attack_trials(cfg, paired_spec)

    assert seen == [(7, True), (7, False), (8, True), (8, False)]


def test_cache_hit_skips_compute(monkeypatch):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    cached = {"status": "complete", "run_id": "cached", "metrics": {"adv": 1.0}}
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c, s=None: cached)

    def _boom(*a, **k):
        raise AssertionError("compute must not run on a cache hit")

    monkeypatch.setattr(runner, "run_attack_trials", _boom)
    assert runner.run_single_experiment(cfg, spec)["run_id"] == "cached"


def test_federated_history_is_list_of_maps_not_nested_arrays(monkeypatch):
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c, s=None: None)
    monkeypatch.setattr(runner.firestore, "save_result", lambda c, r, s=None: False)
    spec = ATTACKS["zlib"]
    result = runner.run_single_experiment(spec.config_cls(attack_trials=2), spec)
    fh = result["federated_history"]
    assert isinstance(fh, list)
    assert all(isinstance(item, dict) and "rounds" in item for item in fh)


def test_failed_run_does_not_clean_artifacts(monkeypatch, tmp_path):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c, s=None: None)
    monkeypatch.setattr(runner.firestore, "mark_result_failed", lambda c, e, s=None: True)
    monkeypatch.setattr(runner, "artifact_dir_for", lambda c, s=None: tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("attack exploded")

    monkeypatch.setattr(runner, "run_attack_trials", _boom)
    cleaned = []
    monkeypatch.setattr(runner, "cleanup_artifacts", lambda p: cleaned.append(p))
    with pytest.raises(RuntimeError, match="attack exploded"):
        runner.run_single_experiment(cfg, spec)
    assert cleaned == []


def test_sweep_continues_after_a_failed_run(monkeypatch):
    first = ATTACKS["zlib"]
    second = ATTACKS["min_k"]
    calls = []
    resets = []

    def fake_run(config, spec, **kwargs):
        calls.append(spec.name)
        if spec.name == "zlib":
            raise RuntimeError("Ray GCS failed to start")
        return {"run_id": "second", "status": "complete"}

    monkeypatch.setattr(runner, "run_single_experiment", fake_run)
    monkeypatch.setattr(runner, "reset_ray_after_failure", lambda: resets.append(True))

    results = runner.run_sweep(
        [(first.config_cls(), first), (second.config_cls(), second)],
        use_firestore=False,
    )

    assert calls == ["zlib", "min_k"]
    assert [result["status"] for result in results] == ["failed", "complete"]
    assert results[0]["run_id"]
    assert results[0]["attack_name"] == "zlib"
    assert results[0]["error"] == "RuntimeError: Ray GCS failed to start"
    assert resets == [True]


def test_ray_reset_is_best_effort(monkeypatch):
    class BrokenRay:
        @staticmethod
        def shutdown():
            raise RuntimeError("cleanup failed")

    monkeypatch.setitem(__import__("sys").modules, "ray", BrokenRay)
    # Cleanup errors must not escape and abort the remaining sweep.
    runner.reset_ray_after_failure()


def test_no_firestore_flag_skips_cache_and_write(monkeypatch):
    spec = ATTACKS["zlib"]
    monkeypatch.setattr(
        runner.firestore, "load_cached_result",
        lambda c, s=None: (_ for _ in ()).throw(AssertionError("cache must not be read")),
    )
    result = runner.run_single_experiment(spec.config_cls(attack_trials=2), spec, use_firestore=False)
    assert result["status"] == "complete"


def test_firestore_calls_thread_spec_for_legacy_key_formula(monkeypatch, tmp_path):
    """amia/loss use non-default key formulas; every firestore call must carry
    spec so the doc id written matches run_id (see key_named_prefix)."""
    seen = {}

    def rec_load(c, s=None):
        seen["load_spec"] = s
        return None

    def rec_save(c, r, s=None):
        seen["save_spec"] = s
        return False

    monkeypatch.setattr(runner.firestore, "load_cached_result", rec_load)
    monkeypatch.setattr(runner.firestore, "save_result", rec_save)

    spec = ATTACKS["loss"]

    def fake_trials(config, artifact_dir=None):
        return [
            {
                "trial_id": 0,
                "truth_member": True,
                "target_client_id": config.target_client_id,
                "score_name": "average_per_token_negative_log_likelihood",
                "score": 1.0,
                "threshold": 0.5,
                "pred_member": True,
                "replacement_text": None,
                "federated_history": [],
                "artifacts": {},
            },
            {
                "trial_id": 1,
                "truth_member": False,
                "target_client_id": config.target_client_id,
                "score_name": "average_per_token_negative_log_likelihood",
                "score": 0.0,
                "threshold": 0.5,
                "pred_member": False,
                "replacement_text": "held out",
                "federated_history": [],
                "artifacts": {},
            },
        ]

    # AttackSpec is frozen; build a variant with the stub trial function
    # instead of monkeypatching the instance attribute.
    spec = replace(spec, custom_trials=fake_trials)
    monkeypatch.setattr(runner, "artifact_dir_for", lambda c, s=None: tmp_path)

    runner.run_single_experiment(spec.config_cls(attack_trials=2), spec)

    assert seen["load_spec"] is not None and seen["load_spec"].name == "loss"
    assert seen["save_spec"] is not None and seen["save_spec"].name == "loss"
