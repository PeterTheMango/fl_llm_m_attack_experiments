"""Proves the runner -> custom_trials -> build_payload handshake works for the
two legacy attacks (amia, loss), whose heavy FL/torch/flwr internals are
monkeypatched out. This is a wiring test, not an end-to-end attack test:
federated_fine_tune / train_ami_probe / run_attack_trials are stubbed with
lightweight fakes so no torch/flwr/GPU is required.
"""
from dataclasses import replace

from master_script.core import runner
from master_script.core.registry import ATTACKS


def test_loss_runs_through_runner_with_mocked_fl(monkeypatch):
    import master_script.core.attacks.loss as lossmod

    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c, s=None: None)
    monkeypatch.setattr(runner.firestore, "save_result", lambda c, r, s=None: False)

    def stub(config, artifact_dir):
        trials = []
        for trial_id, truth_member in ((0, True), (1, False)):
            trials.append({
                "trial_id": trial_id,
                "truth_member": truth_member,
                "target_client_id": config.target_client_id,
                "score_name": "average_per_token_negative_log_likelihood",
                "score": 0.1 if truth_member else 0.9,
                "threshold": 0.5,
                "pred_member": truth_member,
                "replacement_text": None if truth_member else "held-out text",
                "federated_history": [],
                "threshold_info": {"threshold": 0.5},
                "artifacts": {},
            })
        return trials

    # AttackSpec is frozen; build a variant with the stub trial function
    # instead of monkeypatching the instance attribute.
    spec = replace(ATTACKS["loss"], custom_trials=stub)

    res = runner.run_single_experiment(spec.config_cls(), spec)
    assert res["status"] == "complete"
    assert res["run_id"].startswith("loss_federated_llm_adaptation_v1_")
    assert "metrics" in res
    assert res["attack_name"] == "loss"


def test_amia_runs_through_runner_with_mocked_fl(monkeypatch):
    import master_script.core.attacks.amia as amiamod

    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c, s=None: None)
    monkeypatch.setattr(runner.firestore, "save_result", lambda c, r, s=None: False)

    def fake_federated_fine_tune(config, artifact_dir=None):
        return object(), object(), [], [], "model/path"

    def fake_train_ami_probe(model, tokenizer, clients, config, artifact_dir=None):
        return object(), [], "probe/path"

    def fake_run_attack_trials(model, tokenizer, probe, clients, config):
        return [
            {"trial_id": 0, "truth_member": True, "score": 1.0, "pred_member": True, "batch_size": 8},
            {"trial_id": 1, "truth_member": False, "score": 0.0, "pred_member": False, "batch_size": 8},
        ]

    def fake_clear_experiment_objects(*objects):
        return None

    monkeypatch.setattr(amiamod, "federated_fine_tune", fake_federated_fine_tune)
    monkeypatch.setattr(amiamod, "train_ami_probe", fake_train_ami_probe)
    monkeypatch.setattr(amiamod, "run_attack_trials", fake_run_attack_trials)
    monkeypatch.setattr(amiamod, "clear_experiment_objects", fake_clear_experiment_objects)

    spec = ATTACKS["amia"]
    res = runner.run_single_experiment(spec.config_cls(), spec)

    assert len(res["run_id"]) == 24
    assert "probe_training_loss" in res
    assert res["attack_name"] == "amia"
    assert res["config"]["firestore_collection"] == "ami_federated_llm_results"
