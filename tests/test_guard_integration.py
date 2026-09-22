from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
from master_script.core.guard_runtime import parse_guard, prepare_guard, read_guard_events, reject_training_round, PolicyAbortedRound
from master_script.core.pipeline import parse_pipeline
from master_script.core.metrics import guarded_metrics


def settings(tmp_path, mode="rules", architecture="causal_lm"):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "client_guard_v1", "observation_architecture": architecture}))
    value = {"mode": mode, "policy_file": str(policy), "release_budget": 2, "diagnostic": True}
    return replace(parse_guard(value, "<config>"), runtime_directory=str(tmp_path / "local"))


def test_runtime_refusals_and_audit_are_local(tmp_path):
    runtime = prepare_guard(settings(tmp_path), "observation", [np.ones(2)], rounds=4)
    assert not runtime.check([np.ones(2)], 0, 1, architecture="amia_probe", observation=True)
    assert runtime.check([np.ones(2)], 0, 1)
    assert not runtime.check([np.ones(2)], 0, 1)
    assert runtime.check([np.ones(2)], 0, 2)
    assert not runtime.check([np.ones(2)], 0, 3)
    events = read_guard_events(tmp_path / "local")
    assert events[0]["reservation_count"] == 0
    assert events[-1]["reason"] == "budget_exhausted"
    assert events[-1]["reservation_count"] == 2


def test_shadow_logs_without_enforcing_detector_or_architecture(tmp_path):
    runtime = prepare_guard(settings(tmp_path, "shadow"), "observation", [np.ones(2)], rounds=4)
    assert runtime.check([np.ones(2)], 0, 1, architecture="amia_probe", observation=True)
    assert not runtime.check([np.array([np.nan, 0])], 0, 2, architecture="amia_probe", observation=True)
    events = read_guard_events(tmp_path / "local")
    assert events[0]["would_accept"] is False
    assert events[0]["decision"] == "accepted"


def test_policy_abort_is_not_incomplete_aggregation():
    with pytest.raises(PolicyAbortedRound):
        reject_training_round([(None, SimpleNamespace(metrics={"guard_decision": "rejected"}))], [])
    with pytest.raises(RuntimeError, match="infrastructure"):
        reject_training_round([], [RuntimeError()])


def test_null_scores_never_become_nonmember_predictions():
    rows = [{"truth_member": truth, "decision": "rejected", "score": None, "pred_member": None} for truth in (True, False)]
    result = guarded_metrics(rows)
    assert result["roc_auc"] is None and result["adv"] is None
    assert result["release_coverage"] == 0
    assert result["decision_transcript_metrics"]["roc_auc"] == .5
    rows += [{"truth_member": True, "decision": "accepted", "score": 1., "pred_member": True},
             {"truth_member": False, "decision": "accepted", "score": 0., "pred_member": False}]
    result = guarded_metrics(rows)
    assert result["roc_auc"] == 1 and result["release_coverage"] == .5
    rows[0]["score"] = 0
    with pytest.raises(ValueError):
        guarded_metrics(rows)


def test_guard_identity_is_content_based_and_runtime_free(tmp_path):
    guard = settings(tmp_path)
    value = {"client_guard": {"mode": "rules", "policy_file": guard.policy_file, "release_budget": 2, "diagnostic": True}}
    pipeline = parse_pipeline(value)
    copy = tmp_path / "copy.json"; copy.write_text(guard.policy_json)
    moved = parse_pipeline({"client_guard": {**value["client_guard"], "policy_file": str(copy)}})
    assert pipeline.identity() == moved.identity()
    assert replace(pipeline, client_guard=replace(pipeline.client_guard, runtime_directory="/elsewhere")).identity() == pipeline.identity()
    assert "client_guard" not in parse_pipeline({"defense": {"mechanism": "dp_sgd"}}).identity()


def test_early_flower_rejection_never_loads_victim_model(monkeypatch, tmp_path):
    """Exercise the actual nested VictimClient and aggregator without a GPU."""
    import sys
    from types import ModuleType
    from master_script.core.attacks import amia
    from master_script.core import runtime_memory

    class Client:
        def to_client(self):
            return self
    class App:
        def __init__(self, client_fn=None, server_fn=None):
            self.client_fn, self.server_fn = client_fn, server_fn
    class Strategy:
        def __init__(self, **kwargs):
            pass
    exports = {
        "flwr": {}, "flwr.client": {"NumPyClient": Client, "ClientApp": App},
        "flwr.common": {"ndarrays_to_parameters": lambda a: a, "parameters_to_ndarrays": lambda a: a},
        "flwr.server": {"ServerApp": App, "ServerAppComponents": SimpleNamespace, "ServerConfig": SimpleNamespace},
        "flwr.server.strategy": {"FedAvg": Strategy},
        "transformers": {"AutoModelForCausalLM": object, "AutoTokenizer": object},
    }
    for name, values in exports.items():
        module = ModuleType(name); module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(amia, "get_parameters", lambda p: [np.ones(2)])
    def private(*args, **kwargs):
        pytest.fail("Rejected request reached private computation")
    monkeypatch.setattr(amia, "client_loss_gradients", private)
    def simulate(server_app, client_app, **kwargs):
        strategy = server_app.server_fn(None).strategy
        client = client_app.client_fn(SimpleNamespace(node_config={"partition-id": 0}))
        for i in range(2):
            arrays, count, metrics = client.fit([np.ones(2)], {"trial_id": i})
            assert arrays == [] and count == 0
            assert set(metrics) == {"partition_id", "guard_decision", "response_seconds"}
            strategy.aggregate_fit(i + 1, [(None, SimpleNamespace(parameters=arrays, num_examples=count, metrics=metrics))], [])
    monkeypatch.setattr(runtime_memory, "run_simulation", simulate)
    monkeypatch.setattr(runtime_memory, "simulation_backend", lambda cfg: {})
    runtime = prepare_guard(settings(tmp_path), "obs", [np.ones(2)], rounds=1)
    config = SimpleNamespace(num_clients=1, target_client_id=0, attack_trials=2, gradient_threshold=0., seed=7)
    probe = SimpleNamespace(fc1=SimpleNamespace(in_features=2, out_features=2))
    rows = amia.run_attack_trials("unused", probe, [["private"]], config, guard_runtime=runtime, checkpoint_dir=tmp_path / "checkpoints")
    assert len(rows) == 2 and all(r["score"] is None for r in rows)
    assert len(list((tmp_path / "checkpoints").rglob("trial-*.json"))) == 2


def test_matched_worlds_exhaust_identically(tmp_path):
    runtime = prepare_guard(settings(tmp_path, architecture="amia_probe"), "obs", [np.ones(2)], rounds=3)
    outcomes = []
    for index in range(6):
        world = replace(runtime, scope=f"obs:world-{index % 2}")
        outcomes.append(world.check([np.ones(2)], 0, index // 2 + 1, architecture="amia_probe", observation=True))
    assert outcomes == [True, True, True, True, False, False]


def test_missing_artifacts_and_unsafe_supported_combinations(tmp_path):
    from master_script.core.pipeline import validate_pipeline_run
    from master_script.core.registry import ATTACKS
    with pytest.raises(OSError):
        parse_guard({"mode": "rules", "policy_file": str(tmp_path / "missing"), "release_budget": 2}, "<config>")
    spec = ATTACKS["amia"]
    pipeline = parse_pipeline({"client_guard": {"mode": "rules", "policy_file": settings(tmp_path).policy_file, "release_budget": 2}})
    with pytest.raises(ValueError, match="dp_sgd"):
        validate_pipeline_run(spec.config_cls(), spec, pipeline)


def test_runner_persists_all_refused_as_complete(monkeypatch, tmp_path):
    from master_script.core import runner
    from master_script.core.pipeline import Pipeline, Defense
    from master_script.core.attacks.amia import SPEC, AmiaConfig
    monkeypatch.setattr(runner, "resolve_run_config", lambda cfg: cfg)
    rows = [{"trial_id": i, "truth_member": i % 2 == 0, "decision": "rejected", "score": None, "pred_member": None} for i in range(2)]
    spec = replace(SPEC, pipeline=Pipeline(Defense(), client_guard=settings(tmp_path)),
                   custom_trials=lambda *a, **kw: rows,
                   build_payload=lambda cfg, trials, directory: {"metrics": guarded_metrics(trials), "attack_trials": trials})
    result = runner.run_single_experiment(AmiaConfig(attack_trials=2), spec, use_firestore=False, artifact_directory=tmp_path / "run")
    assert result["status"] == "complete"
    assert result["metrics"]["roc_auc"] is None
    saved = json.loads((tmp_path / "run/result.json").read_text())
    assert saved["attack_trials"][0]["score"] is None
    from master_script.webui.results import _trials
    assert _trials(saved) == []


def test_policy_abort_envelope_is_distinct(tmp_path):
    from master_script.core.runner import failed_sweep_result
    from master_script.core.attacks.amia import SPEC, AmiaConfig
    result = failed_sweep_result(AmiaConfig(), SPEC, "run", PolicyAbortedRound("refused"))
    assert result["status"] == "policy_aborted"
