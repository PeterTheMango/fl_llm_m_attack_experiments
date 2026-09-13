from types import SimpleNamespace
import weakref

import numpy as np
import pytest

pytest.importorskip("flwr")
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from master_script.core.aggregation import private_fedavg
from master_script.core.defenses import clipped_mean, strategy_class
from master_script.core.pipeline import Defense


def results(updates):
    return [(None, SimpleNamespace(parameters=ndarrays_to_parameters(u), num_examples=i + 1,
                                  metrics={"partition_id": i})) for i, u in enumerate(updates)]


def test_streaming_private_aggregate_matches_full_vector_formula(monkeypatch):
    from master_script.core import aggregation
    monkeypatch.setattr(aggregation, "_CHUNK", 3)
    previous = [np.arange(10, dtype=np.float32).reshape(2, 5),
                np.array([.1, -.2], dtype=np.float64), np.array(7, dtype=np.int64)]
    updates = [[previous[0] + i, previous[1] - 2*i, np.array(999, dtype=np.int64)]
               for i in (0., 2., -3.)]
    batch = results(updates)
    before = [list(r.parameters.tensors) for _, r in batch]
    expected = clipped_mean(updates, previous, 1.5, .7, np.random.default_rng(11))
    actual = parameters_to_ndarrays(private_fedavg(batch, ndarrays_to_parameters(previous),
                                                  1.5, .7, np.random.default_rng(11)))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        assert a.dtype == b.dtype
    assert [r.parameters.tensors for _, r in batch] == before
    assert actual[-1] == 7


def test_decoder_does_not_retain_all_client_models(monkeypatch):
    import flwr.common
    original = flwr.common.bytes_to_ndarray
    refs, peak = [], [0]
    def decode(raw):
        array = original(raw)
        refs.append(weakref.ref(array))
        peak[0] = max(peak[0], sum(ref() is not None for ref in refs))
        return array
    monkeypatch.setattr(flwr.common, "bytes_to_ndarray", decode)
    initial = [np.zeros(10, dtype=np.float32), np.zeros(5, dtype=np.float32)]
    private_fedavg(results([[p+i for p in initial] for i in range(20)]),
                   ndarrays_to_parameters(initial), 1., 1., np.random.default_rng(1))
    assert peak[0] <= 6  # Independent of the 20 clients and 40 input tensors.


@pytest.mark.parametrize("update", [[np.array([np.nan], dtype=np.float32)],
                                   [np.zeros(2, dtype=np.float32)]])
def test_bad_update_rejected(update):
    with pytest.raises((ValueError, FloatingPointError)):
        private_fedavg(results([update]), ndarrays_to_parameters([np.zeros(1, dtype=np.float32)]),
                       1., 1., np.random.default_rng(1))


def test_private_strategy_releases_one_aggregate_and_retains_no_decoded_models():
    class Capture:
        def __init__(self, **kwargs):
            self.history = []
        def aggregate_fit(self, round_id, batch, failures):
            if len(batch) != 2:
                raise RuntimeError("missing scheduled client")
            self.history.append([r.metrics["partition_id"] for _, r in batch])
            self.captured, metrics = self.aggregate_updates(round_id, batch, failures)
            return self.captured, metrics
        def aggregate_updates(self, *args):
            raise AssertionError("No second FedAvg over an already-noised update")
    initial = ndarrays_to_parameters([np.zeros(2, dtype=np.float32)])
    privacy = {}
    strategy = strategy_class(Capture, Defense(mechanism="dp_fedavg"),
                              None, privacy)(initial_parameters=initial)
    for round_id in (1, 2):
        batch = results([[np.array([3., 4.], dtype=np.float32)], [np.zeros(2, dtype=np.float32)]])
        original = [r.parameters for _, r in batch]
        parameter, _ = strategy.aggregate_fit(round_id, batch, [])
        assert strategy.previous is parameter is strategy.captured
        assert [r.parameters for _, r in batch] == original
    assert privacy["steps"] == 2
    assert strategy.history == [[0, 1], [0, 1]]
    with pytest.raises(RuntimeError, match="incomplete"):
        strategy.aggregate_fit(3, batch, [RuntimeError("worker killed")])
    assert strategy.rounds_released == 2


def test_serialized_state_copy_preserves_tied_weights_and_integer_buffers():
    torch = pytest.importorskip("torch")
    from master_script.core.model_io import set_serialized_parameters
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2, bias=False)
            self.b = self.a
            self.register_buffer("count", torch.tensor(3))
    model = Model()
    arrays = [np.full(tuple(t.shape), 7, dtype=t.numpy().dtype) for t in model.state_dict().values()]
    set_serialized_parameters(model, ndarrays_to_parameters(arrays))
    assert model.a.weight is model.b.weight
    assert all(torch.all(t == 7) for t in model.state_dict().values())


@pytest.mark.parametrize("already_initialized", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_simulation_cleanup_respects_runtime_ownership(monkeypatch, already_initialized, fail):
    import sys
    from master_script.core import runtime_memory
    import flwr.simulation
    calls = []
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(is_initialized=lambda: already_initialized,
                                                           shutdown=lambda: calls.append("shutdown")))
    monkeypatch.setattr(runtime_memory, "release_memory", lambda: calls.append("release"))
    def run(**kwargs):
        if fail:
            raise RuntimeError("worker failed")
        return "done"
    monkeypatch.setattr(flwr.simulation, "run_simulation", run)
    if fail:
        with pytest.raises(RuntimeError, match="worker failed"):
            runtime_memory.run_simulation()
    else:
        assert runtime_memory.run_simulation() == "done"
    assert calls == (["shutdown"] if not already_initialized else []) + ["release"]


def test_trial_model_references_are_gone_before_cleanup(monkeypatch):
    from master_script.core import runner, runtime_memory
    references = []
    class Model: pass
    def trial(*args):
        model = Model()
        references.append(weakref.ref(model))
        return {"score": 1.}
    def release():
        assert all(ref() is None for ref in references)
    monkeypatch.setattr(runner, "_run_attack_trial", trial)
    monkeypatch.setattr(runtime_memory, "release_memory", release)
    runner.run_attack_trial(SimpleNamespace(attack_trials=2), SimpleNamespace(name="reference"), 0, True)
    runner.run_attack_trial(SimpleNamespace(attack_trials=2), SimpleNamespace(name="reference"), 1, False)


def test_focused_config_caps_workers_without_dropping_clients():
    from master_script.core.runtime_memory import simulation_backend
    from master_script.core.yaml_config import load_config_file
    from master_script.paths import CONFIGS_DIR
    for config, spec in load_config_file(CONFIGS_DIR / "amia_reference_rag_master.yaml"):
        assert config.num_clients == config.clients_per_round == 4
        backend = simulation_backend(config)
        assert backend["init_args"] == {"num_cpus": 1, "address": "local"}
        assert backend["client_resources"] == {"num_cpus": 1, "num_gpus": 1.}


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_worker_cap_fails_before_compute(cap):
    from master_script.core.config import validate_attack_config
    from master_script.core.attacks.reference import ReferenceConfig
    with pytest.raises(ValueError, match="sim_max_concurrent_clients"):
        validate_attack_config(ReferenceConfig(sim_max_concurrent_clients=cap))


def test_failed_run_persists_attack_and_defense_identity(monkeypatch):
    from master_script.core import firestore
    from master_script.core.yaml_config import load_config_file
    from master_script.paths import CONFIGS_DIR
    from tests.test_firestore import _FakeDB
    db = _FakeDB()
    monkeypatch.setattr(firestore, "get_firestore_client", lambda *a: db)
    config, spec = load_config_file(CONFIGS_DIR / "amia_reference_rag_master.yaml", only=["amia"])[0]
    firestore.mark_result_failed(config, "worker failed", spec)
    saved = next(iter(db.store.values()))
    assert saved["attack_name"] == "amia"
    assert saved["pipeline"]["condition"] == "baseline"
