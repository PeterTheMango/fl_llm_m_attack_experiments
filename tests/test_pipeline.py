from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import pickle

import pytest

from master_script.core.config import experiment_key
from master_script.core.pipeline import Defense, Pipeline, parse_pipeline
from master_script.core.registry import ATTACKS
from master_script.core.yaml_config import ConfigError, load_config_doc, load_config_file
from master_script.paths import CONFIGS_DIR


def test_disabled_pipeline_keeps_all_historical_hashes_and_configs():
    for name, spec in ATTACKS.items():
        cfg = spec.config_cls()
        pairs = load_config_doc({"pipeline": {"defense": {"mechanism": "none"}}, "attacks": {name: {}}})
        loaded, loaded_spec = pairs[0]
        assert asdict(loaded) == asdict(cfg)
        assert experiment_key(loaded, loaded_spec) == experiment_key(cfg, spec)


def test_defense_options_separate_cache_ids_without_changing_attack_fields():
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    keys = {experiment_key(cfg, spec)}
    for mechanism, noise in [("dp_sgd", 1.0), ("dp_sgd", 2.0), ("dp_fedavg", 1.0)]:
        defended = replace(spec, pipeline=Pipeline(Defense(mechanism=mechanism, noise_multiplier=noise)))
        keys.add(experiment_key(cfg, defended))
        assert asdict(cfg) == asdict(spec.config_cls())
    assert len(keys) == 4


def test_pipeline_is_spawn_pickleable_and_snapshots_study():
    cfg, spec = load_config_file(CONFIGS_DIR / "pipeline_zlib_dp.yaml")[0]
    restored = pickle.loads(pickle.dumps((cfg, spec)))
    assert experiment_key(*restored) == experiment_key(cfg, spec)
    assert "public_documents" in json.loads(spec.pipeline.rag.study_json)
    assert "study_json" not in spec.pipeline.metadata()["rag"]


@pytest.mark.parametrize("defense", [
    {"mechanism": "guess"}, {"noise_multiplier": 0}, {"noise_multiplier": float("nan")},
    {"delta": 1}, {"clip_norm": True}, {"epsilon": 3},
])
def test_invalid_defense_fails_before_training(defense):
    with pytest.raises(ConfigError):
        load_config_doc({"pipeline": {"defense": defense}, "attacks": {"zlib": {}}})


def test_toy_defense_is_explicitly_rejected():
    with pytest.raises(ConfigError, match="real models"):
        load_config_doc({"defaults": {"use_hf_models": False},
                         "pipeline": {"defense": {"mechanism": "dp_sgd"}}, "attacks": {"zlib": {}}})


def test_rag_file_bytes_change_identity_and_location_does_not(tmp_path):
    study = json.loads((CONFIGS_DIR / "pipeline_demo_study.json").read_text())
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    options = {"rag": {"study_file": str(path)}}
    before = parse_pipeline(options)
    duplicate = tmp_path / "copy.json"
    duplicate.write_bytes(path.read_bytes())
    assert before.identity() == parse_pipeline({"rag": {"study_file": str(duplicate)}}).identity()
    study["utility_queries"][0]["answer"] = "a changed answer"
    path.write_text(json.dumps(study))
    assert before.identity() != parse_pipeline(options).identity()


def test_mirabel_hides_an_isolated_match_not_a_flat_corpus():
    from master_script.core.rag import mirabel
    assert mirabel([0.05, 0.1, 0.06, 0.99])["detected"]
    assert mirabel([0.3, 0.3, 0.3, 0.3])["detected"] is False


def test_retrieval_hides_before_refilling_top_k():
    import numpy as np
    from master_script.core.rag import retrieve
    docs = [{"text": str(i)} for i in range(4)]
    vectors = np.array([[0.05], [0.1], [0.06], [0.99]])
    assert retrieve(docs, vectors, np.array([1.]), 2, 0.05, False)[0] == ["3", "1"]
    assert retrieve(docs, vectors, np.array([1.]), 2, 0.05, True) == (["1", "2"], True)


def test_invalid_membership_labels_are_rejected():
    from master_script.core.rag import validate_study
    study = json.loads((CONFIGS_DIR / "pipeline_demo_study.json").read_text())
    study["membership_candidates"][0]["id"] = "different-id-for-same-text"
    with pytest.raises(ValueError, match="another corpus ID"):
        validate_study(study)


def test_privacy_composition_and_noise_scaling():
    from master_script.core.defenses import privacy_bound
    one = privacy_bound(1, 1.0, 1e-5)
    two = privacy_bound(2, 1.0, 1e-5)
    more_noise = privacy_bound(1, 2.0, 1e-5)
    assert two["rho"] == 2 * one["rho"]
    assert more_noise["rho"] == one["rho"] / 4
    assert two["epsilon"] > one["epsilon"] > more_noise["epsilon"]


def test_client_delta_clipping_bounds_replacement_sensitivity():
    import numpy as np
    from master_script.core.defenses import clipped_mean
    previous = [np.array([10., -10.])]
    left = clipped_mean([[np.array([110., -10.])], [np.array([10., -10.])]], previous, 1., 0.)
    right = clipped_mean([[np.array([-90., -10.])], [np.array([10., -10.])]], previous, 1., 0.)
    assert np.linalg.norm(left[0] - right[0]) == pytest.approx(1.)  # 2C / two clients
    assert np.allclose(left[0], [10.5, -10.])


def test_private_training_clips_individual_records_before_averaging(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from master_script.core.defenses import private_train

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(loss=self.weight * input_ids[0, 0])

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[float(text), 0.]]), "attention_mask": torch.ones(1, 2)}

    captured = []

    class Optimizer:
        def __init__(self, params, **kwargs):
            self.params = params

        def zero_grad(self, **kwargs):
            for p in self.params:
                p.grad = None

        def step(self):
            captured.append(float(self.params[0].grad))

    monkeypatch.setattr(torch.optim, "AdamW", Optimizer)
    monkeypatch.setattr(torch, "randn", lambda shape, **kwargs: torch.zeros(shape))
    config = SimpleNamespace(client_lr=0.1, local_epochs=1, local_batch_size=2, max_length=8)
    assert private_train(Model(), Tokenizer(), ["100", "-1"], config, Defense(mechanism="dp_sgd")) == 1
    assert captured == pytest.approx([0.], abs=1e-6)  # mean-then-clip would produce +1


def test_private_strategy_composes_repeated_client_steps_and_rejects_failures():
    pytest.importorskip("flwr")
    import numpy as np
    from types import SimpleNamespace
    from flwr.common import ndarrays_to_parameters
    from master_script.core.defenses import strategy_class

    class Capture:
        def aggregate_fit(self, round_id, results, failures):
            assert all(set(r.metrics) == {"partition_id"} for _, r in results)
            return results[0][1].parameters, {}

    arrays = [np.array([0.])]
    privacy = {}
    strategy = strategy_class(Capture, Defense(mechanism="dp_sgd"), arrays, privacy)()

    def result(cid, steps):
        return None, SimpleNamespace(parameters=ndarrays_to_parameters(arrays), num_examples=1,
                                     metrics={"partition_id": cid, "dp_steps": steps, "loss": 42.})

    strategy.aggregate_fit(1, [result(0, 3), result(1, 2)], [])
    strategy.aggregate_fit(2, [result(1, 4)], [])
    assert privacy["steps"] == 6
    assert privacy["client_steps"] == {"0": 3, "1": 6}
    assert privacy["rho"] == 12
    with pytest.raises(RuntimeError, match="incomplete"):
        strategy.aggregate_fit(3, [result(0, 3)], [RuntimeError("client failed")])
    assert privacy["steps"] == 6


def test_answer_likelihood_masks_every_prompt_and_context_token():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from master_script.core.rag import answer_nll
    captured = {}

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(1, len(text.split()) + 1))

    class Model:
        config = SimpleNamespace(max_position_embeddings=100)

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(loss=torch.tensor(3.))

    settings = SimpleNamespace(max_new_tokens=8, max_context_tokens=20)
    assert answer_nll({"model": Model(), "tokenizer": Tokenizer(), "device": "cpu"},
                      "Which city?", "New York", ["private context"], settings) == 3.
    labels = captured["labels"][0]
    assert (labels[:-2] == -100).all()
    assert (labels[-2:] == captured["input_ids"][0, -2:]).all()


def test_parallel_child_serialization_preserves_pipeline_identity_and_cleans_files(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from master_script.perform_experiments import _run_parallel
    cfg, spec = load_config_file(CONFIGS_DIR / "pipeline_zlib_dp.yaml")[0]
    files = []

    def child(cmd, **kwargs):
        path = Path(cmd[cmd.index("--config") + 1])
        loaded_cfg, loaded_spec = load_config_file(path)[0]
        files.extend([path, Path(loaded_spec.pipeline.rag.study_file)])
        assert experiment_key(loaded_cfg, loaded_spec) == experiment_key(cfg, spec)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    assert _run_parallel([(cfg, spec)], SimpleNamespace(log_level="INFO", no_firestore=True,
                                                       keep_artifacts=False))[0]["status"] == "complete"
    assert all(not p.exists() for p in files)


def test_editor_resolves_rag_study_relative_to_configs_directory():
    from master_script.webui.configs import validate
    assert validate((CONFIGS_DIR / "pipeline_zlib_dp.yaml").read_text())["ok"]
