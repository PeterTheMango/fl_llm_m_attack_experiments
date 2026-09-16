"""Focused FL/RAG study: independent calibration and actual release defenses."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from master_script.core.attacks import amia, reference
from master_script.core.calibration import nonmember_threshold
from master_script.core.config import experiment_key, validate_attack_config
from master_script.core.pipeline import Pipeline, Defense, parse_pipeline
from master_script.core.yaml_config import ConfigError, load_config_file, load_config_doc
from master_script.paths import CONFIGS_DIR


def test_archive_preserves_original_files_and_only_focused_configs_are_active():
    archive = CONFIGS_DIR / "archive"
    manifest = json.loads((archive / "manifest.json").read_text())
    assert len(manifest) == 19
    for row in manifest:
        assert sha256((archive / row["filename"]).read_bytes()).hexdigest() == row["sha256"]
    from master_script.webui.configs import listing
    assert {f["name"] for f in listing()} == {"amia_reference_rag_master.yaml", "amia_reference_rag_light.yaml"}


def test_focused_file_matches_seeds_and_keeps_defense_variants_distinct():
    path = CONFIGS_DIR / "amia_reference_rag_master.yaml"
    pairs = load_config_file(path)
    assert Counter(s.name for c, s in pairs) == {"amia": 27, "reference": 15}
    assert len({experiment_key(c, s) for c, s in pairs}) == 42
    assert len(load_config_file(path, only=["amia"])) == 27
    assert len(load_config_file(path, only=["reference"])) == 15
    assert len(pickle.loads(pickle.dumps(pairs))) == 42
    for c, s in pairs:
        assert c.threshold_mode == "calibrated"
        assert tuple(s.pipeline.rag.defenses) == ("ordinary", "mirabel", "instruction", "mirabel_instruction")
        assert s.pipeline.rag.prompt_format == "chat"
        if s.name == "amia":
            assert c.attack_targets == 5
            assert s.pipeline.defense.mechanism != "dp_fedavg"
    for attack in ("amia", "reference"):
        conditions = {s.pipeline.condition for c, s in pairs if s.name == attack}
        for condition in conditions:
            assert {c.seed for c, s in pairs if s.name == attack and s.pipeline.condition == condition} == {7, 1007, 2007}


@pytest.mark.parametrize("variants", [[], [None], [{"bad": 3}], {"base": {}}, [{"base": [1]}]])
def test_bad_variants_fail_before_compute(variants):
    with pytest.raises(ConfigError):
        load_config_doc({"attacks": {"reference": {"variants": variants}}})


def test_calibration_ties_do_not_raise_false_positive_rate():
    for scores in ([0.] * 100, list(range(100)), [0.] * 80 + [1.] * 20):
        result = nonmember_threshold(scores, .05)
        assert sum(s >= result["threshold"] for s in scores) <= 5
        assert result["population_fpr_guarantee"] is False
    assert nonmember_threshold(list(range(100)), .05)["threshold"] > 94
    with pytest.raises(ValueError, match="resolve"):
        nonmember_threshold([0, 1, 2], .01)
    with pytest.raises(ValueError, match="finite"):
        nonmember_threshold([math.nan] * 100, .05)


def test_reference_calibrates_without_using_candidate_label(monkeypatch):
    from master_script.core import datasets
    cfg = reference.ReferenceConfig(dataset_name="squad", use_hf_models=True,
                                    threshold_mode="calibrated", calibration_nonmember_count=20)
    records = [f"calibration{i}" for i in range(20)]
    monkeypatch.setattr(datasets, "calibration_records", lambda *a: records)
    seen = []
    def score(target, ref, text, max_length):
        seen.append(text)
        return int(text.removeprefix("calibration")) / 100 - 1
    monkeypatch.setattr(reference, "score_candidate_hf", score)
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text.encode())}
    target = {"training_records": ["private"], "target_record": "candidate", "tokenizer": Tokenizer()}
    calibrated = reference.calibrate(cfg, target, object())
    assert seen == records
    assert calibrated["calibration_fpr"] == .05
    assert sum((i / 100 - 1) >= calibrated["threshold"] for i in range(20)) == 1


def test_amia_fit_calibration_and_private_partitions_are_disjoint(monkeypatch):
    from master_script.core import datasets
    monkeypatch.setattr(datasets, "_load_dataset_pool", lambda name, pool_size, **k: tuple(f"record {i}" for i in range(pool_size)))
    cfg = amia.AmiaConfig(dataset_name="squad_research", threshold_mode="calibrated")
    fit = set(amia.adversary_records(cfg))
    calibration = set(amia.calibration_records(cfg))
    private = {r for p in amia.build_client_texts(cfg) for r in p}
    assert len(fit) == 64 and len(calibration) == 200
    assert not fit & calibration and not calibration & private and not fit & private


def test_observation_clipping_and_independent_noise_change_released_gradients():
    from master_script.core.defenses import protect_observation
    cfg = amia.AmiaConfig(observation_defense="clip", observation_clip_norm=2)
    clipped = protect_observation([np.array([30., 40.]), np.array([0.])], cfg)
    assert math.sqrt(sum(float(np.sum(x*x)) for x in clipped)) == pytest.approx(2)
    cfg = replace(cfg, observation_defense="gaussian")
    # An inactive neuron no longer discloses a deterministic exact-zero payload.
    first = protect_observation([np.zeros(128), np.zeros(1)], cfg)
    second = protect_observation([np.zeros(128), np.zeros(1)], cfg)
    assert amia.gradient_score(first) > 0 and not np.array_equal(first[0], second[0])


def test_client_applies_defense_to_the_actual_loss_gradient(monkeypatch):
    torch = pytest.importorskip("torch")
    probe = amia._ami_probe_cls()(1, 1)
    with torch.no_grad():
        for p in probe.parameters(): p.fill_(1)
    monkeypatch.setattr(amia, "sentence_embedding", lambda *a: torch.tensor([[1.]]))
    model = SimpleNamespace(get_output_embeddings=lambda: lambda f: torch.zeros(len(f), 2))
    tokenizer = lambda *a, **k: {"input_ids": [[0, 1]]}
    cfg = amia.AmiaConfig(observation_defense="clip", observation_clip_norm=.01)
    gradients = amia.client_loss_gradients(model, tokenizer, probe, ["private"], cfg)
    assert np.sqrt(sum(np.sum(g*g) for g in gradients)) == pytest.approx(.01)


def test_bfloat16_parameters_and_integer_buffers_roundtrip_for_flower():
    torch = pytest.importorskip("torch")
    from master_script.core.model_io import model_parameters
    model = torch.nn.Linear(2, 1).to(torch.bfloat16)
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    arrays = model_parameters(model)
    assert arrays[0].dtype == np.float32 and arrays[-1].dtype == np.int64
    before = model.weight.detach().clone()
    amia.set_parameters(model, arrays)
    assert torch.equal(model.weight, before)


def test_loading_bfloat16_checkpoint_explicitly_uses_float32(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from master_script.core.model_io import load_causal_model
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16))
    model.to(torch.bfloat16).save_pretrained(tmp_path)
    restored = load_causal_model(tmp_path)
    assert next(restored.parameters()).dtype == torch.float32


def test_multi_target_batch_ids_and_training_evaluations_are_not_duplicated(monkeypatch, tmp_path):
    calls = []
    def run(cfg, directory, pipeline):
        calls.append((cfg.seed, cfg.attack_trials))
        ts = [{"trial_id": i, "truth_member": i % 2 == 0, "pred_member": i % 2 == 0, "score": float(i % 2 == 0)}
              for i in range(cfg.attack_trials)]
        ts[0]["pipeline_evaluation"] = {"training_privacy": {"steps": 3}}
        return {"trials": ts, "context": {}}
    monkeypatch.setattr(amia, "_run_target", run)
    monkeypatch.setattr(amia.dataset_sources, "target_record_for", lambda cfg, default: str(cfg.seed))
    cfg = amia.AmiaConfig(dataset_name="squad", attack_targets=3, attack_trials=12, observation_defense="gaussian")
    out = amia.custom_trials_adapter(cfg, tmp_path, Pipeline(Defense()))
    assert calls == [(7, 4), (8, 4), (9, 4)]
    assert [t["trial_id"] for t in out["trials"]] == list(range(12))
    assert len({t["target_sha256"] for t in out["trials"]}) == 3
    assert sum("pipeline_evaluation" in t for t in out["trials"]) == 3
    assert out["context"]["observation_privacy"]["steps"] == 12
    result = amia.build_payload_adapter(cfg, out["trials"], tmp_path, out["context"])
    assert all("pipeline_evaluation" not in t for t in result["attack_trials"])


def test_rag_unrecognized_answers_are_flagged_and_all_defenses_evaluated(monkeypatch):
    from master_script.core import rag
    options = {"rag": {"study_file": str(CONFIGS_DIR / "archive" / "pipeline_demo_study.json"),
                       "defenses": ["ordinary", "mirabel", "instruction", "mirabel_instruction"]}}
    pipeline = parse_pipeline(options)
    monkeypatch.setattr(rag, "embed", lambda texts, *a: np.ones((len(texts), 1)))
    seen = []
    def generate(bundle, question, contexts, settings):
        seen.append(getattr(settings, "instruction_defense", False))
        return "unrelated output"
    monkeypatch.setattr(rag, "generate_answer", generate)
    monkeypatch.setattr(rag, "answer_nll", lambda *a: 10.)
    output = rag.evaluate_pipeline({}, None, pipeline)
    assert len(output["rag_conditions"]) == 8
    assert True in seen and False in seen
    assert not output["rag_validity"]["ordinary_conditions_informative"]
    for condition in output["rag_conditions"].values():
        assert condition["metrics"]["adv"] == .5
        assert condition["diagnostics"]["attack_status"] == "no_recognized_answers"
        assert condition["diagnostics"]["utility_status"] == "no_correct_answers"


@pytest.mark.parametrize("fields", [{"attack_trials": 6, "attack_targets": 2},
                                   {"observation_defense": "fake"}, {"observation_delta": 1},
                                   {"threshold_mode": "test_labels"}])
def test_invalid_amia_study_inputs_fail_fast(fields):
    with pytest.raises(ValueError):
        validate_attack_config(replace(amia.AmiaConfig(), **fields))
