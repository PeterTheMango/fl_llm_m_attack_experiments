from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from master_script.core.guard_features import parameter_features
from master_script.tools.benchmark_guard_runtime import benchmark, legacy_features
from master_script.tools.analyze_guard_study import summarize_results


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64])
def test_feature_fast_path_preserves_small_updates_and_chunk_boundaries(dtype):
    rng = np.random.default_rng(21)
    prior = rng.normal(size=(257, 257)).astype(dtype)
    value = (prior + .001).astype(dtype)
    # Non-contiguous arrays, a scalar layer and an empty layer.
    values = [value.T, np.array(1, dtype=dtype), np.array([], dtype=dtype)]
    priors = [prior.T, np.array(2, dtype=dtype), np.array([], dtype=dtype)]
    observed, expected = parameter_features(values, priors), legacy_features(values, priors)
    assert observed == pytest.approx(expected, rel=1e-11, abs=1e-25)


@pytest.mark.parametrize("scale", [0., 1e-45, 1e-30, 1e-15, 1., 1e30, 1e38])
def test_float32_extrema_and_denominator_floor(scale):
    prior = np.array([1, -1, 0], dtype=np.float32) * np.float32(scale)
    value = -prior
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        observed = parameter_features([value], [prior])
        expected = legacy_features([value], [prior])
    assert observed == pytest.approx(expected, rel=1e-11, abs=1e-25)
    assert all(np.isfinite(list(observed.values())))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("in_reference", [False, True])
def test_fast_path_still_rejects_nonfinite_values_after_first_chunk(bad, in_reference):
    value, prior = np.ones(65540, dtype=np.float32), np.ones(65540, dtype=np.float32)
    (prior if in_reference else value)[-1] = bad
    with pytest.raises(ValueError, match="finite"):
        parameter_features([value], [prior])


def test_positive_but_below_floor_utility_is_excluded_without_hiding_privacy():
    result = {"run_id": "small", "status": "complete", "attack_name": "amia",
              "metrics": {"roc_auc": 1.},
              "pipeline_evaluations": [{"no_retrieval_utility": {"token_f1": .02}}]}
    report = summarize_results([result])
    assert report["excluded_runs"] == 1
    assert report["conditions"][0]["exclusion"] == "utility_below_absolute_floor"
    assert report["conditions"][0]["membership_auc"] == 1.
    assert not report["conditions"][0]["utility_valid"]


def test_cpu_benchmark_preserves_snapshot_and_historical_ledger_and_detects_tampering(tmp_path):
    from master_script.core.guard_runtime import prepare_guard, read_guard_events
    from tests.test_guard_integration import settings
    name = "0000-example.json"
    directory = tmp_path / "artifacts" / Path(name).stem / "client-guard"
    params = [np.arange(17, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    runtime = prepare_guard(replace(settings(tmp_path), runtime_directory=str(directory)),
                            "training:1000:True", params, rounds=1)
    assert runtime.authorize(params, 0, 1) is not None
    result = {"run_id": "example", "implementation_fingerprint": "old",
              "guard_events": read_guard_events(directory)}
    (tmp_path / name).write_text(json.dumps(result))
    (tmp_path / "manifest.json").write_text(json.dumps({"entries": [
        {"run_id": "example", "result_file": name, "status": "complete"}]}))
    def hashes():
        return {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    before = hashes()
    report = benchmark(tmp_path, repeats=2)
    assert report["features_agree"] and len(report["samples"]["current"]) == 2
    assert report["request_bytes"] == sum(p.nbytes for p in params)
    assert hashes() == before
    np.savez(runtime.reference_path, np.zeros(17, dtype=np.float32), np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="differs"):
        benchmark(tmp_path, repeats=1)


def test_no_context_answers_are_saved_only_in_private_audit(monkeypatch, tmp_path):
    from master_script.core import rag
    from master_script.core.pipeline import parse_pipeline
    study = Path(__file__).parents[1] / "master_script/configs/archive/pipeline_demo_study.json"
    pipeline = parse_pipeline({"rag": {"study_file": str(study), "defenses": ["ordinary"]}})
    pipeline = replace(pipeline, rag=replace(pipeline.rag, runtime_audit_directory=str(tmp_path)))
    monkeypatch.setattr(rag, "embed", lambda texts, *args: np.ones((len(texts), 1)))
    monkeypatch.setattr(rag, "generate_answer", lambda *args: "PRIVATE RAW ANSWER")
    monkeypatch.setattr(rag, "answer_nll", lambda *args: 1.)
    output = rag.evaluate_pipeline({}, None, pipeline, trial_id=0)
    rows = [json.loads(line) for line in (tmp_path / "rag-answers.jsonl").read_text().splitlines()]
    no_context = [r for r in rows if r["kind"] == "no_retrieval_utility"]
    assert len(no_context) == len(json.loads(pipeline.rag.study_json)["utility_queries"])
    assert all(r["answer"] == "PRIVATE RAW ANSWER" and r["contexts"] == [] for r in no_context)
    assert "PRIVATE RAW ANSWER" not in json.dumps(output)
    assert (tmp_path / "rag-answers.jsonl").stat().st_mode & 0o777 == 0o600
