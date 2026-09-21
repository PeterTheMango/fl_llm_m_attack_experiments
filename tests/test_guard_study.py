from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from master_script.core.guard_replay import adaptive_public_requests, interpolate_request, repeated_query_curve
from master_script.core.guard_features import FEATURE_NAMES, FEATURE_SCHEMA
from master_script.tools.analyze_guard_study import clustered_mean_interval, paired_utility_report


def test_bounded_public_evasion_preserves_shapes():
    detector = {"schema": FEATURE_SCHEMA, "features": list(FEATURE_NAMES), "mean": [0]*3, "scale": [1]*3,
                "weights": [1,0,0], "intercept": 0, "threshold": .75}
    requests = list(adaptive_public_requests([np.ones(2)], [np.ones(2)*10], detector, 5))
    assert len(requests) == 5
    assert not requests[0]["public_decision"]
    assert all(r["parameters"][0].shape == (2,) for r in requests)
    assert requests[1]["fraction"] < requests[0]["fraction"]
    with pytest.raises(ValueError):
        interpolate_request([np.ones(2)], [np.ones(3)], .5)


def test_repeated_curve_uses_targets_not_batches():
    rows = [{"trial_id": i, "truth_member": i % 2 == 0, "target_sha256": "one",
             "decision": "rejected", "score": None} for i in range(8)]
    curve = repeated_query_curve(rows)
    assert len(curve) == 4 and all(c["independent_targets"] == 1 for c in curve)
    assert all(c["gradient"] is None and c["refusal"]["roc_auc"] == .5 for c in curve)


def test_cluster_intervals_do_not_count_repeated_questions_as_groups():
    rows = [{"group": "one", "f1": 1.}] * 100
    assert clustered_mean_interval(rows, "f1", "group") == {"mean": 1., "interval_95": None, "groups": 1}


def test_zero_utility_is_not_a_privacy_success():
    common = {"attack_name": "amia", "config": {"seed": 7, "model_id": "m", "dataset_name": "d"}, "status": "complete"}
    base = {**common, "run_id": "base", "pipeline": {"condition": "baseline"},
            "pipeline_evaluations": [{"trial_id": 0, "no_retrieval_utility": {"token_f1": .5, "exact_match": .3}}]}
    result = {**common, "run_id": "guard", "pipeline": {"condition": "guard"},
              "pipeline_evaluations": [{"trial_id": 0, "no_retrieval_utility": {"token_f1": 0., "exact_match": 0.}}]}
    assert paired_utility_report([base, result])[0]["slices"][0]["utility_gate"] is False


def test_paired_rag_datastore_worlds_share_checkpoint_and_size(monkeypatch):
    from master_script.core import rag
    monkeypatch.setattr(rag, "embed", lambda texts, model: np.ones((len(texts), 1)))
    monkeypatch.setattr(rag, "generate_answer", lambda bundle, question, contexts, settings: "Yes" if "target" in contexts else "No")
    settings = SimpleNamespace(embedding_model="unused", top_k=3, significance=.05, defenses=("ordinary",))
    study = {"private_documents": [{"id": str(i), "text": str(i)} for i in range(3)]}
    cells = []
    for training_member in (False, True):
        bundle = {"target_record": "target", "training_records": ["target"] if training_member else []}
        cells += rag.evaluate_overlap(bundle, study, settings)
    assert {(r["training_member"], r["datastore_member"]) for r in cells} == {(False,False),(False,True),(True,False),(True,True)}
    assert all(r["document_count"] == 3 and r["pred_member"] == r["datastore_member"] for r in cells)


def test_pipeline_config_expands_both_attacks_without_experiments():
    from master_script.core.queue import load_batch
    batch = load_batch(["master_script/configs/client_guard_privacy_pilot.yaml"])
    assert len(batch.pairs) == 6
    assert {spec.name for _, spec in batch.pairs} == {"amia", "reference"}
    assert len({entry["run_id"] for entry in batch.entries}) == 6
