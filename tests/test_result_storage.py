import copy
import json
import pytest

from master_script.core.result_storage import pack_result, unpack_result


def large_result():
    trial = {"candidate_index": 1, "truth_member": True, "score": .75,
             "response_kind": "yes", "retrieved": True, "pred_member": True}
    condition = {"metrics": {"adv": .8}, "diagnostics": {"recognition_rate": 1.},
                 "membership_trials": [dict(trial, candidate_index=i) for i in range(200)],
                 "utility_trials": [{"token_f1": .75, "answer_nll": 2.}] * 10}
    return {"run_id": "large", "status": "complete", "attack_name": "amia",
            "metrics": {"adv": .8}, "attack_trials": [trial] * 200,
            "pipeline_evaluations": [{"trial_id": i, "rag_conditions": {
                f"{corpus}_{defense}": condition for corpus in ("public", "private")
                for defense in ("ordinary", "mirabel", "instruction", "mirabel_instruction")}}
                for i in range(5)]}


def test_full_five_target_rag_result_fits_and_roundtrips_without_losing_trials():
    result = large_result()
    assert len(json.dumps(result).encode()) > 1_048_576
    packed = pack_result(result)
    assert len(json.dumps(packed).encode()) < 900_000
    assert packed["metrics"] == result["metrics"]
    assert packed["pipeline_evaluations"][0]["rag_conditions"]["public_ordinary"]["metrics"] == {"adv": .8}
    assert unpack_result(packed) == result
    from master_script.webui.state import DashboardState
    state = DashboardState()
    state.ingest([packed])
    assert state.runs["large"] == result


def test_checksum_failure_and_small_legacy_result():
    packed = pack_result(large_result())
    broken = copy.deepcopy(packed)
    broken["detail_storage"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        unpack_result(broken)
    small = {"status": "complete", "metrics": {"adv": .5}}
    assert pack_result(small) == small
    assert unpack_result(small) == small


def test_firestore_cache_restores_large_results(monkeypatch):
    from master_script.core import firestore
    from master_script.core.attacks.zlib import ZlibConfig
    from tests.test_firestore import _NamedFakeDB
    db = _NamedFakeDB()
    monkeypatch.setattr(firestore, "get_firestore_client", lambda *a, **k: db)
    monkeypatch.setattr(firestore, "_delete_field_sentinel", lambda: None)
    result = large_result()
    assert firestore.save_result(ZlibConfig(), result)
    assert firestore.load_cached_result(ZlibConfig()) == result
