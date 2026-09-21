from dataclasses import replace

import numpy as np
import pytest

from master_script.core.client_guard import GuardPolicy, GuardRequest, GuardResponse, guarded_observation
from master_script.core.release_ledger import ReleaseLedger


def fixtures(tmp_path):
    policy = GuardPolicy("approved", ((2,),), ("weight",), 2)
    request = GuardRequest("one", 0, "gradient_observation", "approved", ("weight",), (np.ones(2),))
    return policy, request, ReleaseLedger(tmp_path / "ledger.sqlite")


@pytest.mark.parametrize("change", [
    {"architecture": "malicious_head"}, {"operation": "execute_code"},
    {"round_id": 1}, {"round_id": False}, {"request_id": ""},
    {"trainable_layers": ("extra",)}, {"parameters": (np.ones(3),)},
    {"parameters": (np.array([np.nan, 1]),)},
    {"parameters": (np.array([np.inf, 1]),)},
    {"parameters": (np.array(["x", "y"]),)},
])
def test_rejection_never_reads_private_data(tmp_path, change):
    policy, request, ledger = fixtures(tmp_path)
    def private(_):
        pytest.fail("Rejected request accessed private data")
    response, audit = guarded_observation(policy, ledger, "client", replace(request, **change), 0, private)
    assert response == GuardResponse("rejected")
    assert response.gradient is response.score is response.pred_member is None
    assert audit.reason != "accepted"
    assert ledger.count("client") == 0


def test_accepted_protected_and_retry_refused(tmp_path):
    policy, request, ledger = fixtures(tmp_path)
    calls = []
    def private(parameters):
        calls.append(parameters)
        return [np.array([3., 4.])]
    response, audit = guarded_observation(policy, ledger, "client", request, 0, private)
    assert response.decision == "accepted"
    assert np.isfinite(response.gradient[0]).all()
    assert not np.array_equal(response.gradient[0], [3., 4.])
    assert not np.shares_memory(calls[0][0], request.parameters[0])
    assert audit.policy_sha256 == policy.sha256
    restarted = ReleaseLedger(tmp_path / "ledger.sqlite")
    response, audit = guarded_observation(policy, restarted, "client", request, 0, private)
    assert response.decision == "rejected"
    assert audit.reason == "duplicate_request"
    assert len(calls) == 1


def test_failure_burns_reservation(tmp_path):
    policy, request, ledger = fixtures(tmp_path)
    def private(_):
        raise RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        guarded_observation(policy, ledger, "client", request, 0, private)
    assert ledger.count("client") == 1
    response, audit = guarded_observation(policy, ledger, "client", request, 0, private)
    assert response.decision == "rejected"


@pytest.mark.parametrize("change", [
    {"release_budget": True}, {"release_budget": 0},
    {"observation_noise_multiplier": 0}, {"observation_clip_norm": float("nan")},
    {"observation_defense": "none"}, {"parameter_shapes": [(2,)]},
    {"trainable_layers": ["weight"]}, {"architecture": ""},
])
def test_invalid_policy(tmp_path, change):
    policy, _, _ = fixtures(tmp_path)
    with pytest.raises(ValueError):
        replace(policy, **change)


def test_response_invariants():
    with pytest.raises(ValueError):
        GuardResponse("rejected", (np.zeros(1),))
    with pytest.raises(ValueError):
        GuardResponse("accepted")
