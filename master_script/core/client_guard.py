"""Stage-A client boundary: declarative rules and protected observation release.

Rules are prevention baselines, not a learned detector or a DP guarantee. The
trusted caller binds the architecture, parameter ordering, trainable layers and
round to client-owned model code. Server declarations alone are not attestation.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np

from .defenses import protect_observation


@dataclass(frozen=True)
class GuardPolicy:
    architecture: str
    parameter_shapes: tuple[tuple[int, ...], ...]
    trainable_layers: tuple[str, ...]
    release_budget: int
    observation_clip_norm: float = 1.0
    observation_noise_multiplier: float = 1.0
    operation: str = "gradient_observation"
    observation_defense: str = "gaussian"

    def __post_init__(self):
        if not isinstance(self.architecture, str) or not self.architecture.strip():
            raise ValueError("architecture must be nonempty")
        if (type(self.parameter_shapes) is not tuple or not self.parameter_shapes
                or any(type(s) is not tuple or any(type(n) is not int or n <= 0 for n in s)
                       for s in self.parameter_shapes)):
            raise ValueError("parameter_shapes must be immutable positive dimensions")
        if (type(self.trainable_layers) is not tuple or not self.trainable_layers
                or any(not isinstance(s, str) or not s.strip() for s in self.trainable_layers)
                or len(set(self.trainable_layers)) != len(self.trainable_layers)):
            raise ValueError("trainable_layers must be distinct nonempty names")
        if type(self.release_budget) is not int or self.release_budget <= 0:
            raise ValueError("release_budget must be a positive integer")
        for name in ("observation_clip_norm", "observation_noise_multiplier"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.operation != "gradient_observation" or self.observation_defense != "gaussian":
            raise ValueError("Stage-A guarded releases require Gaussian observation protection")

    @property
    def sha256(self):
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GuardRequest:
    request_id: str
    round_id: int
    operation: str
    architecture: str
    trainable_layers: tuple[str, ...]
    parameters: tuple


@dataclass(frozen=True)
class GuardResponse:
    decision: str
    gradient: tuple | None = None
    score: None = None
    pred_member: None = None

    def __post_init__(self):
        if self.decision not in ("accepted", "rejected"):
            raise ValueError("Unknown guard decision")
        if (self.decision == "accepted") != (self.gradient is not None):
            raise ValueError("Only accepted responses carry gradients")
        if self.score is not None or self.pred_member is not None:
            raise ValueError("Attacker scores do not belong in client responses")


@dataclass(frozen=True)
class GuardAudit:
    """Local only: never serialize into server-visible Flower metrics."""
    reason: str
    policy_sha256: str
    reservation_count: int
    accounting_scope: str = "client_gradient_observation"


def rule_violation(policy, request, expected_round):
    if not isinstance(request.request_id, str) or not request.request_id.strip():
        return "invalid_request_id"
    if type(expected_round) is not int or expected_round < 0:
        raise ValueError("expected_round must be a client-owned nonnegative integer")
    if type(request.round_id) is not int or request.round_id != expected_round:
        return "round_mismatch"
    if request.operation != policy.operation:
        return "operation_mismatch"
    if request.architecture != policy.architecture:
        return "architecture_mismatch"
    if request.trainable_layers != policy.trainable_layers:
        return "trainable_layers_mismatch"
    if type(request.parameters) is not tuple or len(request.parameters) != len(policy.parameter_shapes):
        return "parameter_shapes_mismatch"
    for array, shape in zip(request.parameters, policy.parameter_shapes):
        if not isinstance(array, np.ndarray) or array.shape != shape:
            return "parameter_shapes_mismatch"
        if array.dtype.kind not in "fi" or not np.isfinite(array).all():
            return "invalid_parameters"
    return None


def guarded_observation(policy, ledger, client_id, request, expected_round, compute_gradient):
    """Invoke the private callback only after validation and a durable debit.

    The callback receives the validated parameter snapshot. It must be trusted
    client code and return raw gradients; protection is applied exactly once
    here. Exceptions propagate as operational errors and never refund a debit.
    No fit_config or server policy overrides are accepted by this interface.
    """
    # Snapshot server arrays before validation to prevent mutation between check
    # and use. This deliberately uses extra memory in the correctness prototype.
    if type(request.parameters) is tuple:
        from dataclasses import replace
        request = replace(request, parameters=tuple(a.copy() if isinstance(a, np.ndarray) else a
                                                    for a in request.parameters))
    reason = rule_violation(policy, request, expected_round)
    if reason is None:
        reason = ledger.reserve(client_id, request.request_id, policy.release_budget)
    if reason is not None:
        return GuardResponse("rejected"), GuardAudit(reason, policy.sha256, ledger.count(client_id))
    gradients = compute_gradient(request.parameters)
    # Validate raw output before noise; numerical failures are not policy refusals.
    if not gradients or not all(np.asarray(g).dtype.kind in "fi" and np.isfinite(g).all() for g in gradients):
        raise ValueError("Private gradients must be finite and nonempty")
    protected = tuple(protect_observation(gradients, policy))
    if not all(np.isfinite(g).all() for g in protected):
        raise FloatingPointError("Nonfinite protected gradient")
    return GuardResponse("accepted", protected), GuardAudit("accepted", policy.sha256, ledger.count(client_id))
