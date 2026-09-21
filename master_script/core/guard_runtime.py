"""Local simulation wiring shared by training and AMIA.

Trust is supplied by the study harness, not by server fit_config. This is not
remote attestation. Each worker reads the same persistent local ledger and pinned
public checkpoint snapshot. Audit events are private research artifacts.
"""
from dataclasses import asdict, dataclass
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from .guard_features import FEATURE_SCHEMA, parameter_features
from .guard_detector import load_detector, detector_score
from .release_ledger import ReleaseLedger


class PolicyAbortedRound(RuntimeError):
    """A scheduled client refused; no partial aggregation is allowed."""


@dataclass(frozen=True)
class GuardSettings:
    mode: str
    policy_file: str
    policy_sha256: str
    policy_json: str
    release_budget: int
    detector_file: str | None = None
    detector_sha256: str | None = None
    diagnostic: bool = False
    rejection_action: str = "refuse"
    runtime_directory: str | None = None

    def metadata(self):
        return {k: v for k, v in asdict(self).items() if k not in ("policy_json", "runtime_directory")}


def parse_guard(value, source):
    allowed = {"mode", "policy_file", "release_budget", "detector_file", "detector_sha256", "diagnostic", "rejection_action"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Invalid client_guard fields")
    if value.get("mode") not in ("rules", "classifier", "rules_classifier", "shadow"):
        raise ValueError("client_guard.mode must be rules, classifier, rules_classifier or shadow")
    if type(value.get("release_budget")) is not int or value["release_budget"] <= 0:
        raise ValueError("client_guard.release_budget must be positive")
    if type(value.get("diagnostic", False)) is not bool or value.get("rejection_action", "refuse") != "refuse":
        raise ValueError("Invalid diagnostic or rejection_action")
    if not isinstance(value.get("policy_file"), str) or not value["policy_file"]:
        raise ValueError("client_guard.policy_file is required")
    base = Path(source).resolve().parent if source != "<config>" else Path.cwd()
    path = (base / value["policy_file"]).resolve()
    raw = path.read_bytes()
    policy = json.loads(raw)
    if not isinstance(policy, dict) or set(policy) != {"schema", "observation_architecture"} or policy["schema"] != "client_guard_v1" or policy["observation_architecture"] not in ("causal_lm", "amia_probe"):
        raise ValueError("Policy requires schema client_guard_v1 and approved observation_architecture")
    detector = value.get("detector_file")
    digest = value.get("detector_sha256")
    if value["mode"] == "rules" and detector is not None:
        raise ValueError("Rules mode must not load a classifier")
    if value["mode"] in ("classifier", "rules_classifier") and not detector:
        raise ValueError("Classifier mode requires a pinned detector")
    if (detector is None) != (digest is None):
        raise ValueError("Detector file and SHA256 must be supplied together")
    if detector is not None:
        if not isinstance(detector, str) or not detector or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid detector pin")
        detector = str((base / detector).resolve())
        load_detector(detector, digest)
    return GuardSettings(**{**value, "policy_file": str(path), "policy_sha256": sha256(raw).hexdigest(),
                            "policy_json": raw.decode(), "detector_file": detector})


def prepare_guard(settings, scope, parameters, *, rounds):
    if settings is None:
        return None
    if settings.runtime_directory is None:
        raise ValueError("Guard runtime requires a client-owned artifact directory")
    root = Path(settings.runtime_directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A scope is chosen by the harness for one model/world, never fit_config.
    key = sha256(scope.encode()).hexdigest()
    path = root / (key + ".npz")
    # Never silently change the approved checkpoint for an existing scope.
    if not path.exists():
        with path.open("xb") as stream:
            np.savez(stream, *parameters)
    else:
        with np.load(path, allow_pickle=False) as stored:
            if len(stored.files) != len(parameters) or any(not np.array_equal(stored[f"arr_{i}"], p) for i, p in enumerate(parameters)):
                raise ValueError("Approved checkpoint changed within guard scope")
    return GuardRuntime(settings, scope, str(path), rounds)


@dataclass(frozen=True)
class GuardRuntime:
    settings: GuardSettings
    scope: str
    reference_path: str
    rounds: int

    def check(self, parameters, client_id, round_id, *, architecture="causal_lm", observation=False, reserve=True):
        start = time.perf_counter()
        ledger = ReleaseLedger(Path(self.settings.runtime_directory) / "release-ledger.sqlite")
        client = f"{self.scope}:{client_id}"
        reason, features, score = None, None, None
        if type(round_id) is not int or not 1 <= round_id <= self.rounds:
            reason = "round_mismatch"
        elif observation and self.settings.mode != "classifier" and architecture != json.loads(self.settings.policy_json)["observation_architecture"]:
            reason = "architecture_mismatch"
        with np.load(self.reference_path, allow_pickle=False) as stored:
            if reason in (None, "architecture_mismatch"):
                if len(parameters) != len(stored.files) or any(np.asarray(p).shape != stored[f"arr_{i}"].shape for i, p in enumerate(parameters)):
                    reason = "parameter_shapes_mismatch"
                elif any(np.asarray(p).dtype != stored[f"arr_{i}"].dtype for i, p in enumerate(parameters)):
                    reason = "parameter_dtype_mismatch"
                elif any(np.asarray(p).dtype.kind not in "fi" or not np.isfinite(p).all() for p in parameters):
                    reason = "invalid_parameters"
                else:
                    features = parameter_features(parameters, (stored[f"arr_{i}"] for i in range(len(parameters))))
        if reason is None and self.settings.detector_file is not None:
            detector = load_detector(self.settings.detector_file, self.settings.detector_sha256)
            score = detector_score(detector, features)
            if score > detector["threshold"]:
                reason = "detector_threshold"
        proposed = reason is None
        # Shadow observes policy decisions but still enforces finite shapes and
        # release accounting. Architecture refusals can be observed diagnostically.
        accepted = proposed or (self.settings.mode == "shadow" and reason in ("architecture_mismatch", "detector_threshold"))
        if accepted and reserve:
            ledger_reason = ledger.reserve(client, str(round_id), self.settings.release_budget)
            if ledger_reason:
                accepted, reason = False, ledger_reason
        event = {"scope": self.scope, "client_id": client_id, "round_id": round_id,
                 "decision": "accepted" if accepted else "rejected", "would_accept": proposed,
                 "reason": reason or "accepted", "mode": self.settings.mode,
                 "features": features, "detector_score": score, "feature_schema": FEATURE_SCHEMA,
                 "policy_sha256": self.settings.policy_sha256, "detector_sha256": self.settings.detector_sha256,
                 "reservation_count": ledger.count(client), "seconds": time.perf_counter() - start,
                 "accounting_scope": "observation" if observation else "training",
                 "audit_outputs_private": True}
        with closing(sqlite3.connect(ledger.path, timeout=30)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS guard_events (event TEXT NOT NULL)")
            db.execute("INSERT INTO guard_events VALUES (?)", (json.dumps(event, allow_nan=False),))
        return accepted


def reject_training_round(results, failures):
    if failures:
        raise RuntimeError("Training round has infrastructure failures")
    if any(result.metrics.get("guard_decision") == "rejected" for _, result in results):
        raise PolicyAbortedRound("Client policy aborted the scheduled round; no partial aggregate released")


def read_guard_events(directory):
    path = Path(directory) / "release-ledger.sqlite"
    if not path.exists():
        return []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='guard_events'").fetchone():
            return []
        return [json.loads(row[0]) for row in db.execute("SELECT event FROM guard_events ORDER BY rowid")]
