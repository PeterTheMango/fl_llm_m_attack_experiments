"""Local simulation wiring shared by training and AMIA.

Trust is supplied by the study harness, not by server fit_config. This is not
remote attestation. Each worker reads the same persistent local ledger and pinned
public checkpoint snapshot. Audit events are private research artifacts.
"""
from dataclasses import asdict, dataclass
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time
from zipfile import BadZipFile
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
    validation_workers: int = 1

    def metadata(self):
        data = {k: v for k, v in asdict(self).items() if k not in ("policy_json", "runtime_directory")}
        if self.validation_workers == 1:
            data.pop("validation_workers")  # Preserve the serial configuration identity.
        return data


def parse_guard(value, source):
    allowed = {"mode", "policy_file", "release_budget", "detector_file", "detector_sha256", "diagnostic", "rejection_action", "validation_workers"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Invalid client_guard fields")
    if value.get("mode") not in ("rules", "classifier", "rules_classifier", "shadow"):
        raise ValueError("client_guard.mode must be rules, classifier, rules_classifier or shadow")
    if type(value.get("release_budget")) is not int or value["release_budget"] <= 0:
        raise ValueError("client_guard.release_budget must be positive")
    if type(value.get("validation_workers", 1)) is not int or value.get("validation_workers", 1) not in (1, 2, 4):
        raise ValueError("validation_workers must be 1, 2 or 4")
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
    if (root.parent / "checkpoint-retirement.json").exists():
        raise ValueError("Guard scope was retired; preserve its ledger and use a new approved run")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A scope is chosen by the harness for one model/world, never fit_config.
    key = sha256(scope.encode()).hexdigest()
    path = root / (key + ".npz")
    # Never silently change the approved checkpoint for an existing scope.
    if not path.exists():
        from .storage import require_space, atomic_binary
        require_space(root, sum(np.asarray(p).nbytes + 65536 for p in parameters))
        try:
            with atomic_binary(path, exclusive=True) as stream:
                np.savez(stream, *parameters)
        except FileExistsError:
            # Another preparer published the same scope; validate below.
            pass
    try:
        with np.load(path, allow_pickle=False) as stored:
            if len(stored.files) != len(parameters) or any(not np.array_equal(stored[f"arr_{i}"], p) for i, p in enumerate(parameters)):
                raise ValueError("Approved checkpoint changed within guard scope")
    except (OSError, EOFError, ValueError, BadZipFile) as exc:
        raise ValueError(f"Invalid or changed guard checkpoint {path}. Preserve its ledger; "
                         "do not reuse a partial snapshot or reset accounting to retry.") from exc
    return GuardRuntime(settings, scope, str(path), rounds,
                        tuple(parameter_digest(p) for p in parameters))


def parameter_digest(value):
    """Hash exact dtype, shape and bytes; never use file timestamps as integrity."""
    array = np.ascontiguousarray(value)
    digest = sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _inspect_layer(stored, index, value, expected_digest):
    """Each task owns one freshly read reference array; never cache its contents."""
    timings = {}
    tick = time.perf_counter()
    prior = stored[f"arr_{index}"]
    timings["reference_load"] = time.perf_counter() - tick
    tick = time.perf_counter()
    if parameter_digest(prior) != expected_digest:
        raise ValueError("Approved checkpoint integrity mismatch")
    timings["reference_hash"] = time.perf_counter() - tick
    tick = time.perf_counter()
    reason = None
    if value.shape != prior.shape:
        reason = "parameter_shapes_mismatch"
    elif value.dtype != prior.dtype:
        reason = "parameter_dtype_mismatch"
    elif value.dtype.kind not in "fi" or not np.isfinite(value).all():
        reason = "invalid_parameters"
    timings["validation"] = time.perf_counter() - tick
    tick = time.perf_counter()
    features = parameter_features([value], [prior]) if reason is None else None
    timings["feature_extraction"] = time.perf_counter() - tick
    return reason, features, timings


@dataclass(frozen=True)
class GuardRuntime:
    settings: GuardSettings
    scope: str
    reference_path: str
    rounds: int
    reference_digests: tuple = ()

    def check(self, parameters, client_id, round_id, **kwargs):
        """Compatibility diagnostic. Execution callers must use authorize's snapshot."""
        return self.authorize(parameters, client_id, round_id, **kwargs) is not None

    def authorize(self, parameters, client_id, round_id, *, architecture="causal_lm", observation=False, reserve=True):
        """Return owned validated arrays, or None, before any private computation.

        Read each reference array once, verify its pinned content, and discard it
        after extracting that layer's features. No stale parameter cache is used.
        The caller executes ONLY the returned snapshot, closing check/use races.
        """
        from .runtime_memory import rss_gib
        start = time.perf_counter()
        workers = self.settings.validation_workers
        if type(workers) is not int or workers not in (1, 2, 4):
            raise ValueError("validation_workers must be 1, 2 or 4")
        rss_before = rss_gib()
        stages = dict.fromkeys(("request_copy", "request_hash", "reference_load", "reference_hash", "validation",
                               "feature_extraction", "detector_load", "detector_inference",
                               "ledger_setup", "ledger_reservation"), 0.)
        tick = time.perf_counter()
        snapshot = [np.array(p, copy=True) for p in parameters]
        stages["request_copy"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if workers == 1:
            digests = [parameter_digest(p) for p in snapshot]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                digests = list(pool.map(parameter_digest, snapshot))
        request_digest = sha256("".join(digests).encode()).hexdigest()
        stages["request_hash"] = time.perf_counter() - tick
        tick = time.perf_counter()
        ledger = ReleaseLedger(Path(self.settings.runtime_directory) / "release-ledger.sqlite")
        stages["ledger_setup"] = time.perf_counter() - tick
        client = f"{self.scope}:{client_id}"
        reason, features, score = None, None, None
        if type(round_id) is not int or not 1 <= round_id <= self.rounds:
            reason = "round_mismatch"
        elif observation and self.settings.mode != "classifier" and architecture != json.loads(self.settings.policy_json)["observation_architecture"]:
            reason = "architecture_mismatch"
        if not self.reference_digests:
            raise ValueError("Guard runtime requires pinned reference contents")
        # The reference stays pinned to the approved initialization. Later FL
        # parameters are measured against it, not silently promoted to trusted.
        layer_features = []
        reference_start = time.perf_counter()
        with np.load(self.reference_path, allow_pickle=False) as stored:
            if len(snapshot) != len(stored.files) or len(snapshot) != len(self.reference_digests):
                reason = "parameter_shapes_mismatch"
            elif reason in (None, "architecture_mismatch"):
                def inspect(i):
                    return _inspect_layer(stored, i, snapshot[i], self.reference_digests[i])
                # ZipFile serializes seeks/reads through its shared-file lock.
                # Hashing and feature reductions can overlap on owned arrays.
                # Results are reduced in tensor order, preserving feature math.
                if workers == 1:
                    inspected = map(inspect, range(len(snapshot)))
                    for layer_reason, layer_values, timings in inspected:
                        for key, seconds in timings.items():
                            stages[key] += seconds
                        if layer_reason:
                            reason = layer_reason
                            break
                        layer_features.append(layer_values)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for layer_reason, layer_values, timings in pool.map(inspect, range(len(snapshot))):
                            for key, seconds in timings.items():
                                stages[key] += seconds
                            if layer_reason and reason in (None, "architecture_mismatch"):
                                reason = layer_reason
                            if layer_values is not None:
                                layer_features.append(layer_values)
                if len(layer_features) == len(snapshot) and snapshot:
                    features = {"relative_delta": float(np.mean([f["relative_delta"] for f in layer_features])),
                                "maximum_layer_delta": max(f["maximum_layer_delta"] for f in layer_features),
                                "parameter_concentration": max(f["parameter_concentration"] for f in layer_features)}
        reference_wall_seconds = time.perf_counter() - reference_start
        # Classifier-only is diagnostic: structural/numerical safety still applies.
        if features is not None and self.settings.detector_file is not None and reason in (None, "architecture_mismatch"):
            tick = time.perf_counter()
            detector = load_detector(self.settings.detector_file, self.settings.detector_sha256)
            stages["detector_load"] = time.perf_counter() - tick
            tick = time.perf_counter()
            score = detector_score(detector, features)
            stages["detector_inference"] = time.perf_counter() - tick
            if reason is None and score > detector["threshold"]:
                reason = "detector_threshold"
        proposed = reason is None
        accepted = proposed or (self.settings.mode == "shadow" and reason in ("architecture_mismatch", "detector_threshold"))
        tick = time.perf_counter()
        if accepted and reserve:
            ledger_reason = ledger.reserve(client, str(round_id), self.settings.release_budget)
            if ledger_reason:
                accepted, reason = False, ledger_reason
        count = ledger.count(client)
        stages["ledger_reservation"] = time.perf_counter() - tick
        event = {"scope": self.scope, "client_id": client_id, "round_id": round_id,
                 "decision": "accepted" if accepted else "rejected", "would_accept": proposed,
                 "reason": reason or "accepted", "mode": self.settings.mode,
                 "features": features, "detector_score": score, "feature_schema": FEATURE_SCHEMA,
                 "policy_sha256": self.settings.policy_sha256, "detector_sha256": self.settings.detector_sha256,
                 "reservation_count": count, "seconds": time.perf_counter() - start,
                 "stages_seconds": stages,
                 "validation_workers": workers,
                 "reference_validation_wall_seconds": reference_wall_seconds,
                 "stage_timing_scope": "request stages are wall time; reference stages sum per-layer elapsed time and may overlap with multiple workers",
                 "request_bytes": sum(p.nbytes for p in snapshot), "request_sha256": request_digest,
                 "rss_gib_before": rss_before, "rss_gib_after": rss_gib(),
                 "memory_scope": "process RSS samples; not peak GPU allocation",
                 "accounting_scope": "observation" if observation else "training",
                 "audit_outputs_private": True}
        tick = time.perf_counter()
        with closing(sqlite3.connect(ledger.path, timeout=30)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS guard_events (event TEXT NOT NULL)")
            db.execute("INSERT INTO guard_events VALUES (?)", (json.dumps(event, allow_nan=False),))
        # Store audit commit latency separately: including the measurement itself
        # in the original timed write would require a recursive measurement.
        with closing(sqlite3.connect(ledger.path, timeout=30)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS guard_audit_timings (seconds REAL NOT NULL)")
            db.execute("INSERT INTO guard_audit_timings VALUES (?)", (time.perf_counter() - tick,))
        return snapshot if accepted else None


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


def record_training_round(runtime, round_id, results, *, status, seconds, failures=0):
    """Durable progress even if Flower propagates an aborted-round exception."""
    if runtime is None:
        return
    row = {"scope": runtime.scope, "round": round_id, "status": status,
           "seconds": seconds, "infrastructure_failures": failures,
           "clients": [{"client_id": r.metrics.get("partition_id"),
                        "examples": r.num_examples,
                        "decision": r.metrics.get("guard_decision", "accepted" if r.num_examples else "not_scheduled"),
                        "train_loss": r.metrics.get("train_loss"),
                        "model_load_seconds": r.metrics.get("model_load_seconds")}
                       for _, r in results],
           "recovery": "stop_world_preserve_ledger_require_new_approved_run" if status != "completed" else None}
    path = Path(runtime.settings.runtime_directory) / "release-ledger.sqlite"
    with closing(sqlite3.connect(path, timeout=30)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS training_rounds (event TEXT NOT NULL)")
        db.execute("INSERT INTO training_rounds VALUES (?)", (json.dumps(row, allow_nan=False),))


def read_training_rounds(directory):
    path = Path(directory) / "release-ledger.sqlite"
    if not path.exists():
        return []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='training_rounds'").fetchone():
            return []
        return [json.loads(r[0]) for r in db.execute("SELECT event FROM training_rounds ORDER BY rowid")]



def read_guard_storage_profile(directory):
    path = Path(directory) / "release-ledger.sqlite"
    if not path.exists():
        return {"audit_commits": 0, "audit_commit_seconds": 0.}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='guard_audit_timings'").fetchone():
            return {"status": "unavailable_legacy"}
        count, total = db.execute("SELECT COUNT(*), SUM(seconds) FROM guard_audit_timings").fetchone()
        return {"audit_commits": count, "audit_commit_seconds": total or 0.,
                "scope": "primary event commit; excludes instrumentation's secondary timing commit"}
