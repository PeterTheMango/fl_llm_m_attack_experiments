"""Opt-in pipeline settings, kept outside the historically hashed attack configs."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Defense:
    mechanism: str = "none"
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5


@dataclass(frozen=True)
class Rag:
    study_file: str
    study_sha256: str
    study_json: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    max_context_tokens: int = 256
    max_new_tokens: int = 32
    significance: float = 0.05
    prompt_format: str = "plain"
    evaluation_trials: int | None = None
    defenses: tuple = ("ordinary", "mirabel")
    membership_overlap: bool = False


@dataclass(frozen=True)
class Pipeline:
    defense: Defense
    rag: Rag | None = None
    condition: str = ""
    client_guard: object | None = None

    def metadata(self):
        data = asdict(self)
        if self.client_guard is None:
            data.pop("client_guard")
        else:
            data["client_guard"] = self.client_guard.metadata()
        if not data["condition"]:
            data.pop("condition")
        if data["rag"]:
            # Record provenance, never private document contents, in results.
            data["rag"].pop("study_json")
            if not data["rag"]["membership_overlap"]:
                data["rag"].pop("membership_overlap")
            if data["rag"]["prompt_format"] == "plain":
                data["rag"].pop("prompt_format")  # Preserve existing pipeline identities.
            if data["rag"]["evaluation_trials"] is None:
                data["rag"].pop("evaluation_trials")
            if tuple(data["rag"]["defenses"]) == ("ordinary", "mirabel"):
                data["rag"].pop("defenses")
            else:
                data["rag"]["defenses"] = list(data["rag"]["defenses"])
        return data

    def identity(self):
        data = self.metadata()
        if data["rag"]:
            data["rag"].pop("study_file")
        if self.client_guard is not None:
            for key in ("policy_file", "detector_file"):
                data["client_guard"].pop(key, None)
        return data


def _mapping(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{name}: unknown fields: {', '.join(sorted(unknown))}")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def parse_pipeline(value, source="<config>"):
    if value is None:
        return None
    value = _mapping(value, ("defense", "rag", "condition", "client_guard"), "pipeline")
    condition = value.get("condition", "")
    if not isinstance(condition, str):
        raise ValueError("pipeline.condition must be a string")
    d = _mapping(value.get("defense", {}), Defense.__dataclass_fields__, "pipeline.defense")
    defense = Defense(**d)
    if defense.mechanism not in ("none", "dp_sgd", "dp_fedavg"):
        raise ValueError("defense.mechanism must be none, dp_sgd, or dp_fedavg")
    for name in ("clip_norm", "noise_multiplier", "delta"):
        _positive(getattr(defense, name), f"defense.{name}")
    if defense.delta >= 1:
        raise ValueError("defense.delta must be below 1")
    rag = None
    if value.get("rag") is not None:
        r = _mapping(value["rag"], ("study_file", "embedding_model", "top_k", "max_context_tokens",
                                     "max_new_tokens", "significance", "prompt_format",
                                     "evaluation_trials", "defenses", "membership_overlap"), "pipeline.rag")
        if not isinstance(r.get("study_file"), str) or not r["study_file"]:
            raise ValueError("rag.study_file is required")
        base = Path(source).resolve().parent if source != "<config>" else Path.cwd()
        path = (base / r["study_file"]).resolve()
        raw = path.read_bytes()
        study = json.loads(raw)
        from .rag import validate_study
        validate_study(study)
        rag = Rag(**{**r, "study_file": str(path), "study_sha256": sha256(raw).hexdigest(),
                     "study_json": raw.decode("utf-8")})
        if type(rag.membership_overlap) is not bool:
            raise ValueError("rag.membership_overlap must be a boolean")
        for name in ("top_k", "max_context_tokens", "max_new_tokens"):
            v = getattr(rag, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"rag.{name} must be a positive integer")
        _positive(rag.significance, "rag.significance")
        if rag.significance >= 1:
            raise ValueError("rag.significance must be below 1")
        if not isinstance(rag.embedding_model, str) or not rag.embedding_model.strip():
            raise ValueError("rag.embedding_model must be a nonempty model identifier")
        if rag.prompt_format not in ("plain", "chat"):
            raise ValueError("rag.prompt_format must be plain or chat")
        if rag.evaluation_trials is not None and (type(rag.evaluation_trials) is not int or rag.evaluation_trials <= 0):
            raise ValueError("rag.evaluation_trials must be a positive integer or null")
        if (not isinstance(rag.defenses, (list, tuple)) or not rag.defenses
                or any(d not in ("ordinary", "mirabel", "instruction", "mirabel_instruction") for d in rag.defenses)
                or len(set(rag.defenses)) != len(rag.defenses)):
            raise ValueError("rag.defenses must contain distinct ordinary/mirabel/instruction/mirabel_instruction values")
    from .guard_runtime import parse_guard
    guard = parse_guard(value["client_guard"], source) if value.get("client_guard") is not None else None
    if defense.mechanism == "none" and rag is None and guard is None:
        return None
    return Pipeline(defense, rag, condition, guard)


def validate_pipeline_run(config, spec, pipeline):
    if pipeline is None:
        return
    for name in ("num_clients", "clients_per_round", "federated_rounds", "local_epochs",
                 "local_batch_size", "max_length", "attack_trials"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer for pipeline runs")
    if config.clients_per_round > config.num_clients:
        raise ValueError("clients_per_round cannot exceed num_clients for pipeline runs")
    if not 0 <= config.target_client_id < config.num_clients:
        raise ValueError("target_client_id must identify a configured client")
    if config.max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not getattr(config, "use_hf_models", True):
        raise ValueError("Defense/RAG pipeline runs require real models; the toy model is not a DP or RAG implementation")

    if pipeline.client_guard is not None:
        guard = pipeline.client_guard
        if spec.name not in ("amia", "reference"):
            raise ValueError("Client guard currently supports AMIA and Reference only")
        if not guard.diagnostic and guard.mode != "shadow":
            if pipeline.defense.mechanism != "dp_sgd":
                raise ValueError("Enforced guarded training requires client-side dp_sgd or explicit diagnostic ablation")
            if spec.name == "amia" and config.observation_defense != "gaussian":
                raise ValueError("Guarded AMIA requires Gaussian observation noise or explicit diagnostic ablation")
