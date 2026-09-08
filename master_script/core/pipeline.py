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


@dataclass(frozen=True)
class Pipeline:
    defense: Defense
    rag: Rag | None = None

    def metadata(self):
        data = asdict(self)
        if data["rag"]:
            # Record provenance, never private document contents, in results.
            data["rag"].pop("study_json")
        return data

    def identity(self):
        data = self.metadata()
        if data["rag"]:
            data["rag"].pop("study_file")
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
    value = _mapping(value, ("defense", "rag"), "pipeline")
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
                                     "max_new_tokens", "significance"), "pipeline.rag")
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
        for name in ("top_k", "max_context_tokens", "max_new_tokens"):
            v = getattr(rag, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"rag.{name} must be a positive integer")
        _positive(rag.significance, "rag.significance")
        if rag.significance >= 1:
            raise ValueError("rag.significance must be below 1")
        if not isinstance(rag.embedding_model, str) or not rag.embedding_model.strip():
            raise ValueError("rag.embedding_model must be a nonempty model identifier")
    if defense.mechanism == "none" and rag is None:
        return None
    return Pipeline(defense, rag)


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
