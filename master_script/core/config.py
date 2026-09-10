# master_script/core/config.py
"""Versioned scientific identity, source revisions, validation and sweep expansion.

Legacy digest shapes are retained inside a method/source-version namespace;
corrected algorithms deliberately cannot reuse historical completed results.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence
import json
import math
import re

from ..paths import ARTIFACTS_DIR

METHOD_VERSION = "theory_v2"


@dataclass(frozen=True)
class AttackConfig:
    """Resolved source revisions are part of the corrected scientific identity."""
    model_revision: Optional[str] = None
    reference_revision: Optional[str] = None
    dataset_revision: Optional[str] = None


def stable_json(payload: Any) -> str:
    """AMIA/LOSS variant: tolerates non-JSON types via default=str."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def key_sha16(config: AttackConfig) -> str:
    """The nine modern notebooks. Note: no default=str."""
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def key_sha24_default_str(config: AttackConfig) -> str:
    """AMIA_adaptation.ipynb: 24 chars, not 16."""
    return sha256(stable_json(asdict(config)).encode("utf-8")).hexdigest()[:24]


def key_named_prefix(config: AttackConfig) -> str:
    """LOSS_adaptation.ipynb: f'{experiment_name}_{digest16}'."""
    digest = sha256(stable_json(asdict(config)).encode("utf-8")).hexdigest()[:16]
    return f"{config.experiment_name}_{digest}"


def legacy_experiment_key(config: AttackConfig, spec: Optional[Any] = None) -> str:
    """Dispatch to the attack's own key formula.

    The spec=None fallback is the modern 16-char formula, correct only for the
    nine modern attacks. Always pass spec when you have it.
    """
    legacy = spec.key_fn(config) if spec is not None and getattr(spec, "key_fn", None) is not None else key_sha16(config)
    pipeline = getattr(spec, "pipeline", None)
    if pipeline is None:
        return legacy
    digest = sha256(stable_json({"version": 1, "attack_key": legacy,
                                "pipeline": pipeline.identity()}).encode()).hexdigest()[:24]
    return f"pipeline_v1_{digest}"


def experiment_key(config: AttackConfig, spec: Optional[Any] = None) -> str:
    """Corrected methods never reuse completed results from the old algorithms."""
    return f"{METHOD_VERSION}_{implementation_fingerprint()}_{legacy_experiment_key(config, spec)}"


def implementation_fingerprint():
    """Hash maintained scientific source, not generated files or credentials."""
    root = Path(__file__).parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def resolve_run_config(config):
    """Resolve mutable Hub refs before cache lookup; executed only on the server."""
    from .datasets import uses_real_dataset, dataset_spec
    updates = {}

    def pin(identifier, revision, repo_type):
        if Path(identifier).is_dir():
            digest = sha256()
            for file in sorted(Path(identifier).rglob("*")):
                if file.is_file():
                    digest.update(str(file.relative_to(identifier)).encode())
                    with file.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
            return "local-sha256-" + digest.hexdigest()
        if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            return revision
        from huggingface_hub import HfApi
        info = HfApi().repo_info(identifier, repo_type=repo_type, revision=revision or "main")
        if not info.sha:
            raise RuntimeError(f"Could not resolve an immutable revision for {identifier}")
        return info.sha

    if getattr(config, "use_hf_models", True):
        updates["model_revision"] = pin(config.model_id, config.model_revision, "model")
        reference_id = getattr(config, "reference_model_id", None) or config.model_id
        updates["reference_revision"] = (updates["model_revision"] if reference_id == config.model_id and config.reference_revision is None
                                          else pin(reference_id, config.reference_revision, "model"))
    if getattr(config, "use_hf_models", True) and hasattr(config, "neighbor_model_id"):
        updates["neighbor_model_revision"] = pin(config.neighbor_model_id, config.neighbor_model_revision, "model")
    if uses_real_dataset(config):
        updates["dataset_revision"] = pin(dataset_spec(config.dataset_name).hub_path, config.dataset_revision, "dataset")
    return replace(config, **updates)


def validate_attack_config(config, spec=None):
    """Validate scientific input ranges equally for direct, YAML and pipeline runs."""
    for name in ("num_clients", "clients_per_round", "federated_rounds", "local_epochs",
                 "local_batch_size", "max_length", "attack_trials", "attack_batch_size",
                 "probe_epochs", "num_samples", "num_neighbours", "num_paraphrases",
                 "self_prompt_tokens", "calibration_nonmember_count", "rouge_n",
                 "reference_samples", "reference_epochs", "reference_batch_size",
                 "reference_generation_length", "generation_max_length", "adversary_negative_count",
                 "ldp_target_samples", "certificate_samples"):
        value = getattr(config, name, None)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if config.attack_trials < 2 or config.max_length < 2:
        raise ValueError("At least two trials and two tokens are required")
    if config.clients_per_round > config.num_clients:
        raise ValueError("clients_per_round cannot exceed num_clients")
    if type(config.target_client_id) is not int or not 0 <= config.target_client_id < config.num_clients:
        raise ValueError("target_client_id must identify a configured client")
    for name in ("client_lr", "probe_lr", "loss_bound", "reference_lr", "embedding_noise_scale", "epsilon"):
        value = getattr(config, name, None)
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    for name in ("threshold", "gradient_threshold", "sim_num_gpus"):
        value = getattr(config, name, None)
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    for name, lower, upper in (("min_k_percent", 0, 100), ("mask_ratio", 0, 1)):
        value = getattr(config, name, None)
        if value is not None and (isinstance(value, bool) or not lower < value <= upper):
            raise ValueError(f"{name} must be in ({lower}, {upper}]")
    if hasattr(config, "num_shots") and (type(config.num_shots) is not int or not 1 <= config.num_shots <= 4):
        raise ValueError("num_shots must select between one and four available public shots")
    if hasattr(config, "neighbour_swaps") and config.neighbour_swaps != 1:
        raise ValueError("Only the paper's single-position neighborhood setting is supported")
    if hasattr(config, "window_sizes"):
        sizes = config.window_sizes
        if not sizes or any(type(w) is not int or w <= 0 for w in sizes) or len(set(sizes)) != len(sizes):
            raise ValueError("window_sizes must contain distinct positive integers")
    if getattr(config, "decision_rule", "bounded_randomized") not in ("bounded_randomized", "nonmember_quantile"):
        raise ValueError("Unknown LOSS decision_rule")
    if getattr(config, "ldp_mechanism", "none") not in ("none", "BitRand", "OME"):
        raise ValueError("Unknown AMIA LDP mechanism")
    if hasattr(config, "certificate_delta") and not 0 < config.certificate_delta < 1:
        raise ValueError("certificate_delta must be in (0, 1)")
    if hasattr(config, "reference_generation_length") and config.reference_generation_length <= config.self_prompt_tokens:
        raise ValueError("Reference generation must have a positive continuation budget")
    if hasattr(config, "threshold_quantile") and not 0 <= config.threshold_quantile <= 1:
        raise ValueError("threshold_quantile must be in [0, 1]")


def expand_sweep(base_config, sweep: Dict[str, Sequence]) -> Iterator:
    keys = list(sweep.keys())
    if not keys:
        yield base_config
        return
    for values in product(*(sweep[key] for key in keys)):
        yield replace(base_config, **dict(zip(keys, values)))


def artifact_dir_for(config: AttackConfig, spec: Optional[Any] = None) -> Path:
    root = getattr(config, "artifact_root", None) or getattr(config, "local_artifact_dir")
    return ARTIFACTS_DIR / Path(root).name / experiment_key(config, spec)
