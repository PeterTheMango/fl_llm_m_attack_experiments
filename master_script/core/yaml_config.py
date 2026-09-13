# master_script/core/yaml_config.py
"""YAML -> (config, spec) pairs, with fail-fast validation.

load_config_doc is the real loader; load_config_file only reads and parses.
Unknown keys inside an attack's base/sweep are errors: catching a typo here
saves a multi-hour run that would otherwise produce a differently-hashed,
silently-wrong experiment.
"""
from dataclasses import fields, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

from .config import expand_sweep, validate_attack_config
from .datasets import validate_dataset_name
from .firestore import RESULTS_COLLECTION
from .registry import ATTACKS


class ConfigError(ValueError):
    """Raised for any malformed config, always before compute starts."""


def _field_names(cls) -> set:
    return {f.name for f in fields(cls)}


def load_config_file(path, only: Optional[Sequence[str]] = None) -> List[Tuple[object, object]]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    return load_config_doc(yaml.safe_load(path.read_text()) or {}, only, source=str(path))


def load_config_doc(doc: dict, only: Optional[Sequence[str]] = None,
                    source: str = "<config>") -> List[Tuple[object, object]]:
    """Expand an already-parsed config document.

    Split out from load_config_file so the dashboard's manual mode -- which
    builds the same dict from a form and has no file to point at -- reaches
    this validation and expansion rather than reimplementing it. `source` is
    whatever the caller wants errors to name.
    """
    if not isinstance(doc, dict):
        raise ConfigError(f"{source}: expected a YAML mapping")
    # Explicit per-attack variants avoid meaningless cross products (e.g. an
    # epsilon sweep for a no-LDP attack) and work through the same CLI/UI loader.
    attack_sections = doc.get("attacks") or {}
    if not isinstance(attack_sections, dict):
        raise ConfigError(f"{source}: attacks must be a mapping")
    if any(isinstance(section, dict) and "variants" in section for section in attack_sections.values()):
        pairs = []
        for name, section in attack_sections.items():
            if name not in ATTACKS:
                raise ConfigError(f"{source}: unknown attack: {name}")
            if only is not None and name not in only:
                continue
            section = section or {}
            if not isinstance(section, dict):
                raise ConfigError(f"{source}: attack '{name}' must be a mapping")
            variants = section.get("variants", [{}])
            if not isinstance(variants, list) or not variants:
                raise ConfigError(f"{source}: {name}.variants must be a nonempty list")
            for variant in variants:
                if not isinstance(variant, dict) or set(variant) - {"base", "sweep", "pipeline"}:
                    raise ConfigError(f"{source}: {name} variant accepts base, sweep and pipeline only")
                expanded_section = {k: v for k, v in section.items() if k != "variants"}
                for key in ("base", "sweep"):
                    inherited, override = section.get(key) or {}, variant.get(key) or {}
                    if not isinstance(inherited, dict) or not isinstance(override, dict):
                        raise ConfigError(f"{source}: {name}.{key} must be a mapping")
                    expanded_section[key] = {**inherited, **override}
                if "pipeline" in variant:
                    expanded_section["pipeline"] = variant["pipeline"]
                pairs.extend(load_config_doc({**doc, "attacks": {name: expanded_section}}, only, source))
        return pairs
    # A list reuses the same attack grid for matched training conditions.
    if isinstance(doc.get("pipeline"), list):
        if not doc["pipeline"]:
            raise ConfigError(f"{source}: pipeline list must not be empty")
        pairs = []
        for variant in doc["pipeline"]:
            if not isinstance(variant, dict):
                raise ConfigError(f"{source}: each pipeline variant must be a mapping")
            pairs.extend(load_config_doc({**doc, "pipeline": variant}, only, source))
        return pairs
    defaults = doc.get("defaults") or {}
    attacks = doc.get("attacks") or {}
    if not attacks:
        raise ConfigError(f"{source}: no 'attacks:' section")

    unknown = sorted(set(attacks) - set(ATTACKS))
    if unknown:
        raise ConfigError(
            f"{source}: unknown attack(s): {', '.join(unknown)}. Known: {', '.join(sorted(ATTACKS))}"
        )

    selected = list(attacks) if only is None else [a for a in attacks if a in set(only)]
    pairs: List[Tuple[object, object]] = []
    for name in selected:
        spec = ATTACKS[name]
        allowed = _field_names(spec.config_cls)
        section = attacks[name] or {}
        from .pipeline import parse_pipeline, validate_pipeline_run
        try:
            pipeline = parse_pipeline(section.get("pipeline", doc.get("pipeline")), source)
        except (ValueError, TypeError, OSError) as exc:
            raise ConfigError(f"{source}: attack '{name}': {exc}") from exc
        if pipeline is not None:
            spec = replace(spec, pipeline=pipeline)
        base_over = dict(section.get("base") or {})
        sweep = dict(section.get("sweep") or {})

        configured_collections = []
        if "firestore_collection" in defaults:
            configured_collections.append(defaults["firestore_collection"])
        if "firestore_collection" in base_over:
            configured_collections.append(base_over["firestore_collection"])
        configured_collections.extend(sweep.get("firestore_collection") or [])
        if any(value != RESULTS_COLLECTION for value in configured_collections):
            raise ConfigError(
                f"{source}: attack '{name}': firestore_collection is fixed to "
                f"{RESULTS_COLLECTION!r}"
            )

        # amia/loss have no toy path and no use_hf_models field on their
        # dataclasses at all. Treat use_hf_models there as a virtual switch:
        # an explicit false is the "toy" rejection below, not an unknown-field
        # error, and any other value is stripped before field validation.
        if not spec.supports_toy:
            if base_over.get("use_hf_models") is False or any(
                v is False for v in (sweep.get("use_hf_models") or [])
            ):
                raise ConfigError(
                    f"attack '{name}' has no toy path and requires use_hf_models: true"
                )
            base_over.pop("use_hf_models", None)
            sweep.pop("use_hf_models", None)

        bad = sorted((set(base_over) | set(sweep)) - allowed)
        if bad:
            raise ConfigError(
                f"{source}: attack '{name}' has unknown field(s): {', '.join(bad)}. "
                f"Valid fields: {', '.join(sorted(allowed))}"
            )

        # defaults are cross-attack: silently skip keys this attack lacks.
        merged = {k: v for k, v in defaults.items() if k in allowed}
        merged.update(base_over)
        if "firestore_collection" in allowed:
            merged["firestore_collection"] = RESULTS_COLLECTION
        cfg = replace(spec.config_cls(), **merged) if merged else spec.config_cls()

        for expanded in expand_sweep(cfg, sweep):
            if not spec.supports_toy and not getattr(expanded, "use_hf_models", True):
                raise ConfigError(
                    f"attack '{name}' has no toy path and requires use_hf_models: true"
                )
            try:
                validate_attack_config(expanded, spec)
                validate_dataset_name(expanded.dataset_name)
                validate_pipeline_run(expanded, spec, pipeline)
            except ValueError as exc:
                raise ConfigError(f"{source}: attack '{name}': {exc}") from exc
            pairs.append((expanded, spec))
    return pairs
