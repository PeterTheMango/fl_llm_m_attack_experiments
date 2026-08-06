# master_script/webui/manual.py
"""Manual launch mode: a form payload -> the same doc a YAML config parses to.

Deliberately thin. Everything past build_doc is core.yaml_config's job, so a
manual sweep and a saved-file sweep cannot expand differently, and an error
the CLI would raise reads the same in the browser.
"""
from typing import Any, Dict, List, Tuple

import yaml

from ..core.yaml_config import ConfigError, load_config_doc
from . import attackfields

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


class ManualError(ValueError):
    """Raised for a payload that cannot be turned into a config document."""


def _coerce(raw: Any, type_name: str, attack: str, field: str) -> Any:
    text = str(raw).strip()
    if type_name == "bool":
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ManualError(f"attack '{attack}': {field} expects true or false, got {text!r}.")
    if type_name in ("int", "float"):
        caster = int if type_name == "int" else float
        try:
            return caster(text)
        except ValueError:
            wanted = "a whole number" if type_name == "int" else "a number"
            raise ManualError(
                f"attack '{attack}': {field} expects {wanted}, got {text!r}."
            ) from None
    return text


def build_doc(payload: dict) -> dict:
    """Turn the form payload into a config document.

    Only fields the user actually changed, or marked as a sweep, are emitted:
    a manual config stays as small as the equivalent hand-written YAML instead
    of freezing today's defaults into a file.
    """
    attacks: Dict[str, dict] = {}
    for entry in payload.get("attacks") or []:
        name = (entry.get("name") or "").strip()
        if not name:
            raise ManualError("An attack entry has no name.")
        if name in attacks:
            raise ManualError(f"attack '{name}' appears more than once.")

        schema = attackfields.by_name(name)
        sweeps = set(entry.get("sweeps") or [])
        base: Dict[str, Any] = {}
        sweep: Dict[str, list] = {}

        for field, raw in (entry.get("values") or {}).items():
            info = schema.get(field)
            # An unknown field keeps its text and travels to the loader, which
            # rejects it by name and lists what would have been valid.
            type_name = info["type"] if info else "str"

            if info is not None and info["readonly"]:
                # Dropping it silently would run something other than what was
                # asked for -- under a different key, or on a path the attack
                # does not have. Only an unchanged echo is ignored.
                if _coerce(raw, type_name, name, field) != info["default"]:
                    raise ManualError(
                        f"attack '{name}': {field} cannot be changed ({info['reason']})."
                    )
                continue

            if field in sweeps:
                items = [part for part in str(raw).split(",") if part.strip()]
                if not items:
                    raise ManualError(f"attack '{name}': sweep {field} has no values.")
                sweep[field] = [_coerce(part, type_name, name, field) for part in items]
                continue

            value = _coerce(raw, type_name, name, field)
            if info is not None and value == info["default"]:
                continue  # unchanged
            base[field] = value

        section: Dict[str, dict] = {}
        if base:
            section["base"] = base
        if sweep:
            section["sweep"] = sweep
        attacks[name] = section

    if not attacks:
        raise ManualError("Add at least one attack before starting a sweep.")
    return {"attacks": attacks}


def to_yaml(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def pairs(payload: dict) -> List[Tuple[object, object]]:
    return load_config_doc(build_doc(payload), source="<manual>")


def validate(payload: dict) -> dict:
    """Expand without running or writing. Never raises."""
    try:
        doc = build_doc(payload)
        expanded = load_config_doc(doc, source="<manual>")
    except (ManualError, ConfigError) as exc:
        return {"ok": False, "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    per_attack: Dict[str, int] = {}
    for _cfg, spec in expanded:
        per_attack[spec.name] = per_attack.get(spec.name, 0) + 1
    return {
        "ok": True,
        "message": f"Valid · expands to {len(expanded)} run(s).",
        "runs": len(expanded),
        "per_attack": per_attack,
        "yaml": to_yaml(doc),
    }
