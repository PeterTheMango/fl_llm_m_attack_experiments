# master_script/webui/attackfields.py
"""Attack dataclasses -> a form schema for the Launch page's manual mode.

Pure introspection: no I/O and no dashboard state, so the form can never
offer a field the runner would reject, or miss one it accepts. Grouping is
presentation; the field names and defaults are the dataclasses' own.
"""
from dataclasses import MISSING, fields as dc_fields
from typing import Any, Dict, List, get_type_hints

from ..core.registry import ATTACKS

GROUP_ORDER = ["Model & data", "Federation", "Attack", "Advanced"]

# Only the shared and plumbing names are enumerated. Anything else -- which in
# practice means the attack's own knobs -- falls through to "Attack", so a new
# attack-specific field shows up in the form without editing this table.
_GROUPS = {
    "model_id": "Model & data",
    "dataset_name": "Model & data",
    "max_length": "Model & data",
    "seed": "Model & data",
    "use_hf_models": "Model & data",
    "num_clients": "Federation",
    "clients_per_round": "Federation",
    "federated_rounds": "Federation",
    "local_epochs": "Federation",
    "local_batch_size": "Federation",
    "client_lr": "Federation",
    "target_client_id": "Federation",
    "fl_framework": "Advanced",
    "sim_num_gpus": "Advanced",
    "keep_artifacts": "Advanced",
    "firestore_collection": "Advanced",
    "artifact_root": "Advanced",
    "attack_name": "Advanced",
    "paper_source": "Advanced",
}
_FALLBACK_GROUP = "Attack"

# Read-only fields carry the reason they are read-only: it is what the form
# shows beside the value, and what the error names if a caller sends one
# changed anyway. These two are part of the hashed config, so an edit does not
# vary the experiment -- it renames it into a document nobody will look for.
_READONLY = {
    "attack_name": "part of the experiment key",
    "paper_source": "part of the experiment key",
    "firestore_collection": "shared result collection",
}


def _type_name(annotation: Any) -> str:
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


def fields_for(name: str) -> List[dict]:
    spec = ATTACKS[name]
    hints = get_type_hints(spec.config_cls)
    out: List[dict] = []
    for field in dc_fields(spec.config_cls):
        default = None if field.default is MISSING else field.default
        out.append({
            "name": field.name,
            "type": _type_name(hints.get(field.name, str)),
            "default": default,
            "group": _GROUPS.get(field.name, _FALLBACK_GROUP),
            "readonly": field.name in _READONLY,
            "reason": _READONLY.get(field.name, ""),
        })

    # amia/loss carry no use_hf_models field at all. Omitting it from the form
    # would read as "the toy path is an option here"; it is not.
    if not spec.supports_toy and not any(f["name"] == "use_hf_models" for f in out):
        out.append({
            "name": "use_hf_models", "type": "bool", "default": True,
            "group": "Model & data", "readonly": True, "reason": "no toy path",
        })
    return out


def by_name(name: str) -> Dict[str, dict]:
    """Field dicts keyed by field name. Unknown attacks give {} so callers can
    fall through to the loader's own unknown-attack error."""
    if name not in ATTACKS:
        return {}
    return {field["name"]: field for field in fields_for(name)}


def schema() -> dict:
    return {
        "group_order": GROUP_ORDER,
        "attacks": {
            name: {"supports_toy": spec.supports_toy, "fields": fields_for(name)}
            for name, spec in sorted(ATTACKS.items())
        },
    }
