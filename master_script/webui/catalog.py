# master_script/webui/catalog.py
"""Display metadata for the 11 attacks: label, colour, one-line title.

Presentation only. Nothing here feeds a run_id, a config, or a Firestore
document -- it exists so the dashboard can colour and label a run without the
attack modules having to care that a dashboard exists.
"""
from typing import Dict

# Three of these are the CANARY design's own picks (amia/loss/reference); the
# rest extend the same palette so all 11 attacks stay distinguishable.
_COLORS = {
    "amia": "#2bb3c0",
    "loss": "#8b6fe0",
    "reference": "#d8943a",
    "zlib": "#3fcf8e",
    "min_k": "#4aa8ff",
    "min_k_plus_plus": "#7ec8ff",
    "neighborhood": "#e06c9f",
    "recall": "#c9a227",
    "samia": "#5fd0b0",
    "spv_mia": "#f0606a",
    "wbc": "#9aa6b2",
}

_LABELS = {
    "amia": "AMIA",
    "loss": "LOSS",
    "reference": "REF",
    "zlib": "ZLIB",
    "min_k": "MIN-K",
    "min_k_plus_plus": "MIN-K++",
    "neighborhood": "NBHD",
    "recall": "RECALL",
    "samia": "SAMIA",
    "spv_mia": "SPV-MIA",
    "wbc": "WBC",
}

_FALLBACK_COLOR = "#5f6b78"
_TITLE_MAX = 96


def label_for(name: str) -> str:
    return _LABELS.get(name, name.replace("_", "-").upper())


def color_for(name: str) -> str:
    return _COLORS.get(name, _FALLBACK_COLOR)


def _title_from(methodology: dict) -> str:
    """Derive a one-line heading from the attack's own methodology text.

    Deliberately derived rather than hand-written: the methodology dict is what
    every result document already carries, so the dashboard heading can never
    disagree with the recorded methodology.
    """
    text = (methodology or {}).get("paper_attack", "") or ""
    head = text.split(":")[0].split(";")[0].strip()
    if not head:
        return ""
    return head if len(head) <= _TITLE_MAX else head[: _TITLE_MAX - 1].rstrip() + "…"


def catalog() -> Dict[str, dict]:
    """{attack_name: {key,label,color,title,methodology}} for every registered attack."""
    from ..core.registry import ATTACKS

    out = {}
    for name, spec in ATTACKS.items():
        out[name] = {
            "key": name,
            "label": label_for(name),
            "color": color_for(name),
            "title": _title_from(spec.methodology),
            "methodology": dict(spec.methodology or {}),
        }
    return out


def entry_for(name: str, methodology: dict | None = None) -> dict:
    """Catalog entry for a name that may not be registered (e.g. a stale doc)."""
    return {
        "key": name,
        "label": label_for(name),
        "color": color_for(name),
        "title": _title_from(methodology or {}),
        "methodology": dict(methodology or {}),
    }
