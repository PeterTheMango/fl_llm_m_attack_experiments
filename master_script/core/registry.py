# master_script/core/registry.py
"""Attack registry. One AttackSpec per adaptation notebook."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .config import key_sha16


@dataclass(frozen=True)
class AttackSpec:
    name: str
    config_cls: type
    methodology: dict
    # key_fn: this attack's OWN document-id formula. Three variants exist across
    # the notebooks; unifying them would orphan completed Firestore documents.
    key_fn: Callable = key_sha16
    score_toy: Optional[Callable] = None
    score_hf: Optional[Callable] = None
    extra_metrics: Optional[Callable] = None
    needs_reference: bool = False
    supports_toy: bool = True


def _load() -> Dict[str, AttackSpec]:
    from .attacks import (
        amia, loss, min_k, min_k_plus_plus, neighborhood, recall,
        reference, samia, spv_mia, wbc, zlib,
    )

    specs = [
        zlib.SPEC, min_k.SPEC, min_k_plus_plus.SPEC, neighborhood.SPEC,
        recall.SPEC, reference.SPEC, samia.SPEC, spv_mia.SPEC, wbc.SPEC,
        amia.SPEC, loss.SPEC,
    ]
    return {spec.name: spec for spec in specs}


ATTACKS: Dict[str, AttackSpec] = _load()


def get_attack(name: str) -> AttackSpec:
    if name not in ATTACKS:
        raise KeyError(f"Unknown attack {name!r}. Known: {', '.join(sorted(ATTACKS))}")
    return ATTACKS[name]
