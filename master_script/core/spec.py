# master_script/core/spec.py
"""AttackSpec dataclass. Leaf module: no attack module needs to import
registry to get this, breaking the registry<->attacks import cycle."""
from dataclasses import dataclass
from typing import Callable, Optional

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
