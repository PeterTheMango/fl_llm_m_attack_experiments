# master_script/core/config.py
"""Config base, stable hashing, and grid expansion.

The key formula is load-bearing: it is the Firestore document id, and it must
stay byte-identical to each notebook's. THREE formulas exist -- the nine modern
notebooks, AMIA, and LOSS each hash differently. Unifying them would orphan
completed documents. See tests/test_hash_equivalence.py.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence
import json

from ..paths import ARTIFACTS_DIR


@dataclass(frozen=True)
class AttackConfig:
    """Marker base. Declares NO fields on purpose.

    Any field here would be emitted first by asdict() and would change every
    subclass's key, orphaning completed Firestore documents.
    """


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


def experiment_key(config: AttackConfig, spec: Optional[Any] = None) -> str:
    """Dispatch to the attack's own key formula.

    The spec=None fallback is the modern 16-char formula, correct only for the
    nine modern attacks. Always pass spec when you have it.
    """
    if spec is not None and getattr(spec, "key_fn", None) is not None:
        return spec.key_fn(config)
    return key_sha16(config)


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
