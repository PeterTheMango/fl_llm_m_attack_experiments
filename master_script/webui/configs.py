# master_script/webui/configs.py
"""Browse, validate and save sweep configs from the dashboard.

Validation goes through core.yaml_config.load_config_doc -- the same loader
the CLI uses -- so a config the editor calls valid is a config the runner
accepts, and the run count shown is the real expanded grid, not an estimate.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..core.registry import ATTACKS
from ..core.yaml_config import ConfigError, load_config_doc
from ..paths import CONFIGS_DIR

SUFFIX = ".yaml"
# Plain basenames only: this writes into CONFIGS_DIR and nowhere else.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MAX_BYTES = 256_000

TEMPLATE = """# New sweep config.
# defaults apply to every attack below; keys an attack doesn't define are skipped.
defaults:
  use_hf_models: false
  attack_trials: 4

attacks:
  zlib: {}
  # min_k:
  #   base: {min_k_percent: 20}
  #   sweep:
  #     federated_rounds: [1, 2, 4]
  #     seed: [7, 11, 23]
"""


class ConfigNameError(ValueError):
    """Raised for a filename that would escape CONFIGS_DIR or is malformed."""


def safe_path(name: str) -> Path:
    """Resolve a user-supplied name to a path inside CONFIGS_DIR, or refuse."""
    name = (name or "").strip()
    if not name.endswith(SUFFIX):
        name += SUFFIX
    stem = name[: -len(SUFFIX)]
    if not _NAME_RE.match(stem) or "/" in name or "\\" in name or ".." in stem:
        raise ConfigNameError(
            f"Invalid config name {name!r}. Use letters, digits, '.', '_' or '-'."
        )
    path = (CONFIGS_DIR / name).resolve()
    if path.parent != CONFIGS_DIR.resolve():
        raise ConfigNameError(f"Config name {name!r} would write outside the configs directory.")
    return path


def listing() -> List[dict]:
    out = []
    for path in sorted(CONFIGS_DIR.glob(f"*{SUFFIX}")):
        stat = path.stat()
        out.append({"name": path.name, "bytes": stat.st_size, "modified_unix": stat.st_mtime})
    return out


def read(name: str) -> str:
    path = safe_path(name)
    if not path.exists():
        raise ConfigNameError(f"No such config: {path.name}")
    return path.read_text()


def validate(text: str) -> dict:
    """Expand a config exactly as the runner would, without writing anything.

    Reports the real number of runs and their per-attack breakdown, so the
    size of a sweep is visible before it costs GPU hours.
    """
    if len(text.encode("utf-8")) > _MAX_BYTES:
        return {"ok": False, "message": f"Config is larger than {_MAX_BYTES // 1000} KB."}

    try:
        pairs = load_config_doc(yaml.safe_load(text) or {})
    except ConfigError as exc:
        return {"ok": False, "message": str(exc)}
    except Exception as exc:  # malformed YAML surfaces as a parser error
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    per_attack: Dict[str, int] = {}
    for _cfg, spec in pairs:
        per_attack[spec.name] = per_attack.get(spec.name, 0) + 1
    return {
        "ok": True,
        "message": f"Valid · expands to {len(pairs)} run(s).",
        "runs": len(pairs),
        "per_attack": per_attack,
    }


def save(name: str, text: str, overwrite: bool = False) -> dict:
    """Validate, then write. An invalid config never reaches disk."""
    try:
        path = safe_path(name)
    except ConfigNameError as exc:
        return {"ok": False, "message": str(exc)}

    if path.exists() and not overwrite:
        return {"ok": False, "message": f"{path.name} already exists.", "exists": True}

    result = validate(text)
    if not result["ok"]:
        return {"ok": False, "message": f"Not saved — {result['message']}"}

    path.write_text(text if text.endswith("\n") else text + "\n")
    return {
        "ok": True,
        "message": f"Saved {path.name} · {result['runs']} run(s).",
        "name": path.name,
        "runs": result["runs"],
        "per_attack": result["per_attack"],
    }


def payload(name: Optional[str] = None) -> dict:
    """Everything the editor needs: the file list, one file's text, the template."""
    files = listing()
    text = None
    if name:
        try:
            text = read(name)
        except ConfigNameError:
            text = None
    return {
        "files": files,
        "name": name,
        "text": text,
        "template": TEMPLATE,
        "attacks": sorted(ATTACKS),
    }
