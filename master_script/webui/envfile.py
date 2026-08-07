# master_script/webui/envfile.py
"""Read, edit and hot-reload the .env the program runs on.

Editing credentials from a browser is deliberate (the VM is often only
reachable through the tunnel), so two properties matter here:

1. Secret values are masked on the way out. Reading a secret back in the
   clear is a separate, explicit request -- it never rides along with the
   settings list.
2. A save is not finished until the running process actually uses the new
   values: os.environ is updated in place and the cached Firestore client is
   torn down so the next call re-authenticates with the new credentials.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..paths import MASTER_DIR

# Keys the program actually reads, with what they're for. Anything else the
# file already contains is still shown and editable -- this list drives the
# "add a known key" affordance and the secret masking, not what's allowed.
KNOWN_KEYS: Dict[str, dict] = {
    "FIREBASE_SERVICE_ACCOUNT_JSON": {
        "secret": True,
        "help": "Service-account JSON, inline. Either this or GOOGLE_APPLICATION_CREDENTIALS is required for Firestore.",
    },
    "GOOGLE_APPLICATION_CREDENTIALS": {
        "secret": False,
        "help": "Path to a service-account JSON file. Alternative to FIREBASE_SERVICE_ACCOUNT_JSON.",
    },
    "FIREBASE_PROJECT_ID": {
        "secret": False,
        "help": "Firebase project id passed to the Firestore client.",
    },
    "EXPERIMENT_GPU": {
        "secret": False,
        "help": "Pin runs to one GPU index, or 'cpu' to force CPU. Empty = pick the GPU with the most free memory.",
    },
    "TUNNEL_PROVIDER": {
        "secret": False,
        "help": "Tunnel provider the Access page starts with: cloudflare or ngrok.",
    },
    "TUNNEL_API_KEY": {
        "secret": True,
        "help": "Cloudflare API token or ngrok authtoken, saved from the Access page.",
    },
    "TUNNEL_CODE": {
        # Cloudflare's is a connector token; ngrok's is only a domain name. The
        # stricter of the two decides, since one key holds both.
        "secret": True,
        "help": "Cloudflare named-tunnel connector token, or ngrok reserved domain. Empty = ephemeral URL.",
    },
    "TUNNEL_PORT": {
        "secret": False,
        "help": "Local port the tunnel points at. Defaults to the port the dashboard is served on.",
    },
}

_SECRET_HINTS = ("token", "secret", "password", "key", "credential")
_MASK_KEEP = 4
_BACKUP_SUFFIX = ".bak"


def is_secret(key: str) -> bool:
    known = KNOWN_KEYS.get(key)
    if known is not None:
        return known["secret"]
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def mask(value: str) -> str:
    """A shape-preserving hint, never the value itself."""
    if not value:
        return ""
    if len(value) <= _MASK_KEEP:
        return "•" * len(value)
    return "•" * min(24, len(value) - _MASK_KEEP) + value[-_MASK_KEEP:]


def env_path() -> Path:
    """The .env the program loads: the nearest one at or above master_script/."""
    for base in [MASTER_DIR, *MASTER_DIR.parents]:
        candidate = base / ".env"
        if candidate.exists():
            return candidate
    # Nothing on disk yet: the repo root is where python-dotenv would find it.
    return MASTER_DIR.parent / ".env"


def backup_path() -> Path:
    """Where write() copies the previous file before editing it."""
    path = env_path()
    return path.parent / (path.name + _BACKUP_SUFFIX)


def _split(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    return key, value.strip().strip('"').strip("'")


def read_pairs() -> Dict[str, str]:
    path = env_path()
    if not path.exists():
        return {}
    pairs: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        parsed = _split(line)
        if parsed:
            pairs[parsed[0]] = parsed[1]
    return pairs


def entries() -> List[dict]:
    """Every key in the file plus any known key it's still missing."""
    pairs = read_pairs()
    out = []
    for key in list(pairs) + [k for k in KNOWN_KEYS if k not in pairs]:
        value = pairs.get(key, "")
        secret = is_secret(key)
        out.append({
            "key": key,
            "set": bool(value),
            "secret": secret,
            "known": key in KNOWN_KEYS,
            "help": KNOWN_KEYS.get(key, {}).get("help", ""),
            # Non-secret values are shown as-is; secrets only ever as a shape.
            "value": mask(value) if secret else value,
            "masked": secret and bool(value),
        })
    return out


def reveal(key: str) -> Optional[str]:
    """Plaintext for one key. Deliberately a separate call from entries()."""
    return read_pairs().get(key)


def _needs_quoting(value: str) -> bool:
    return value != value.strip() or any(c in value for c in ' #"\'\n')


def _format(key: str, value: str) -> str:
    if _needs_quoting(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'{key}="{escaped}"'
    return f"{key}={value}"


def write(updates: Dict[str, str], deletes: Optional[List[str]] = None) -> Path:
    """Apply updates in place, keeping comments, ordering and untouched keys.

    The previous file is copied to .env.bak first: this edits live credentials,
    and a fat-fingered save should be recoverable.
    """
    path = env_path()
    deletes = set(deletes or [])
    lines = path.read_text().splitlines() if path.exists() else []

    if path.exists():
        # Sibling name, not with_suffix(): ".env" has no stem to suffix onto.
        shutil.copy2(path, backup_path())

    seen = set()
    out: List[str] = []
    for line in lines:
        parsed = _split(line)
        if parsed is None:
            out.append(line)
            continue
        key = parsed[0]
        if key in deletes:
            seen.add(key)
            continue
        if key in updates:
            out.append(_format(key, updates[key]))
            seen.add(key)
            continue
        out.append(line)

    added = [k for k in updates if k not in seen]
    if added:
        if out and out[-1].strip():
            out.append("")
        out.append(f"# added via the CANARY Monitor dashboard {time.strftime('%Y-%m-%d %H:%M')}")
        out.extend(_format(k, updates[k]) for k in added)

    path.write_text("\n".join(out).rstrip("\n") + "\n")
    return path


def reload_into_process() -> List[str]:
    """Push the file's values into os.environ and drop stale clients.

    Returns the keys that changed. Without the client reset, new credentials
    would sit in os.environ while firebase_admin kept using the app it cached
    at first use.
    """
    from ..core import firestore

    pairs = read_pairs()
    changed = []
    for key, value in pairs.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            changed.append(key)

    # Keys dropped from the file should stop applying to this process too, but
    # only ones we manage -- never touch unrelated inherited environment.
    for key in list(KNOWN_KEYS):
        if key not in pairs and key in os.environ:
            del os.environ[key]
            changed.append(key)

    if changed:
        firestore.reset_client()
    return sorted(set(changed))
