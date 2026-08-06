# master_script/webui/api.py
"""JSON API behind the CANARY Monitor single-page dashboard.

Every endpoint is a read of DashboardState (or a delegate into launch/tunnel).
No experiment logic lives here.
"""
import math
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from . import (
    attackfields, configs, envfile, launch, localguard, manual, monitor, results, tunnel,
)
from .state import DashboardState

router = APIRouter(prefix="/api")

STATE = DashboardState()


def _finite(value):
    """Firestore stores NaN/Infinity; JSON cannot. Emit null instead of crashing."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


@router.get("/live")
def live():
    STATE.ensure_fresh()
    return _finite({
        **monitor.live_payload(STATE),
        "source": {"listener": STATE.live, "error": STATE.error,
                   "last_sync_unix": STATE.last_sync_unix},
    })


@router.get("/results")
def results_view():
    STATE.ensure_fresh()
    return _finite(results.results_payload(STATE))


@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    STATE.ensure_fresh()
    payload = results.detail_payload(STATE, run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id={run_id!r}")
    return _finite(payload)


def _tunnel_status() -> dict:
    """Live status plus the saved config the Access page prefills from."""
    return {**tunnel.MANAGER.status, "saved": tunnel.saved_payload()}


@router.get("/tunnel")
def tunnel_status():
    return _tunnel_status()


class TunnelStart(BaseModel):
    provider: Optional[str] = None
    # None means "keep what is saved in the .env"; "" means "clear it". The
    # browser is never sent these values, so it can only send back what the
    # user actually retyped.
    api_key: Optional[str] = None
    code: Optional[str] = None
    port: Optional[int] = None


def _resolved(body: TunnelStart) -> dict:
    """Merge what the browser sent over what the .env already holds."""
    saved = tunnel.load_saved()
    return {
        "provider": body.provider or saved["provider"],
        "api_key": saved["api_key"] if body.api_key is None else body.api_key,
        "code": saved["code"] if body.code is None else body.code,
        "port": int(body.port or saved["port"]),
    }


@router.post("/tunnel/save")
def tunnel_save(body: TunnelStart):
    """Persist the credentials without opening the tunnel (§4.4 stays intact).

    Unlike a start this accepts an empty api_key -- clearing a stored token is
    exactly what someone revoking access from this page is trying to do.
    """
    try:
        tunnel.save_config(**_resolved(body))
    except (ValueError, OSError) as exc:
        return {"ok": False, "message": f"Could not save: {exc}", "status": _tunnel_status()}
    return {"ok": True, "message": f"Tunnel settings saved to {envfile.env_path().name}.",
            "status": _tunnel_status()}


@router.post("/tunnel/start")
def tunnel_start(body: TunnelStart):
    """§4.4: exposure is always explicit and user-initiated. Never auto-started."""
    try:
        cfg = tunnel.TunnelConfig(**_resolved(body))
        # Saved before the spawn, not after: a missing cloudflared binary says
        # nothing about whether the credentials were worth keeping.
        tunnel.save_config(cfg.provider, cfg.api_key, cfg.code, cfg.port)
        tunnel.MANAGER.start(cfg)
    except (ValueError, FileNotFoundError, OSError) as exc:
        tunnel.MANAGER.error = f"Could not start tunnel: {exc}"
        return {"ok": False, "message": tunnel.MANAGER.error, "status": _tunnel_status()}
    return {"ok": True, "message": "Tunnel starting.", "status": _tunnel_status()}


@router.post("/tunnel/stop")
def tunnel_stop():
    tunnel.MANAGER.stop()
    return {"ok": True, "message": "Tunnel stopped.", "status": _tunnel_status()}


@router.get("/launch")
def launch_options():
    return launch.launch_payload()


class LaunchRequest(BaseModel):
    config_file: str
    attacks: Optional[List[str]] = None
    use_firestore: bool = True


@router.post("/launch")
def launch_start(body: LaunchRequest):
    return launch.start_sweep(body.config_file, body.attacks, body.use_firestore)


# ---------- manual launch mode ----------
#
# The form payload is all strings; typing happens in manual.build_doc against
# the attack's own dataclass, not here.

@router.get("/attacks/fields")
def attack_fields():
    return attackfields.schema()


class ManualAttack(BaseModel):
    name: str
    values: Dict[str, str] = {}
    sweeps: List[str] = []


class ManualRequest(BaseModel):
    attacks: List[ManualAttack] = []
    use_firestore: bool = True


@router.post("/launch/manual/validate")
def launch_manual_validate(body: ManualRequest):
    """Expand the form exactly as a start would. Writes nothing, runs nothing."""
    return manual.validate(body.model_dump())


@router.post("/launch/manual")
def launch_manual_start(body: ManualRequest):
    return launch.start_manual(body.model_dump(), body.use_firestore)


# ---------- config editor ----------

@router.get("/configs")
def config_list(name: Optional[str] = None):
    return configs.payload(name)


@router.get("/configs/{name}")
def config_read(name: str):
    try:
        return {"ok": True, "name": name, "text": configs.read(name)}
    except configs.ConfigNameError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class ConfigText(BaseModel):
    text: str


@router.post("/configs/validate")
def config_validate(body: ConfigText):
    """Dry run through the CLI's own loader. Writes nothing."""
    return configs.validate(body.text)


class ConfigSave(ConfigText):
    name: str
    overwrite: bool = False


@router.post("/configs/save")
def config_save(body: ConfigSave):
    return configs.save(body.name, body.text, body.overwrite)


# ---------- settings (.env) ----------
#
# Local-only. This is the one surface that reads and writes credentials, so it
# stays on the machine running the dashboard even while the tunnel is up.

LOCAL_ONLY = [Depends(localguard.require_local)]


@router.get("/session")
def session(request: Request):
    """What this particular caller is allowed to see. Drives the nav, not access."""
    return {"local": localguard.is_local(request), "client_host": localguard.client_host(request)}


@router.get("/settings", dependencies=LOCAL_ONLY)
def settings():
    """Secrets come back masked; use /settings/reveal to read one in the clear."""
    path = envfile.env_path()
    return {
        "path": str(path),
        "exists": path.exists(),
        "entries": envfile.entries(),
        "tunnel_live": bool(tunnel.MANAGER.status["connected"]),
    }


@router.get("/settings/reveal/{key}", dependencies=LOCAL_ONLY)
def settings_reveal(key: str):
    value = envfile.reveal(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{key} is not set in the .env file")
    return {"key": key, "value": value}


class SettingsSave(BaseModel):
    updates: dict = {}
    deletes: List[str] = []


@router.post("/settings", dependencies=LOCAL_ONLY)
def settings_save(body: SettingsSave):
    """Write the .env, then make the running process actually use it."""
    updates = {k: ("" if v is None else str(v)) for k, v in body.updates.items()}
    bad = [k for k in list(updates) + list(body.deletes) if not k or "=" in k or "\n" in k]
    if bad:
        return {"ok": False, "message": f"Invalid key name(s): {', '.join(bad)}"}

    try:
        path = envfile.write(updates, body.deletes)
    except OSError as exc:
        return {"ok": False, "message": f"Could not write {envfile.env_path()}: {exc}"}

    changed = envfile.reload_into_process()
    # New credentials mean the projection was built against the old ones.
    STATE.detach_listener()
    STATE.last_sync_unix = None
    return {
        "ok": True,
        "message": (f"Saved {path.name} · reloaded {len(changed)} value(s) into the running process."
                    if changed else f"Saved {path.name} · no value changed."),
        "changed": changed,
        "entries": envfile.entries(),
    }
