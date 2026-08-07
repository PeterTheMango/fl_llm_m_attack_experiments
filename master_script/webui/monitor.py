# master_script/webui/monitor.py
"""Live view payload: running set, sweep progress, recently-finished feed.

Thin by design (spec §2): all data logic lives in DashboardState. This module
only projects state into the JSON the Live view renders.
"""
import time
from typing import List, Optional

from . import catalog, gpustats, logtail
from .state import attack_name

# The three phases a run moves through, in the order runner.py performs them.
STAGES = ("fine-tune", "attack", "measure")

# A run-state entry is only believable while something keeps re-stamping it.
# The reporter beats every launch.HEARTBEAT_SECONDS; past several missed beats
# the writer is gone (crash, kill, VM reboot) and the entry is a ghost. This is
# the half that survives a hard kill, where on_run_end never got to run.
STALE_AFTER_SECONDS = 150

_DISTINGUISHING_KEYS = ("model_id", "federated_rounds", "num_clients", "seed")


def running_set_available(state) -> bool:
    """Without the script's run-state report, in-progress runs are unknowable (§2.4)."""
    return bool(state.running) or state.manifest is not None


def _elapsed_seconds(run: dict) -> Optional[float]:
    started = run.get("started_unix") or run.get("start_time_unix")
    return None if not started else max(0.0, time.time() - started)


def _chips(cfg: dict) -> List[str]:
    chips = []
    for key in _DISTINGUISHING_KEYS:
        value = cfg.get(key)
        if value is None:
            continue
        chips.append(str(value).split("/")[-1] if key == "model_id" else f"{key}={value}")
    mech = cfg.get("ldp_mechanism")
    eps = cfg.get("epsilon")
    if mech and mech != "none" and eps is not None:
        chips.append(f"{mech} ε{eps}")
    elif mech is not None or eps is not None:
        chips.append("no-DP")
    return chips


def last_heartbeat(run: dict) -> Optional[float]:
    """When the writer last confirmed this run alive.

    Falls back to the start time: an entry that never carried a heartbeat was
    written by something that isn't confirming anything, so ageing it out is
    the honest reading.
    """
    return run.get("heartbeat_unix") or run.get("started_unix") or run.get("start_time_unix")


def is_stale(run: dict, now: Optional[float] = None) -> bool:
    """True when nothing has re-stamped this entry for STALE_AFTER_SECONDS."""
    beat = last_heartbeat(run)
    if beat is None:
        return True  # no timestamp at all: nothing vouches for it
    return (time.time() if now is None else now) - beat > STALE_AFTER_SECONDS


def _running_rows(state) -> List[dict]:
    rows = []
    now = time.time()
    for run in state.running:
        cfg = run.get("config") or {}
        name = cfg.get("attack_name") or run.get("attack") or run.get("attack_name") or "?"
        stage = run.get("stage")
        beat = last_heartbeat(run)
        stale = is_stale(run, now)
        rows.append({
            "stale": stale,
            "last_heartbeat_unix": beat,
            "heartbeat_age_seconds": None if beat is None else max(0.0, now - beat),
            "run_id": run.get("run_id", "?"),
            "attack": name,
            "gpu": run.get("gpu"),
            "config": cfg,
            "chips": _chips(cfg),
            "stages": list(STAGES),
            # -1 means the run-state report did not say; the UI shows every bar
            # muted rather than guessing which phase is in flight.
            "stage_index": STAGES.index(stage) if stage in STAGES else -1,
            "stage_label": stage or "stage unreported",
            "elapsed_seconds": _elapsed_seconds(run),
        })
    return rows


def _sweep_rows(state) -> List[dict]:
    """Per-attack progress. Denominators exist only when a manifest was published."""
    manifest_total = {}
    for entry in state.manifest or []:
        key = entry.get("attack") or entry.get("attack_name") or "?"
        manifest_total[key] = manifest_total.get(key, 0) + 1

    running_count = {}
    for run in state.running:
        if is_stale(run):
            continue  # a ghost must not be counted as occupying a slot
        cfg = run.get("config") or {}
        key = cfg.get("attack_name") or run.get("attack") or "?"
        running_count[key] = running_count.get(key, 0) + 1

    resolved = {}
    for run in state.runs.values():
        key = attack_name(run)
        bucket = resolved.setdefault(key, {"complete": 0, "failed": 0})
        if run.get("status") == "complete":
            bucket["complete"] += 1
        elif run.get("status") == "failed":
            bucket["failed"] += 1

    rows = []
    for key in sorted(set(manifest_total) | set(running_count) | set(resolved)):
        bucket = resolved.get(key, {"complete": 0, "failed": 0})
        complete, failed = bucket["complete"], bucket["failed"]
        running = running_count.get(key, 0)
        total = manifest_total.get(key)
        rows.append({
            "attack": key,
            "complete": complete,
            "failed": failed,
            "running": running,
            "total": total,
            "pending": None if total is None else max(0, total - complete - failed - running),
        })
    return rows


def _recent_rows(state, limit: int) -> List[dict]:
    rows = []
    for run in state.recently_finished(limit):
        metrics = run.get("metrics") or {}
        rows.append({
            "run_id": run.get("run_id", "?"),
            "attack": attack_name(run),
            "status": run.get("status", "?"),
            "adv": metrics.get("adv"),
            "updated_at_unix": run.get("updated_at_unix"),
        })
    return rows


def live_payload(state, *, recent_limit: int = 8, log_limit: int = 60) -> dict:
    """Everything the Live view renders, as plain JSON."""
    running = _running_rows(state)
    return {
        "now_unix": time.time(),
        "run_state_available": running_set_available(state),
        "running": running,
        # Counts for the header pill and sweep bars: a stale entry is a ghost,
        # not a run, and must not inflate either.
        "running_live": sum(1 for r in running if not r["stale"]),
        "running_stale": sum(1 for r in running if r["stale"]),
        "stale_after_seconds": STALE_AFTER_SECONDS,
        "sweeps": _sweep_rows(state),
        "recent": _recent_rows(state, recent_limit),
        "gpus": gpustats.poll(),
        "logs": logtail.tail(log_limit),
        "attacks": catalog.catalog(),
    }
