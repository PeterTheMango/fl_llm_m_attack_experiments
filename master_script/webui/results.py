# master_script/webui/results.py
"""Results + detail payloads: filterable grid, aggregates, per-run breakdown.

Thin by design (spec §3): all data logic lives in DashboardState. This module
only projects state into the JSON the Results and Detail views render.
"""
from pathlib import Path
import shutil
from statistics import mean
from typing import Any, Dict, List, Optional

from ..core import firestore
from ..paths import ARTIFACTS_DIR
from . import catalog
from .state import COLLECTION, attack_name

_TOLERANCE = 0.02

# Config keys the grid shows as columns, in the order the detail view lists them.
CONFIG_ORDER = (
    "model_id", "dataset_name", "num_clients", "clients_per_round",
    "federated_rounds", "local_epochs", "local_batch_size", "client_lr",
    "ldp_mechanism", "epsilon", "seed", "attack_trials", "target_client_id",
    "max_length", "use_hf_models", "threshold",
)

# X-axis factors the comparison scatter offers.
X_FACTORS = ("epsilon", "federated_rounds", "num_clients", "local_epochs", "client_lr", "seed")

_LOSS_HINTS = ("loss", "nll", "perplexity")


def privacy_direction(baseline_adv: float, current_adv: float) -> str:
    """§3.4: report the direction the runs imply. Makes no privacy claim of its own."""
    delta = current_adv - baseline_adv
    if abs(delta) <= _TOLERANCE:
        return f"Privacy unchanged (ΔAdv = {delta:+.3f})"
    if delta < 0:
        return f"Privacy improved — attack advantage fell (ΔAdv = {delta:+.3f})"
    return f"Privacy declined — attack advantage rose (ΔAdv = {delta:+.3f})"


def _grid_config(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in CONFIG_ORDER if k in cfg}


def current_error(run: dict) -> Optional[str]:
    """The run's error, or None if it succeeded.

    Documents written before save_result cleared the field can carry an `error`
    from an earlier failed attempt alongside `status: complete`. That is history,
    not this run's outcome -- see prior_error().
    """
    return None if run.get("status") == "complete" else run.get("error")


def prior_error(run: dict) -> Optional[str]:
    """An earlier attempt's error on a run that has since succeeded."""
    return run.get("error") if run.get("status") == "complete" else None


def _row(run: dict) -> dict:
    metrics = run.get("metrics") or {}
    return {
        "run_id": run.get("run_id", "?"),
        "attack": attack_name(run),
        "status": run.get("status", "?"),
        "updated_at_unix": run.get("updated_at_unix"),
        "config": _grid_config(run.get("config") or {}),
        "metrics": {
            "adv": metrics.get("adv"),
            "tpr": metrics.get("tpr"),
            "tnr": metrics.get("tnr"),
            "num_trials": metrics.get("num_trials"),
        } if metrics else None,
        "error": current_error(run),
    }


def results_payload(state) -> dict:
    """Every run the dashboard knows about; the client filters, sorts and plots."""
    runs = [_row(r) for r in state.runs.values()]
    models = sorted({r["config"].get("model_id") for r in runs if r["config"].get("model_id")})
    mechs = sorted({str(r["config"].get("ldp_mechanism")) for r in runs
                    if r["config"].get("ldp_mechanism") is not None})
    return {
        "runs": runs,
        "attacks": catalog.catalog(),
        "aggregates": state.aggregate_by("attack_name"),
        "models": models,
        "mechanisms": mechs,
        "x_factors": list(X_FACTORS),
    }


def _numeric_loss(entry: Any) -> Optional[float]:
    """Pull a per-round loss out of one history entry, whatever the attack called it."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if not isinstance(entry, dict):
        return None
    for key, value in entry.items():
        if any(hint in key.lower() for hint in _LOSS_HINTS) and isinstance(value, (int, float)):
            return float(value)
    for key, value in entry.items():
        if any(hint in key.lower() for hint in _LOSS_HINTS) and isinstance(value, list):
            nums = [v for v in value if isinstance(v, (int, float))]
            if nums:
                return float(mean(nums))
    return None


def _flatten_rounds(history: Any) -> List[List[Any]]:
    """Return a list of per-trial round-lists. Attacks disagree on the wrapper shape."""
    if not isinstance(history, list) or not history:
        return []
    if isinstance(history[0], dict) and ("rounds" in history[0] or "history" in history[0]):
        return [entry.get("rounds") or entry.get("history") or [] for entry in history
                if isinstance(entry, dict)]
    return [history]


def normalize_history(history: Any) -> List[dict]:
    """Per-round mean loss across trials. Empty when no attack recorded a loss."""
    per_round: Dict[int, List[float]] = {}
    for rounds in _flatten_rounds(history):
        if not isinstance(rounds, list):
            continue
        for index, entry in enumerate(rounds):
            value = _numeric_loss(entry)
            if value is None:
                continue
            number = entry.get("round") if isinstance(entry, dict) else None
            per_round.setdefault(int(number) if isinstance(number, int) else index, []).append(value)
    return [{"round": r, "mean_loss": mean(v)} for r, v in sorted(per_round.items())]


def _trials(run: dict) -> List[dict]:
    out = []
    for trial in run.get("attack_trials") or []:
        if not isinstance(trial, dict) or "score" not in trial:
            continue
        out.append({
            "trial_id": trial.get("trial_id"),
            "truth_member": bool(trial.get("truth_member")),
            "score": float(trial["score"]),
            "pred_member": bool(trial.get("pred_member")),
        })
    return out


def _artifacts(run: dict) -> List[dict]:
    artifacts = run.get("artifacts")
    if isinstance(artifacts, dict):
        items = list(artifacts.items())
    elif isinstance(artifacts, list):
        items = [(f"artifact[{i}]", a) for i, a in enumerate(artifacts)]
    else:
        items = []
    rows = [{"k": k, "v": "—" if v is None else str(v)} for k, v in items]
    # The Firestore document is the authoritative record; local paths may be gone.
    rows.append({"k": "firestore_doc", "v": f"ami_federated_llm_results/{run.get('run_id', '?')}",
                 "authoritative": True})
    return rows


def detail_payload(state, run_id: str) -> Optional[dict]:
    """One run in full. None when the dashboard has never seen that run_id."""
    run = state.runs.get(run_id)
    if run is None:
        return None

    cfg = run.get("config") or {}
    name = attack_name(run)
    methodology = run.get("methodology") or cfg.get("methodology") or {}
    known = catalog.catalog().get(name)
    meta = known if known else catalog.entry_for(name, methodology)

    return {
        "run_id": run.get("run_id", run_id),
        "attack": name,
        "meta": meta,
        "status": run.get("status", "?"),
        "updated_at_unix": run.get("updated_at_unix"),
        "metrics": run.get("metrics") or None,
        # Prefer the methodology the document recorded; fall back to the
        # registry so an older document still shows the attack's method.
        "methodology": methodology or meta["methodology"],
        "config": [{"k": k, "v": cfg[k]} for k in CONFIG_ORDER if k in cfg]
                  + [{"k": k, "v": v} for k, v in sorted(cfg.items())
                     if k not in CONFIG_ORDER and k != "methodology"],
        "federated_history": normalize_history(run.get("federated_history")),
        "trials": _trials(run),
        "artifacts": _artifacts(run),
        "error": current_error(run),
        "prior_error": prior_error(run),
    }


def _safe_artifact_dir(run: dict) -> Optional[Path]:
    """Return a deletable artifact directory only when it is below our root."""
    artifacts = run.get("artifacts") or {}
    raw = artifacts.get("artifact_dir") if isinstance(artifacts, dict) else None
    if not raw:
        return None
    root = ARTIFACTS_DIR.resolve()
    candidate = Path(str(raw)).expanduser().resolve()
    if candidate == root:
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def delete_run(state, run_id: str) -> dict:
    """Delete the Firestore result and any safely-scoped local artifacts."""
    run = state.runs.get(run_id)
    if run is None:
        return {"ok": False, "message": f"No run found for run_id={run_id!r}"}

    # Firestore is authoritative: if this fails, keep local state and artifacts
    # intact so the page never claims the result was removed when it was not.
    firestore.delete_result(run_id, COLLECTION)

    artifact_dir = _safe_artifact_dir(run)
    artifact_removed = False
    artifact_warning = ""
    if artifact_dir is not None and artifact_dir.exists():
        try:
            shutil.rmtree(artifact_dir)
            artifact_removed = True
        except OSError as exc:
            artifact_warning = f" Local artifacts could not be removed: {exc}"

    state.remove_run(run_id)
    return {
        "ok": True,
        "message": (f"Deleted result {run_id} from Firestore."
                    + (" Removed its local artifacts." if artifact_removed else "")
                    + artifact_warning),
        "run_id": run_id,
        "artifact_removed": artifact_removed,
        "artifact_warning": artifact_warning,
    }
