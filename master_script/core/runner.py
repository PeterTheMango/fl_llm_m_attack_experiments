"""Experiment orchestration. The single code path shared by the CLI and web UI.

Ordering is load-bearing and mirrors the notebooks exactly:
cache check -> FL fine-tune -> attack -> measure -> persist -> cleanup.
"""
from dataclasses import asdict, replace
from pathlib import Path
import logging
import shutil
import time

from . import federation, firestore
from .config import artifact_dir_for, experiment_key
from .metrics import summarize
from .scoring import ScoreContext

log = logging.getLogger(__name__)


def cleanup_artifacts(artifact_dir) -> None:
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def run_attack_trial(config, spec, trial_id: int, truth_member: bool) -> dict:
    # Per-trial reseed, exactly as the notebooks do. NOTE: trial_config is never
    # hashed -- the run_id belongs to the original config.
    trial_config = replace(config, seed=config.seed + trial_id)
    if trial_config.use_hf_models:
        target, history = federation.run_hf_federated_finetune(trial_config, truth_member=truth_member)
        reference = federation.load_reference_bundle(trial_config) if spec.needs_reference else None
        score = spec.score_hf(ScoreContext(trial_config, target, federation.TARGET_RECORD, reference))
    else:
        target, history = federation.run_toy_federated_finetune(trial_config, truth_member=truth_member)
        reference = None
        if spec.needs_reference:
            reference, _ = federation.run_toy_federated_finetune(trial_config, truth_member=False)
        score = spec.score_toy(ScoreContext(trial_config, target, federation.TARGET_RECORD, reference))

    return {
        "trial_id": trial_id,
        "truth_member": bool(truth_member),
        "score": float(score),
        "pred_member": bool(score >= trial_config.threshold),
        "federated_history": history,
    }


def run_attack_trials(config, spec) -> list:
    return [
        run_attack_trial(config, spec, trial_id=i, truth_member=(i % 2 == 0))
        for i in range(config.attack_trials)
    ]


def run_single_experiment(config, spec, *, use_firestore: bool = True, keep_artifacts=None) -> dict:
    # run_id always uses spec: amia and loss have their own key formulas.
    run_id = experiment_key(config, spec)
    if use_firestore:
        cached = firestore.load_cached_result(config, spec)
        if cached and cached.get("status") == "complete":
            log.info("cache hit %s (%s); skipping compute", run_id, spec.name)
            return cached

    artifact_dir = artifact_dir_for(config, spec)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        if spec.custom_trials is not None:
            trials = spec.custom_trials(config, artifact_dir)
        else:
            trials = run_attack_trials(config, spec)
    except Exception as exc:
        log.exception("run %s (%s) failed", run_id, spec.name)
        if use_firestore:
            firestore.mark_result_failed(config, str(exc), spec)
        raise  # NOTE: no cleanup -- a failed run keeps its artifacts.

    if spec.build_payload is not None:
        result = spec.build_payload(config, trials, artifact_dir)
        result.setdefault("run_id", run_id)
        result.setdefault("status", "complete")
        result.setdefault("updated_at_unix", int(time.time()))
        result.setdefault("attack_name", getattr(config, "attack_name", spec.name))
    else:
        result = {
            "run_id": run_id,
            "status": "complete",
            "updated_at_unix": int(time.time()),
            "attack_name": getattr(config, "attack_name", spec.name),
            "config": asdict(config),
            "methodology": spec.methodology,
            # Firestore forbids directly nested arrays: wrap each trial's
            # per-round history (a list) inside a map.
            "federated_history": [
                {"trial_id": r["trial_id"], "rounds": r["federated_history"]} for r in trials
            ],
            "metrics": summarize(trials, spec),
            "attack_trials": [
                {k: r[k] for k in ("trial_id", "truth_member", "score", "pred_member")}
                for r in trials
            ],
            "artifacts": {"artifact_dir": str(artifact_dir), "federated_model_path": None},
        }

    saved = firestore.save_result(config, result, spec) if use_firestore else False
    result["firestore_saved"] = saved
    keep = config.keep_artifacts if keep_artifacts is None else keep_artifacts
    if saved and not keep:
        cleanup_artifacts(artifact_dir)
    return result


def run_sweep(pairs, *, use_firestore: bool = True, keep_artifacts=None,
              on_run_start=None, on_run_end=None) -> list:
    """pairs: iterable of (config, spec). Sequential; --max-parallel is the CLI's job.

    on_run_start/on_run_end bracket each run so an observer (the dashboard's
    run-state report) can say which run is in flight *now*. on_run_end fires in
    a finally: a run that raises must not leave the observer believing it is
    still running. A hard kill still can -- that is what the reader-side
    staleness check is for.
    """
    results = []
    for config, spec in pairs:
        run_id = experiment_key(config, spec)
        if on_run_start is not None:
            on_run_start(run_id, spec.name, config)
        try:
            result = run_single_experiment(
                config, spec, use_firestore=use_firestore, keep_artifacts=keep_artifacts
            )
        finally:
            if on_run_end is not None:
                on_run_end(run_id, spec.name, config)
        results.append(result)
    return results
