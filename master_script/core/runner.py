"""Experiment orchestration. The single code path shared by the CLI and web UI.

Ordering is load-bearing and mirrors the notebooks exactly:
cache check -> FL fine-tune -> attack -> measure -> persist -> cleanup.
"""
from dataclasses import asdict, replace
from pathlib import Path
import logging
import math
from hashlib import sha256
import shutil
import time

from . import federation, firestore
from . import datasets as dataset_sources
from .config import artifact_dir_for, experiment_key, validate_attack_config, resolve_run_config, METHOD_VERSION, implementation_fingerprint
from .metrics import summarize
from .scoring import ScoreContext

log = logging.getLogger(__name__)


def cleanup_artifacts(artifact_dir) -> None:
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def reset_ray_after_failure() -> None:
    """Best-effort reset of a Ray driver left unhealthy by a failed simulation.

    Ray is an optional dependency for toy/no-HF runs, so importing it here must
    remain lazy.  Cleanup must never replace the experiment error or prevent the
    rest of a sweep from running.
    """
    try:
        import ray
    except ImportError:
        return

    try:
        # Safe even when Ray failed before completing initialization.  This
        # disconnects the current driver and clears its local global state so
        # the next Flower simulation can initialize a fresh backend.
        ray.shutdown()
    except Exception:
        log.exception("could not reset Ray after a failed experiment")


def failed_sweep_result(config, spec, run_id: str, exc: Exception) -> dict:
    """Small result envelope used to keep a sweep moving after one run fails."""
    try:
        config_payload = asdict(config)
    except TypeError:
        # Real runs use dataclass configs.  Keeping this defensive makes the
        # orchestration boundary tolerant of lightweight callers and tests.
        config_payload = dict(config) if isinstance(config, dict) else {}

    message = f"{type(exc).__name__}: {exc}"
    return {
        "run_id": run_id,
        "status": "failed",
        "updated_at_unix": int(time.time()),
        "attack_name": getattr(config, "attack_name", spec.name),
        "config": config_payload,
        "error": message[:2000],
        **({"pipeline": spec.pipeline.metadata()} if getattr(spec, "pipeline", None) is not None else {}),
    }


def run_attack_trial(config, spec, trial_id: int, truth_member: bool) -> dict:
    # Per-trial reseed, exactly as the notebooks do. NOTE: trial_config is never
    # hashed -- the run_id belongs to the original config. Real-data trials use
    # the same seed in adjacent positive/negative trials so each pair differs
    # only in target membership, not in the sampled target or client corpus.
    seed_offset = trial_id // 2
    trial_config = replace(config, seed=config.seed + seed_offset)
    if trial_config.use_hf_models:
        target, history = federation.run_hf_federated_finetune(
            trial_config, truth_member=truth_member,
            **({"pipeline": spec.pipeline} if spec.pipeline is not None else {}))
        reference = federation.load_reference_bundle(trial_config) if spec.needs_reference else None
        candidate_text = target.get("target_record", federation.TARGET_RECORD)
        score = spec.score_hf(ScoreContext(trial_config, target, candidate_text, reference))
    else:
        target, history = federation.run_toy_federated_finetune(trial_config, truth_member=truth_member)
        reference = None
        if spec.needs_reference:
            reference = federation.ToyFederatedLM()
        candidate_text = getattr(target, "target_record", federation.TARGET_RECORD)
        score = spec.score_toy(ScoreContext(trial_config, target, candidate_text, reference))

    if not math.isfinite(float(score)):
        raise FloatingPointError(f"{spec.name} produced a nonfinite membership score")
    exposed = any(config.target_client_id in r.get("selected_clients", []) for r in history)
    trial = {
        "trial_id": trial_id,
        "truth_member": bool(truth_member),
        "score": float(score),
        "pred_member": bool(score >= trial_config.threshold),
        "federated_history": history,
        "membership_target": "assigned_training_record",
        "target_exposed": bool(truth_member and exposed),
        "seed": trial_config.seed,
        "candidate_sha256": sha256(candidate_text.encode()).hexdigest(),
        **({"training_provenance": target.get("training_provenance", {}),
            "attack_provenance": target.get("attack_provenance", {})} if trial_config.use_hf_models else {}),
    }
    if spec.pipeline is not None:
        from .rag import evaluate_pipeline
        trial["pipeline_evaluation"] = evaluate_pipeline(target, trial_config, spec.pipeline, trial_id=trial_id)
    return trial


def run_attack_trials(config, spec) -> list:
    return [
        run_attack_trial(config, spec, trial_id=i, truth_member=(i % 2 == 0))
        for i in range(config.attack_trials)
    ]


def run_single_experiment(config, spec, *, use_firestore: bool = True, keep_artifacts=None,
                          artifact_directory=None) -> dict:
    validate_attack_config(config, spec)
    config = resolve_run_config(config)
    # run_id always uses spec: amia and loss have their own key formulas.
    run_id = experiment_key(config, spec)
    cache_error = None
    if use_firestore:
        try:
            cached = firestore.load_cached_result(config, spec)
        except Exception as exc:
            if spec.pipeline is None and artifact_directory is None:
                raise
            cached = None
            cache_error = f"{type(exc).__name__}: {exc}"[:2000]
            log.warning("cache unavailable for %s; computing with local persistence", run_id)
        if cached and cached.get("status") == "complete":
            log.info("cache hit %s (%s); skipping compute", run_id, spec.name)
            return cached

    artifact_dir = Path(artifact_directory) if artifact_directory is not None else artifact_dir_for(config, spec)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        context = None
        if spec.custom_trials is not None:
            trials = spec.custom_trials(config, artifact_dir,
                                        **({"pipeline": spec.pipeline} if spec.pipeline is not None else {}))
            if isinstance(trials, dict):
                context, trials = trials["context"], trials["trials"]
        else:
            trials = run_attack_trials(config, spec)
        if not trials or not all(math.isfinite(float(t["score"])) for t in trials):
            raise FloatingPointError("A completed experiment requires finite nonempty trial scores")
    except Exception as exc:
        log.exception("run %s (%s) failed", run_id, spec.name)
        if use_firestore:
            firestore.mark_result_failed(config, str(exc), spec)
        raise  # NOTE: no cleanup -- a failed run keeps its artifacts.

    if spec.build_payload is not None:
        result = spec.build_payload(config, trials, artifact_dir, **({"context": context} if context is not None else {}))
        result.setdefault("run_id", run_id)
        result.setdefault("status", "complete")
        result.setdefault("updated_at_unix", int(time.time()))
        result.setdefault("attack_name", getattr(config, "attack_name", spec.name))
        result.setdefault("config", asdict(config))
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
                {k: value for k, value in r.items() if k not in ("federated_history", "pipeline_evaluation")}
                for r in trials
            ],
            "artifacts": {"artifact_dir": str(artifact_dir), "federated_model_path": None},
        }

    result["method_version"] = METHOD_VERSION
    result["implementation_fingerprint"] = implementation_fingerprint()
    result["evaluation_scope"] = "FL adaptation; not a reproduction of the source benchmark"
    if spec.name == "wbc":
        result["wbc_protocol"] = {
            "window_sizes": list(config.window_sizes),
            "schedule": ("local-v2-appendix-author-config" if tuple(config.window_sizes) == (2,3,4,6,9,13,18,25,32,40)
                         else "equation-12" if tuple(config.window_sizes) == (2,3,4,5,8,11,15,21,29,40) else "explicit-custom"),
            "aggregation": "uniform_mean_of_per_size_positive_fractions",
            "short_input_policy": "clamp_each_requested_size_to_scored_length",
            "source_discrepancy": "Local v2 Eq.12 and Appendix C.1.5 give different schedules"}
    if spec.pipeline is not None:
        from .metrics import scientific_metrics
        result["run_id"] = run_id
        result["pipeline"] = spec.pipeline.metadata()
        result["pipeline_evaluation_scope"] = "Study-specific FL defense/RAG adaptation; not a source benchmark reproduction"
        result["pipeline_evaluations"] = [
            {"trial_id": t["trial_id"], **t["pipeline_evaluation"]}
            for t in trials if "pipeline_evaluation" in t
        ]
        labels = [t["truth_member"] for t in trials]
        scores = [(-1 if spec.name == "loss" else 1) * t["score"] for t in trials]
        result["batch_membership_metrics" if spec.name == "amia" else "training_membership_metrics"] = scientific_metrics(labels, scores)
        # Sequentially released models on overlapping records must compose.
        from .defenses import privacy_bound
        protected = [e.get("training_privacy", {}) for e in result["pipeline_evaluations"]]
        if spec.pipeline.defense.mechanism != "none":
            if not protected or any(p.get("steps", 0) <= 0 for p in protected):
                raise RuntimeError("Private training returned no accounting evidence; refusing a privacy claim")
            result["training_privacy_composed"] = privacy_bound(
                sum(p.get("steps", 0) for p in protected),
                spec.pipeline.defense.noise_multiplier, spec.pipeline.defense.delta)
            result["training_privacy_composed"]["scope"] = (
                "Training releases in this experiment only; audit scores, probe training, "
                "private retrieval outputs and repeated experiments are not covered.")
    if spec.pipeline is not None or artifact_directory is not None:
        from .queue import write_json
        # Keep completed computation even if the optional remote write fails.
        if cache_error:
            result["cache_error"] = cache_error
        result["firestore_saved"] = False
        write_json(artifact_dir / "result.json", result)
    try:
        saved = firestore.save_result(config, result, spec) if use_firestore else False
    except Exception as exc:
        if spec.pipeline is None and artifact_directory is None:
            raise
        saved = False
        result["persistence_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        log.exception("computed run %s retained locally after Firestore write failed", run_id)
    result["firestore_saved"] = saved
    if spec.pipeline is not None or artifact_directory is not None:
        from .queue import write_json
        write_json(artifact_dir / "result.json", result)
    keep = config.keep_artifacts if keep_artifacts is None else keep_artifacts
    if saved and not keep and spec.pipeline is None and artifact_directory is None:
        cleanup_artifacts(artifact_dir)
    return result


def run_sweep(pairs, *, use_firestore: bool = True, keep_artifacts=None,
              on_run_start=None, on_run_end=None, on_result=None, artifact_base=None) -> list:
    """pairs: iterable of (config, spec). Sequential; --max-parallel is the CLI's job.

    on_run_start/on_run_end bracket each run so an observer (the dashboard's
    run-state report) can say which run is in flight *now*. A failed run is
    returned with status=failed and the next run proceeds. on_run_end fires in
    a finally so failures never leave the observer believing a run is active.
    A hard kill still can -- that is what reader-side staleness detection is
    for.
    """
    results = []
    for index, (config, spec) in enumerate(pairs):
        run_id = experiment_key(config, spec)
        try:
            config = resolve_run_config(config)
            run_id = experiment_key(config, spec)
            if on_run_start is not None:
                on_run_start(run_id, spec.name, config)
            result = run_single_experiment(
                config, spec, use_firestore=use_firestore, keep_artifacts=keep_artifacts,
                **({"artifact_directory": Path(artifact_base) / f"{index:04d}-{run_id}"}
                   if artifact_base is not None else {}),
            )
        except Exception as exc:
            # run_single_experiment logs the traceback and persists the failed
            # status for execution failures.  The sweep boundary is deliberately
            # fail-soft: retain a compact local result, reset a possibly broken
            # Ray driver, and advance to the next independent attack/config.
            log.error(
                "continuing sweep after run %s (%s) failed: %s: %s",
                run_id, spec.name, type(exc).__name__, exc,
            )
            reset_ray_after_failure()
            result = failed_sweep_result(config, spec, run_id, exc)
        finally:
            if on_run_end is not None:
                on_run_end(run_id, spec.name, config)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
