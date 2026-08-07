#!/usr/bin/env python3
"""Run MIA adaptation experiments from a YAML config.

Consolidates the 11 notebooks in code_experiments/adaptations/. Attack and
evaluation behavior is identical to the notebooks; run_id hashes are preserved
so existing Firestore results still cache-hit.

Examples
--------
List every registered attack:
    python perform_experiments.py --list-attacks

See what a sweep would do, without spending GPU hours:
    python perform_experiments.py --config configs/example_sweep.yaml --dry-run

Run the smoke suite (no GPU, no credentials):
    python perform_experiments.py --config configs/smoke.yaml

Run one attack for real, on both GPUs, with charts:
    python perform_experiments.py --config configs/example_sweep.yaml \\
        --attack zlib --max-parallel 2 --charts
"""
from pathlib import Path
import argparse
import sys

from .paths import CONFIGS_DIR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perform_experiments.py",
        description="Run membership-inference attack experiments against federated LLM fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "smoke.yaml",
                   help="YAML config file (default: configs/smoke.yaml)")
    p.add_argument("--attack", action="append", dest="attacks", metavar="NAME",
                   help="Attack to run; repeatable. Default: every attack in the config.")
    p.add_argument("--list-attacks", action="store_true",
                   help="Print the attack registry and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Expand the grid, print run_ids and cache status, run nothing.")
    p.add_argument("--max-parallel", type=int, default=1, choices=(1, 2),
                   help="Concurrent runs. 2 spawns one GPU-pinned subprocess per run (2-GPU VM).")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="Keep local model/probe artifacts after a successful persist.")
    p.add_argument("--no-firestore", action="store_true",
                   help="Local-only: skip the cache read and the result write.")
    p.add_argument("--charts", dest="charts", action="store_true", default=True,
                   help="Render Adv charts into outputs/Charts/ (default).")
    p.add_argument("--no-charts", dest="charts", action="store_false",
                   help="Skip chart rendering.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from .core.registry import ATTACKS

    if args.list_attacks:
        print(f"{len(ATTACKS)} registered attacks:\n")
        for name in sorted(ATTACKS):
            spec = ATTACKS[name]
            toy = "toy+hf" if spec.supports_toy else "hf only"
            print(f"  {name:<18} {toy:<8} {spec.config_cls.__name__}")
        return 0

    from .core.yaml_config import ConfigError, load_config_file

    try:
        pairs = load_config_file(args.config, only=args.attacks)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(pairs, use_firestore=not args.no_firestore)

    # GPU pinning MUST precede any torch import.
    from .core.gpu import apply_gpu_selection

    apply_gpu_selection()
    from .logging_setup import setup_session_logging

    log_path = setup_session_logging(args.log_level)
    print(f"session log: {log_path}")
    return _run(pairs, args)


def _dry_run(pairs, use_firestore: bool) -> int:
    from .core.config import experiment_key
    from .core.firestore import load_cached_result

    cached_n = 0
    print(f"{len(pairs)} run(s) planned\n")
    for cfg, spec in pairs:
        run_id = experiment_key(cfg, spec)
        status = "pending"
        if use_firestore and load_cached_result(cfg, spec):
            status = "cached"
            cached_n += 1
        print(f"  {run_id}  {spec.name:<18} {status}")
    print(f"\n{cached_n} cached, {len(pairs) - cached_n} pending")
    return 0


def _run(pairs, args) -> int:
    from .core.runner import run_sweep
    from .core.runstate import RunStateReporter, publish_manifest, suppressed

    # The run-state report is what lets the dashboard show a CLI sweep as it
    # happens. Off without Firestore (nowhere to publish) and in GPU-pinned
    # children (the parent reports for them).
    report = not args.no_firestore and not suppressed()
    reporter = RunStateReporter() if report else None
    if report:
        publish_manifest(pairs)

    try:
        if args.max_parallel == 2:
            results = _run_parallel(pairs, args, reporter)
        else:
            results = run_sweep(
                pairs,
                use_firestore=not args.no_firestore,
                keep_artifacts=args.keep_artifacts or None,
                **(reporter.hooks if reporter else {}),
            )
    finally:
        # Whatever happened, stop claiming anything is running.
        if reporter is not None:
            reporter.stop()

    if args.charts:
        from .core.charts import render_sweep_summary

        for path in render_sweep_summary([r for r in results if r]):
            print(f"chart: {path}")

    ok = sum(1 for r in results if r and r.get("status") == "complete")
    print(f"\n{ok}/{len(pairs)} run(s) complete")
    for r in results:
        if r and r.get("metrics"):
            print(f"  {r.get('run_id')}  adv={r['metrics'].get('adv'):.3f}")
    return 0 if ok == len(pairs) else 1


def _run_parallel(pairs, args, reporter=None) -> list:
    """One GPU-pinned subprocess per run, at most two in flight (2-GPU VM).

    Subprocesses (not threads): Flower/Ray and CUDA do not share a process
    cleanly, and each run must pin its own GPU before torch initializes.
    """
    import os
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from .core.config import experiment_key

    from .core.runstate import SUPPRESS_ENV

    def _one(item):
        index, (cfg, spec) = item
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            from dataclasses import asdict

            import yaml

            yaml.safe_dump({"attacks": {spec.name: {"base": asdict(cfg)}}}, fh)
            single = fh.name
        # The parent owns the run-state report for the whole sweep: the report
        # is one array, so a child publishing its own would clobber its sibling.
        env = {**os.environ, "EXPERIMENT_GPU": str(index % 2), SUPPRESS_ENV: "1"}
        cmd = [sys.executable, "-m", "master_script.perform_experiments",
               "--config", single, "--max-parallel", "1", "--no-charts",
               "--log-level", args.log_level]
        if args.no_firestore:
            cmd.append("--no-firestore")
        if args.keep_artifacts:
            cmd.append("--keep-artifacts")
        run_id = experiment_key(cfg, spec)
        if reporter is not None:
            reporter.on_run_start(run_id, spec.name, cfg)
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        finally:
            if reporter is not None:
                reporter.on_run_end(run_id)
        os.unlink(single)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return None
        return {"run_id": run_id, "status": "complete", "config": asdict(cfg)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(_one, enumerate(pairs)))


if __name__ == "__main__":
    raise SystemExit(main())
