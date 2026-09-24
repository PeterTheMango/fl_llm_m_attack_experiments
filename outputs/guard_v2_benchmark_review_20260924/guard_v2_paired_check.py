"""Prepare two matched GPU runs; --run explicitly launches and packages them.

Copy this file to the remote repository root and run there. Without --run,
only configuration generation and the existing CLI dry run are performed.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from uuid import uuid4


def check_output_space(path, minimum_gib):
    """Check the destination volume before allocating a new experiment."""
    location = Path(path)
    while not location.exists():
        location = location.parent
    free_gib = shutil.disk_usage(location).free / 1024 ** 3
    if free_gib < minimum_gib:
        raise ValueError(f"Only {free_gib:.2f} GiB free on the output filesystem; "
                         f"this pair requires {minimum_gib:g} GiB of preflight headroom. "
                         "Choose --output-root on a larger volume or archive old artifacts first. "
                         "No training started.")
    return free_gib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Allocated physical GPU index")
    parser.add_argument("--run", action="store_true", help="Launch the two runs after validation")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"),
                        help="Destination for the new run, including an optional larger mounted volume")
    parser.add_argument("--min-free-gib", type=float, default=20.,
                        help="Required free output space before --run (default: 20 GiB; estimated headroom, not a storage guarantee)")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("--gpu must be a numeric physical GPU index")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 1:
        parser.error("--min-free-gib must be finite and at least 1")
    root = Path.cwd()
    if not (root / "master_script/core/guard_features.py").is_file():
        parser.error("Run from the repository root")
    feature_hash = sha256((root / "master_script/core/guard_features.py").read_bytes()).hexdigest()
    if feature_hash != "db7edc658b1113ba6b80cd14b5366008581309892c8f97321115db39cffd1797":
        parser.error("Feature source differs from the benchmarked patch; reconcile it before this comparison")
    sys.path.insert(0, str(root))
    import yaml
    from master_script.tools.build_guard_stage import build_stage, ROOT
    from master_script.core.yaml_config import load_config_doc
    from master_script.tools.analyze_guard_study import pairing_key

    plan = json.loads((ROOT / "guard/stages/smoke_v2.json").read_text())
    plan.update(conditions=["baseline", "rules_only"], variants=["causal_gradient_alignment"],
                reference_conditions=[], purpose="Paired runtime and utility regression after CPU optimization; one diagnostic target")
    doc = build_stage(plan)
    doc["attacks"] = {"amia": doc["attacks"]["amia"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_parent = args.output_root.expanduser().resolve()
    free_gib = None
    if args.run:
        try:
            free_gib = check_output_space(output_parent, args.min_free_gib)
        except ValueError as exc:
            parser.error(str(exc))
    output = output_parent / f"guard-v2-paired-check-{stamp}-{uuid4().hex[:8]}"
    queue = output / "experiments.yaml"
    pairs = load_config_doc(doc, source=str(queue))
    if len(pairs) != 2 or {s.pipeline.condition for _, s in pairs} != {"baseline", "rules_only"}:
        raise ValueError("Expected exactly baseline and rules-only")
    if any(c.attack_variant != "causal_gradient_alignment" or c.federated_rounds != 3
           or c.attack_targets != 1 or c.attack_trials != 4 for c, _ in pairs):
        raise ValueError("Unexpected scope for the bounded check")
    output.mkdir(parents=True, exist_ok=False)
    queue.write_text(yaml.safe_dump(doc, sort_keys=False))
    (output / "launch.json").write_text(json.dumps({"plan": plan, "expected_runs": 2,
        "gpu": args.gpu, "feature_source_sha256": feature_hash,
        "output_root": str(output_parent), "free_output_gib_before_launch": free_gib,
        "minimum_free_output_gib": args.min_free_gib,
        "storage_note": "Output-volume check only; model caches, logs and Ray temporary files may use other volumes. 20 GiB is estimated headroom for retained models and guard snapshots, not a reservation or guarantee.",
        "runtime_estimate_minutes": [25, 35],
        "estimate_basis": "Prior causal baseline 626s and rules 842s; hardware/load may change timing",
        "purpose": "Measure actual FL overhead and collect private no-context answer audit",
        "limitations": "One reused diagnostic target; no classifier or independent efficacy evaluation"}, indent=2) + "\n")
    command = [sys.executable, "-m", "master_script.perform_experiments", "--queue", str(queue),
               "--attack", "amia", "--queue-output", str(output / "results"), "--no-firestore", "--no-charts"]
    subprocess.run(command + ["--dry-run"], check=True)
    print(f"Prepared two runs: {output}", flush=True)
    if not args.run:
        print("No training started. Add --run to prepare a new pair and launch it.")
        return
    subprocess.run(command, env={**os.environ, "EXPERIMENT_GPU": args.gpu}, check=True)
    manifests = list((output / "results").glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one new batch manifest")
    batch = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    if len(manifest["entries"]) != 2 or any(e["status"] != "complete" for e in manifest["entries"]):
        raise ValueError(f"The pair did not complete; inspect {manifests[0]}")
    results = [batch / e["result_file"] for e in manifest["entries"]]
    payloads = [json.loads(p.read_text()) for p in results]
    if pairing_key(payloads[0]) != pairing_key(payloads[1]):
        raise ValueError("Completed results do not form a matched pair")
    subprocess.run([sys.executable, "-m", "master_script.tools.analyze_guard_study",
                    *map(str, results), "--output", str(batch / "paired-analysis.json")], check=True)
    # Include only the missing no-context answers, not all raw audit content.
    audits = []
    for index, result_path in enumerate(results):
        for path in (batch / "artifacts" / result_path.stem).rglob("rag-answers.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("kind") == "no_retrieval_utility":
                    audits.append({**row, "run_id": payloads[index]["run_id"]})
    audit_path = output / "no-context-answers.jsonl"
    with os.fdopen(os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        for row in audits:
            stream.write(json.dumps(row) + "\n")
    if not audits:
        print("No no-context audit rows found; keep the original artifacts for diagnosis.", flush=True)
    archive = output / "paired-check-review.tar.gz"
    with os.fdopen(os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            for path in [queue, output / "launch.json", audit_path, *sorted(batch.glob("*.json"))]:
                tar.add(path, arcname=str(path.relative_to(output)))
    print(f"Attach this review archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
