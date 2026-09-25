"""Prepare and execute bounded independent-target trace collection.

Each job is one AMIA target or one paired Reference target. Checkpoints from
new successful jobs are retired after result/trace verification. Failed jobs
and all historical outputs are preserved. Execution defaults to one job.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import yaml
from master_script.core.config import implementation_fingerprint
from master_script.core.queue import write_json, load_batch
from master_script.core.checkpoint_retention import retire_checkpoints, file_digest
from master_script.tools.build_guard_stage import build_stage, ROOT


SMOKE_TARGET = "2e016354fa6fa91efc79d0ba65e673f5392708522aed8ad3dee8cf86f5f9b6ad"


def prepare(output, *, workers=1, targets=8, seed=2000):
    if workers not in (1, 2, 4) or not 3 <= targets <= 24 or seed < 2000:
        raise ValueError("Use 1/2/4 workers, 3–24 targets, and a fresh seed band >=2000")
    output = Path(output).resolve()
    plan = json.loads((ROOT / "guard/stages/collection_v2.json").read_bytes())
    plan.update(targets=1)
    doc = build_stage(plan)
    train_count = targets // 2
    val_count = max(1, (targets - train_count) // 2)
    jobs = []
    documents = []
    for i in range(targets):
        role = "train" if i < train_count else "validation" if i < train_count + val_count else "test"
        for attack, arms in doc["attacks"].items():
            for arm in arms["variants"]:
                job = deepcopy(doc)
                job["defaults"]["seed"] = seed + i
                job["attacks"] = {attack: {**deepcopy(arms), "variants": [deepcopy(arm)]}}
                job["attacks"][attack]["variants"][0]["pipeline"]["client_guard"]["validation_workers"] = workers
                name = f"job-{len(jobs):03d}.yaml"
                text = yaml.safe_dump(job, sort_keys=False)
                from master_script.core.yaml_config import load_config_doc
                pairs = load_config_doc(job, source=str(output / name))
                if len(pairs) != 1:
                    raise ValueError("Each storage-bounded job must expand to one run")
                config, spec = pairs[0]
                jobs.append({"config": name, "sha256": sha256(text.encode()).hexdigest(), "seed": seed + i,
                             "role": role, "attack": attack, "variant": getattr(config, "attack_variant", None)})
                documents.append((name, text))
    protocol_raw = (ROOT / "guard/study_protocol_v3.json").read_bytes()
    launch = {"schema": "guard_collection_v3", "created_unix": time.time(),
              "implementation_fingerprint": implementation_fingerprint(), "validation_workers": workers,
              "planned_targets": targets, "jobs": jobs, "held_out_variants": ["causal_gradient_alignment"],
              "excluded_targets": [SMOKE_TARGET], "protocol_file": "utility-protocol.json",
              "protocol_sha256": sha256(protocol_raw).hexdigest(),
              "retention": "retire completed new-job model/probe weights and guard NPZ after result/ledger verification; keep JSON/JSONL/SQLite",
              "purpose": "Diagnostic detector data only, not a claim that privacy/utility/overhead gates pass",
              "duration": "Default execution is one job; the complete plan is a multi-hour study requiring deliberate bounded execution",
              "storage": "20 GiB free output and 4 GiB scratch preflight per job; estimates, not reservations or guarantees"}
    output.mkdir(parents=True, exist_ok=False)
    for name, text in documents:
        (output / name).write_text(text)
    (output / "utility-protocol.json").write_bytes(protocol_raw)
    write_json(output / "launch.json", launch)
    return launch


def require_headroom(path, gib):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < gib * 1024 ** 3:
        raise ValueError(f"Need {gib} GiB free on {path} before starting another job")


def execute(launch_path, *, gpu, max_jobs=1, scratch_root=None):
    lock = Path(launch_path).resolve().parent / ".collector.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Collector is running or its lock survived interruption; inspect before removing the lock") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
        return _execute(launch_path, gpu=gpu, max_jobs=max_jobs, scratch_root=scratch_root)
    finally:
        lock.unlink()


def _execute(launch_path, *, gpu, max_jobs=1, scratch_root=None):
    launch_path = Path(launch_path).resolve()
    root = launch_path.parent
    launch = json.loads(launch_path.read_bytes())
    if launch["schema"] != "guard_collection_v3" or implementation_fingerprint() != launch["implementation_fingerprint"]:
        raise ValueError("Collection code changed; prepare a new frozen launch")
    if file_digest(root / launch["protocol_file"]) != launch["protocol_sha256"]:
        raise ValueError("Utility protocol changed after declaration")
    if not str(gpu).isdigit() or type(max_jobs) is not int or max_jobs <= 0:
        raise ValueError("Select a numeric GPU and positive max_jobs")
    batches = []
    for job in launch["jobs"]:
        p = root / job["config"]
        if p.parent != root or file_digest(p) != job["sha256"]:
            raise ValueError("Collection configuration changed")
        batch = load_batch([p])
        if len(batch.pairs) != 1:
            raise ValueError("Job must contain exactly one run")
        batches.append(batch)
    state_path = root / "progress.json"
    state = json.loads(state_path.read_bytes()) if state_path.exists() else {
        "launch_sha256": file_digest(launch_path), "completed": [], "active_job": None}
    if state["launch_sha256"] != file_digest(launch_path) or state["active_job"] is not None:
        raise ValueError("Changed launch or interrupted/failed job: preserve artifacts and inspect before continuing")
    for completed_job in state["completed"]:
        saved = (root / completed_job["result"]).resolve()
        if root not in saved.parents or file_digest(saved) != completed_job["sha256"]:
            raise ValueError("A completed trace changed; preserve it and inspect before continuing")
    # Resolve all identities before GPU/private work. Repeated seeds are not
    # assumed independent, and the same target can never acquire two roles.
    from master_script.core.datasets import target_record_for
    targets, groups = {}, {}
    for job, batch in zip(launch["jobs"], batches):
        config, _ = batch.pairs[0]
        target = sha256(target_record_for(config, "").encode()).hexdigest()
        seed = str(job["seed"])
        if target in launch["excluded_targets"]:
            raise ValueError("Collection overlaps the diagnostic smoke target")
        if seed in targets and targets[seed] != target:
            raise ValueError("Related attack variants selected different targets")
        if seed not in targets and target in targets.values():
            raise ValueError("Different seeds selected the same target; prepare a disjoint cohort")
        targets[seed] = target
        if target in groups and groups[target] != job["role"]:
            raise ValueError("A target cannot cross split roles")
        groups[target] = job["role"]
    if state.get("targets", targets) != targets:
        raise ValueError("Target identities changed since the first collection job")
    state["targets"] = targets
    write_json(state_path, state)
    scratch_root = Path(scratch_root or tempfile.gettempdir()).resolve()
    if not scratch_root.is_dir():
        raise ValueError("Scratch root must be an existing writable directory")
    completed = {r["job"] for r in state["completed"]}
    for index in [i for i in range(len(batches)) if i not in completed][:max_jobs]:
        require_headroom(root, 20)
        require_headroom(scratch_root, 4)
        job = launch["jobs"][index]
        scratch = Path(tempfile.mkdtemp(prefix="gcv3-", dir=scratch_root))
        job_output = root / "results" / f"job-{index:03d}"
        state["active_job"] = {"index": index, "scratch": str(scratch)}
        write_json(state_path, state)
        command = [sys.executable, "-m", "master_script.perform_experiments", "--queue", str(root / job["config"]),
                   "--queue-output", str(job_output), "--no-firestore", "--no-charts"]
        # Each job is a new process; no private-world or ledger retry occurs.
        subprocess.run(command, env={**os.environ, "EXPERIMENT_GPU": str(gpu), "RAY_TMPDIR": str(scratch)}, check=True)
        manifests = list(job_output.glob("*/manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Unexpected batch layout; preserve artifacts")
        manifest = json.loads(manifests[0].read_bytes())
        if manifest["status"] != "complete" or len(manifest["entries"]) != 1 or manifest["entries"][0]["status"] != "complete":
            raise ValueError("Job did not complete; its artifacts are preserved")
        entry = manifest["entries"][0]
        if Path(entry["result_file"]).name != entry["result_file"] or entry.get("source_sha256") != job["sha256"]:
            raise ValueError("Executed job source or result location differs from the declaration")
        result_path = manifests[0].parent / entry["result_file"]
        result = json.loads(result_path.read_bytes())
        found = {t.get("target_sha256") or t.get("candidate_sha256") for t in result["attack_trials"]}
        if found != {targets[str(job["seed"])]}:
            raise ValueError("Actual private run target differs from the declared target")
        if result.get("implementation_fingerprint") != launch["implementation_fingerprint"]:
            raise ValueError("Executed code differs from frozen launch")
        retirement = retire_checkpoints(result_path)
        # Preserve Ray text logs, then remove only the temporary directory
        # created by this invocation, after its experiment process has exited.
        with tarfile.open(job_output / "ray-logs.tar.gz", "x:gz") as tar:
            for log in scratch.rglob("*"):
                if log.is_file() and not log.is_symlink() and log.suffix in (".log", ".out", ".err"):
                    tar.add(log, arcname=str(log.relative_to(scratch)))
        shutil.rmtree(scratch)
        state["completed"].append({"job": index, "result": str(result_path.relative_to(root)),
                                   "sha256": file_digest(result_path), "reclaimed_bytes": retirement["reclaimed_bytes"]})
        state["active_job"] = None
        write_json(state_path, state)
        print(f"Completed and retired {index + 1}/{len(batches)}; retained trace: {result_path}", flush=True)
    splits = {"schema": "guard_splits_v2", "sources": [{"path": r["result"], "sha256": r["sha256"]} for r in state["completed"]],
              "groups": groups, "held_out_variants": launch["held_out_variants"]}
    write_json(root / "splits.partial.json", splits)
    if len(state["completed"]) == len(batches):
        write_json(root / "splits.complete.json", splits)
        print("Collection complete. Use splits.complete.json with assemble_guard_dataset.")
    else:
        print("Bounded invocation complete; rerun the same launch to execute the next unstarted job(s).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("output"); prep.add_argument("--workers", type=int, choices=[1, 2, 4], default=1)
    prep.add_argument("--targets", type=int, default=8); prep.add_argument("--seed", type=int, default=2000)
    run = sub.add_parser("run")
    run.add_argument("launch"); run.add_argument("--gpu", required=True)
    run.add_argument("--max-jobs", type=int, default=1); run.add_argument("--scratch-root")
    args = p.parse_args()
    if args.action == "prepare":
        launch = prepare(args.output, workers=args.workers, targets=args.targets, seed=args.seed)
        print(f"Prepared {len(launch['jobs'])} one-target jobs. No training started. Default execution is one job.")
    else:
        execute(args.launch, gpu=args.gpu, max_jobs=args.max_jobs, scratch_root=args.scratch_root)


if __name__ == "__main__":
    main()
