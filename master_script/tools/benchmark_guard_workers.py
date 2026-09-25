"""Compare serial and concurrent full guard checks on a saved public snapshot."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import tempfile
import numpy as np
from master_script.core.guard_runtime import GuardRuntime, GuardSettings, parameter_digest, read_guard_events
from master_script.core.config import implementation_fingerprint
from master_script.core.storage import atomic_binary
from master_script.tools.benchmark_guard_runtime import select_snapshot


def benchmark(run_dir, workers=(1, 2, 4), repeats=2):
    if not workers or len(set(workers)) != len(workers) or any(type(w) is not int or w not in (1, 2, 4) for w in workers):
        raise ValueError("Use distinct worker counts from 1, 2, 4")
    if type(repeats) is not int or not 1 <= repeats <= 6:
        raise ValueError("repeats must be between 1 and 6")
    path, historical, result = select_snapshot(run_dir)
    with np.load(path, allow_pickle=False) as stored:
        parameters = [stored[f"arr_{i}"] for i in range(len(stored.files))]
    digests = tuple(parameter_digest(p) for p in parameters)
    request_hash = sha256("".join(digests).encode()).hexdigest()
    if request_hash != historical["request_sha256"]:
        raise ValueError("Snapshot differs from the recorded initial training request")
    samples = {str(w): [] for w in workers}
    policy = json.dumps({"schema": "client_guard_v1", "observation_architecture": "causal_lm"})
    with tempfile.TemporaryDirectory(prefix="guard-workers-") as temporary:
        settings = GuardSettings("rules", "benchmark", sha256(policy.encode()).hexdigest(), policy, 1,
                                 runtime_directory=temporary, diagnostic=True)
        for repeat in range(repeats):
            order = list(workers) if repeat % 2 == 0 else list(reversed(workers))
            for w in order:
                runtime = GuardRuntime(replace(settings, validation_workers=w), f"benchmark:{repeat}:{w}",
                                       str(path), 1, digests)
                owned = runtime.authorize(parameters, 0, 1)
                if owned is None:
                    raise ValueError("Public benchmark request was rejected")
                del owned
                samples[str(w)].append(read_guard_events(temporary)[-1])
    first = next(iter(samples.values()))[0]
    for events in samples.values():
        for event in events:
            if event["request_sha256"] != first["request_sha256"] or event["features"] != first["features"]:
                raise ValueError("Worker settings changed request identity or features")
    medians = {w: statistics.median(e["seconds"] for e in events) for w, events in samples.items()}
    return {"schema": "guard_worker_benchmark_v1", "implementation_fingerprint": implementation_fingerprint(),
            "snapshot_source_run": result["run_id"], "snapshot_request_sha256": request_hash,
            "request_bytes": historical["request_bytes"], "repeats": repeats,
            "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
            "median_wall_seconds": medians, "features_and_hashes_identical": True, "samples": samples,
            "limits": ["Initial approved weights only; warm-cache CPU measurements",
                       "Reference task times overlap; use event seconds for total elapsed cost",
                       "Workers hold up to one reference layer each in addition to the owned request",
                       "No private computation, GPU, original-ledger mutation or historical checkpoint writes",
                       "No automatic worker selection or 10% FL overhead claim"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new benchmark output")
    report = benchmark(args.run_dir, args.workers, args.repeats)
    with atomic_binary(output, exclusive=True) as stream:
        stream.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"output": str(output), "median_wall_seconds": report["median_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
