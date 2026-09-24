"""CPU-only A/B guard benchmark using an existing approved training snapshot.

No model loading, training, GPU, private records or historical ledger writes.
The previous two-pass feature calculation is compared with the current one;
both perform all request copies, hashes, reference checks and fresh accounting.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import tempfile

import numpy as np
from master_script.core import guard_runtime
from master_script.core.guard_features import FEATURE_NAMES, _scaled_layer_features
from master_script.core.guard_runtime import GuardRuntime, GuardSettings, parameter_digest, read_guard_events
from master_script.core.storage import atomic_binary


def legacy_features(parameters, reference):
    layers = [_scaled_layer_features(v, p) for v, p in zip(parameters, reference, strict=True)]
    return dict(zip(FEATURE_NAMES, (float(np.mean([d for d, _ in layers])),
                                  max(d for d, _ in layers), max(c for _, c in layers))))


def select_snapshot(run_dir):
    """Resolve a recorded first-round training snapshot, never a probe request."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    for entry in manifest["entries"]:
        if entry.get("status") != "complete":
            continue
        name = entry["result_file"]
        if Path(name).name != name:
            raise ValueError("Manifest result_file must be a filename")
        result = json.loads((run_dir / name).read_bytes())
        if result["run_id"] != entry["run_id"]:
            raise ValueError("Manifest/result identity mismatch")
        for event in result.get("guard_events", []):
            if (event.get("accounting_scope") == "training" and event.get("round_id") == 1
                    and event.get("features", {}).get("relative_delta") == 0
                    and event.get("decision") == "accepted"):
                key = sha256(event["scope"].encode()).hexdigest()
                path = run_dir / "artifacts" / Path(name).stem / "client-guard" / (key + ".npz")
                if path.is_file():
                    return path, event, result
    raise ValueError("No approved training snapshot found. Run on the original host with artifacts retained.")


@contextmanager
def feature_calculation(fn):
    # This single-process benchmark changes only its own function binding.
    # It never changes the source or the production experiment's settings.
    previous = guard_runtime.parameter_features
    guard_runtime.parameter_features = fn
    try:
        yield
    finally:
        guard_runtime.parameter_features = previous


def benchmark(run_dir, repeats=2):
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    path, historical, result = select_snapshot(run_dir)
    with np.load(path, allow_pickle=False) as stored:
        parameters = [stored[f"arr_{i}"] for i in range(len(stored.files))]
    digests = tuple(parameter_digest(p) for p in parameters)
    digest = sha256("".join(digests).encode()).hexdigest()
    if digest != historical["request_sha256"]:
        raise ValueError("Snapshot differs from the recorded initial training request")
    current = guard_runtime.parameter_features
    samples = {"previous": [], "current": []}
    policy = json.dumps({"schema": "client_guard_v1", "observation_architecture": "causal_lm"})
    with tempfile.TemporaryDirectory(prefix="guard-cpu-benchmark-") as temporary:
        settings = GuardSettings(mode="rules", policy_file="benchmark-client-owned",
                                 policy_sha256=sha256(policy.encode()).hexdigest(), policy_json=policy,
                                 release_budget=1, diagnostic=True, runtime_directory=temporary)
        for repeat in range(repeats):
            # Counterbalance order; these are warm-cache measurements, not a
            # controlled cold-disk or actual FL-round performance experiment.
            order = ("previous", "current") if repeat % 2 == 0 else ("current", "previous")
            for name in order:
                runtime = GuardRuntime(settings, f"benchmark:{repeat}:{name}", str(path), 1, digests)
                with feature_calculation(legacy_features if name == "previous" else current):
                    snapshot = runtime.authorize(parameters, 0, 1)
                if snapshot is None:
                    raise ValueError("Benchmark request unexpectedly rejected")
                del snapshot
                event = read_guard_events(temporary)[-1]
                samples[name].append(event)
    for old, new in zip(samples["previous"], samples["current"], strict=True):
        if not np.allclose([old["features"][k] for k in FEATURE_NAMES],
                           [new["features"][k] for k in FEATURE_NAMES], rtol=1e-10, atol=1e-14):
            raise ValueError("Feature calculation disagrees with previous implementation")
    medians = {name: statistics.median(e["seconds"] for e in events) for name, events in samples.items()}
    return {"schema": "guard_cpu_benchmark_v1", "run_id": result["run_id"],
            "historical_implementation": result["implementation_fingerprint"],
            "feature_source_sha256": sha256(Path(__file__).parents[1].joinpath("core/guard_features.py").read_bytes()).hexdigest(),
            "snapshot": str(path), "snapshot_request_sha256": digest,
            "request_bytes": sum(p.nbytes for p in parameters), "repeats": repeats,
            "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
            "historical_request_seconds": historical["seconds"], "median_seconds": medians,
            "current_over_previous": medians["current"] / medians["previous"],
            "features_agree": True, "samples": samples,
            "limits": ["Approved initial parameters only; no malicious/late-round tensor replay",
                       "Warm-cache CPU microbenchmark; not GPU or FL-round overhead",
                       "Each request rehashes reference and request; separate temporary ledger",
                       "Historical ledgers and snapshots are read-only; no private releases"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new output file: {output}")
    report = benchmark(args.run_dir, args.repeats)
    with atomic_binary(output, exclusive=True) as stream:
        stream.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"output": str(output), "median_seconds": report["median_seconds"],
                      "features_agree": report["features_agree"]}, indent=2))


if __name__ == "__main__":
    main()
