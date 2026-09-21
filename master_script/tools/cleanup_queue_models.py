"""Preview removal of model weights/snapshots from completed queue runs.

Default is read-only. --apply permanently deletes only the printed model files.
Results, logs, configs, probes, trial checkpoints and release ledgers are kept.
Never use while an experiment or another process is reading those model files.
"""
import argparse
import json
from pathlib import Path
from master_script.core.queue import write_json


TERMINAL = {"complete", "failed", "interrupted", "storage_exhausted"}


def candidates(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") not in TERMINAL:
        raise ValueError("Queue is not finalized; refusing cleanup of a running or stale queue")
    found = []
    for entry in manifest["entries"]:
        if entry.get("status") != "complete":
            continue
        root = directory / "artifacts" / f"{entry['index']:04d}-{entry['run_id']}"
        if not root.resolve().is_relative_to(directory / "artifacts"):
            raise ValueError("Artifact path escapes the queue")
        for path in root.rglob("*"):
            if any(p.is_symlink() for p in (path, *path.parents) if p != directory.parent):
                continue
            if not path.is_file():
                continue
            model = any(p.name == "federated_model" for p in path.relative_to(root).parents)
            weights = model and (path.suffix == ".safetensors" or (path.name.startswith("pytorch_model") and path.suffix == ".bin"))
            snapshot = path.parent.name == "client-guard" and path.suffix == ".npz"
            if weights or snapshot:
                found.append(path)
    return sorted(found)


def cleanup(directory, apply=False):
    directory = Path(directory).resolve()
    files = candidates(directory)
    total = sum(p.stat().st_size for p in files)
    for path in files:
        print(f"{path.stat().st_size / 1024**3:.3f} GiB  {path}")
    print(f"{'Delete' if apply else 'Would delete'} {len(files)} model files, {total / 1024**3:.2f} GiB")
    if apply and files:
        # Keep an append-only history of cleanup attempts, even after a partial failure.
        import time
        report = directory / f"model-cleanup-{time.time_ns()}.json"
        payload = {"status": "in_progress", "planned": [str(p.relative_to(directory)) for p in files], "deleted": []}
        write_json(report, payload)
        for path in files:
            # Revalidate the queue and paths immediately before unlinking.
            if path not in candidates(directory):
                raise RuntimeError("Cleanup candidate changed; stopping")
            path.unlink()
            payload["deleted"].append(str(path.relative_to(directory)))
            write_json(report, payload)
        payload["status"] = "complete"
        write_json(report, payload)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_directory")
    parser.add_argument("--apply", action="store_true", help="Permanently delete listed model files; run without this flag first")
    args = parser.parse_args()
    cleanup(args.queue_directory, args.apply)


if __name__ == "__main__":
    main()
