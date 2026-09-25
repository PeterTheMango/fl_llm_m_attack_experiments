"""Retire replay weights only after a new job's complete results are durable.

Results, raw answer audits, request features, configuration and release ledgers
are retained. A tombstone prevents reinitializing the retired guard scope.
"""
from hashlib import sha256
import json
from pathlib import Path
from .queue import write_json


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retire_checkpoints(result_path):
    """Only the conventional artifact directory belonging to this result."""
    path = Path(result_path).resolve()
    result = json.loads(path.read_bytes())
    if result.get("status") != "complete":
        raise ValueError("Cannot retire incomplete experiment checkpoints")
    artifact = path.parent / "artifacts" / path.stem
    if artifact.is_symlink() or not artifact.is_dir():
        raise ValueError("Missing or symlinked owned artifact directory")
    local = json.loads((artifact / "result.json").read_bytes())
    if local != result:
        raise ValueError("Queue result and artifact result disagree; preserve checkpoints")
    if result.get("pipeline", {}).get("client_guard"):
        from .guard_runtime import read_guard_events
        if not result.get("guard_events") or read_guard_events(artifact / "client-guard") != result["guard_events"]:
            raise ValueError("Durable guard trace disagrees with result; preserve checkpoints")
    tombstone = artifact / "checkpoint-retirement.json"
    if tombstone.exists():
        raise ValueError("Retirement already recorded; do not automatically retry partial deletion")
    files = []
    for p in sorted(artifact.rglob("*")):
        if p.is_symlink():
            raise ValueError("Unexpected symlink in owned artifact directory")
        if p.is_file() and (p.suffix == ".safetensors" or p.match("pytorch_model*.bin")
                            or p.name == "ami_probe.pt" or (p.suffix == ".npz" and p.parent.name == "client-guard")):
            files.append({"path": str(p.relative_to(artifact)), "bytes": p.stat().st_size, "sha256": file_digest(p)})
    report = {"schema": "checkpoint_retirement_v1", "run_id": result["run_id"],
              "result_sha256": file_digest(path), "status": "retiring", "files": files,
              "retained": "result/config/trace JSON, private audit JSONL, tokenizer metadata and SQLite ledgers",
              "replay": "unavailable after retirement; never recreate this scope"}
    # Publish the deletion inventory before unlinking anything. If interrupted,
    # retain the partial tombstone and stop; the new collector never resumes it.
    write_json(tombstone, report)
    for item in files:
        (artifact / item["path"]).unlink()
    report.update(status="retired", reclaimed_bytes=sum(f["bytes"] for f in files))
    write_json(tombstone, report)
    return report
