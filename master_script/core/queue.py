"""Validated FIFO batches using the existing sequential experiment runner."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import yaml

from .config import experiment_key
from .yaml_config import ConfigError, load_config_doc
from ..paths import OUTPUTS_DIR


@dataclass
class Batch:
    pairs: list
    entries: list
    directory: str | None = None


def load_batch(paths, only=None):
    """Snapshot and validate every file before any execution or output creation."""
    if not paths:
        raise ConfigError("A queue requires at least one config file")
    pairs, entries = [], []
    for path in paths:
        path = Path(path).resolve()
        try:
            raw = path.read_bytes()
            doc = yaml.safe_load(raw)
            if not isinstance(doc, dict):
                raise ConfigError(f"{path}: expected a YAML mapping")
            expanded = load_config_doc(doc, only, source=str(path))
        except (OSError, yaml.YAMLError, TypeError, AttributeError) as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        if not expanded:
            raise ConfigError(f"{path}: no runs match the selected attacks")
        digest = sha256(raw).hexdigest()
        for cfg, spec in expanded:
            entries.append({"index": len(entries), "source": str(path),
                            "source_sha256": digest,
                            "run_id": experiment_key(cfg, spec), "attack": spec.name,
                            "status": "pending"})
            pairs.append((cfg, spec))
    return Batch(pairs, entries)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def write_json(path, payload):
    """Atomic replacement within a newly allocated batch directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def interrupt_batch(directory):
    """Called only after the dashboard has terminated its owned process tree."""
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    if manifest["status"] != "running":
        return
    manifest.update(status="interrupted", ended_unix=time.time())
    for entry in manifest["entries"]:
        if entry["status"] == "running":
            entry.update(status="interrupted", ended_unix=time.time())
        elif entry["status"] == "pending":
            entry["status"] = "not_run"
    write_json(path, manifest)


def reserve_directory(output_root=None):
    root = Path(output_root) if output_root is not None else OUTPUTS_DIR / "queues"
    directory = root / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def run_batch(batch, *, output_root=None, use_firestore=True, keep_artifacts=None,
              on_run_start=None, on_run_end=None, on_progress=None):
    from .runner import run_sweep

    directory = Path(batch.directory) if batch.directory else reserve_directory(output_root)
    if any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing batch directory")
    manifest = {"batch_id": directory.name, "status": "running",
                "started_unix": time.time(), "entries": [dict(e) for e in batch.entries]}
    index = 0

    def persist():
        write_json(directory / "manifest.json", manifest)
        if on_progress:
            on_progress({"batch_dir": str(directory), "finished": index,
                         "planned": len(batch.pairs)})

    def start(run_id, attack, config):
        manifest["entries"][index].update(run_id=run_id, status="running", started_unix=time.time())
        persist()
        if on_run_start:
            on_run_start(run_id, attack, config)

    def result_ready(result):
        nonlocal index
        entry = manifest["entries"][index]
        entry["run_id"] = result["run_id"]
        filename = f"{index:04d}-{entry['run_id']}.json"
        write_json(directory / filename, result)
        entry.update(status=result["status"], ended_unix=time.time(), result_file=filename)
        index += 1
        persist()

    persist()
    try:
        results = run_sweep(batch.pairs, use_firestore=use_firestore,
                            keep_artifacts=keep_artifacts, on_run_start=start,
                            on_run_end=on_run_end, on_result=result_ready,
                            artifact_base=directory / "artifacts")
        manifest["status"] = "complete" if all(r.get("status") == "complete" for r in results) else "failed"
        return results, directory
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        for entry in manifest["entries"]:
            if entry["status"] == "running":
                entry.update(status="interrupted", ended_unix=time.time())
            elif entry["status"] == "pending":
                entry["status"] = "not_run"
        raise
    finally:
        manifest["ended_unix"] = time.time()
        persist()
