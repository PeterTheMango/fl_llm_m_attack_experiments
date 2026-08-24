# master_script/webui/state.py
"""In-memory projection of Firestore. Deliberately non-durable (spec §1.2).

On restart the dashboard rebuilds entirely by re-reading Firestore. Nothing
here is ever persisted, and nothing here is ever used to resume a run.
"""
import threading
import time
from statistics import mean
from typing import Callable, Dict, List, Optional

from ..core.firestore import MONITOR_STATE_DOC, get_firestore_client

COLLECTION = "ami_federated_llm_results"


def attack_name(run: dict) -> Optional[str]:
    """Resolve a run's display/group name.

    Nine attacks store `attack_name` on the frozen config; amia/loss don't
    (and MUST NOT, per the hash-preservation contract), so fall back to the
    payload-level `attack_name` the runner stamps on every result.
    """
    return run.get("config", {}).get("attack_name") or run.get("attack_name") or "?"


class DashboardState:
    def __init__(self) -> None:
        self.runs: Dict[str, dict] = {}
        self.running: List[dict] = []
        self.manifest: Optional[List[dict]] = None
        self.last_sync_unix: Optional[float] = None
        self.error: str = ""
        self._listener = None
        self._lock = threading.Lock()

    @property
    def live(self) -> bool:
        """True once a Firestore listener is pushing changes to this projection."""
        return self._listener is not None

    def ingest(self, docs) -> None:
        for doc in docs:
            if doc.get("run_id") == MONITOR_STATE_DOC:
                self.running = doc.get("running") or []
                self.manifest = doc.get("manifest")
                continue
            if "run_id" in doc:
                self.runs[doc["run_id"]] = doc
        self.last_sync_unix = time.time()

    def remove_run(self, run_id: str) -> bool:
        """Remove one run from the live projection after an authoritative delete."""
        with self._lock:
            existed = run_id in self.runs
            self.runs.pop(run_id, None)
            self.last_sync_unix = time.time()
        return existed

    def clear_run_state(self) -> None:
        """Clear transient running/manifest state after an owned sweep stops."""
        with self._lock:
            self.running = []
            self.manifest = []
            self.last_sync_unix = time.time()

    def refresh(self, collection: str = COLLECTION) -> bool:
        """One-shot re-read of the collection. False when credentials are missing.

        The listener is the normal path; this is the fallback for environments
        where on_snapshot isn't usable, and it is what makes the dashboard
        rebuild itself from Firestore alone after a restart (§1.2).
        """
        try:
            db = get_firestore_client()
        except Exception as exc:
            self.error = str(exc)
            self.last_sync_unix = time.time()  # throttle retries; don't hammer on every read
            return False
        try:
            docs = [s.to_dict() | {"run_id": s.id} for s in db.collection(collection).stream()]
        except Exception as exc:
            self.error = str(exc)
            self.last_sync_unix = time.time()  # throttle retries; don't hammer on every read
            return False
        with self._lock:
            self.runs.clear()
            self.running = []
            self.manifest = None
            self.ingest(docs)
        self.error = ""
        return True

    def ensure_fresh(self, max_age_seconds: float = 5.0, collection: str = COLLECTION) -> None:
        """Refresh on read when nothing is pushing updates. No-op under a listener."""
        if self.live:
            return
        if self.last_sync_unix is not None and time.time() - self.last_sync_unix < max_age_seconds:
            return
        self.refresh(collection)

    def attach_listener(self, collection: str, on_change: Callable[[], None]) -> None:
        """Firestore real-time listener; on_change fires on the SDK's thread."""
        db = get_firestore_client()

        def _cb(snapshots, changes, read_time):
            # A delete is absent from snapshots, so ingesting the remaining
            # documents cannot remove it from the projection on its own.
            for change in changes or []:
                kind = getattr(getattr(change, "type", None), "name", "")
                if kind == "REMOVED":
                    self.remove_run(change.document.id)
            self.ingest([s.to_dict() | {"run_id": s.id} for s in snapshots])
            on_change()

        self._listener = db.collection(collection).on_snapshot(_cb)

    def detach_listener(self) -> None:
        """Drop the listener so the next read re-connects.

        Called after the credentials change: a watch opened with the old
        service account would keep streaming (or keep failing) against it.
        """
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.unsubscribe()
            except Exception:
                pass  # already closed, or the SDK tore it down with the app

    def filtered(self, attack: Optional[str] = None, status: Optional[str] = None, **factors) -> List[dict]:
        out = []
        for run in self.runs.values():
            cfg = run.get("config", {})
            if attack and attack_name(run) != attack:
                continue
            if status and run.get("status") != status:
                continue
            if any(cfg.get(k) != v for k, v in factors.items()):
                continue
            out.append(run)
        return out

    def aggregate_by(self, field: str) -> Dict[str, dict]:
        buckets: Dict[str, List[float]] = {}
        for run in self.runs.values():
            adv = (run.get("metrics") or {}).get("adv")
            key = attack_name(run) if field == "attack_name" else run.get("config", {}).get(field)
            if adv is None or key is None:
                continue
            buckets.setdefault(str(key), []).append(adv)
        return {
            k: {"mean_adv": mean(v), "max_adv": max(v), "min_adv": min(v), "count": len(v)}
            for k, v in buckets.items()
        }

    def recently_finished(self, limit: int = 10) -> List[dict]:
        return sorted(
            self.runs.values(), key=lambda r: r.get("updated_at_unix", 0), reverse=True
        )[:limit]

    def sweep_progress(self) -> dict:
        """Complete/failed come from Firestore; total/pending need the manifest (§2.2)."""
        complete = sum(1 for r in self.runs.values() if r.get("status") == "complete")
        failed = sum(1 for r in self.runs.values() if r.get("status") == "failed")
        total = len(self.manifest) if self.manifest is not None else None
        pending = None if total is None else max(0, total - complete - failed - len(self.running))
        return {
            "complete": complete, "failed": failed, "running": len(self.running),
            "total": total, "pending": pending,
        }
