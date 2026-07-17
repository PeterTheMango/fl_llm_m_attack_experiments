# master_script/webui/state.py
"""In-memory projection of Firestore. Deliberately non-durable (spec §1.2).

On restart the dashboard rebuilds entirely by re-reading Firestore. Nothing
here is ever persisted, and nothing here is ever used to resume a run.
"""
from statistics import mean
from typing import Callable, Dict, List, Optional

from ..core.firestore import MONITOR_STATE_DOC, get_firestore_client


class DashboardState:
    def __init__(self) -> None:
        self.runs: Dict[str, dict] = {}
        self.running: List[dict] = []
        self.manifest: Optional[List[dict]] = None
        self._listener = None

    def ingest(self, docs) -> None:
        for doc in docs:
            if doc.get("run_id") == MONITOR_STATE_DOC:
                self.running = doc.get("running") or []
                self.manifest = doc.get("manifest")
                continue
            if "run_id" in doc:
                self.runs[doc["run_id"]] = doc

    def attach_listener(self, collection: str, on_change: Callable[[], None]) -> None:
        """Firestore real-time listener; on_change fires on the SDK's thread."""
        db = get_firestore_client()

        def _cb(snapshots, changes, read_time):
            self.ingest([s.to_dict() | {"run_id": s.id} for s in snapshots])
            on_change()

        self._listener = db.collection(collection).on_snapshot(_cb)

    def filtered(self, attack: Optional[str] = None, status: Optional[str] = None, **factors) -> List[dict]:
        out = []
        for run in self.runs.values():
            cfg = run.get("config", {})
            if attack and cfg.get("attack_name") != attack:
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
            key = run.get("config", {}).get(field)
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
