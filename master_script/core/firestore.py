"""Firestore cache, persistence, and monitoring for attack experiments.

Ported from zlib_adaptations.ipynb cells 7 and 19, with spec parameter support
for AMIA/LOSS key-formula dispatch (see config.experiment_key).
"""
from dataclasses import asdict
from typing import Any, Dict, Optional
import json
import os
import time

# Load environment variables from .env at import time (optional dependency).
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from .config import experiment_key


MONITOR_STATE_DOC = "monitor_state"
RESULTS_COLLECTION = "ami_federated_llm_results"
LEGACY_LOSS_COLLECTION = "loss_federated_llm_results"


def _read_collections(spec: Optional[Any] = None) -> tuple[str, ...]:
    """Canonical result collection followed by any attack-specific legacy home."""
    if getattr(spec, "name", None) == "loss":
        return RESULTS_COLLECTION, LEGACY_LOSS_COLLECTION
    return (RESULTS_COLLECTION,)


def get_firestore_client(project_id: Optional[str] = None):
    """Initialize and return a Firestore client.

    Imports firebase_admin and firestore only when called (function-local),
    so importing this module never requires the firebase-admin package.

    Raises RuntimeError if credentials are not found.
    """
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        raw_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if raw_json:
            cred = credentials.Certificate(json.loads(raw_json))
        elif cred_path:
            cred = credentials.Certificate(cred_path)
        else:
            raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS.")

        options = {"projectId": project_id} if project_id else None
        firebase_admin.initialize_app(cred, options=options)

    return firestore.client()


def reset_client() -> bool:
    """Drop the cached firebase_admin app so the next call re-authenticates.

    get_firestore_client() initializes at most once per process. Editing
    credentials at runtime (the settings page) therefore has no effect until
    that app is deleted -- without this, new credentials sit in os.environ
    while the client keeps using the ones it started with.

    Returns True if an app was actually deleted.
    """
    try:
        import firebase_admin
    except ImportError:
        return False

    deleted = False
    for app in list(firebase_admin._apps.values()):
        try:
            firebase_admin.delete_app(app)
            deleted = True
        except Exception:
            # Already torn down, or in use by a listener that outlived it.
            pass
    return deleted


def load_cached_result(config: Any, spec: Optional[Any] = None) -> Optional[Dict]:
    """Load a cached result from Firestore if it exists and is complete.

    Returns None if:
    - Credentials are missing (missing-credentials case only, caught and swallowed)
    - The document does not exist
    - The document status is not "complete"

    Args:
        config: The attack configuration object.
        spec: Optional spec object for key-formula dispatch (e.g., AMIA, LOSS).
              Passed to experiment_key(config, spec).

    Returns:
        The document dict if found and status=="complete", else None.
    """
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        # Missing-credentials case only: run locally, uncached.
        return None

    run_id = experiment_key(config, spec)
    for collection in _read_collections(spec):
        snapshot = db.collection(collection).document(run_id).get()
        if snapshot.exists:
            payload = snapshot.to_dict()
            if payload.get("status") == "complete":
                return payload

    return None


def save_result(config: Any, result: Dict, spec: Optional[Any] = None) -> bool:
    """Save a result to Firestore, returning success status.

    Returns False ONLY if credentials are missing (missing-credentials case).
    A genuine write/serialization error (e.g., nested-array rejection by Firestore)
    MUST propagate/re-raise so it fails fast rather than masquerading as "not saved".

    Args:
        config: The attack configuration object.
        result: The result dict to save.
        spec: Optional spec object for key-formula dispatch.

    Returns:
        True if saved successfully, False if credentials are missing.

    Raises:
        Any non-credentials exception (e.g., ValueError from Firestore).
    """
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        # Missing-credentials case ONLY: fine to skip writing locally.
        return False

    payload = _clear_stale_error(result, _delete_field_sentinel())

    # A real write/serialization error (e.g. nested-array rejection) propagates
    # so it fails fast rather than masquerading as "not saved". Do not widen.
    db.collection(RESULTS_COLLECTION).document(
        experiment_key(config, spec)
    ).set(payload, merge=True)
    return True


def _delete_field_sentinel():
    """firestore.DELETE_FIELD, or None where firebase-admin isn't importable.

    Kept function-local like every other firebase_admin use here: importing
    this module must never require the package (see module docstring).
    """
    try:
        from firebase_admin import firestore

        return firestore.DELETE_FIELD
    except Exception:
        return None


def _clear_stale_error(result: Dict, delete_sentinel: Any) -> Dict:
    """Drop a previous attempt's `error` when this attempt succeeded.

    Writes are merges, and the success payload has no `error` key -- so without
    this, a run that failed and was later re-run keeps the old error string
    forever and reads as failed. Deleting the field (rather than writing
    `error: None`) restores exactly the shape a never-failed run has.
    """
    if delete_sentinel is None or result.get("status") != "complete" or "error" in result:
        return result
    return {**result, "error": delete_sentinel}


def mark_result_failed(config: Any, error: str, spec: Optional[Any] = None) -> bool:
    """Record a failure in Firestore.

    Never triggers artifact cleanup (see runner). Returns False if credentials
    are missing, True if saved successfully.

    Args:
        config: The attack configuration object.
        error: Error message or exception to record.
        spec: Optional spec object for key-formula dispatch.

    Returns:
        True if saved, False if credentials are missing.
    """
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        return False

    db.collection(RESULTS_COLLECTION).document(
        experiment_key(config, spec)
    ).set(
        {
            "run_id": experiment_key(config, spec),
            "status": "failed",
            "updated_at_unix": int(time.time()),
            "config": asdict(config),
            "error": str(error)[:2000],
        },
        merge=True,
    )
    return True


def delete_result(run_id: str, collection: str = RESULTS_COLLECTION) -> bool:
    """Delete one persisted experiment result.

    Unlike cache reads and optional writes, an explicit destructive request
    must surface credential and network failures to its caller. Silently
    treating a failed delete as success would leave the dashboard lying about
    what remains authoritative in Firestore.
    """
    db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    db.collection(collection).document(run_id).delete()
    return True


def publish_monitor_state(
    state: Dict, collection: str = RESULTS_COLLECTION
) -> bool:
    """Publish the optional run-state report (ideation doc §1.1).

    Coarse and transient: the currently-running run_ids and the planned sweep
    manifest. Never used to resume or reconstruct a run (§1.2).

    Args:
        state: State dict to publish.
        collection: Collection name (default: "ami_federated_llm_results").

    Returns:
        True if saved, False if credentials are missing.
    """
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        return False

    db.collection(collection).document(MONITOR_STATE_DOC).set(
        {**state, "updated_at_unix": int(time.time())}, merge=True
    )
    return True
