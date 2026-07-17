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

    snapshot = db.collection(config.firestore_collection).document(
        experiment_key(config, spec)
    ).get()

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

    # A real write/serialization error (e.g. nested-array rejection) propagates
    # so it fails fast rather than masquerading as "not saved". Do not widen.
    db.collection(config.firestore_collection).document(
        experiment_key(config, spec)
    ).set(result, merge=True)
    return True


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

    db.collection(config.firestore_collection).document(
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


def publish_monitor_state(
    state: Dict, collection: str = "ami_federated_llm_results"
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
