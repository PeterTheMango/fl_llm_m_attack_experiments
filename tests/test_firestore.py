import pytest

from master_script.core import firestore as fs
from master_script.core.attacks.zlib import ZlibConfig


class _FakeDoc:
    def __init__(self, store, key):
        self._store, self._key = store, key

    def get(self):
        return _FakeSnap(self._store.get(self._key))

    def set(self, payload, merge=False):
        if any(isinstance(v, list) and v and isinstance(v[0], list) for v in payload.values()):
            raise ValueError("nested arrays are not supported")
        self._store[self._key] = payload


class _FakeSnap:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return self._data


class _FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, key):
        return _FakeDoc(self._store, key)


class _FakeDB:
    def __init__(self):
        self.store = {}

    def collection(self, _name):
        return _FakeCollection(self.store)


def test_load_cached_returns_none_without_credentials(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS.")

    monkeypatch.setattr(fs, "get_firestore_client", _boom)
    assert fs.load_cached_result(ZlibConfig()) is None


def test_save_returns_false_without_credentials(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS.")

    monkeypatch.setattr(fs, "get_firestore_client", _boom)
    assert fs.save_result(ZlibConfig(), {"status": "complete"}) is False


def test_genuine_write_error_reraises_not_swallowed(monkeypatch):
    """A nested-array rejection must fail fast, not look like 'not saved'."""
    db = _FakeDB()
    monkeypatch.setattr(fs, "get_firestore_client", lambda *a, **k: db)
    with pytest.raises(ValueError, match="nested arrays"):
        fs.save_result(ZlibConfig(), {"federated_history": [[1, 2], [3]]})


def test_cache_hit_only_when_status_complete(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(fs, "get_firestore_client", lambda *a, **k: db)
    cfg = ZlibConfig()
    from master_script.core.config import experiment_key

    db.store[experiment_key(cfg)] = {"status": "running"}
    assert fs.load_cached_result(cfg) is None
    db.store[experiment_key(cfg)] = {"status": "complete", "run_id": "x"}
    assert fs.load_cached_result(cfg)["run_id"] == "x"


# ---------- a recovered run must not keep reading as failed ----------

_DELETE = object()  # stand-in for firestore.DELETE_FIELD


def test_success_clears_a_previous_attempts_error():
    """Writes are merges and the success payload has no `error` key, so without
    this the old error string survives forever and the run reads as failed."""
    payload = fs._clear_stale_error({"status": "complete", "metrics": {}}, _DELETE)
    assert payload["error"] is _DELETE


def test_a_failed_write_keeps_its_error():
    payload = fs._clear_stale_error({"status": "failed", "error": "boom"}, _DELETE)
    assert payload["error"] == "boom"


def test_a_payload_carrying_its_own_error_is_left_alone():
    payload = fs._clear_stale_error({"status": "complete", "error": "warned"}, _DELETE)
    assert payload["error"] == "warned"


def test_clearing_is_skipped_when_the_sentinel_is_unavailable():
    """firebase-admin is an optional import; absence must not corrupt the write."""
    result = {"status": "complete", "metrics": {}}
    assert fs._clear_stale_error(result, None) == result


def test_clearing_does_not_mutate_the_callers_result():
    result = {"status": "complete"}
    fs._clear_stale_error(result, _DELETE)
    assert "error" not in result


def test_save_result_sends_the_cleared_payload(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(fs, "get_firestore_client", lambda *a, **k: db)
    monkeypatch.setattr(fs, "_delete_field_sentinel", lambda: _DELETE)
    from master_script.core.config import experiment_key

    cfg = ZlibConfig()
    assert fs.save_result(cfg, {"status": "complete", "metrics": {"adv": 0.6}}) is True
    assert db.store[experiment_key(cfg)]["error"] is _DELETE
