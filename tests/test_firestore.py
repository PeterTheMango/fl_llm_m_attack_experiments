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
