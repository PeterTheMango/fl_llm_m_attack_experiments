# tests/test_webui_settings.py
"""Settings page: .env edits keep the file's shape, secrets stay masked unless
explicitly revealed, and a save actually reaches the running process."""
import os

import pytest

from master_script.webui import envfile

ORIGINAL = """# credentials
FIREBASE_PROJECT_ID=my-project
FIREBASE_SERVICE_ACCOUNT_JSON=super-secret-value-1234

# unrelated
SOME_OTHER_TOOL=keep-me
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(ORIGINAL)
    monkeypatch.setattr(envfile, "env_path", lambda: path)
    return path


def test_mask_never_contains_the_whole_secret():
    masked = envfile.mask("super-secret-value-1234")
    assert "super-secret" not in masked
    assert masked.endswith("1234")


def test_short_secrets_are_masked_entirely():
    assert set(envfile.mask("abc")) == {"•"}


@pytest.mark.parametrize("key,secret", [
    ("FIREBASE_SERVICE_ACCOUNT_JSON", True),
    ("GOOGLE_APPLICATION_CREDENTIALS", False),  # a path, not a credential
    ("FIREBASE_PROJECT_ID", False),
    ("SOME_API_TOKEN", True),
    ("MY_PASSWORD", True),
])
def test_secret_classification(key, secret):
    assert envfile.is_secret(key) is secret


def test_entries_mask_secrets_and_list_known_missing_keys(env):
    entries = {e["key"]: e for e in envfile.entries()}
    assert "super-secret-value-1234" not in entries["FIREBASE_SERVICE_ACCOUNT_JSON"]["value"]
    assert entries["FIREBASE_SERVICE_ACCOUNT_JSON"]["masked"] is True
    assert entries["FIREBASE_PROJECT_ID"]["value"] == "my-project"  # not a secret
    assert entries["EXPERIMENT_GPU"]["set"] is False  # known but absent
    assert entries["SOME_OTHER_TOOL"]["value"] == "keep-me"  # unknown keys survive


def test_reveal_is_the_only_path_to_a_plaintext_secret(env):
    assert envfile.reveal("FIREBASE_SERVICE_ACCOUNT_JSON") == "super-secret-value-1234"
    assert envfile.reveal("NOT_A_KEY") is None


def test_write_preserves_comments_ordering_and_untouched_keys(env):
    envfile.write({"FIREBASE_PROJECT_ID": "new-project"})
    text = env.read_text()
    assert "# credentials" in text and "# unrelated" in text
    assert "SOME_OTHER_TOOL=keep-me" in text
    assert "FIREBASE_PROJECT_ID=new-project" in text
    assert "my-project" not in text


def test_write_backs_up_the_previous_file(env):
    """This edits live credentials; a fat-fingered save must be recoverable."""
    envfile.write({"FIREBASE_PROJECT_ID": "new-project"})
    backup = envfile.backup_path()
    assert backup.name == ".env.bak"
    assert backup.read_text() == ORIGINAL


def test_write_quotes_values_that_would_otherwise_reparse_wrongly(env):
    envfile.write({"NOTE": "hello world # not a comment"})
    assert 'NOTE="hello world # not a comment"' in env.read_text()
    assert envfile.read_pairs()["NOTE"] == "hello world # not a comment"


def test_write_deletes_requested_keys(env):
    envfile.write({}, deletes=["FIREBASE_SERVICE_ACCOUNT_JSON"])
    assert "FIREBASE_SERVICE_ACCOUNT_JSON" not in envfile.read_pairs()
    assert "SOME_OTHER_TOOL" in envfile.read_pairs()


def test_new_keys_are_appended_not_dropped(env):
    envfile.write({"EXPERIMENT_GPU": "1"})
    assert envfile.read_pairs()["EXPERIMENT_GPU"] == "1"


def test_reload_pushes_values_into_the_running_process(env, monkeypatch):
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    changed = envfile.reload_into_process()
    assert "FIREBASE_PROJECT_ID" in changed
    assert os.environ["FIREBASE_PROJECT_ID"] == "my-project"


def test_reload_drops_a_managed_key_that_was_removed_from_the_file(env, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU", "3")
    envfile.reload_into_process()
    assert "EXPERIMENT_GPU" not in os.environ


def test_reload_leaves_unrelated_inherited_environment_alone(env, monkeypatch):
    monkeypatch.setenv("PATH_LIKE_THING", "untouched")
    envfile.reload_into_process()
    assert os.environ["PATH_LIKE_THING"] == "untouched"


def test_reload_resets_the_cached_firestore_client(env, monkeypatch):
    """New credentials are useless while firebase_admin holds the old app."""
    from master_script.core import firestore

    called = {}
    monkeypatch.setattr(firestore, "reset_client", lambda: called.setdefault("reset", True))
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    envfile.reload_into_process()
    assert called.get("reset") is True


def test_reset_client_is_safe_when_no_app_was_ever_initialized():
    from master_script.core import firestore

    assert firestore.reset_client() in (True, False)  # never raises
