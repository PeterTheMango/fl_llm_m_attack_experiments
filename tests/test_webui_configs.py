# tests/test_webui_configs.py
"""Config editor: validation goes through the CLI's own loader, and nothing
invalid or outside CONFIGS_DIR ever reaches disk."""
import pytest

from master_script.webui import configs

VALID = "attacks:\n  zlib:\n    sweep:\n      seed: [7, 11]\n"


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(configs, "CONFIGS_DIR", tmp_path)
    return tmp_path


def test_missing_suffix_is_added(config_dir):
    assert configs.safe_path("my-sweep").name == "my-sweep.yaml"


@pytest.mark.parametrize("name", [
    "../escape.yaml", "../../etc/passwd", "/tmp/pwned.yaml",
    "sub/dir.yaml", "back\\slash.yaml", "", "  ", ".hidden",
])
def test_names_that_would_escape_the_configs_dir_are_refused(config_dir, name):
    with pytest.raises(configs.ConfigNameError):
        configs.safe_path(name)


def test_validate_reports_the_real_expanded_run_count(config_dir):
    """The grid size is what a sweep costs; it must not be an estimate."""
    result = configs.validate(VALID)
    assert result["ok"] is True
    assert result["runs"] == 2
    assert result["per_attack"] == {"zlib": 2}


def test_validate_rejects_an_unknown_attack(config_dir):
    result = configs.validate("attacks:\n  not_an_attack: {}\n")
    assert result["ok"] is False
    assert "unknown attack" in result["message"]


def test_validate_rejects_a_typoed_field(config_dir):
    """A typo here would otherwise produce a differently-hashed experiment."""
    result = configs.validate("attacks:\n  zlib:\n    base: {federated_roundz: 3}\n")
    assert result["ok"] is False
    assert "federated_roundz" in result["message"]


def test_validate_error_does_not_leak_the_temp_path(config_dir):
    result = configs.validate("attacks:\n  not_an_attack: {}\n")
    assert "/tmp" not in result["message"] and "<config>" in result["message"]


def test_validate_reports_broken_yaml_without_raising(config_dir):
    result = configs.validate("attacks:\n  zlib: [oops\n")
    assert result["ok"] is False
    assert "Error" in result["message"]


def test_validate_writes_nothing(config_dir):
    configs.validate(VALID)
    assert list(config_dir.iterdir()) == []


def test_save_writes_a_valid_config(config_dir):
    result = configs.save("my-sweep", VALID)
    assert result["ok"] is True
    assert (config_dir / "my-sweep.yaml").read_text() == VALID


def test_save_refuses_to_clobber_without_overwrite(config_dir):
    configs.save("my-sweep", VALID)
    result = configs.save("my-sweep", "attacks:\n  zlib: {}\n")
    assert result["ok"] is False and result["exists"] is True
    assert (config_dir / "my-sweep.yaml").read_text() == VALID  # untouched


def test_save_overwrites_when_asked(config_dir):
    configs.save("my-sweep", VALID)
    result = configs.save("my-sweep", "attacks:\n  zlib: {}\n", overwrite=True)
    assert result["ok"] is True and result["runs"] == 1


def test_an_invalid_config_never_reaches_disk(config_dir):
    result = configs.save("broken", "attacks:\n  not_an_attack: {}\n")
    assert result["ok"] is False
    assert not (config_dir / "broken.yaml").exists()


def test_save_refuses_a_traversing_name_without_writing(config_dir, tmp_path):
    result = configs.save("../escaped", VALID)
    assert result["ok"] is False
    assert not (tmp_path.parent / "escaped.yaml").exists()


def test_the_new_config_template_is_itself_valid(config_dir):
    """'New config' must not hand the user something that fails validation."""
    assert configs.validate(configs.TEMPLATE)["ok"] is True


def test_validate_creates_no_temp_file(config_dir, monkeypatch):
    """Validation is a pure expansion; it has no reason to touch the disk."""
    import tempfile

    def _refuse(*a, **kw):
        raise AssertionError("validate must not create a temporary file")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _refuse)
    assert configs.validate(VALID)["ok"] is True
