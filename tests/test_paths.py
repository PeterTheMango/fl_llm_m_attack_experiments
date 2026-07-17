import os
from pathlib import Path

from master_script import paths


def test_dirs_resolve_relative_to_master_script_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib

    importlib.reload(paths)
    assert paths.MASTER_DIR.name == "master_script"
    assert paths.LOGS_DIR == paths.MASTER_DIR / "logs"
    assert paths.CHARTS_DIR == paths.MASTER_DIR / "outputs" / "Charts"


def test_dirs_exist_after_import():
    assert paths.LOGS_DIR.is_dir()
    assert paths.CHARTS_DIR.is_dir()


def test_charts_dir_is_capitalized():
    assert paths.CHARTS_DIR.name == "Charts"
