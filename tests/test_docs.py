# tests/test_docs.py
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "master_script" / "README.md"


def test_readme_exists():
    assert README.exists()


def test_readme_documents_every_cli_flag():
    from master_script.perform_experiments import build_parser

    text = README.read_text()
    for action in build_parser()._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                assert opt in text, f"{opt} is undocumented in README.md"


def test_readme_explains_hash_preservation():
    text = README.read_text().lower()
    assert "run_id" in text and "hash" in text
