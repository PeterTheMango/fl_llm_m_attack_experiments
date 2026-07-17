"""Filesystem anchors. Everything resolves from this file, never from cwd."""
from pathlib import Path

MASTER_DIR = Path(__file__).resolve().parent
LOGS_DIR = MASTER_DIR / "logs"
OUTPUTS_DIR = MASTER_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "Charts"
CONFIGS_DIR = MASTER_DIR / "configs"
ARTIFACTS_DIR = MASTER_DIR / "artifacts"

for _d in (LOGS_DIR, OUTPUTS_DIR, CHARTS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
