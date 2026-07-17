"""Session-wide and per-run logging, both rooted in master_script/logs/."""
import logging
import time
from pathlib import Path

from .paths import LOGS_DIR

_FMT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_session_logging(level: str = "INFO") -> Path:
    """Attach a session log file to the root logger. Returns the log path."""
    log_path = LOGS_DIR / f"session-{time.strftime('%Y%m%d-%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(_FMT))
    root.addHandler(file_handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(_FMT))
    root.addHandler(stream)
    return log_path


def run_log_handler(run_id: str) -> logging.Handler:
    """A per-run file handler; caller adds/removes it around a run."""
    handler = logging.FileHandler(LOGS_DIR / f"{run_id}.log")
    handler.setFormatter(logging.Formatter(_FMT))
    return handler
