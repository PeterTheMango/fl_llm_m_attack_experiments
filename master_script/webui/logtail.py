# master_script/webui/logtail.py
"""Live tail of the newest session log in master_script/logs/.

The dashboard's log pane shows what the runner actually wrote. It never
invents lines: with no session log on disk the tail is simply empty.
"""
import re
from typing import List, Optional

from ..paths import LOGS_DIR

# "2026-08-05 14:32:01,123 WARNING  master_script.core.runner: message"
_LINE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})[,.]?\d*\s+"
    r"(?P<level>[A-Z]+)\s+(?P<logger>\S+?):\s*(?P<msg>.*)$"
)
_ERR_LEVELS = {"WARNING", "ERROR", "CRITICAL"}
_TAIL_BYTES = 64_000


def newest_session_log() -> Optional[object]:
    logs = sorted(LOGS_DIR.glob("session-*.log"))
    return logs[-1] if logs else None


def tail(limit: int = 60) -> List[dict]:
    """Last `limit` log records as {t, stream, text}. stream is 'err' or 'out'."""
    path = newest_session_log()
    if path is None:
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _TAIL_BYTES))
            raw = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    lines = raw.splitlines()
    if lines and not raw.endswith("\n"):
        lines = lines[:-1]  # drop a half-written trailing line

    out: List[dict] = []
    for line in lines[-limit:]:
        match = _LINE_RE.match(line)
        if match:
            out.append({
                "t": match["time"],
                "stream": "err" if match["level"] in _ERR_LEVELS else "out",
                "text": f"{match['logger'].split('.')[-1]} · {match['msg']}",
            })
        elif line.strip():
            # Continuation line (tracebacks); keep it attached to the pane.
            out.append({"t": "", "stream": "err", "text": line.rstrip()})
    return out
