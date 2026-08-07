# master_script/webui/gpustats.py
"""VM GPU telemetry via nvidia-smi. Nice-to-have: absent GPUs are not an error."""
import shutil
import subprocess
from typing import List

_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
_TIMEOUT_S = 4


def poll() -> List[dict]:
    """Current per-GPU utilisation. Returns [] when nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            text=True, timeout=_TIMEOUT_S,
        )
    except Exception:
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "id": int(parts[0]),
                "name": parts[1],
                "util": float(parts[2]),
                "mem_used_gib": float(parts[3]) / 1024.0,
                "mem_total_gib": float(parts[4]) / 1024.0,
                "temp_c": float(parts[5]),
            })
        except ValueError:
            continue
    return gpus
