# master_script/webui/gpustats.py
"""VM GPU telemetry via nvidia-smi. Nice-to-have: absent GPUs are not an error.

Fields are parsed INDIVIDUALLY, and an unparseable one becomes None rather than
discarding the whole GPU. nvidia-smi answers "[N/A]" for fields the host does
not expose, and the two it withholds most often are the two this module wants
most: utilization.gpu is unavailable at the parent-device level once MIG is
enabled (it exists per GPU-instance, which only DCGM reports), and
temperature.gpu is unavailable inside a vGPU guest. An all-or-nothing parse
turned a perfectly working GRID A100D-1-20C into "no GPU on this host".
"""
import shutil
import subprocess
from typing import List, Optional

_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
_TIMEOUT_S = 4


def _num(value: str) -> Optional[float]:
    """A field's value, or None for nvidia-smi's "[N/A]" / "[Not Supported]"."""
    try:
        return float(value)
    except ValueError:
        return None


def _gib(value: str) -> Optional[float]:
    mib = _num(value)
    return None if mib is None else mib / 1024.0


def poll() -> List[dict]:
    """Current per-GPU utilisation. Returns [] when nvidia-smi is unavailable.

    A row is dropped only when it has no usable identity (no integer index).
    Every other field may be None; callers must render "unknown" rather than 0,
    since 0% utilisation and unknown utilisation are different claims.
    """
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
            gpu_id = int(parts[0])
        except ValueError:
            continue  # No index: nothing to key telemetry to.
        gpus.append({
            "id": gpu_id,
            "name": parts[1] or f"GPU {gpu_id}",
            "util": _num(parts[2]),
            "mem_used_gib": _gib(parts[3]),
            "mem_total_gib": _gib(parts[4]),
            "temp_c": _num(parts[5]),
        })
    return gpus
