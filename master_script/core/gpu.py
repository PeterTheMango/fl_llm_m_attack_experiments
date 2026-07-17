# Pin to one GPU before torch initializes CUDA (avoids OOM on a shared box).
import os
import shutil
import subprocess
from pathlib import Path


def _read_env_file_var(name: str):
    # Minimal .env reader so GPU pinning works before python-dotenv is loaded.
    for base in [Path.cwd(), *Path.cwd().parents]:
        env_path = base / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == name:
                    return value.strip().strip('"').strip("'")
            break
    return None


def _gpu_free_memory():
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, free = line.split(",")
        rows.append((idx.strip(), int(free)))
    return rows


def select_gpu() -> str | None:
    forced = os.environ.get("EXPERIMENT_GPU") or _read_env_file_var("EXPERIMENT_GPU")
    if forced is not None and forced.strip() != "":
        return forced.strip()
    rows = _gpu_free_memory()
    if not rows:
        return None
    return max(rows, key=lambda r: r[1])[0]


def apply_gpu_selection() -> str | None:
    """Set CUDA_VISIBLE_DEVICES from select_gpu(). Call before importing torch."""
    chosen = select_gpu()
    if chosen is None:
        print("GPU selection: no GPU detected; using default device visibility.")
    elif chosen.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("GPU selection: forced CPU (CUDA_VISIBLE_DEVICES='').")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen
        _free = dict(_gpu_free_memory()).get(chosen)
        _detail = f" ({_free} MiB free)" if _free is not None else ""
        print(f"GPU selection: pinned to physical GPU {chosen}{_detail}; it appears as cuda:0 in this process.")
    return chosen
