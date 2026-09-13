"""Release simulation workers and reclaim trial-local model allocations."""
import gc
import logging
import sys

log = logging.getLogger(__name__)


def release_memory():
    gc.collect()
    # Do not import torch just to clean up a toy/config-only run.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    # glibc can retain freed NumPy/model buffers in the process heap. This is
    # best effort and deliberately optional on macOS and non-glibc systems.
    if sys.platform.startswith("linux"):
        import ctypes
        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
        except (AttributeError, OSError):
            pass


def run_simulation(**kwargs):
    import ray
    from flwr.simulation import run_simulation as flower_run
    already_initialized = ray.is_initialized()
    if already_initialized and kwargs.get("backend_config", {}).get("init_args", {}).get("num_cpus") is not None:
        raise RuntimeError("A capped simulation requires a fresh local Ray runtime; restart the experiment process")
    try:
        return flower_run(**kwargs)
    finally:
        # Never disconnect an independently supplied Ray runtime.
        if not already_initialized:
            ray.shutdown()
        release_memory()


def simulation_backend(config):
    backend = {"client_resources": {"num_cpus": 1, "num_gpus": float(config.sim_num_gpus)}}
    cap = getattr(config, "sim_max_concurrent_clients", None)
    if cap is not None:
        if type(cap) is not int or cap <= 0:
            raise ValueError("sim_max_concurrent_clients must be a positive integer")
        # One logical CPU per actor and a bounded local CPU resource pool cap
        # simultaneous ClientApps without changing the participating clients.
        backend["init_args"] = {"num_cpus": cap, "address": "local"}
    return backend


def rss_gib():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024 ** 3, 3)
    except (ImportError, OSError):
        return None
