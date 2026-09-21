"""Storage checks for large artifacts; headroom is a guardrail, not a reservation."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import tempfile

MIN_FREE_BYTES = 1024 ** 3  # Leave 1 GiB for logs, SQLite and result manifests.


class StorageCapacityError(OSError):
    def __init__(self, message):
        super().__init__(errno.ENOSPC, message)


def require_space(path, write_bytes=0):
    location = Path(path).resolve()
    while not location.exists():
        location = location.parent
    available = shutil.disk_usage(location).free
    required = int(write_bytes) + MIN_FREE_BYTES
    if available < required:
        raise StorageCapacityError(
            f"Insufficient disk space at {location}: {available / 1024**3:.2f} GiB free; "
            f"need {required / 1024**3:.2f} GiB including 1 GiB headroom. "
            "Archive old model artifacts or select an output volume with more space before retrying.")


def storage_exhausted(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        # Ray/SQLite can preserve only the message across process boundaries.
        if "no space left on device" in str(exc).lower() or "database or disk is full" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@contextmanager
def atomic_binary(path, *, exclusive=False):
    """Publish only a fully written/fsynced file; never overwrite an exclusive pin."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)  # Atomic create-if-absent on the same filesystem.
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_model_save(model, path):
    # Conservative estimate counts tied parameters twice and allows serialization overhead.
    size = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    require_space(path, size + max(size // 20, 1024 ** 2))
