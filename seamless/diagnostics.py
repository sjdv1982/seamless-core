"""Opt-in materialisation tracing. Recording never resolves or hashes content."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock, get_ident

_lock = RLock()
_recorders = []

@dataclass
class MaterialisationLog:
    ops: list = field(default_factory=list)
    threads: list = field(default_factory=list)


def record(op, checksum=None, celltype=None, nbytes=None):
    if not _recorders:
        return
    entry = (op, checksum.hex() if checksum is not None else None, celltype, nbytes)
    with _lock:
        for log in _recorders:
            log.ops.append(entry)
            log.threads.append(get_ident())


@contextmanager
def record_materialisation():
    """Record across threads, including controller and side-work threads."""
    log = MaterialisationLog()
    with _lock:
        _recorders.append(log)
    try:
        yield log
    finally:
        with _lock:
            _recorders.remove(log)
