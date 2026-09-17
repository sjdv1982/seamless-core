import os
import signal
import sys
from typing import Callable, List


class CacheMissError(Exception):
    """Exception for when a checksum cannot be mapped to a buffer"""

    @property
    def checksum(self):
        return self.args[0] if self.args else None


_IS_WORKER = False
_closed = False
_require_close = False
_DEBUG_SHUTDOWN = bool(os.environ.get("SEAMLESS_DEBUG_SHUTDOWN"))
_close_hooks: List[Callable[[], None]] = []

if _DEBUG_SHUTDOWN:
    try:
        import faulthandler

        faulthandler.enable()
        if hasattr(signal, "SIGUSR2"):
            faulthandler.register(
                signal.SIGUSR2,
                file=sys.stderr,
                all_threads=True,
                chain=False,
            )
    except Exception:
        pass


def set_is_worker(value: bool = True) -> None:
    """Mark the current process as a Seamless worker process."""

    global _IS_WORKER
    _IS_WORKER = bool(value)


def is_worker() -> bool:
    """Return True when running inside a Seamless worker process."""

    return _IS_WORKER


def ensure_open(op: str | None = None, *, mark_required: bool = True) -> None:
    """Raise RuntimeError if Seamless was closed; optionally mark that close is required."""

    global _require_close
    if mark_required:
        _require_close = True
    if _closed:
        action = f" for {op}" if op else ""
        raise RuntimeError(
            f"Seamless has been closed; cannot perform further operations{action}."
        )


def register_close_hook(hook: Callable[[], None]) -> None:
    """Register a callable to be run when seamless.close() executes."""

    if not callable(hook):
        raise TypeError("hook must be callable")
    _close_hooks.append(hook)


from .checksum_class import Checksum as _Checksum
from .buffer_class import Buffer as _Buffer
from .expression_class import Expression as _Expression
from .cell_class import Cell as _Cell, SubCell as _SubCell, CellBase
from .cell_errors import BoundStateError, ProjectionError, AuthorityError
from .shutdown import close

# Expose classes under the top-level module so their repr shows seamless.<Class>
Checksum = _Checksum
Checksum.__module__ = __name__
Buffer = _Buffer
Buffer.__module__ = __name__
Expression = _Expression
Expression.__module__ = __name__
Cell = _Cell
Cell.__module__ = __name__
SubCell = _SubCell
SubCell.__module__ = __name__

__all__ = [
    "Checksum",
    "Buffer",
    "Expression",
    "Cell",
    "CellBase",
    "AuthorityError",
    "SubCell",
    "BoundStateError",
    "ProjectionError",
    "CacheMissError",
    "set_is_worker",
    "is_worker",
    "ensure_open",
    "close",
    "register_close_hook",
]

from . import diagnostics
