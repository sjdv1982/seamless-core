"""Centralized shutdown routine for Seamless."""

from __future__ import annotations

import atexit
import asyncio
import gc
import logging
import os
import signal
import sys
import multiprocessing as mp
import time
from typing import Any, Dict, List

from . import is_worker

_closing = False
_closed = False
_DEBUG = bool(os.environ.get("SEAMLESS_DEBUG_SHUTDOWN"))


def _module(name: str):
    """Return a module from sys.modules without importing it."""

    return sys.modules.get(name)


def _run_coro(coro: Any, loop: asyncio.AbstractEventLoop | None, timeout: float | None):
    """Run an async callable on the provided loop."""

    if coro is None:
        return None
    if loop and loop.is_running():
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)
    return (
        asyncio.run(coro)
        if timeout is None
        else asyncio.run(asyncio.wait_for(coro, timeout))
    )


def _compute_flush_timeout(
    entries: Dict[Any, Any], *, multiplier: float = 1.0
) -> float:
    total_bytes = 0
    for entry in entries.values():
        buf = getattr(entry, "buffer", None)
        if buf is None:
            continue
        try:
            total_bytes += len(buf.content)
        except Exception:
            pass
    seconds = total_bytes / 1_000_000.0
    return multiplier * max(3.0, seconds if seconds > 0 else 3.0)


def _flush_buffers_short_then_long(failures: List[str]) -> List[str]:
    remaining: List[str] = []
    bw = _module("seamless.caching.buffer_writer")
    if bw is None:
        return remaining
    entries = getattr(bw, "_entries", {})
    if not entries:
        return remaining

    timeout1 = _compute_flush_timeout(entries, multiplier=1.0)
    try:
        bw.flush(timeout=timeout1)
    except Exception as exc:
        failures.append(f"buffer flush (short) raised: {exc}")

    entries = getattr(bw, "_entries", {})
    if entries:
        timeout2 = _compute_flush_timeout(entries, multiplier=2.0)
        try:
            bw.flush(timeout=timeout2)
        except Exception as exc:
            failures.append(f"buffer flush (long) raised: {exc}")
        entries = getattr(bw, "_entries", {})

    remaining = [str(cs) for cs in getattr(bw, "_entries", {}).keys()]
    return remaining


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[seamless.close DEBUG] {msg}", file=sys.stderr, flush=True)


def _mark_module_closed() -> None:
    """Mark the top-level seamless module as closed without importing it."""

    try:
        mod = sys.modules.get("seamless")
        if mod is not None:
            setattr(mod, "_closed", True)
    except Exception:
        pass


def _run_close_hooks() -> None:
    """Run any close hooks registered on the seamless module."""

    try:
        mod = _module("seamless")
        hooks = [] if mod is None else list(getattr(mod, "_close_hooks", []) or [])
    except Exception:
        hooks = []
    for hook in hooks:
        try:
            hook()
        except Exception:
            pass


def _close_remote_clients():
    client_mod = _module("seamless_remote.client")
    if client_mod is None:
        return
    close_clients = getattr(client_mod, "close_all_clients", None)
    if close_clients is not None:
        try:
            close_clients()
        except Exception:
            pass


def _sweep_worker_shared_memory(worker_manager: Any, failures: List[str]) -> None:
    if worker_manager is None:
        return
    pointers = list(getattr(worker_manager, "_pointers", {}).values())
    manager = getattr(worker_manager, "_manager", None)
    loop = getattr(worker_manager, "loop", None)
    memory_registry = getattr(manager, "memory_registry", None)
    handles = _iter_worker_handles(worker_manager)
    pids = [getattr(h, "pid", None) for h in handles if getattr(h, "pid", None)]

    if memory_registry is not None:
        for pid in pids:
            try:
                _run_coro(memory_registry.reset_pid(pid), loop, timeout=1.0)
            except Exception as exc:
                failures.append(f"reset_pid failed for PID {pid}: {exc}")
        try:
            _run_coro(memory_registry.close(), loop, timeout=1.0)
        except Exception as exc:
            failures.append(f"memory_registry.close failed: {exc}")

    for ptr in pointers:
        try:
            ptr.shm.close()
        except Exception:
            pass
        try:
            ptr.shm.unlink()
        except Exception:
            pass
    try:
        getattr(worker_manager, "_pointers", {}).clear()
    except Exception:
        pass


def _quiet_workers(worker_manager: Any, failures: List[str]) -> None:
    if worker_manager is None:
        return
    handles = _iter_worker_handles(worker_manager)
    loop = getattr(worker_manager, "loop", None)
    debug = bool(os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION"))
    for handle in handles:
        try:
            if debug:
                print(
                    f"[seamless.close] sending quiet to {getattr(handle, 'name', '?')}",
                    file=sys.stderr,
                )
            _run_coro(handle.request("quiet", None, timeout=0.5), loop, timeout=0.5)
            if debug:
                print(
                    f"[seamless.close] quiet ack from {getattr(handle, 'name', '?')}",
                    file=sys.stderr,
                )
        except Exception as exc:
            failures.append(
                f"quiet request failed for {getattr(handle, 'name', '?')}: {exc}"
            )


def _wait_workers_ready(
    worker_manager: Any, failures: List[str], timeout: float = 0.2
) -> None:
    """Give workers a brief chance to finish their ready handshake to avoid broken pipes."""

    if worker_manager is None:
        return
    loop = getattr(worker_manager, "loop", None)
    handles = _iter_worker_handles(worker_manager)
    for handle in handles:
        try:
            if handle.ready_event.is_set():
                continue
        except Exception:
            continue
        try:
            assert loop is not None
            fut = asyncio.run_coroutine_threadsafe(handle.wait_until_ready(), loop)
            fut.result(timeout=timeout)
        except Exception as exc:
            failures.append(
                f"wait_until_ready timed out for {getattr(handle, 'name', '?')}: {exc}"
            )


def close(*, from_atexit: bool = False) -> None:
    """Close Seamless subsystems to avoid leaks and hangs."""

    global _closing, _closed
    if _closed or _closing:
        return
    if is_worker():
        return
    try:
        proc = mp.current_process()
        if proc is not None and proc.name != "MainProcess":
            return
    except Exception:
        pass

    in_child_process = False
    try:
        in_child_process = mp.parent_process() is not None
    except Exception:
        pass

    if from_atexit and not getattr(
        sys.modules.get("seamless"), "_require_close", False
    ):
        # Nothing in Seamless was used; skip shutdown noise.
        _closed = True
        _closing = False
        return

    _closing = True
    _closed = True
    _mark_module_closed()
    _debug(f"close(from_atexit={from_atexit}) start")
    _run_close_hooks()
    failures: List[str] = []
    pending_buffers: List[str] = []
    refholders: List[Any] = []
    warned_manual: set[Any] = set()
    worker_shutdown = False
    try:
        for logger_name in (
            "seamless_transformer.process.manager",
            "seamless_transformer.process.channel",
        ):
            try:
                logging.getLogger(logger_name).setLevel(logging.ERROR)
            except Exception:
                pass
        if from_atexit:
            if not in_child_process:
                print(
                    "[seamless.close] called via atexit; call seamless.close() earlier to ensure all buffers/workers shut down cleanly (atexit is best-effort/ noisier)",
                    file=sys.stderr,
                )
        worker_mod = _module("seamless_transformer.worker")
        worker_manager = (
            getattr(worker_mod, "_worker_manager", None) if worker_mod else None
        )
        has_workers = bool(
            worker_mod and getattr(worker_mod, "has_spawned", lambda: False)()
        )
        _debug(
            f"worker_mod={'present' if worker_mod else 'absent'}, worker_manager={'present' if worker_manager else 'absent'}, has_workers={has_workers}"
        )

        if worker_manager is not None:
            try:
                setattr(worker_manager, "_closing", True)
            except Exception:
                pass
            manager_impl = getattr(worker_manager, "_manager", None)
            if manager_impl is not None:
                try:
                    setattr(manager_impl, "_closing", True)
                except Exception:
                    pass
            for handle in _iter_worker_handles(worker_manager):
                try:
                    handle.restart = False
                except Exception:
                    pass
                try:
                    handle.closing = True
                except Exception:
                    pass
            _debug("worker manager marked closing and restarts disabled")

        if worker_manager is not None:
            _wait_workers_ready(worker_manager, failures)
            _quiet_workers(worker_manager, failures)
            _debug("workers quieted")

        # Phase 1: cancel/call manager cleanup
        if worker_mod and has_workers:
            try:
                _debug("calling worker_mod.shutdown_workers()")
                worker_mod.shutdown_workers()
                worker_shutdown = True
                _debug("worker_mod.shutdown_workers() returned")
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "Seamless has been closed" in str(
                    exc
                ):
                    # If close() already marked the session closed, suppress noisy retry error.
                    pass
                else:
                    failures.append(f"shutdown_workers raised: {exc}")

        # Lifecycle audit is deliberately after activity has settled and before
        # buffer flushing and holder cleanup, so it sees the final semantic state.
        try:
            from .reference_lifecycle import audit_reference_accounting, registered_refholders

            refholders = registered_refholders()
            audit_reference_accounting(holders=refholders, warned_manual=warned_manual)
        except Exception as exc:
            logging.getLogger("seamless.references").warning(
                "Reference lifecycle audit raised: %s", exc
            )

        # Phase 2: buffer flush attempts
        _debug("flushing pending buffers (short/long)")
        pending_buffers = _flush_buffers_short_then_long(failures)

        # Release explicit lifecycle ownership while buffers are still available
        # to the existing writer flush, then allow finalizers to settle.
        for holder in refholders:
            try:
                holder._release_refholds()
            except Exception as exc:
                logging.getLogger("seamless.references").warning(
                    "Refholder %s 0x%x cleanup raised: %s",
                    type(holder).__name__,
                    id(holder),
                    exc,
                )
        refholders = []
        gc.collect()
        gc.collect()
        try:
            from .reference_lifecycle import audit_reference_accounting
            from .caching.buffer_cache import get_buffer_cache

            audit_reference_accounting(
                cache=get_buffer_cache(),
                holders=[],
                warned_manual=warned_manual,
            )
            get_buffer_cache().force_clear_reference_accounting()
        except Exception as exc:
            logging.getLogger("seamless.references").warning(
                "Reference lifecycle post-cleanup raised: %s", exc
            )

        # Close remote client sessions/keepalive
        _debug("closing remote clients")
        _close_remote_clients()

        # Sweep shared memory even if workers were force-killed
        if worker_manager is not None:
            _debug("sweeping worker shared memory")
            _sweep_worker_shared_memory(worker_manager, failures)
        # Fire any pending multiprocessing SemLock finalizers (e.g. tqdm's RLock)
        # while the resource tracker is still alive so they unregister cleanly.
        # Without this, _stop_resource_tracker closes the tracker pipe first, the
        # tracker treats live SemLocks as "leaked" and unlinks them itself, and
        # the parent's atexit-driven SemLock._cleanup then raises FileNotFoundError.
        try:
            from multiprocessing import util as _mp_util

            _mp_util._run_finalizers(0)
        except Exception:
            pass
        _stop_resource_tracker(failures)

    finally:
        summary_parts: List[str] = []
        debug = bool(os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION"))
        write_clients = False
        try:
            br_mod = _module("seamless_remote.buffer_remote")
            if br_mod is not None:
                clients = getattr(br_mod, "_write_server_clients", []) or []
                write_clients = any(not getattr(c, "readonly", True) for c in clients)
        except Exception:
            pass
        if pending_buffers and write_clients:
            summary_parts.append(f"unwritten buffers: {', '.join(pending_buffers)}")
        if failures:
            summary_parts.extend(failures)
        if worker_shutdown and (failures or pending_buffers or debug):
            summary_parts.append("workers shut down")
        if summary_parts:
            print(
                "[seamless.close] " + " ; ".join(summary_parts),
                file=sys.stderr,
            )
        _debug("close() finished")
        _closing = False


def _atexit_close():
    try:
        close(from_atexit=True)
    except Exception:
        pass


atexit.register(_atexit_close)


def _stop_resource_tracker(failures: List[str]) -> None:
    try:
        from multiprocessing import resource_tracker
    except Exception:
        return
    tracker = getattr(resource_tracker, "_resource_tracker", None)
    if tracker is None:
        return
    pid = getattr(tracker, "_pid", None)
    fd = getattr(tracker, "_fd", None)
    if fd is None and pid is None:
        return
    _debug(f"stopping resource tracker pid={pid} fd={fd}")
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            setattr(tracker, "_fd", None)
        except Exception:
            pass
    if pid is None:
        return

    def _wait_nonblock(timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                waited_pid, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                return True
            if waited_pid:
                return True
            time.sleep(0.05)
        return False

    if not _wait_nonblock(0):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if not _wait_nonblock(0.5):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not _wait_nonblock(0.5):
            failures.append(
                f"resource tracker pid {pid} did not exit even after SIGKILL"
            )
            return
    try:
        setattr(tracker, "_pid", None)
    except Exception:
        pass
    try:
        resource_tracker._resource_tracker = None
        resource_tracker.register = lambda *_, **__: None
        resource_tracker.unregister = lambda *_, **__: None
        resource_tracker.ensure_running = lambda *_a, **_k: None
        resource_tracker.getfd = lambda *_a, **_k: None
    except Exception:
        pass


def _iter_worker_handles(worker_manager: Any) -> List[Any]:
    handles = getattr(worker_manager, "_handles", None)
    if not handles:
        return []
    if isinstance(handles, dict):
        return list(handles.values())
    return list(handles)
