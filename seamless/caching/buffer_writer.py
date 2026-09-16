"""Background buffer writer
Whenever a buffer (not just a checksum) is entered into the strong buffer_cache.py cache,
buffer_writer.register(buffer) will be invoked. This adds the buffer to a queue,
unless the buffer is already in the queue (note that the checksum of the buffer is always known).

In the background, a dedicated thread hosts an asyncio event loop that runs the writer coroutine.
All queued buffers are written remotely/asynchronously in the same way that Buffer.write does.
The asynchronous tasks are stored in the queue.

Whenever Buffer.write is invoked, it checks first with the buffer writer if the asynchronous task
for that buffer is already in the queue. If so, it awaits that task and returns.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
import traceback
from http.client import HTTPConnection
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from seamless import ensure_open

if TYPE_CHECKING:
    from seamless.buffer_class import Buffer
    from seamless.checksum_class import Checksum


@dataclass(slots=True)
class _QueueEntry:
    checksum: "Checksum"
    buffer: "Buffer | None"
    future: concurrent.futures.Future
    queued: bool = False
    hash_type: int | None = None
    metadata_writer: object = None


_entries: Dict["Checksum | tuple[Checksum, int]", _QueueEntry] = {}
_has_checked: Dict[str, set[str]] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None
_queue: Optional[asyncio.Queue[_QueueEntry]] = None
_thread: Optional[threading.Thread] = None
_lock = threading.RLock()
_idle_event: Optional[threading.Event] = None


def init() -> None:
    """Ensure that the background writer thread is running."""
    ensure_open("buffer writer init", mark_required=False)
    _ensure_worker()


def register(buffer: "Buffer") -> None:
    """Register a buffer for background writing."""
    mark_required = False
    try:
        import seamless_remote.buffer_remote as buffer_remote

        mark_required = buffer_remote.has_write_server()
    except Exception:
        pass
    ensure_open("buffer writer register", mark_required=mark_required)
    checksum = buffer.checksum
    with _lock:
        entry = _entries.get(checksum)
        if entry is not None:
            entry.buffer = buffer
        else:
            future: concurrent.futures.Future = concurrent.futures.Future()
            entry = _QueueEntry(checksum=checksum, buffer=buffer, future=future)
            _entries[checksum] = entry
    _enqueue_entry(entry)


def register_hash_type(checksum, word):
    """Queue a metadata write on the buffer writer's existing worker."""
    ensure_open("HashType writer register")
    try:
        from seamless_remote import database_remote
    except ImportError:
        return
    key = (checksum, word)
    with _lock:
        if key in _entries:
            return
        entry = _QueueEntry(
            checksum,
            None,
            concurrent.futures.Future(),
            hash_type=word,
            metadata_writer=database_remote.set_hash_type,
        )
        _entries[key] = entry
    _enqueue_entry(entry)


async def await_existing_task(checksum: "Checksum") -> Optional[bool]:
    """Await the shared write task for checksum if it exists."""
    ensure_open("buffer writer await", mark_required=False)
    entry = _entries.get(checksum)
    if entry is None:
        return None
    future = entry.future
    if future.cancelled():
        return None
    _enqueue_entry(entry)
    if future.done():
        return future.result()
    wrapped = asyncio.wrap_future(future)
    return await wrapped


def purge() -> None:
    """Stop the worker thread and clear all queued buffers."""
    _stop_worker()
    with _lock:
        entries = list(_entries.values())
        _entries.clear()
    for entry in entries:
        if not entry.future.done():
            entry.future.cancel()


def flush(timeout: Optional[float] = None) -> None:
    """Drain queued HashTypes, then flush buffers using direct HTTP calls.

    Metadata uses the existing worker; buffer writes retain the shutdown-safe
    HTTP path, which does not create asynchronous futures at interpreter exit.
    """

    try:
        import seamless_remote.buffer_remote as buffer_remote
    except ImportError:
        return

    with _lock:
        metadata = [entry for entry in _entries.values() if entry.hash_type is not None]
    for entry in metadata:
        entry.future.result(timeout=timeout)

    buffers = {
        checksum: entry.buffer
        for checksum, entry in list(_entries.items())
        if entry.hash_type is None
    }

    clients = []
    clients0 = getattr(buffer_remote, "_write_server_clients", [])
    for c in clients0:
        try:
            init_sync = getattr(c, "ensure_initialized_sync", None)
            if init_sync is not None:
                init_sync()
        except Exception:
            traceback.print_exc()
            continue
        if not getattr(c, "url", None):
            continue
        try:
            setattr(c, "_shutdown", True)
        except Exception:
            pass
        clients.append(c)

    if not clients0:
        return

    start = time.time()

    def remaining() -> Optional[float]:
        if timeout is None:
            return None
        return max(0.0, timeout - (time.time() - start))

    health_ok: Dict[str, bool] = {}
    # Pre-check /has-now in bulk per client to avoid redundant PUTs.
    present_per_client: Dict[str, set[str]] = {}
    for client in clients:
        url = client.url.rstrip("/")
        checked = _has_checked.setdefault(url, set())
        to_check: list[str] = [cs.hex() for cs in buffers]
        if to_check:
            present = _has_sync(url, to_check, remaining(), now=True)
            present_per_client[url] = present
            checked.update(to_check)

    for checksum, buffer in buffers.items():
        success = False
        cs_hex = checksum.hex()
        for client in clients:
            url = client.url.rstrip("/")
            if cs_hex in present_per_client.get(url, set()):
                success = True
                break
            ok = health_ok.get(url)
            if ok is None:
                ok = _healthcheck_sync(url, remaining())
                health_ok[url] = ok
            if not ok:
                continue
            if _put_sync(url, checksum, buffer, remaining()):
                success = True

        if success:
            entry = _entries.get(checksum)
            if entry is not None:
                fut = entry.future
                if not fut.done():
                    fut.set_result(success)
                with _lock:
                    _entries.pop(checksum, None)

    for client in clients:
        try:
            close_sessions = getattr(client, "_close_sessions", None)
            if close_sessions is not None:
                close_sessions()
        except Exception:
            pass

    _stop_worker()


# --- worker thread management -------------------------------------------------
def _ensure_worker() -> None:
    global _thread
    with _lock:
        thread = _thread
        if thread is not None and thread.is_alive():
            return
        worker = threading.Thread(
            target=_thread_main,
            name="SeamlessBufferWriter",
            daemon=True,
        )
        _thread = worker
    worker.start()


def _stop_worker() -> None:
    thread = None
    with _lock:
        loop = _loop
        queue = _queue
        thread = _thread
    if loop is None or queue is None or thread is None:
        return

    def _request_stop() -> None:
        queue.put_nowait(None)

    loop.call_soon_threadsafe(_request_stop)
    thread.join()
    with _lock:
        for entry in _entries.values():
            entry.queued = False
        if _idle_event is not None:
            _idle_event.set()


def _close_worker_thread_sessions(thread: threading.Thread) -> None:
    """Ensure aiohttp sessions owned by the worker thread get closed."""
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except ImportError:
        return
    clients = getattr(buffer_remote, "_write_server_clients", [])
    for client in clients:
        close_for_thread = getattr(client, "_close_sessions_for_thread", None)
        if close_for_thread is None:
            continue
        try:
            close_for_thread(thread)
        except Exception:
            traceback.print_exc()


def _thread_main() -> None:
    global _loop, _queue, _thread
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue[_QueueEntry] = asyncio.Queue()
    try:
        asyncio.set_event_loop(loop)
        with _lock:
            _loop = loop
            _queue = queue
            pending = [
                entry
                for entry in _entries.values()
                if not entry.future.done()
                and not entry.future.cancelled()
                and not entry.queued
            ]
            for entry in pending:
                entry.queued = True
        for entry in pending:
            queue.put_nowait(entry)
        loop.run_until_complete(_worker_loop(queue))
    finally:
        try:
            _close_worker_thread_sessions(threading.current_thread())
        except Exception:
            traceback.print_exc()
        asyncio.set_event_loop(None)
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        loop.close()
        with _lock:
            _loop = None
            _queue = None
            _thread = None
            for entry in _entries.values():
                entry.queued = False


async def _worker_loop(queue: asyncio.Queue[_QueueEntry]) -> None:
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except ImportError:
        buffer_remote = None

    loop = asyncio.get_running_loop()
    pending: set[asyncio.Task] = set()

    def _on_done(task: asyncio.Task) -> None:
        pending.discard(task)

    while True:
        entry = await queue.get()
        if entry is None:
            break
        if entry.future.cancelled() or entry.future.done():
            continue
        if buffer_remote is not None and entry.hash_type is None:
            try:
                await buffer_remote.promise(entry.checksum)
            except Exception:
                pass
        task = loop.create_task(_process_entry(entry))
        pending.add(task)
        task.add_done_callback(_on_done)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _process_entry(entry: _QueueEntry) -> None:
    try:
        if entry.hash_type is not None:
            result = await entry.metadata_writer(entry.checksum, entry.hash_type)
        else:
            try:
                import seamless_remote.buffer_remote as buffer_remote
            except ImportError:
                result = False
            else:
                result = await buffer_remote.write_buffer(entry.checksum, entry.buffer)
        error = None
    except Exception as exc:
        result, error = None, exc

    future = entry.future
    if not future.done():
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)
    if error is None:
        with _lock:
            key = (
                entry.checksum
                if entry.hash_type is None
                else (entry.checksum, entry.hash_type)
            )
            _entries.pop(key, None)


# --- queue submission helpers -------------------------------------------------
def _enqueue_entry(entry: _QueueEntry) -> None:
    _ensure_worker()
    with _lock:
        loop = _loop
        queue = _queue
        already_queued = entry.queued
    if loop is None or queue is None or already_queued or entry.future.done():
        return

    def _put(e: _QueueEntry) -> None:
        if e.future.cancelled() or e.future.done() or e.queued:
            return
        e.queued = True
        queue.put_nowait(e)

    loop.call_soon_threadsafe(_put, entry)


def _healthcheck_sync(url: str, timeout: Optional[float]) -> bool:
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (80 if parts.scheme == "http" else 443)
    try:
        conn = HTTPConnection(host, port, timeout=timeout or 1.0)
        conn.request("GET", "/healthcheck")
        resp = conn.getresponse()
        ok = 200 <= resp.status < 300
        conn.close()
        return ok
    except Exception:
        return False


def _has_sync(
    url: str, checksums: list[str], timeout: Optional[float], *, now: bool
) -> set[str]:
    """Return the subset of checksums that the server reports as present/promised."""

    if not checksums:
        return set()
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (80 if parts.scheme == "http" else 443)
    try:
        conn = HTTPConnection(host, port, timeout=timeout or 1.0)
        body = json.dumps(checksums)
        endpoint = "/has-now" if now else "/has"
        conn.request(
            "GET", endpoint, body=body, headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        if not (200 <= resp.status < 300):
            conn.close()
            return set()
        data = resp.read()
        conn.close()
        result = json.loads(data)
        if not isinstance(result, list) or len(result) != len(checksums):
            return set()
        present = {cs for cs, ok in zip(checksums, result) if bool(ok)}
        return present
    except Exception:
        return set()


def _put_sync(
    url: str, checksum: Checksum, buffer: Buffer, timeout: Optional[float]
) -> bool:
    checksum_hex = checksum.hex()
    # Skip if already present/promised
    present = _has_sync(url, [checksum_hex], timeout, now=True)
    if checksum_hex in present:
        return True

    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (80 if parts.scheme == "http" else 443)
    path = "/" + checksum_hex
    try:
        conn = HTTPConnection(host, port, timeout=timeout or 1.0)
        conn.request("PUT", path, body=buffer.content)
        resp = conn.getresponse()
        ok = 200 <= resp.status < 300
        conn.close()
        return ok
    except Exception:
        if timeout is not None:
            import traceback

            traceback.print_exc()
        return False


__all__ = ["register", "await_existing_task", "init", "purge", "flush"]


init()
