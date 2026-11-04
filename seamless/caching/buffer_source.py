"""Local buffer sources outside the in-memory buffer cache.

Buffer sources describe data that can be read from another local store, such
as a mounted directory or a configured buffer folder. They do not own buffer
references; each lookup validates the returned bytes against the requested
checksum before creating a ``Buffer``.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from seamless.checksum_class import Checksum


BufferResult = Any
BufferGetter = Callable[[Checksum], BufferResult | None]
AsyncBufferGetter = Callable[[Checksum], Awaitable[BufferResult | None]]
BufferPresence = Callable[[Checksum], bool]
BufferLength = Callable[[Checksum], int | None]


@dataclass(frozen=True)
class BufferSource:
    """Callbacks for one local store.

    ``has_buffer`` and ``get_buffer_length`` should use cheap metadata only.
    Retrieval callbacks may return raw ``bytes`` or a ``Buffer``. By default
    the registry hashes the returned content. Set ``returns_verified_buffer``
    only when every retrieval callback verifies the bytes against its requested
    checksum and returns a ``Buffer`` carrying that verified checksum; the
    registry then reuses that validation instead of hashing a second time.
    """

    get_buffer: BufferGetter | None = None
    get_buffer_async: AsyncBufferGetter | None = None
    has_buffer: BufferPresence | None = None
    get_buffer_length: BufferLength | None = None
    returns_verified_buffer: bool = False

    def __post_init__(self):
        for callback in (
            self.get_buffer,
            self.get_buffer_async,
            self.has_buffer,
            self.get_buffer_length,
        ):
            if callback is not None and not callable(callback):
                raise TypeError("Buffer source callbacks must be callable")
        if not isinstance(self.returns_verified_buffer, bool):
            raise TypeError("returns_verified_buffer must be a bool")


class BufferSourceRegistry:
    """Owner-scoped registry for local stores of checksum-addressed buffers."""

    def __init__(self):
        self._lock = threading.RLock()
        self._sources_by_owner: dict[Any, tuple[BufferSource, ...]] = {}
        self._pending: dict[
            tuple[asyncio.AbstractEventLoop, Checksum, bool],
            tuple[asyncio.Task, int],
        ] = {}

    @staticmethod
    def _normalize_sources(
        sources: BufferSource | Iterable[BufferSource],
    ) -> tuple[BufferSource, ...]:
        if isinstance(sources, BufferSource):
            sources = (sources,)
        else:
            sources = tuple(sources)
        if any(not isinstance(source, BufferSource) for source in sources):
            raise TypeError("Buffer sources must be BufferSource instances")
        return sources

    def replace(self, owner: Any, sources: BufferSource | Iterable[BufferSource]):
        """Atomically replace all providers registered by ``owner``."""
        sources = self._normalize_sources(sources)
        with self._lock:
            self._sources_by_owner.pop(owner, None)
            if sources:
                self._sources_by_owner[owner] = sources

    def unregister(self, owner: Any):
        with self._lock:
            self._sources_by_owner.pop(owner, None)

    def has_owner(self, owner: Any) -> bool:
        with self._lock:
            return owner in self._sources_by_owner

    def has_async_only_sources(self) -> bool:
        return any(
            source.get_buffer is None and source.get_buffer_async is not None
            for source in self._snapshot()
        )

    def _snapshot(self) -> tuple[BufferSource, ...]:
        with self._lock:
            return tuple(
                source
                for sources in self._sources_by_owner.values()
                for source in sources
            )

    @staticmethod
    def _to_buffer(
        result: BufferResult | None,
        checksum: Checksum,
        *,
        returns_verified_buffer=False,
    ):
        if result is None:
            return None

        from seamless import Buffer
        from seamless.checksum.calculate_checksum import calculate_checksum

        if isinstance(result, Buffer):
            if returns_verified_buffer and result._checksum == checksum:
                return result
            content = result.content
        elif isinstance(result, bytes):
            content = result
        else:
            return None

        if calculate_checksum(content) != checksum.hex():
            return None
        return Buffer(content, checksum=checksum)

    def get_buffer(self, checksum: Checksum):
        """Try synchronous local sources in registration order."""
        checksum = Checksum(checksum)
        for source in self._snapshot():
            if source.get_buffer is None:
                continue
            try:
                result = source.get_buffer(checksum)
            except Exception:
                continue
            buffer = self._to_buffer(
                result,
                checksum,
                returns_verified_buffer=source.returns_verified_buffer,
            )
            if buffer is not None:
                return buffer
        return None

    async def _get_buffer_uncached(self, checksum: Checksum, *, async_only=False):
        for source in self._snapshot():
            if async_only and source.get_buffer is not None:
                continue
            if source.get_buffer_async is not None:
                try:
                    result = await source.get_buffer_async(checksum)
                except Exception:
                    continue
                try:
                    buffer = await asyncio.to_thread(
                        self._to_buffer,
                        result,
                        checksum,
                        returns_verified_buffer=source.returns_verified_buffer,
                    )
                except Exception:
                    continue
            elif source.get_buffer is not None and not async_only:
                try:
                    buffer = await asyncio.to_thread(
                        self._get_from_sync_source, source, checksum
                    )
                except Exception:
                    continue
            else:
                continue
            if buffer is not None:
                return buffer
        return None

    @classmethod
    def _get_from_sync_source(cls, source: BufferSource, checksum: Checksum):
        result = source.get_buffer(checksum)
        return cls._to_buffer(
            result,
            checksum,
            returns_verified_buffer=source.returns_verified_buffer,
        )

    async def get_buffer_async(self, checksum: Checksum, *, async_only=False):
        """Try local sources without blocking the event loop.

        Concurrent requests for the same checksum on the same loop share one
        lookup. A canceled waiter does not cancel work still used by others.
        With no registered sources, this returns ``None`` at once: no shared
        lookup is created and no event-loop turn is taken.
        """
        checksum = Checksum(checksum)
        with self._lock:
            if not self._sources_by_owner:
                return None
        loop = asyncio.get_running_loop()
        key = (loop, checksum, async_only)
        pending = self._pending.get(key)
        if pending is None:
            task = loop.create_task(
                self._get_buffer_uncached(checksum, async_only=async_only)
            )

            def finished(fetch):
                current = self._pending.get(key)
                if current is not None and current[0] is fetch:
                    self._pending.pop(key, None)
                if not fetch.cancelled():
                    fetch.exception()

            task.add_done_callback(finished)
            waiters = 0
        else:
            task, waiters = pending

        self._pending[key] = (task, waiters + 1)
        try:
            return await asyncio.shield(task)
        finally:
            current = self._pending.get(key)
            if current is not None and current[0] is task:
                remaining = current[1] - 1
                if remaining == 0:
                    self._pending.pop(key, None)
                    if not task.done():
                        task.cancel()
                else:
                    self._pending[key] = (task, remaining)

    def has_buffer(self, checksum: Checksum) -> bool:
        """Return metadata-only availability; never fetch buffer contents."""
        checksum = Checksum(checksum)
        for source in self._snapshot():
            if source.has_buffer is not None:
                try:
                    if source.has_buffer(checksum):
                        return True
                except Exception:
                    continue
            if source.get_buffer_length is not None:
                try:
                    length = source.get_buffer_length(checksum)
                except Exception:
                    continue
                if isinstance(length, int) and not isinstance(length, bool) and length >= 0:
                    return True
        return False

    def get_buffer_lengths(self, checksums: Iterable[Checksum]):
        """Return cheap local lengths, consulting sources in registration order."""
        checksums = [Checksum(checksum) for checksum in checksums]
        results: list[int | None] = [None] * len(checksums)
        pending = list(range(len(checksums)))
        for source in self._snapshot():
            if source.get_buffer_length is None or not pending:
                continue
            remaining = []
            for index in pending:
                try:
                    length = source.get_buffer_length(checksums[index])
                except Exception:
                    remaining.append(index)
                    continue
                if isinstance(length, int) and not isinstance(length, bool) and length >= 0:
                    results[index] = length
                else:
                    remaining.append(index)
            pending = remaining
        return results


_buffer_source_registry = BufferSourceRegistry()


def get_buffer_source_registry() -> BufferSourceRegistry:
    return _buffer_source_registry


def replace_buffer_sources(
    owner: Any, sources: BufferSource | Iterable[BufferSource]
) -> None:
    _buffer_source_registry.replace(owner, sources)


def unregister_buffer_sources(owner: Any) -> None:
    _buffer_source_registry.unregister(owner)


def get_buffer_source(checksum: Checksum):
    return _buffer_source_registry.get_buffer(checksum)


async def get_buffer_source_async(checksum: Checksum):
    return await _buffer_source_registry.get_buffer_async(checksum)


async def get_async_only_buffer_source(checksum: Checksum):
    """Resolve sources with no synchronous getter, for sync API parity."""
    return await _buffer_source_registry.get_buffer_async(checksum, async_only=True)


def has_async_only_buffer_sources() -> bool:
    return _buffer_source_registry.has_async_only_sources()


def has_buffer_source(checksum: Checksum) -> bool:
    return _buffer_source_registry.has_buffer(checksum)


def get_buffer_source_lengths(checksums: Iterable[Checksum]):
    return _buffer_source_registry.get_buffer_lengths(checksums)
