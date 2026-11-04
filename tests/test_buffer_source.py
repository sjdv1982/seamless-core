import asyncio
import threading

from seamless import Checksum
from seamless.caching.buffer_source import (
    BufferSource,
    get_buffer_source,
    get_buffer_source_async,
    get_buffer_source_lengths,
    has_buffer_source,
    replace_buffer_sources,
    unregister_buffer_sources,
)
from seamless.checksum.calculate_checksum import calculate_checksum


def _checksum(content):
    return Checksum(calculate_checksum(content))


def test_local_source_lookup_validates_and_tries_next_provider():
    content = b"buffer source validation"
    checksum = _checksum(content)
    first_owner, second_owner = object(), object()
    replace_buffer_sources(first_owner, BufferSource(get_buffer=lambda _: b"stale"))
    replace_buffer_sources(
        second_owner, BufferSource(get_buffer=lambda requested: content if requested == checksum else None)
    )
    try:
        buffer = get_buffer_source(checksum)
        assert buffer is not None
        assert buffer.content == content
        assert buffer.get_checksum() == checksum
    finally:
        unregister_buffer_sources(first_owner)
        unregister_buffer_sources(second_owner)


def test_presence_and_lengths_use_metadata_callbacks_only():
    content = b"metadata-only"
    checksum = _checksum(content)
    owner = object()
    calls = {"read": 0, "presence": 0, "length": 0}

    def get_buffer(_):
        calls["read"] += 1
        raise AssertionError("metadata lookup must not read contents")

    def has_buffer(requested):
        calls["presence"] += 1
        return requested == checksum

    def get_length(requested):
        calls["length"] += 1
        return len(content) if requested == checksum else None

    replace_buffer_sources(
        owner,
        BufferSource(
            get_buffer=get_buffer,
            has_buffer=has_buffer,
            get_buffer_length=get_length,
        ),
    )
    try:
        assert has_buffer_source(checksum)
        assert get_buffer_source_lengths([checksum, _checksum(b"missing")]) == [
            len(content),
            None,
        ]
        assert calls["read"] == 0
        assert calls["presence"] == 1
        assert calls["length"] == 2
    finally:
        unregister_buffer_sources(owner)


def test_async_lookup_is_shared_between_waiters():
    content = b"shared async source lookup"
    checksum = _checksum(content)
    owner = object()
    calls = 0

    async def get_buffer(requested):
        nonlocal calls
        calls += 1
        assert requested == checksum
        await asyncio.sleep(0.02)
        return content

    replace_buffer_sources(owner, BufferSource(get_buffer_async=get_buffer))
    try:
        async def run():
            return await asyncio.gather(
                get_buffer_source_async(checksum),
                get_buffer_source_async(checksum),
            )

        first, second = asyncio.run(run())
        assert first is second
        assert first.content == content
        assert calls == 1
    finally:
        unregister_buffer_sources(owner)


def test_sync_source_is_read_off_the_event_loop_thread():
    content = b"sync source is offloaded"
    checksum = _checksum(content)
    owner = object()
    callback_thread = None

    def get_buffer(requested):
        nonlocal callback_thread
        assert requested == checksum
        callback_thread = threading.get_ident()
        return content

    replace_buffer_sources(owner, BufferSource(get_buffer=get_buffer))
    try:
        async def run():
            loop_thread = threading.get_ident()
            buffer = await get_buffer_source_async(checksum)
            return buffer, loop_thread

        buffer, loop_thread = asyncio.run(run())
        assert buffer.content == content
        assert callback_thread != loop_thread
    finally:
        unregister_buffer_sources(owner)


def test_sync_checksum_resolve_supports_async_only_sources():
    from seamless.caching.buffer_source import BufferSource, replace_buffer_sources

    content = b"async-only source resolves synchronously"
    checksum = _checksum(content)
    owner = object()

    async def get_buffer(requested):
        assert requested == checksum
        await asyncio.sleep(0)
        return content

    replace_buffer_sources(owner, BufferSource(get_buffer_async=get_buffer))
    try:
        assert Checksum(checksum).resolve().content == content
    finally:
        unregister_buffer_sources(owner)


def test_canceling_one_waiter_does_not_cancel_shared_lookup():
    content = b"one waiter can leave"
    checksum = _checksum(content)
    owner = object()
    started = None
    release = None
    calls = 0
    canceled = False

    async def get_buffer(requested):
        nonlocal calls, canceled
        calls += 1
        assert requested == checksum
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            canceled = True
            raise
        return content

    replace_buffer_sources(owner, BufferSource(get_buffer_async=get_buffer))
    try:
        async def run():
            nonlocal started, release
            started = asyncio.Event()
            release = asyncio.Event()
            first = asyncio.create_task(get_buffer_source_async(checksum))
            second = asyncio.create_task(get_buffer_source_async(checksum))
            await started.wait()
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            assert not canceled
            release.set()
            return await second

        buffer = asyncio.run(run())
        assert buffer.content == content
        assert calls == 1
        assert not canceled
    finally:
        unregister_buffer_sources(owner)


def test_canceling_last_waiter_cancels_source_lookup():
    checksum = _checksum(b"last waiter cancellation")
    owner = object()
    started = None
    canceled = False

    async def get_buffer(_):
        nonlocal canceled
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled = True
            raise

    replace_buffer_sources(owner, BufferSource(get_buffer_async=get_buffer))
    try:
        async def run():
            nonlocal started
            started = asyncio.Event()
            waiter = asyncio.create_task(get_buffer_source_async(checksum))
            await started.wait()
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            await asyncio.sleep(0)

        asyncio.run(run())
        assert canceled
    finally:
        unregister_buffer_sources(owner)
