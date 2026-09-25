"""Local Expression cancellation follows the materialization waiting-set contract."""

import asyncio

import pytest

from seamless import Buffer, Checksum, Expression
from seamless.checksum import expression as expression_mod


def _softcancel(expression):
    return expression.softcancel()


def test_local_in_flight_softcancel_deregisters_without_interrupting_fetch(monkeypatch):
    # Isolate caches without discarding state owned by other tests.
    monkeypatch.setattr(expression_mod, "_expression_cache", {})
    monkeypatch.setattr(expression_mod, "_expression_result_buffers", {})
    monkeypatch.setattr(expression_mod, "_active_expressions", {})
    source = Buffer({"a": "local cancellation witness"}, "plain")
    source_checksum = source.get_checksum()
    expression = Expression(
        source_checksum, "a", input_celltype="plain", celltype="str"
    )
    original_resolution = Checksum.resolution

    async def main():
        started = asyncio.Event()
        release = asyncio.Event()
        interrupted = asyncio.Event()

        async def delayed_resolution(checksum, *args, **kwargs):
            if checksum == source_checksum:
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
            return await original_resolution(checksum, *args, **kwargs)

        monkeypatch.setattr(Checksum, "resolution", delayed_resolution)
        task = asyncio.create_task(expression.compute_async(execution="local"))
        try:
            # Wait for actual materialization, not an arbitrary scheduling delay.
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not task.done()
            assert expression_mod._active_expressions

            assert _softcancel(expression) is True
            await asyncio.sleep(0)
            assert not interrupted.is_set()
            with pytest.raises(asyncio.CancelledError):
                await task

            release.set()
            assert not interrupted.is_set()
            assert not expression_mod._active_expressions
            assert _softcancel(expression) is False
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
