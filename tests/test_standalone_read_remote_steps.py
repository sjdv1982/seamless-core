import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from seamless import Buffer, Cell, Checksum
from seamless.checksum import expression as expression_mod
from seamless.checksum.expression import (
    _active_expressions,
    evaluate_expression_async,
    get_expression_cache,
)

from tests.helpers.fake_remotes import drop_buffer, install_fake_remotes


def _cell(checksum):
    return Cell("plain", checksum=checksum)["a"].as_celltype("str")


def test_database_hit_satisfies_getter(monkeypatch):
    get_expression_cache().clear()
    source_checksum = Checksum("1" * 64)
    result_checksum = Buffer("cached", "str").get_checksum()
    drop_buffer(source_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    install_fake_remotes(monkeypatch, {key: result_checksum}, {}, calls)

    assert _cell(source_checksum).checksum == result_checksum
    assert calls == ["database:get"]


def test_hashserver_only_input_dispatches_from_getter(monkeypatch):
    get_expression_cache().clear()
    source = Buffer({"a": "remote"}, "plain")
    source_checksum = source.get_checksum()
    result_checksum = Buffer("remote", "str").get_checksum()
    drop_buffer(source_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    install_fake_remotes(monkeypatch, {}, {key: result_checksum}, calls)

    assert _cell(source_checksum).checksum == result_checksum
    assert calls == ["database:get", "jobserver:run", "database:set"]


def test_concurrent_getters_share_one_dispatch(monkeypatch):
    get_expression_cache().clear()
    _active_expressions.clear()
    source = Buffer({"a": "remote"}, "plain")
    source_checksum = source.get_checksum()
    result_checksum = Buffer("remote", "str").get_checksum()
    drop_buffer(source_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    dispatch_gate = threading.Event()
    dispatch_started = threading.Event()
    start_gate = threading.Barrier(3)
    install_fake_remotes(
        monkeypatch,
        {},
        {key: result_checksum},
        calls,
        run_expression_gate=dispatch_gate,
        run_expression_started=dispatch_started,
    )
    cells = [_cell(source_checksum), _cell(source_checksum)]

    def read(cell):
        start_gate.wait()
        return cell.checksum

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(read, cell) for cell in cells]
        start_gate.wait()
        assert dispatch_started.wait(timeout=5)
        dispatch_gate.set()
        results = [future.result(timeout=5) for future in futures]

    assert results == [result_checksum, result_checksum]
    assert calls.count("jobserver:run") == 1


def test_concurrent_evaluate_expression_async_calls_share_one_evaluation(monkeypatch):
    get_expression_cache().clear()
    source = Buffer({"a": "local"}, "plain")
    source_checksum = source.get_checksum()
    source_ref = source.tempref()
    original_resolution = Checksum.resolution
    gate = asyncio.Event()
    entered = asyncio.Event()
    resolutions = 0

    async def gated_resolution(self, *args, **kwargs):
        nonlocal resolutions
        resolutions += 1
        entered.set()
        await gate.wait()
        return await original_resolution(self, *args, **kwargs)

    monkeypatch.setattr(Checksum, "resolution", gated_resolution)

    async def main():
        first = asyncio.create_task(
            evaluate_expression_async(source_checksum, "a", "plain", "str")
        )
        await entered.wait()
        second = asyncio.create_task(
            evaluate_expression_async(source_checksum, "a", "plain", "str")
        )
        await asyncio.sleep(0)
        gate.set()
        return await asyncio.gather(first, second)

    try:
        expected = Buffer("local", "str").get_checksum()
        assert asyncio.run(main()) == [expected, expected]
        assert resolutions == 1
    finally:
        source_ref.clear()


def test_remote_getter_in_running_loop_is_not_a_failure(monkeypatch):
    get_expression_cache().clear()
    remote_source = Buffer({"a": "remote"}, "plain")
    remote_checksum = remote_source.get_checksum()
    remote_content = remote_source.content
    drop_buffer(remote_checksum)
    calls = []
    install_fake_remotes(
        monkeypatch,
        {},
        {},
        calls,
        buffers={remote_checksum: remote_content},
    )

    local_source = Buffer({"a": "local"}, "plain")
    local_ref = local_source.tempref()

    async def read_cells():
        remote = _cell(remote_checksum)
        assert remote.checksum is None
        assert remote.state == "waiting"
        assert remote.exception is None
        assert "jobserver:run" not in calls

        local = _cell(local_source.get_checksum())
        assert local.checksum == Buffer("local", "str").get_checksum()

    try:
        asyncio.run(read_cells())
    finally:
        local_ref.clear()


@pytest.mark.parametrize("cancel", [False, True])
def test_local_active_failure_and_cancellation_cleanup(monkeypatch, cancel):
    from seamless import CacheMissError
    from seamless.checksum import expression as expression_mod

    missing = Checksum("8" * 64)
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    async def evaluate(*args, **kwargs):
        calls.append(1)
        entered.set()
        await finish.wait()
        raise CacheMissError(missing)

    monkeypatch.setattr(expression_mod, "_evaluate_expression_async", evaluate)

    async def main():
        tasks = [asyncio.create_task(evaluate_expression_async(missing, "a", "plain", "str"))
                 for _ in range(2)]
        await entered.wait()
        if cancel:
            for task in tasks:
                task.cancel()
        else:
            finish.set()
        errors = await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert calls == [1]
        expected = asyncio.CancelledError if cancel else CacheMissError
        assert all(isinstance(error, expected) for error in errors)
        if not cancel:
            assert all(error.args[0] == missing for error in errors)
        assert not expression_mod._active_expressions

    asyncio.run(main())
