import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.checksum import expression as expression_mod
from seamless.checksum.expression import (
    _active_expressions,
    evaluate_expression_local_async,
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


@pytest.mark.parametrize("scratch", [False, True])
def test_intermediate_is_scratch_then_materialized_here(monkeypatch, scratch):
    source = Buffer("remote", "text")
    source_checksum = source.get_checksum()
    intermediate = Expression(
        source_checksum, input_celltype="text", celltype="mixed",
    ).compute(execution="local")
    drop_buffer(source_checksum)
    drop_buffer(intermediate)
    get_expression_cache().clear()
    first_key = (source_checksum.hex(), "", "text", "mixed")
    calls, sent = [], []
    buffers = {source_checksum: source.content}
    install_fake_remotes(
        monkeypatch, {}, {first_key: intermediate}, calls, buffers=buffers,
    )
    jobserver_remote = sys.modules["seamless_remote.jobserver_remote"]
    fake_dispatch = jobserver_remote.run_expression

    async def run_expression(*args, scratch=False):
        sent.append(scratch)
        return await fake_dispatch(*args, scratch=scratch)

    monkeypatch.setattr(jobserver_remote, "run_expression", run_expression)
    cell = Cell("text", checksum=source_checksum).as_celltype("mixed")[0]
    cell.scratch = scratch

    result = cell.compute()
    assert result is not None, (cell.exception, sent, calls)
    assert result == Buffer("r", "mixed").get_checksum(), (cell.exception, sent, calls)
    assert sent == [True]
    assert intermediate not in buffers


@pytest.mark.parametrize("scratch", [False, True])
def test_getter_dispatch_carries_the_cell_scratch_policy(monkeypatch, scratch):
    # cells.md, *Scratch*: a non-scratch Cell's requests carry scratch=False,
    # so the executing side writes the end result and .value works after a
    # remote evaluation; a scratch Cell's dispatched result is never written.
    get_expression_cache().clear()
    source = Buffer({"a": "remote"}, "plain")
    source_checksum = source.get_checksum()
    result = Buffer("remote", "str")
    result_checksum = result.get_checksum()
    drop_buffer(source_checksum)
    drop_buffer(result_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls, buffers, sent = [], {}, []
    install_fake_remotes(monkeypatch, {}, {key: result_checksum}, calls, buffers=buffers)
    jobserver_remote = sys.modules["seamless_remote.jobserver_remote"]
    fake_run_expression = jobserver_remote.run_expression

    async def run_expression(*args, scratch=False):
        sent.append(scratch)
        answer = await fake_run_expression(*args, scratch=scratch)
        if not scratch:  # the executing side writes the end result
            buffers[answer] = result.content
        return answer

    monkeypatch.setattr(jobserver_remote, "run_expression", run_expression)
    cell = _cell(source_checksum)
    cell.scratch = scratch

    assert cell.checksum == result_checksum
    assert sent == [scratch]
    if scratch:
        assert buffers == {}
    else:
        assert cell.value == "remote"


@pytest.mark.parametrize("scratch", [False, True])
def test_getter_accepts_a_recorded_checksum_whose_buffer_is_unreachable(
    monkeypatch, scratch
):
    # A getter asks for a checksum, so a recorded result answers it even when
    # its buffer is reachable nowhere, whatever the Cell's scratch. Only a
    # value request (run(), fingertip) insists on reachable bytes.
    get_expression_cache().clear()
    source_checksum = Checksum("2" * 64)
    result_checksum = Buffer("recorded, not stored", "str").get_checksum()
    drop_buffer(source_checksum)
    drop_buffer(result_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    install_fake_remotes(monkeypatch, {key: result_checksum}, {}, calls)
    cell = _cell(source_checksum)
    cell.scratch = scratch

    assert cell.checksum == result_checksum
    assert cell.exception is None
    assert "jobserver:run" not in calls


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
            evaluate_expression_local_async(source_checksum, "a", "plain", "str")
        )
        await entered.wait()
        second = asyncio.create_task(
            evaluate_expression_local_async(source_checksum, "a", "plain", "str")
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
        tasks = [asyncio.create_task(evaluate_expression_local_async(missing, "a", "plain", "str"))
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
