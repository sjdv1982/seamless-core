import asyncio
import sys
from types import ModuleType

from seamless import Buffer, Checksum, Expression
from seamless.checksum.expression import (
    _active_expressions,
    evaluate_expression_remote,
    get_expression_cache,
)


def _install_fake_remotes(monkeypatch, expression_rows, jobserver_results, calls):
    seamless_remote = ModuleType("seamless_remote")
    seamless_remote.__path__ = []
    database_remote = ModuleType("seamless_remote.database_remote")
    jobserver_remote = ModuleType("seamless_remote.jobserver_remote")
    buffer_remote = ModuleType("seamless_remote.buffer_remote")

    def key(input_checksum, path, celltype, target_celltype):
        return (
            Checksum(input_checksum).hex(),
            path,
            celltype,
            target_celltype,
        )

    async def get_expression_result(input_checksum, path, celltype, target_celltype):
        calls.append("database:get")
        return expression_rows.get(key(input_checksum, path, celltype, target_celltype))

    async def set_expression_result(
        input_checksum, path, celltype, target_celltype, result_checksum
    ):
        calls.append("database:set")
        expression_rows[key(input_checksum, path, celltype, target_celltype)] = Checksum(
            result_checksum
        )
        return True

    async def run_expression(input_checksum, path, celltype, target_celltype):
        calls.append("jobserver:run")
        return Checksum(jobserver_results[key(input_checksum, path, celltype, target_celltype)])

    database_remote.get_expression_result = get_expression_result
    database_remote.set_expression_result = set_expression_result
    jobserver_remote.run_expression = run_expression
    async def get_buffer(checksum):
        return None

    buffer_remote.get_buffer = get_buffer
    seamless_remote.database_remote = database_remote
    seamless_remote.jobserver_remote = jobserver_remote
    seamless_remote.buffer_remote = buffer_remote
    monkeypatch.setitem(sys.modules, "seamless_remote", seamless_remote)
    monkeypatch.setitem(sys.modules, "seamless_remote.database_remote", database_remote)
    monkeypatch.setitem(sys.modules, "seamless_remote.jobserver_remote", jobserver_remote)
    monkeypatch.setitem(sys.modules, "seamless_remote.buffer_remote", buffer_remote)


def test_remote_expression_dispatch_writes_expression_cache(monkeypatch):
    get_expression_cache().clear()
    source_checksum = Checksum("1" * 64)
    result_checksum = Buffer("remote", "str").get_checksum()
    expression_rows = {}
    calls = []
    key = (source_checksum.hex(), "a", "plain", "str")
    _install_fake_remotes(
        monkeypatch,
        expression_rows,
        {key: result_checksum.hex()},
        calls,
    )

    result = asyncio.run(
        evaluate_expression_remote(
            source_checksum,
            "a",
            "plain",
            "str",
            execution="remote",
        )
    )

    assert result == result_checksum
    assert expression_rows[key] == result_checksum
    assert calls == ["database:get", "jobserver:run", "database:set"]


def test_remote_expression_cache_hit_skips_dispatch(monkeypatch):
    get_expression_cache().clear()
    source_checksum = Checksum("2" * 64)
    result_checksum = Buffer("cached", "str").get_checksum()
    key = (source_checksum.hex(), "a", "plain", "str")
    expression_rows = {key: result_checksum}
    calls = []
    _install_fake_remotes(monkeypatch, expression_rows, {}, calls)

    result = asyncio.run(
        evaluate_expression_remote(
            source_checksum,
            "a",
            "plain",
            "str",
            execution="remote",
        )
    )

    assert result == result_checksum
    assert calls == ["database:get"]


def test_auto_expression_uses_local_buffer_before_remote_dispatch(monkeypatch):
    get_expression_cache().clear()
    source_checksum = Buffer({"a": "local"}, "plain").get_checksum()
    result_checksum = Buffer("local", "str").get_checksum()
    key = (source_checksum.hex(), "a", "plain", "str")
    expression_rows = {}
    calls = []
    _install_fake_remotes(monkeypatch, expression_rows, {key: result_checksum.hex()}, calls)

    result = asyncio.run(
        evaluate_expression_remote(
            source_checksum,
            "a",
            "plain",
            "str",
            execution="auto",
        )
    )

    assert result == result_checksum
    assert calls == ["database:get", "database:set"]


def test_expression_cancel_fast_path_returns_false():
    get_expression_cache().clear()
    source_checksum = Buffer({"a": "local"}, "plain").get_checksum()
    expression = Expression(source_checksum, "a", input_celltype="plain", target_celltype="str")

    assert expression.cancel() is False


def test_remote_expression_members_share_one_active_request(monkeypatch):
    get_expression_cache().clear()
    _active_expressions.clear()
    source_checksum = Checksum("3" * 64)
    result_checksum = Buffer("remote", "str").get_checksum()
    key = (source_checksum.hex(), "a", "plain", "str")
    expression_rows = {}
    calls = []
    started = asyncio.Event()
    release = asyncio.Event()
    _install_fake_remotes(monkeypatch, expression_rows, {key: result_checksum.hex()}, calls)

    from seamless_remote import jobserver_remote

    async def run_expression(input_checksum, path, celltype, target_celltype):
        calls.append("jobserver:run")
        started.set()
        await release.wait()
        return result_checksum

    jobserver_remote.run_expression = run_expression
    expr1 = Expression(source_checksum, "a", input_celltype="plain", target_celltype="str")
    expr2 = Expression(source_checksum, "a", input_celltype="plain", target_celltype="str")

    async def main():
        task1 = asyncio.create_task(expr1.compute_async(execution="remote"))
        await started.wait()
        task2 = asyncio.create_task(expr2.compute_async(execution="remote"))
        for _ in range(100):
            active = _active_expressions.get(key)
            if active is not None and len(active.members) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(_active_expressions[key].members) == 2
        assert expr1.cancel() is True
        assert len(_active_expressions[key].members) == 1
        release.set()
        assert await task1 == result_checksum
        assert await task2 == result_checksum

    asyncio.run(main())
    assert calls == ["database:get", "jobserver:run", "database:get", "database:set"]


def test_remote_expression_last_member_cancel_stops_active_request(monkeypatch):
    get_expression_cache().clear()
    _active_expressions.clear()
    source_checksum = Checksum("4" * 64)
    key = (source_checksum.hex(), "a", "plain", "str")
    expression_rows = {}
    calls = []
    started = asyncio.Event()
    _install_fake_remotes(monkeypatch, expression_rows, {key: "5" * 64}, calls)

    from seamless_remote import jobserver_remote

    async def run_expression(input_checksum, path, celltype, target_celltype):
        calls.append("jobserver:run")
        started.set()
        await asyncio.Future()

    jobserver_remote.run_expression = run_expression
    expression = Expression(source_checksum, "a", input_celltype="plain", target_celltype="str")

    async def main():
        task = asyncio.create_task(expression.compute_async(execution="remote"))
        await started.wait()
        assert expression.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await task

    import pytest

    asyncio.run(main())
    assert calls == ["database:get", "jobserver:run"]
