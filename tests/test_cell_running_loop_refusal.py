"""A synchronous Cell read must not turn remote-loop refusal into failure."""

import asyncio

import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.checksum.expression import get_expression_cache
from seamless.error_envelope import RunningLoopRefusal
from tests.helpers.fake_remotes import drop_buffer, install_fake_remotes


def test_checksum_running_loop_refusal_returns_none_without_exception(monkeypatch):
    get_expression_cache().clear()
    source = Checksum("de" * 32)
    drop_buffer(source)
    result = Buffer("remote cell read", "str")
    result_ref = result.tempref()
    calls = []
    key = (source.hex(), "", "plain", "str")
    install_fake_remotes(
        monkeypatch, {}, {key: result.get_checksum().hex()}, calls,
        jobserver_available=True,
    )
    cell = Cell("str", checksum=source, input_celltype="plain")

    async def read_on_running_loop():
        # Exercise the genuine refusal before checking Cell's suppression of it.
        with pytest.raises(RunningLoopRefusal):
            Expression(source, input_celltype="plain", celltype="str").compute()
        assert cell.checksum is None
        assert cell.exception is None
        assert cell.checksum is None
        assert "jobserver:run" not in calls

    try:
        asyncio.run(read_on_running_loop())
        # Refusal must leave the handle retryable once blocking is permitted.
        assert cell.checksum == result.get_checksum()
        assert cell.exception is None
        assert calls.count("jobserver:run") == 1
    finally:
        result_ref.clear()
