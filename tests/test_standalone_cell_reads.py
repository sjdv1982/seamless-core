import asyncio
import sys
import threading

import pytest

from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.checksum import expression as expression_mod
from seamless.checksum.expression import get_expression_cache

from tests.helpers.fake_remotes import drop_buffer


def _identity(value):
    return value


def test_checksum_does_not_start_source_transformation():
    from seamless_transformer import delayed

    transformation = delayed(_identity)(12)
    cell = Cell("int", source=transformation)

    assert cell.checksum is None
    assert cell.state == "waiting"
    assert transformation._evaluated is False

    assert cell.compute() == Buffer(12, "int").get_checksum()
    assert transformation._evaluated is True


def test_checksum_evaluates_own_conversion_from_bare_checksum():
    get_expression_cache().clear()
    source = Buffer(12, "int")
    source_ref = source.tempref()
    try:
        cell = Cell(
            "str", checksum=source.get_checksum(), input_celltype="int"
        )
        assert cell.checksum is not None
        key = (source.get_checksum().hex(), "", "int", "str")
        assert get_expression_cache()[key] == cell.checksum
        assert cell.value == "12"
    finally:
        source_ref.clear()


def test_checksum_waits_for_running_source_expression(monkeypatch):
    get_expression_cache().clear()
    source = Buffer({"a": "ready"}, "plain")
    source_ref = source.tempref()
    expression = Expression(
        source.get_checksum(), path="a", input_celltype="plain", celltype="str"
    )
    result = Buffer("ready", "str")
    result_ref = result.tempref()
    entered = threading.Event()
    release = threading.Event()

    async def evaluate(*args, **kwargs):
        entered.set()
        await asyncio.to_thread(release.wait)
        return result.get_checksum()

    monkeypatch.setattr(expression_mod, "_evaluate_expression_async", evaluate)
    worker = threading.Thread(target=expression.compute)
    worker.start()
    assert entered.wait(timeout=5)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        cell = Cell("str", source=expression)
        assert cell.checksum == result.get_checksum()
    finally:
        release.set()
        worker.join(timeout=5)
        timer.cancel()
        source_ref.clear()
        result_ref.clear()


def test_missing_input_failure_is_handle_local_and_retryable(monkeypatch):
    monkeypatch.setitem(sys.modules, "seamless_remote", None)
    source = Buffer({"a": "later"}, "plain")
    checksum, content = source.get_checksum(), source.content
    drop_buffer(checksum)
    cell = Cell("str", checksum=checksum, input_celltype="plain", path="a")

    assert cell.checksum is None
    assert cell.state == "failed"
    assert isinstance(cell.exception, CacheMissError)
    assert cell.exception.args == (checksum,)
    assert cell.exception.__traceback__ is None

    restored = Buffer(content, checksum=checksum).tempref()
    try:
        cell.clear_exception()
        assert cell.checksum == Buffer("later", "str").get_checksum()
        assert cell.state == "complete"
        assert cell.exception is None
    finally:
        restored.clear()


def test_unavailable_result_buffer_does_not_fail_cell(monkeypatch):
    source = Buffer({"a": "gone"}, "plain")
    source_ref = source.tempref()
    cell = Cell(
        "str", checksum=source.get_checksum(), input_celltype="plain", path="a"
    )
    result = cell.checksum
    assert result is not None
    drop_buffer(result)

    def no_fingertip(self):
        raise AssertionError("Cell.value must not fingertip")

    monkeypatch.setattr(Checksum, "fingertip", no_fingertip)
    try:
        with pytest.raises(CacheMissError) as exc_info:
            cell.value
        assert exc_info.value.args == (result,)
        assert cell.state == "complete"
        assert cell.exception is None
    finally:
        source_ref.clear()
