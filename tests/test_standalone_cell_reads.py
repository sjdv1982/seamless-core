import asyncio
import sys
import threading

import pytest

from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.checksum import expression as expression_mod
from seamless.checksum.expression import get_expression_cache

from tests.helpers.fake_remotes import drop_buffer


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
    cell = Cell("plain", checksum=checksum)["a"]

    assert cell.checksum is None
    assert cell.state == "failed"
    assert cell.exception is not None
    assert checksum.hex() in str(cell.exception)

    restored = Buffer(content, checksum=checksum).tempref()
    try:
        cell.clear_exception()
        assert cell.checksum == Buffer("later", "plain").get_checksum()
        assert cell.state == "complete"
        assert cell.exception is None
    finally:
        restored.clear()


def test_unavailable_result_buffer_does_not_fail_cell(monkeypatch):
    source = Buffer({"a": "gone"}, "plain")
    source_ref = source.tempref()
    cell = Cell("plain", checksum=source.get_checksum())["a"]
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


def test_read_uses_completed_dependency_without_starting_it():
    source = Buffer("completed dependency", "str")
    hold = source.tempref()

    class Dependency:
        celltype = "str"

        def _compute_dependency(self):
            pytest.fail("a Cell property started its source Transformation")

        def _result_checksum_internal(self):
            return source.get_checksum()

    try:
        # A barrier keeps the dependency below more than one Expression.
        cell = Cell(source=Dependency()).as_celltype("text")[0]
        assert cell.value == "c"
    finally:
        hold.clear()


@pytest.mark.parametrize("celltype", ["python", "ipython", "deepcell", "deepfolder", "folder", "module"])
def test_null_conversion_child_builds_and_evaluates(celltype):
    root = Cell(celltype)
    root.set(None)
    child = root.as_celltype("int")
    assert child.build().compute() == root.checksum
    assert child.checksum == root.checksum
    assert child.value is None


def test_fingertip_uses_only_the_existing_result(monkeypatch):
    root = Cell("str")
    root.set("result to recover")
    child = root.as_celltype("text")
    assert child.fingertip() is None
    result = child.compute()
    recovered = Buffer("result to recover", "text")
    requests = []

    def recover(checksum):
        requests.append(checksum)
        return recovered

    def forbidden(*args, **kwargs):
        pytest.fail("fingertip started Cell evaluation")

    monkeypatch.setattr(Checksum, "fingertip_sync", recover)
    monkeypatch.setattr(Expression, "compute", forbidden)
    assert child.fingertip() is recovered
    assert requests == [result]
    root.set("changed source")
    assert child.fingertip() is None
    assert requests == [result]
