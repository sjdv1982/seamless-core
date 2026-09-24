"""A non-scratch request asks for the bytes: materialize, never answer from a cache alone.

``scratch=False`` (``Expression.run()``, a non-scratch Cell) is answered only by a
result checksum whose buffer the requester can reach. A cached checksum without a
reachable buffer is evaluated again: locally, or by a non-scratch dispatch that the
executing side materializes and writes. With the input reachable nowhere, the answer
is ``CacheMissError`` on the result; recovering it is ``fingertip()``'s job.
"""

import gc

import pytest

from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum import expression as expression_mod
from seamless.checksum.expression import _active_expressions, get_expression_cache

from tests.helpers.fake_remotes import drop_buffer, install_fake_remotes


@pytest.fixture(autouse=True)
def clean_expression_cache():
    gc.collect()
    get_expression_cache().clear()
    _active_expressions.clear()
    yield
    get_expression_cache().clear()
    gc.collect()


def _record_scratch(monkeypatch):
    import sys

    jobserver_remote = sys.modules["seamless_remote.jobserver_remote"]
    original = jobserver_remote.run_expression
    sent = []

    async def run_expression(*args, scratch):
        sent.append(scratch)
        return await original(*args, scratch=scratch)

    monkeypatch.setattr(jobserver_remote, "run_expression", run_expression)
    return sent


def test_local_run_reevaluates_an_evicted_result():
    source = Buffer({"a": "local value"}, "plain")
    source.tempref()
    expression = Expression(
        source.get_checksum(), path="a", input_celltype="plain", celltype="str"
    )
    result = expression.compute(execution="local")
    drop_buffer(result)
    assert not expression_mod._has_local_buffer(result)

    assert expression.run(execution="local") == "local value"
    assert expression_mod._has_local_buffer(result)


def test_compute_then_run_dispatches_non_scratch_when_result_unreachable(monkeypatch):
    source = Buffer({"a": "remote"}, "plain")
    source_checksum = source.get_checksum()
    result_checksum = Buffer("remote", "str").get_checksum()
    drop_buffer(source_checksum)
    drop_buffer(result_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    install_fake_remotes(
        monkeypatch, {}, {key: result_checksum}, calls,
        buffers={source_checksum: source.content},
    )
    sent = _record_scratch(monkeypatch)

    expression = Expression(source_checksum, path="a", input_celltype="plain", celltype="str")
    assert expression.compute() == result_checksum
    assert sent == [True]
    # The scratch dispatch wrote nothing, so run() must ask again, non-scratch.
    with pytest.raises(CacheMissError):
        expression.run()
    assert sent == [True, False]


def test_reachable_cached_result_is_an_answer_for_run(monkeypatch):
    source_checksum = Checksum("2" * 64)
    result_content = b'"stored"\n'
    result_checksum = Buffer(result_content).get_checksum()
    drop_buffer(result_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    install_fake_remotes(
        monkeypatch, {key: result_checksum}, {}, calls,
        buffers={result_checksum: result_content},
    )
    sent = _record_scratch(monkeypatch)

    expression = Expression(source_checksum, path="a", input_celltype="plain", celltype="str")
    assert expression.run() == "stored"
    assert sent == []


def test_input_reachable_nowhere_raises_on_the_result(monkeypatch):
    source_checksum = Checksum("3" * 64)
    result_checksum = Checksum("4" * 64)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    install_fake_remotes(monkeypatch, {key: result_checksum}, {}, calls)
    sent = _record_scratch(monkeypatch)

    expression = Expression(source_checksum, path="a", input_celltype="plain", celltype="str")
    with pytest.raises(CacheMissError) as info:
        expression.run()
    assert info.value.args == (result_checksum,)
    assert sent == []


def test_cell_carries_its_scratch_policy_on_dispatch(monkeypatch):
    source = Buffer({"a": "remote"}, "plain")
    source_checksum = source.get_checksum()
    result_checksum = Buffer("remote", "str").get_checksum()
    drop_buffer(source_checksum)
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    database_rows = {}
    install_fake_remotes(
        monkeypatch, database_rows, {key: result_checksum}, calls,
        buffers={source_checksum: source.content},
    )
    sent = _record_scratch(monkeypatch)

    cell = Cell("plain", checksum=source_checksum)["a"].as_celltype("str")
    assert cell.scratch is False
    assert cell.compute() == result_checksum
    assert sent[-1] is False
    assert not get_buffer_cache().is_scratch_ref(result_checksum)

    get_expression_cache().clear()
    database_rows.clear()
    scratch_cell = Cell("plain", checksum=source_checksum)["a"].as_celltype("str")
    scratch_cell.scratch = True
    assert scratch_cell.compute() == result_checksum
    assert sent[-1] is True
    assert get_buffer_cache().is_scratch_ref(result_checksum)


def test_cell_scratch_is_per_cell():
    cell = Cell("plain", checksum=Checksum("5" * 64))
    cell.scratch = True
    assert cell.scratch is True
    # A projection or a retyping is a new Cell that owns its own result.
    assert cell.as_celltype("mixed").scratch is False
    assert cell["a"].scratch is False
