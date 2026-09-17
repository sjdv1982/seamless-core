"""Expression errors keep their type, and run() materializes like .value.

Review decisions §5: a missing buffer raises ``CacheMissError(checksum)``, not
``ExpressionEvaluationError``, and ``Expression.run()`` (so ``Cell.run()``)
materializes through ``Checksum.resolve()``, which also asks the hashserver.
"""

import gc
import hashlib
import sys
from types import ModuleType

import pytest

from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.checksum.expression import (
    ExpressionEvaluationError,
    choose_expression_evaluation_location,
    evaluate_expression,
    get_expression_cache,
)


def _never_stored(content):
    """A checksum computed without touching any Seamless cache."""
    return Checksum(hashlib.sha256(content).hexdigest())


MISSING = _never_stored(b'{"a": "never stored anywhere"}\n')
REMOTE_CONTENT = b'"only on the hashserver"\n'
REMOTE = _never_stored(REMOTE_CONTENT)


@pytest.fixture(autouse=True)
def clean_expression_cache():
    # Failed computes leave Expressions in traceback cycles. Collect them here, on the
    # main thread: finalized in the buffer-writer thread, their refholder release can
    # deadlock against a main thread that registers a buffer under the cache lock.
    gc.collect()
    get_expression_cache().clear()
    yield
    get_expression_cache().clear()
    gc.collect()


@pytest.fixture
def hashserver(monkeypatch):
    """A fake hashserver holding the given buffers; records every request."""
    buffers, requests = {}, []
    seamless_remote = ModuleType("seamless_remote")
    seamless_remote.__path__ = []
    buffer_remote = ModuleType("seamless_remote.buffer_remote")

    async def get_buffer(checksum):
        requests.append(Checksum(checksum))
        content = buffers.get(Checksum(checksum))
        return None if content is None else Buffer(content, checksum=Checksum(checksum))

    buffer_remote.get_buffer = get_buffer
    seamless_remote.buffer_remote = buffer_remote
    monkeypatch.setitem(sys.modules, "seamless_remote", seamless_remote)
    monkeypatch.setitem(sys.modules, "seamless_remote.buffer_remote", buffer_remote)
    return buffers, requests


@pytest.mark.parametrize("path,input_celltype,celltype", [
    ("a", "plain", "plain"),
    ("[0:3]", "text", "text"),
    ("", "text", "plain"),
], ids=["item", "slice", "conversion"])
def test_missing_input_buffer_raises_cache_miss_error(path, input_celltype, celltype):
    with pytest.raises(CacheMissError) as info:
        evaluate_expression(MISSING, path, input_celltype, celltype)
    assert not isinstance(info.value, ExpressionEvaluationError)
    assert info.value.args == (MISSING,)

    expression = Expression(MISSING, path=path, input_celltype=input_celltype, celltype=celltype)
    with pytest.raises(CacheMissError):
        expression.compute()


def test_invalid_path_still_raises_expression_evaluation_error():
    source = Buffer({"a": 1}, "plain")
    source.tempref()
    with pytest.raises(ExpressionEvaluationError, match="Unclosed path bracket"):
        evaluate_expression(source.get_checksum(), "a[", "plain", "plain")


def test_auto_location_treats_a_missing_buffer_as_remote():
    assert choose_expression_evaluation_location(MISSING, "a", "plain", "plain") == "remote"


def test_run_resolves_a_result_through_the_hashserver(hashserver):
    buffers, requests = hashserver
    buffers[REMOTE] = REMOTE_CONTENT
    expression = Expression(MISSING, path="a", input_celltype="plain", celltype="str")
    get_expression_cache()[expression.database_key] = REMOTE

    assert expression.compute() == REMOTE
    assert expression.run() == "only on the hashserver"
    assert REMOTE in requests
    assert Cell("str", source=expression).run() == "only on the hashserver"


def test_run_without_any_buffer_raises_cache_miss_error(hashserver):
    # Not REMOTE: once fetched, its buffer stays in this process's buffer cache.
    nowhere = _never_stored(b'"on no hashserver"\n')
    expression = Expression(MISSING, path="a", input_celltype="plain", celltype="str")
    get_expression_cache()[expression.database_key] = nowhere

    assert expression.compute() == nowhere
    with pytest.raises(CacheMissError) as info:
        expression.run()
    assert info.value.args == (nowhere,)
