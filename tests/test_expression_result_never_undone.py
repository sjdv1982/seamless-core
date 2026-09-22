"""Expression results that can't be deserialized as their celltype are never undone.

Review decisions §3.2 item 7: such an expression is not invalid. It
deterministically returns its result checksum, which stays stored; reading the
value fails, and fails the same way every time. A Cell built on it is `failed`,
and `clear_exception()` reproduces the failure (§8.4).

Only the text celltypes (python, ipython, yaml) reach this: their validity is
checked at parse time, outside HashType. For the other celltypes, serializing
the result already fails, so no result exists.
"""

import pytest

from seamless import Buffer, Cell, Expression
from seamless.checksum import hash_type as ht
from seamless.checksum.expression import get_expression_cache, resolve_expression_value
from seamless.checksum.hash_type_validation import HashTypeValidationError


CASES = [
    pytest.param("text", "x = (\ny = 1\n", "[0:5]", "python", "x = (", id="text-slice-python"),
    pytest.param("plain", {"code": "x = ("}, "code", "python", "x = (", id="plain-item-python"),
    pytest.param("plain", {"code": "value: ["}, "code", "yaml", "value: [", id="plain-item-yaml"),
]


@pytest.fixture(autouse=True)
def clean_expression_cache():
    get_expression_cache().clear()
    yield
    get_expression_cache().clear()


def _expression(input_celltype, value, path, celltype):
    buffer = Buffer(value, input_celltype)
    buffer.tempref()
    checksum = buffer.get_checksum()
    return Expression(checksum, path=path, input_celltype=input_celltype, celltype=celltype)


def _cache_key(expression):
    return (expression.input_checksum.hex(), expression.path, expression.input_celltype, expression.celltype)


@pytest.mark.parametrize("input_celltype,value,path,celltype,text", CASES)
def test_failed_read_keeps_the_stored_result(input_celltype, value, path, celltype, text):
    expression = _expression(input_celltype, value, path, celltype)
    result = expression.compute()
    assert result is not None
    assert get_expression_cache()[_cache_key(expression)] == result

    messages = set()
    for _ in range(2):
        with pytest.raises(HashTypeValidationError, match="Cannot deserialize") as exc_info:
            expression.run()
        messages.add(str(exc_info.value))
        assert expression.checksum == result
        assert get_expression_cache()[_cache_key(expression)] == result
    assert len(messages) == 1


@pytest.mark.parametrize("input_celltype,value,path,celltype,text", CASES)
def test_reevaluation_returns_the_same_valid_checksum(input_celltype, value, path, celltype, text):
    first = _expression(input_celltype, value, path, celltype)
    result = first.compute()
    with pytest.raises(HashTypeValidationError):
        first.run()

    get_expression_cache().clear()
    again = _expression(input_celltype, value, path, celltype)
    assert again.compute() == result
    # The checksum itself is sound: it only fails as the requested celltype.
    assert resolve_expression_value(result, "text") == text
    hash_type = ht.get_hash_type(result)
    assert hash_type is not None and hash_type.is_utf8
    with pytest.raises(HashTypeValidationError):
        again.run()


@pytest.mark.parametrize("input_celltype,value,path,celltype,text", CASES)
def test_standalone_cell_fails_and_clear_exception_reproduces(input_celltype, value, path, celltype, text):
    expression = _expression(input_celltype, value, path, celltype)
    cell = Cell(celltype, source=expression)
    result = cell.compute()
    assert result is not None

    for _ in range(2):
        with pytest.raises(HashTypeValidationError):
            cell.value
        assert cell.state == "failed"
        assert cell.exception is not None
        assert get_expression_cache()[_cache_key(expression)] == result
        cell.clear_exception()
