"""Celltype checksum: the value is a Checksum, serialized as the bare hex digest."""
import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.checksum.hash_type_validation import HashTypeValidationError


def _held(value, celltype):
    buffer = Buffer(value, celltype)
    buffer.tempref()
    return buffer.get_checksum()


# A leading zero or a letter makes the digest raw text; only digits make it a JSON number.
@pytest.mark.parametrize("digest", ["ab" * 32, "1" * 64, "0123456789" * 6 + "abcd"])
def test_value_serializes_to_the_bare_digest(digest):
    checksum = Checksum(digest)
    buffer = Buffer(checksum, "checksum")
    assert buffer.content == digest.encode()
    assert Buffer(digest.upper(), "checksum").content == digest.encode()
    value = buffer.get_value("checksum")
    assert isinstance(value, Checksum) and value == checksum


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        Buffer("z" * 64, "checksum")
    with pytest.raises(TypeError):
        Buffer(42, "checksum")
    with pytest.raises(TypeError, match="only for celltype 'checksum'"):
        Buffer(Checksum("ab" * 32), "plain")
    # The former quoted-JSON form is no longer a checksum value.
    with pytest.raises(HashTypeValidationError):
        Buffer(b'"' + b"ab" * 32 + b'"\n').get_value("checksum")


def test_cell_expression_and_resolve_agree():
    target = _held("pointed", "str")
    checksum = _held(target, "checksum")
    cell = Cell("checksum", checksum=checksum)
    assert cell.value == cell.run() == cell.build().run() == target
    assert isinstance(cell.value, Checksum)
    assert checksum.resolve("checksum") == target
    assert Expression(checksum, input_celltype="checksum", celltype="checksum").run() == target


def test_set_stores_a_checksum_as_the_value():
    checksum = Checksum("ab" * 32)
    cell = Cell("checksum")
    cell.set(checksum)
    assert cell.checksum.resolve().content == checksum.hex().encode()
    assert isinstance(cell.value, Checksum) and cell.value == checksum
    assert cell.value == cell.run() == cell.build().run()
    by_hex = Cell("checksum")
    by_hex.set(checksum.hex())
    by_value = Cell("checksum")
    by_value.value = checksum
    assert by_hex.checksum == by_value.checksum == cell.checksum
    with pytest.raises(TypeError, match=r"use \.set_checksum\(\)"):
        Cell("int").set(checksum)

