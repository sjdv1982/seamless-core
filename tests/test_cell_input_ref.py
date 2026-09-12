"""A Cell's input_ref is a reference, never a value; values go through set()."""

import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.caching.buffer_cache import get_buffer_cache


def _count(checksum):
    return get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0]


VALUES = ["x", "ab" * 32, b"\0" * 32, 12, 1.5, True, {"a": 1}, [1], bytearray(b"x")]


@pytest.mark.parametrize("value", VALUES, ids=lambda v: type(v).__name__)
def test_constructor_rejects_values(value):
    with pytest.raises(TypeError, match=r"\.set\(\)"):
        Cell(input_ref=value)


@pytest.mark.parametrize("value", VALUES, ids=lambda v: type(v).__name__)
def test_input_ref_property_rejects_values_and_keeps_old_hold(value):
    checksum = Buffer(b"input_ref property").get_checksum()
    cell = Cell(input_ref=checksum)
    with pytest.raises(TypeError):
        cell.input_ref = value
    assert cell.input_ref == checksum
    assert _count(checksum) == 1
    cell._release_refholds()


def test_with_input_and_build_override_reject_values():
    cell = Cell(input_celltype="plain")
    with pytest.raises(TypeError):
        cell.with_input({"a": 1})
    with pytest.raises(TypeError):
        cell.build({"a": 1})
    with pytest.raises(TypeError):
        cell.compute({"a": 1})


def test_checksum_input_forms():
    checksum = Buffer({"a": 1}, "plain").get_checksum()
    by_keyword = Cell(input_ref=checksum, input_celltype="plain")
    by_property = Cell(input_celltype="plain")
    by_property.input_ref = checksum
    by_set_checksum = Cell(input_celltype="plain")
    by_set_checksum.set_checksum(checksum.hex())
    for cell in (by_keyword, by_property, by_set_checksum):
        assert isinstance(cell.input_ref, Checksum)
        assert cell.input_ref == checksum
    assert _count(checksum) == 3
    for cell in (by_keyword, by_property, by_set_checksum):
        cell._release_refholds()
    assert _count(checksum) == 0


def test_reference_inputs_are_accepted():
    checksum = Buffer(b"reference inputs").get_checksum()
    expression = Expression(checksum, input_celltype="bytes")
    upstream = Cell(input_ref=checksum)
    assert Cell(input_ref=None).input_ref is None
    assert Cell(input_ref=expression).input_ref is expression
    assert Cell(input_ref=upstream).input_ref is upstream


def test_set_serializes_value_with_current_celltype():
    value = {"a": 1, "b": [5, 6]}
    buffer = Buffer(value, "plain")
    cell = Cell(input_celltype="plain")
    cell.set(value)
    assert cell.input_ref == buffer.get_checksum()
    assert _count(cell.input_ref) == 1
    # Previously a value input could not be evaluated standalone.
    assert cell.a.run() == 1
    assert cell.b.run() == [5, 6]


def test_set_hex_string_is_a_value_not_a_checksum():
    text = "ab" * 32
    cell = Cell(input_celltype="text")
    cell.set(text)
    assert cell.input_ref != Checksum(text)
    assert cell.run() == text


def test_set_checksum_is_passed_through():
    checksum = Buffer(b"set checksum").get_checksum()
    cell = Cell(input_celltype="bytes")
    cell.set(checksum)
    assert cell.input_ref is checksum


def test_set_value_replaces_and_releases_previous_hold():
    cell = Cell(input_celltype="int")
    cell.set(1)
    first = cell.input_ref
    cell.set(2)
    assert _count(first) == 0
    assert cell.run() == 2
    cell.set(None)
    assert cell.input_ref is None


def test_failed_set_leaves_input_unchanged():
    checksum = Buffer(b"failed set").get_checksum()
    cell = Cell(input_ref=checksum, input_celltype="not-a-celltype")
    with pytest.raises(TypeError):
        cell.set("value")
    assert cell.input_ref == checksum
    assert _count(checksum) == 1
    cell._release_refholds()


@pytest.mark.parametrize("celltype", ["mixed", "plain", "int", "float", "str", "text", "bytes"])
def test_positional_argument_is_celltype(celltype):
    cell = Cell(celltype)
    assert cell.input_celltype == celltype
    assert cell.target_celltype == celltype
    assert cell.input_ref is None


def test_positional_celltype_with_keyword_reference():
    checksum = Buffer(42, "int").get_checksum()
    cell = Cell("int", input_ref=checksum)
    assert cell.run() == 42
    assert cell.input_ref == checksum
    assert Cell().input_celltype == "mixed"


def test_input_reference_cannot_be_second_positional_argument():
    with pytest.raises(TypeError):
        Cell("int", Buffer(42, "int").get_checksum())
