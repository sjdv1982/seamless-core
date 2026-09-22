"""Contract coverage for the celltype hierarchy and checksum-level values."""

import hashlib
import math

import numpy as np
import pytest

from seamless import Buffer, Checksum, Expression
from seamless.checksum.calculate_checksum import TRIVIAL_CHECKSUMS
from seamless.checksum.celltypes import celltypes
from seamless.checksum.conversion import conversion_trivial
from seamless.checksum.null import (
    NULL_BUFFER,
    NULL_CHECKSUM,
    canonicalize_checksum,
    is_null,
    is_null_value,
)
from seamless.checksum.parse_buffer import _parse_buffer
from seamless.checksum.serialize import _serialize
from seamless.checksum.virtual import NOT_VIRTUAL, virtual_value


EXPECTED_CELLTYPES = (
    "binary",
    "mixed",
    "text",
    "python",
    "ipython",
    "plain",
    "yaml",
    "str",
    "bytes",
    "int",
    "float",
    "bool",
    "checksum",
)

EXPECTED_TRIVIAL_CONVERSIONS = {
    ("binary", "mixed"),
    ("bool", "plain"),
    ("bool", "str"),
    ("float", "int"),
    ("float", "plain"),
    ("float", "str"),
    ("int", "float"),
    ("int", "plain"),
    ("int", "str"),
    ("ipython", "text"),
    ("plain", "bytes"),
    ("plain", "mixed"),
    ("plain", "yaml"),
    ("python", "ipython"),
    ("python", "text"),
    ("str", "plain"),
    ("text", "bytes"),
    ("yaml", "text"),
}


def _parse(raw, celltype):
    buffer = Buffer(raw)
    return _parse_buffer(buffer, buffer.get_checksum(), celltype)


def test_celltype_registry_and_checksum_hierarchy_are_exact():
    assert tuple(celltypes) == EXPECTED_CELLTYPES
    assert conversion_trivial == EXPECTED_TRIVIAL_CONVERSIONS
    assert all("checksum" not in edge for edge in conversion_trivial)


def test_parser_and_serializer_reject_unknown_celltypes():
    raw = Buffer(b"1")
    with pytest.raises(TypeError):
        _parse_buffer(raw, raw.get_checksum(), "not-a-celltype")
    with pytest.raises(TypeError):
        _serialize(1, "not-a-celltype")


@pytest.mark.xfail(
    strict=False,
    reason="_serialize returns canonical null before validating the celltype",
)
def test_serializer_rejects_unknown_celltype_for_none_too():
    with pytest.raises(TypeError):
        _serialize(None, "not-a-celltype")


@pytest.mark.parametrize("celltype", EXPECTED_CELLTYPES)
def test_none_has_one_canonical_serialization_for_every_celltype(celltype):
    assert _serialize(None, celltype) == NULL_BUFFER
    assert Buffer(None, celltype).get_checksum() == Checksum(NULL_CHECKSUM)


def test_empty_bytes_canonicalization_is_celltype_specific():
    empty_checksum = Buffer(b"").get_checksum()
    null_checksum = Checksum(NULL_CHECKSUM)

    assert empty_checksum != null_checksum
    assert Buffer(b"", "bytes").content == NULL_BUFFER
    assert Buffer(b"", "bytes").get_checksum() == null_checksum
    assert canonicalize_checksum(empty_checksum, "bytes") == null_checksum
    for celltype in EXPECTED_CELLTYPES:
        if celltype != "bytes":
            assert canonicalize_checksum(empty_checksum, celltype) == empty_checksum



@pytest.mark.xfail(
    strict=False,
    reason="public Expression identity keys do not yet canonicalize empty bytes",
)
def test_expression_key_canonicalizes_empty_bytes_input():
    empty_checksum = Buffer(b"").get_checksum()
    null_checksum = Checksum(NULL_CHECKSUM)
    expression = Expression(empty_checksum, input_celltype="bytes", celltype="bytes")
    assert expression.identity_key[0] == null_checksum.hex()


@pytest.mark.parametrize("raw", [b"null", NULL_BUFFER])
@pytest.mark.parametrize("celltype", EXPECTED_CELLTYPES)
def test_both_null_checksums_are_virtual_for_every_celltype(raw, celltype):
    expected = b"" if celltype == "bytes" else None
    assert _parse(raw, celltype) == expected


def test_null_predicates_distinguish_canonical_and_noncanonical_null():
    canonical = Checksum(NULL_CHECKSUM)
    noncanonical = Buffer(b"null").get_checksum()

    assert is_null(canonical)
    assert is_null_value(canonical)
    assert not is_null(noncanonical)
    assert is_null_value(noncanonical)
    assert not is_null(None)
    assert not is_null_value(None)


def test_null_content_collisions_and_text_empty_string():
    for raw in (b"null", b"null\n"):
        assert Buffer(raw, "bytes").get_value("bytes") == b""

    for celltype in ("text", "python", "ipython", "yaml"):
        assert Buffer("null", celltype).get_value(celltype) is None
        empty = Buffer("", celltype)
        assert empty.content == b"\n"
        assert empty.get_value(celltype) == ""

    for celltype in ("str", "plain"):
        value = Buffer("null", celltype)
        assert value.content == b'"null"\n'
        assert value.get_value(celltype) == "null"


def test_only_canonical_null_has_the_null_display():
    canonical = Checksum(NULL_CHECKSUM)
    noncanonical = Buffer(b"null").get_checksum()

    assert str(canonical) == "NULL"
    assert str(noncanonical) != "NULL"
    assert canonical.hex() == NULL_CHECKSUM
    assert repr(canonical) == repr(NULL_CHECKSUM)


@pytest.mark.parametrize(
    "raw,value",
    [(b"true", True), (b"true\n", True), (b"false", False), (b"false\n", False)],
)
def test_virtual_boolean_reading_matrix(raw, value):
    checksum = Buffer(raw).get_checksum()
    for celltype in ("bool", "plain", "mixed"):
        assert virtual_value(checksum, celltype) is value
    assert virtual_value(checksum, "str") == str(value)
    for celltype in ("int", "float"):
        with pytest.raises(ValueError):
            virtual_value(checksum, celltype)
    nonvirtual_celltypes = (
        "binary",
        "text",
        "python",
        "ipython",
        "yaml",
        "bytes",
        "checksum",
    )
    for celltype in nonvirtual_celltypes:
        assert virtual_value(checksum, celltype) is NOT_VIRTUAL


def test_trivial_checksum_buffer_table_is_exact():
    buffers = (
        b"null",
        b"null\n",
        b"true",
        b"false",
        b"true\n",
        b"false\n",
        b"",
        b"{}",
        b"{}\n",
        b"[]",
        b"[]\n",
    )
    expected = {hashlib.sha256(buffer).hexdigest(): buffer for buffer in buffers}
    assert TRIVIAL_CHECKSUMS == expected


@pytest.mark.parametrize(
    "raw,celltype,expected",
    [
        (b"hello\n\n", "text", "hello"),
        (b"x = 1\n\n", "python", "x = 1"),
        (b"%time 1 + 1\n", "ipython", "%time 1 + 1"),
        (b"a: 1\n\n", "yaml", "a: 1"),
        (b'{"b":2,"a":1}\n', "plain", {"b": 2, "a": 1}),
        (b"1e3", "str", "1000.0"),
        (b'"4.5"', "float", 4.5),
        (b"-4.5", "int", -4),
    ],
)
def test_reference_parser_success_rules(raw, celltype, expected):
    assert _parse(raw, celltype) == expected


def test_bytes_parser_returns_the_buffer_object():
    buffer = Buffer(b"raw bytes")
    result = _parse_buffer(buffer, buffer.get_checksum(), "bytes")
    assert result is buffer


@pytest.mark.parametrize(
    "value,celltype,expected",
    [
        ({"b": 2, "a": 1}, "plain", b'{\n  "a": 1,\n  "b": 2\n}\n'),
        (12, "str", b'"12"\n'),
        (True, "str", b"true\n"),
        (-4.5, "int", b"-4\n"),
        (4, "float", b"4.0\n"),
        (0, "bool", b"false\n"),
        ("hello\n\n", "text", b"hello\n"),
        (b"hello\n\n", "python", b"hello\n"),
        (b"raw", "bytes", b"raw"),
    ],
)
def test_canonical_scalar_and_text_serialization(value, celltype, expected):
    assert _serialize(value, celltype) == expected


@pytest.mark.parametrize("celltype", ["plain", "float"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.xfail(
    strict=False,
    reason="JSON serialization currently turns non-finite numbers into null",
)
def test_json_celltypes_reject_nonfinite_values(celltype, value):
    with pytest.raises((TypeError, ValueError)):
        _serialize(value, celltype)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.xfail(
    strict=False,
    reason="mixed currently serializes non-finite Python floats as raw text",
)
def test_mixed_preserves_nonfinite_values_as_numpy(value):
    top_level = Buffer(value, "mixed").get_value("mixed")
    nested = Buffer([value], "mixed").get_value("mixed")[0]

    assert isinstance(top_level, np.float64)
    assert isinstance(nested, np.float64)
    if math.isnan(value):
        assert math.isnan(top_level) and math.isnan(nested)
    else:
        assert top_level == value
        assert nested == value


def test_numpy_scalar_and_array_storage_contract():
    mixed_float = Buffer(np.float32(1.5), "mixed").get_value("mixed")
    assert type(mixed_float) is float

    scalar = np.float32(1.5)
    for celltype in ("binary", "mixed"):
        binary_scalar = Buffer(scalar, "binary").get_value(celltype)
        assert isinstance(binary_scalar, np.float32)
        assert binary_scalar == scalar

        array = np.array([1, 2], dtype=np.int16)
        restored = Buffer(array, celltype).get_value(celltype)
        assert restored.dtype == array.dtype
        np.testing.assert_array_equal(restored, array)


@pytest.mark.xfail(
    strict=False,
    reason="mixed does not yet accept NumPy boolean scalars",
)
def test_mixed_numpy_boolean_scalar_becomes_python_bool():
    restored = Buffer(np.bool_(True), "mixed").get_value("mixed")
    assert type(restored) is bool


@pytest.mark.parametrize("celltype", ["binary", "mixed"])
@pytest.mark.xfail(
    strict=False,
    reason="the mixed serializer does not yet support complex dtypes",
)
def test_complex_values_use_numpy_storage(celltype):
    scalar = np.complex64(1 + 2j)
    restored = Buffer(scalar, celltype).get_value(celltype)
    assert isinstance(restored, np.complex64)
    assert restored == scalar
