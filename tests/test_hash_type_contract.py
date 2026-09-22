"""Contract coverage for checksum-level HashType classification and queries."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from seamless.buffer_class import Buffer
from seamless.checksum.hash_type import (
    DType,
    Flag,
    HashType,
    Kind,
    Length,
    Rank,
    capabilities,
    deserializable_as,
    has_numeric_items,
    has_string_items,
    is_valid_word,
)
from seamless.checksum.hash_type_validation import conversion_feasible


CHECKSUM = "a" * 64
CELLTYPES = (
    "bytes",
    "text",
    "str",
    "int",
    "float",
    "bool",
    "plain",
    "binary",
    "mixed",
    "checksum",
    "python",
    "ipython",
    "yaml",
)
OUTSIDE_HASHTYPE_DOMAIN = ("deepcell", "deepfolder", "folder", "module")


def test_packed_field_values_and_bit_positions_are_stable():
    assert {member.name: member.value for member in Kind} == {
        "RAW_BYTES": 0,
        "NUMPY": 1,
        "MIXED_OBJECT": 2,
        "MIXED_ARRAY": 3,
        "RAW_TEXT": 4,
        "JSON_OBJECT": 5,
        "JSON_ARRAY": 6,
        "JSON_STRING": 7,
        "JSON_NUMBER": 8,
        "UNTESTED": 9,
        "UTF8_UNTESTED": 10,
        "JSON_UNTESTED": 11,
    }
    assert {member.name: member.value for member in Length} == {
        "SHORT": 0,
        "EQ64": 1,
        "MEDIUM": 2,
        "LONG": 3,
    }
    assert {member.name: member.value for member in DType} == {
        "NA": 0,
        "NUMERIC": 1,
        "NONNUMERIC": 2,
        "STRUCTURED": 3,
    }
    assert {member.name: member.value for member in Rank} == {
        "SCALAR": 0,
        "D1": 1,
        "D2": 2,
        "D3PLUS": 3,
    }
    assert {member.name: member.value for member in Flag} == {
        "NUMERIC_SCALAR": 1,
        "NUMPY_BYTES": 2,
        "SEMANTIC": 4,
    }

    value = HashType(
        Kind.JSON_STRING,
        Length.LONG,
        DType.NA,
        Rank.SCALAR,
        Flag.NUMERIC_SCALAR,
    )
    assert value.word == 7 | (3 << 4) | (1 << 10)
    assert value.word < 2**13
    assert HashType.unpack(value.word) == value


@pytest.mark.parametrize(
    "value",
    [
        HashType(Kind.RAW_TEXT, Length.SHORT, DType.NUMERIC),
        HashType(Kind.NUMPY, Length.SHORT, DType.NA),
        HashType(Kind.RAW_TEXT, Length.SHORT, rank=Rank.D1),
        HashType(
            Kind.NUMPY,
            Length.SHORT,
            DType.NONNUMERIC,
            Rank.D1,
            Flag.NUMPY_BYTES,
        ),
        HashType(Kind.JSON_NUMBER, Length.SHORT),
        HashType(
            Kind.JSON_OBJECT,
            Length.SHORT,
            flags=Flag.NUMERIC_SCALAR,
        ),
        HashType(Kind.RAW_TEXT, Length.SHORT, flags=Flag.SEMANTIC),
    ],
)
def test_well_formedness_rejects_each_forbidden_field_combination(value):
    assert is_valid_word(value.word) is False


CONCRETE_WORDS = {
    "raw-bytes": HashType(Kind.RAW_BYTES, Length.EQ64),
    "numpy": HashType(Kind.NUMPY, Length.EQ64, DType.NUMERIC),
    "mixed-object": HashType(Kind.MIXED_OBJECT, Length.EQ64),
    "mixed-array": HashType(Kind.MIXED_ARRAY, Length.EQ64),
    "raw-text": HashType(Kind.RAW_TEXT, Length.EQ64),
    "json-object": HashType(Kind.JSON_OBJECT, Length.EQ64),
    "json-array": HashType(Kind.JSON_ARRAY, Length.EQ64),
    "json-string": HashType(Kind.JSON_STRING, Length.EQ64),
    "json-number": HashType(
        Kind.JSON_NUMBER, Length.EQ64, flags=Flag.NUMERIC_SCALAR
    ),
}

CONCRETE_DESERIALIZATION = {
    "bytes": set(CONCRETE_WORDS),
    "text": {"raw-text", "json-object", "json-array", "json-string", "json-number"},
    "python": {"raw-text", "json-object", "json-array", "json-string", "json-number"},
    "ipython": {"raw-text", "json-object", "json-array", "json-string", "json-number"},
    "yaml": {"raw-text", "json-object", "json-array", "json-string", "json-number"},
    "plain": {"json-object", "json-array", "json-string", "json-number"},
    "str": {"json-string", "json-number"},
    "int": {"json-number"},
    "float": {"json-number"},
    "binary": {"numpy"},
    "mixed": {
        "numpy",
        "mixed-object",
        "mixed-array",
        "json-object",
        "json-array",
        "json-string",
        "json-number",
    },
    "checksum": {"raw-text", "json-number"},
    "bool": set(),
}


@pytest.mark.parametrize("celltype", CELLTYPES)
@pytest.mark.parametrize("word_name", CONCRETE_WORDS)
def test_concrete_deserializable_as_matrix(word_name, celltype):
    result = deserializable_as(
        CONCRETE_WORDS[word_name], celltype, checksum=CHECKSUM
    )
    assert result is (word_name in CONCRETE_DESERIALIZATION[celltype])


@pytest.mark.parametrize("raw", (b"true", b"true\n", b"false", b"false\n"))
def test_bool_deserialization_depends_on_the_four_canonical_checksums(raw):
    checksum = Buffer(raw).get_checksum()
    word = HashType.from_buffer(raw)
    assert word.deserializable_as("bool", checksum=checksum) is True


@pytest.mark.parametrize("raw", (b"null", b"null\n"))
@pytest.mark.parametrize("celltype", CELLTYPES)
def test_null_checksum_is_deserializable_as_every_celltype(raw, celltype):
    checksum = Buffer(raw).get_checksum()
    word = HashType.from_buffer(raw)
    assert word.deserializable_as(celltype, checksum=checksum) is True


@pytest.mark.parametrize(
    "source,word,expected",
    [
        ("bytes", HashType(Kind.RAW_BYTES, Length.SHORT), {"SEQ"}),
        ("text", HashType(Kind.RAW_TEXT, Length.SHORT), {"SEQ"}),
        ("str", HashType(Kind.JSON_STRING, Length.SHORT), {"SEQ"}),
        ("python", HashType(Kind.RAW_TEXT, Length.SHORT), {"SEQ"}),
        ("ipython", HashType(Kind.RAW_TEXT, Length.SHORT), {"SEQ"}),
        ("yaml", HashType(Kind.RAW_TEXT, Length.SHORT), {"SEQ"}),
        ("plain", HashType(Kind.JSON_OBJECT, Length.SHORT), {"MAP"}),
        ("plain", HashType(Kind.JSON_ARRAY, Length.SHORT), {"SEQ"}),
        ("plain", HashType(Kind.JSON_STRING, Length.SHORT), {"SEQ"}),
        ("mixed", HashType(Kind.MIXED_OBJECT, Length.SHORT), {"MAP"}),
        ("mixed", HashType(Kind.MIXED_ARRAY, Length.SHORT), {"SEQ"}),
        (
            "binary",
            HashType(Kind.NUMPY, Length.SHORT, DType.NUMERIC, Rank.D1),
            {"SEQ"},
        ),
        (
            "binary",
            HashType(Kind.NUMPY, Length.SHORT, DType.STRUCTURED, Rank.SCALAR),
            {"MAP"},
        ),
        (
            "binary",
            HashType(Kind.NUMPY, Length.SHORT, DType.STRUCTURED, Rank.D1),
            {"SEQ", "MAP"},
        ),
        ("int", HashType(Kind.JSON_NUMBER, Length.SHORT, flags=Flag.NUMERIC_SCALAR), set()),
        ("float", HashType(Kind.JSON_NUMBER, Length.SHORT, flags=Flag.NUMERIC_SCALAR), set()),
        ("bool", HashType(Kind.JSON_STRING, Length.SHORT), set()),
        ("checksum", HashType(Kind.RAW_TEXT, Length.EQ64), set()),
    ],
)
def test_capabilities_matrix(source, word, expected):
    assert capabilities(word, source) == expected


def _deserializable_query(word: HashType, celltype: str) -> object:
    return deserializable_as(word, celltype, checksum=CHECKSUM)


def _capabilities_query(word: HashType, celltype: str) -> object:
    return capabilities(word, celltype)


def _numeric_items_query(word: HashType, celltype: str) -> object:
    return has_numeric_items(word, celltype)


def _string_items_query(word: HashType, celltype: str) -> object:
    return has_string_items(word, celltype)


@pytest.mark.xfail(
    strict=False,
    reason="contract ahead of code: HashType queries must reject names outside the 13",
)
@pytest.mark.parametrize(
    "query",
    (
        _deserializable_query,
        _capabilities_query,
        _numeric_items_query,
        _string_items_query,
    ),
    ids=("deserializable", "capabilities", "numeric-items", "string-items"),
)
@pytest.mark.parametrize("celltype", OUTSIDE_HASHTYPE_DOMAIN)
def test_non_hash_type_celltypes_are_caller_errors(
    query: Callable[[HashType, str], object], celltype: str
):
    word = HashType(Kind.JSON_OBJECT, Length.SHORT)
    with pytest.raises(ValueError):
        query(word, celltype)


@pytest.mark.xfail(
    strict=False,
    reason="contract ahead of code: conversion feasibility rejects names outside the 13",
)
@pytest.mark.parametrize("celltype", OUTSIDE_HASHTYPE_DOMAIN)
@pytest.mark.parametrize("position", ("source", "target"))
def test_conversion_feasibility_rejects_non_hash_type_celltypes(celltype, position):
    source, target = (celltype, "plain") if position == "source" else ("plain", celltype)
    word = HashType(Kind.JSON_OBJECT, Length.SHORT)
    with pytest.raises(ValueError):
        conversion_feasible(word, source, target, checksum=CHECKSUM)


@pytest.mark.parametrize(
    "word,source,target,expected",
    [
        (HashType(Kind.JSON_OBJECT, Length.SHORT), "mixed", "str", False),
        (HashType(Kind.MIXED_OBJECT, Length.SHORT), "mixed", "str", True),
        (HashType(Kind.JSON_OBJECT, Length.SHORT), "plain", "binary", False),
        (
            HashType(Kind.NUMPY, Length.SHORT, DType.NUMERIC),
            "binary",
            "plain",
            True,
        ),
        (
            HashType(Kind.NUMPY, Length.SHORT, DType.STRUCTURED),
            "binary",
            "plain",
            True,
        ),
        (
            HashType(Kind.NUMPY, Length.SHORT, DType.NONNUMERIC),
            "binary",
            "plain",
            None,
        ),
        (HashType(Kind.RAW_TEXT, Length.EQ64), "checksum", "plain", None),
        (HashType(Kind.UNTESTED, Length.SHORT), "plain", "binary", None),
        (HashType(Kind.JSON_STRING, Length.SHORT), "mixed", "bool", None),
    ],
)
def test_conversion_possible_and_value_branches(word, source, target, expected):
    assert conversion_feasible(word, source, target, checksum=CHECKSUM) is expected
