"""Exhaustive contract tests for the checksum conversion engine."""

from collections.abc import Callable

import numpy as np
import pytest

from seamless import Buffer, CacheMissError, Checksum
from seamless.checksum.celltypes import celltypes
from seamless.checksum.convert import convert_checksum, conversion_needs_buffer
from seamless.checksum.conversion import (
    SeamlessConversionError,
    conversion_chain,
    conversion_equivalent,
    conversion_forbidden,
    conversion_possible,
    conversion_reformat,
    conversion_reinterpret,
    conversion_trivial,
    conversion_values,
)
from seamless.checksum.null import NULL_CHECKSUM


# This is the complete matrix in contracts/celltypes-and-conversion.md. Keeping
# the indirections here matters: conversion composition is deliberately not
# associative, so category counts or terminal outcomes alone are insufficient.
MATRIX_CELLTYPES = (
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
EXPECTED_MATRIX = {
    "binary": "id T >plain >text >text V >text >bytes RF P P P V".split(),
    "mixed": "RI id >plain >text >text RI >text P RF P P P V".split(),
    "text": " >mixed =text:str id RI RI RF RI RF T >plain >plain >plain V".split(),
    "python": "=text:binary =text:str T id T =text:str X =text:str =python:text X X X V".split(),
    "ipython": "=text:binary =text:str T RF id =text:str X =text:str =ipython:text X X X V".split(),
    "plain": "V T RF >text >text id T RI T RI RI RI V".split(),
    "yaml": ">plain =text:str T X X RF id =text:str =text:bytes >plain >plain >plain V".split(),
    "str": (
        "=plain:binary =str:plain RF =str:text =str:text T =str:text "
        "id =plain:bytes RI RI RI V"
    ).split(),
    "bytes": "RF RF RI >text >text RI >text >plain id >plain >plain >plain V".split(),
    "int": "=plain:binary =int:plain =plain:text X X T >plain T =plain:bytes id T V V".split(),
    "float": "=plain:binary =float:plain =plain:text X X T >plain T =plain:bytes T id V V".split(),
    "bool": "=plain:binary =bool:plain =plain:text X X T >plain T =plain:bytes V V id V".split(),
    "checksum": "V V V V V V V V V V V V id".split(),
}
CATEGORY_TABLES = {
    "T": conversion_trivial,
    "RI": conversion_reinterpret,
    "RF": conversion_reformat,
    "P": conversion_possible,
    "V": conversion_values,
    "X": conversion_forbidden,
}


def _fail_if_fetched():
    raise AssertionError("conversion unexpectedly fetched its source buffer")


def _counting_getter(buffer: Buffer) -> tuple[Callable[[], Buffer], list[None]]:
    calls = []

    def get_buffer():
        calls.append(None)
        return buffer

    return get_buffer, calls


def test_rule_table_matches_the_complete_contract_matrix():
    assert tuple(celltypes) == MATRIX_CELLTYPES
    assert set(EXPECTED_MATRIX) == set(MATRIX_CELLTYPES)

    classified = set().union(*CATEGORY_TABLES.values())
    classified.update(conversion_equivalent)
    classified.update(conversion_chain)
    assert len(classified) == len(MATRIX_CELLTYPES) * (len(MATRIX_CELLTYPES) - 1)

    for source, row in EXPECTED_MATRIX.items():
        assert len(row) == len(MATRIX_CELLTYPES)
        for target, expected in zip(MATRIX_CELLTYPES, row, strict=True):
            pair = (source, target)
            memberships = {
                code for code, table in CATEGORY_TABLES.items() if pair in table
            }
            memberships.update("=" for table in (conversion_equivalent,) if pair in table)
            memberships.update(">" for table in (conversion_chain,) if pair in table)

            if expected == "id":
                assert source == target
                assert not memberships
            elif expected.startswith("="):
                assert memberships == {"="}
                assert conversion_equivalent[pair] == tuple(expected[1:].split(":"))
            elif expected.startswith(">"):
                assert memberships == {">"}
                assert conversion_chain[pair] == expected[1:]
            else:
                assert memberships == {expected}


@pytest.mark.parametrize(
    "source,target",
    [("not-a-celltype", "plain"), ("plain", "not-a-celltype")],
)
def test_executor_rejects_unknown_celltypes(source, target):
    with pytest.raises(TypeError):
        convert_checksum(Checksum(bytes(32)), source, target, _fail_if_fetched)


def test_executor_requires_a_callable_buffer_getter():
    with pytest.raises(TypeError, match="get_buffer must be callable"):
        convert_checksum(Checksum(bytes(32)), "plain", "text", None)


def test_identity_is_unvalidated_and_buffer_free():
    checksum = Checksum(bytes.fromhex("11" * 32))
    assert convert_checksum(checksum, "python", "python", _fail_if_fetched) == (
        checksum,
        None,
    )
    assert conversion_needs_buffer(checksum, "python", "python") is False


def test_cache_miss_from_buffer_getter_propagates_unchanged():
    source = Buffer("not-json", "text")
    checksum = source.get_checksum()
    miss = CacheMissError(checksum)

    def get_buffer():
        raise miss

    with pytest.raises(CacheMissError) as exc_info:
        convert_checksum(checksum, "text", "plain", get_buffer)
    assert exc_info.value is miss


def test_other_buffer_getter_errors_are_wrapped_with_conversion_identity():
    source = Buffer("not-json", "text")
    checksum = source.get_checksum()

    def get_buffer():
        raise RuntimeError("buffer backend failed")

    with pytest.raises(SeamlessConversionError) as exc_info:
        convert_checksum(checksum, "text", "plain", get_buffer)
    message = str(exc_info.value)
    assert f"{checksum.hex()} cannot be converted from text to plain" in message
    assert "buffer backend failed" in message


def test_existing_conversion_error_from_buffer_getter_propagates_unchanged():
    source = Buffer("not-json", "text")
    checksum = source.get_checksum()
    error = SeamlessConversionError("already classified")

    def get_buffer():
        raise error

    with pytest.raises(SeamlessConversionError) as exc_info:
        convert_checksum(checksum, "text", "plain", get_buffer)
    assert exc_info.value is error


@pytest.mark.parametrize("source", [name for name in MATRIX_CELLTYPES if name != "checksum"])
def test_every_source_to_checksum_conversion_serializes_the_source_identity(source):
    checksum = Checksum(bytes.fromhex("22" * 32))
    result_checksum, result_buffer = convert_checksum(
        checksum, source, "checksum", _fail_if_fetched
    )
    assert result_buffer is not None
    assert result_buffer.content == checksum.hex().encode()
    assert result_buffer.get_checksum() == result_checksum


@pytest.mark.parametrize("target", [name for name in MATRIX_CELLTYPES if name != "checksum"])
def test_checksum_to_every_target_dereferences_without_target_validation(target):
    referenced = Checksum(bytes.fromhex("33" * 32))
    source = Buffer(referenced, "checksum")
    get_buffer, calls = _counting_getter(source)
    result_checksum, result_buffer = convert_checksum(
        source.get_checksum(), "checksum", target, get_buffer
    )
    assert calls == [None]
    assert result_checksum == referenced
    assert result_buffer is None


def test_executor_does_not_short_circuit_null_when_target_is_checksum():
    null_checksum = Checksum(NULL_CHECKSUM)
    result_checksum, result_buffer = convert_checksum(
        null_checksum, "plain", "checksum", _fail_if_fetched
    )
    assert result_buffer is not None
    assert result_buffer.get_value("checksum") == null_checksum
    assert result_checksum != null_checksum


@pytest.mark.parametrize(
    "source,target,source_buffer,expected",
    [
        ("binary", "bool", lambda: Buffer(np.array(2), "binary"), True),
        ("binary", "float", lambda: Buffer(np.array(2), "binary"), 2.0),
        ("binary", "int", lambda: Buffer(np.array(2), "binary"), 2),
        ("mixed", "bool", lambda: Buffer(2, "mixed"), True),
        ("mixed", "float", lambda: Buffer(2, "mixed"), 2.0),
        ("mixed", "int", lambda: Buffer(2, "mixed"), 2),
        ("mixed", "str", lambda: Buffer(2, "mixed"), "2"),
    ],
)
def test_every_possible_conversion_accepts_a_scalar(
    source, target, source_buffer, expected
):
    buffer = source_buffer()
    result_checksum, result_buffer = convert_checksum(
        buffer.get_checksum(), source, target, lambda: buffer
    )
    assert result_buffer is not None
    assert result_buffer.get_checksum() == result_checksum
    assert result_buffer.get_value(target) == expected


@pytest.mark.parametrize(
    "source,buffer",
    [
        ("binary", lambda: Buffer(np.array([1, 2]), "binary")),
        ("mixed", lambda: Buffer([1, 2], "mixed")),
        ("mixed", lambda: Buffer({"a": 1}, "mixed")),
    ],
)
def test_possible_conversion_rejects_containers_and_nonscalar_arrays(source, buffer):
    source_buffer = buffer()
    with pytest.raises(SeamlessConversionError):
        convert_checksum(
            source_buffer.get_checksum(), source, "int", lambda: source_buffer
        )


@pytest.mark.parametrize(
    "source,target,value,expected",
    [
        ("bool", "int", True, 1),
        ("bool", "float", True, 1.0),
        ("int", "bool", 0, False),
        ("float", "bool", 1.5, True),
    ],
)
def test_all_boolean_numeric_value_rules_reformat(source, target, value, expected):
    source_buffer = Buffer(value, source)
    result_checksum, result_buffer = convert_checksum(
        source_buffer.get_checksum(), source, target, lambda: source_buffer
    )
    assert result_buffer is not None
    assert result_buffer.get_checksum() == result_checksum
    assert result_buffer.get_value(target) == expected


@pytest.mark.parametrize(
    "value",
    [1, 1.5, True, [1, 2, 3]],
)
def test_plain_to_binary_accepts_exactly_the_documented_json_shapes(value):
    source = Buffer(value, "plain")
    result_checksum, result_buffer = convert_checksum(
        source.get_checksum(), "plain", "binary", lambda: source
    )
    assert result_buffer is not None
    assert result_buffer.get_checksum() == result_checksum
    np.testing.assert_array_equal(result_buffer.get_value("binary"), np.array(value))


@pytest.mark.parametrize("value", [{"a": 1}, [1, {"a": 2}]])
def test_plain_to_binary_rejects_mapping_and_object_dtype(value):
    source = Buffer(value, "plain")
    with pytest.raises(SeamlessConversionError):
        convert_checksum(source.get_checksum(), "plain", "binary", lambda: source)


def test_binary_to_plain_uses_canonical_plain_serialization():
    source = Buffer(np.array([2, 1]), "binary")
    result_checksum, result_buffer = convert_checksum(
        source.get_checksum(), "binary", "plain", lambda: source
    )
    expected = Buffer([2, 1], "plain")
    assert result_buffer is not None
    assert result_buffer.content == expected.content
    assert result_checksum == expected.get_checksum()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_binary_to_plain_rejects_nonfinite_values(value):
    source = Buffer(np.array([value]), "binary")
    with pytest.raises(SeamlessConversionError, match="non-finite"):
        convert_checksum(source.get_checksum(), "binary", "plain", lambda: source)


@pytest.mark.parametrize(
    "source,target,source_buffer,expected_buffer",
    [
        ("bytes", "mixed", lambda: Buffer(b"hello"), lambda: Buffer("hello", "str")),
        (
            "bytes",
            "mixed",
            lambda: Buffer(b"\xff"),
            lambda: Buffer(np.array(b"\xff"), "binary"),
        ),
        (
            "binary",
            "bytes",
            lambda: Buffer(np.array(b"hello"), "binary"),
            lambda: Buffer(b"hello"),
        ),
        (
            "mixed",
            "bytes",
            lambda: Buffer(np.array(b"hello"), "mixed"),
            lambda: Buffer(b"hello"),
        ),
        ("text", "str", lambda: Buffer("hello", "text"), lambda: Buffer("hello", "str")),
        ("str", "text", lambda: Buffer("hello", "str"), lambda: Buffer("hello", "text")),
    ],
)
def test_uncovered_reformat_branches_create_the_documented_buffer(
    source, target, source_buffer, expected_buffer
):
    buffer = source_buffer()
    result_checksum, result_buffer = convert_checksum(
        buffer.get_checksum(), source, target, lambda: buffer
    )
    expected = expected_buffer()
    assert result_buffer is not None
    assert result_buffer.content == expected.content
    assert result_checksum == expected.get_checksum()


@pytest.mark.parametrize(
    "source,target,source_buffer,expected_calls",
    [
        ("bytes", "mixed", lambda: Buffer(b'{"a": 1}\n'), []),
        ("mixed", "bytes", lambda: Buffer({"a": 1}, "mixed"), [None]),
    ],
)
def test_uncovered_reformat_keep_branches_preserve_the_checksum(
    source, target, source_buffer, expected_calls
):
    buffer = source_buffer()
    get_buffer, calls = _counting_getter(buffer)
    result = convert_checksum(buffer.get_checksum(), source, target, get_buffer)
    assert calls == expected_calls
    assert result == (buffer.get_checksum(), None)


@pytest.mark.xfail(
    strict=False,
    reason="contract ahead of code: deep zero-path conversions are not implemented",
)
@pytest.mark.parametrize(
    "source,target",
    [
        ("deepcell", "deepcell"),
        ("deepfolder", "deepfolder"),
        ("folder", "folder"),
        ("deepcell", "plain"),
        ("deepfolder", "plain"),
        ("folder", "deepfolder"),
        ("deepfolder", "folder"),
        ("deepcell", "deepfolder"),
    ],
)
def test_deep_free_zero_path_conversions_preserve_checksum_without_fetch(
    source, target
):
    checksum = Checksum(bytes.fromhex("44" * 32))
    assert convert_checksum(checksum, source, target, _fail_if_fetched) == (
        checksum,
        None,
    )


@pytest.mark.xfail(
    strict=False,
    reason="contract ahead of code: folder-to-mixed materialization is not implemented",
)
def test_empty_folder_to_mixed_materializes_the_empty_mapping():
    source = Buffer({}, "folder")
    result_checksum, result_buffer = convert_checksum(
        source.get_checksum(), "folder", "mixed", lambda: source
    )
    expected = Buffer({}, "mixed")
    assert result_buffer is not None
    assert result_buffer.content == expected.content
    assert result_checksum == expected.get_checksum()
