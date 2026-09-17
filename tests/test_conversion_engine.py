"""Checksum conversion-engine contracts, including lazy buffer access."""

import asyncio
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pytest

from seamless import Buffer, Checksum
from seamless.checksum.convert import convert_checksum, conversion_needs_buffer
from seamless.checksum.conversion import SeamlessConversionError, conversion_trivial
from seamless.checksum.expression import (
    ExpressionKey,
    choose_expression_evaluation_location,
    evaluate_expression,
    evaluate_expression_async,
    get_expression_cache,
)


def _serialized_buffer(value, celltype=None):
    return Buffer(value) if celltype is None else Buffer(value, celltype)


def _npy_bytes_buffer():
    binary = Buffer(np.array([1, 2]), "binary")
    return Buffer(binary.content)


def _source_buffer(celltype):
    values = {
        "plain": ({"a": 1}, "plain"),
        "text": ("hello", "text"),
        "str": ("42", "str"),
        "int": (4.5, "int"),
        "float": (4.5, "float"),
        "bool": (True, "bool"),
        "bytes": (b"hello", None),
        "binary": (np.array([1, 2]), "binary"),
        "mixed": ({"a": 1}, "mixed"),
        "yaml": ("a: 1\n", "yaml"),
        "ipython": ("x = 1", "ipython"),
        "python": ("x = 1", "python"),
    }
    value, serialization_celltype = values[celltype]
    return _serialized_buffer(value, serialization_celltype)


def _raise_if_fetched():
    raise AssertionError("this conversion must not fetch its source buffer")


def _counting_getter(buffer):
    calls = []

    def get_buffer():
        calls.append(None)
        return buffer

    return get_buffer, calls


def _assert_target_value(value, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(value, expected)
    else:
        assert value == expected


@pytest.mark.parametrize(
    "source,target,value,source_celltype,expected",
    [
        ("float", "int", 4.5, "float", lambda buffer: 4),
        ("int", "str", b"4.5\n", None, lambda buffer: "4.5"),
        ("bool", "str", True, "bool", lambda buffer: "True"),
        ("int", "mixed", b"4.5\n", None, lambda buffer: 4.5),
        ("plain", "bytes", {"a": 1}, "plain", lambda buffer: buffer.content),
        ("int", "float", b"4.5\n", None, lambda buffer: 4.5),
    ],
)
def test_no_fetch_keep_conversion_rules_preserve_checksum(
    source, target, value, source_celltype, expected
):
    buffer = _serialized_buffer(value, source_celltype)
    checksum = buffer.get_checksum()
    result_checksum, result_buffer = convert_checksum(
        checksum, source, target, _raise_if_fetched
    )
    assert result_checksum == checksum
    assert result_buffer is None
    target_value = buffer.get_value(target)
    if target == "bytes":
        target_value = target_value.content
    _assert_target_value(target_value, expected(buffer))


@pytest.mark.parametrize(
    "source,target,buffer,expected",
    [
        ("plain", "int", lambda: Buffer("42", "plain"), lambda buffer: 42),
        ("bytes", "str", lambda: Buffer(b'"42"'), lambda buffer: "42"),
        ("text", "plain", lambda: Buffer(b'{"a":1}'), lambda buffer: {"a": 1}),
        (
            "plain",
            "text",
            lambda: Buffer({"a": 1}, "plain"),
            lambda buffer: buffer.decode().rstrip("\n"),
        ),
        ("binary", "bytes", lambda: Buffer(np.array([1, 2]), "binary"), lambda buffer: buffer.content),
    ],
)
def test_checksum_preserving_conversion_rules_fetch_input_once(
    source, target, buffer, expected
):
    input_buffer = buffer()
    checksum = input_buffer.get_checksum()
    get_buffer, calls = _counting_getter(input_buffer)
    assert conversion_needs_buffer(checksum, source, target) is True
    result_checksum, result_buffer = convert_checksum(
        checksum, source, target, get_buffer
    )
    assert calls == [None]
    assert result_checksum == checksum
    assert result_buffer is None
    target_value = input_buffer.get_value(target)
    if target == "bytes":
        target_value = target_value.content
    _assert_target_value(target_value, expected(input_buffer))


def test_bytes_npy_to_binary_preserves_checksum_after_fetching_input_once():
    buffer = _npy_bytes_buffer()
    checksum = buffer.get_checksum()
    get_buffer, calls = _counting_getter(buffer)
    assert conversion_needs_buffer(checksum, "bytes", "binary") is True
    result_checksum, result_buffer = convert_checksum(
        checksum, "bytes", "binary", get_buffer
    )
    assert calls == [None]
    assert result_checksum == checksum
    assert result_buffer is None
    _assert_target_value(buffer.get_value("binary"), np.array([1, 2]))


@pytest.mark.parametrize(
    "source,target,source_buffer,expected_buffer",
    [
        ("plain", "text", lambda: Buffer("hi", "plain"), lambda: Buffer("hi\n", "text")),
        ("text", "plain", lambda: Buffer(b"hi"), lambda: Buffer("hi", "plain")),
        ("bytes", "binary", lambda: Buffer(b"\xff"), lambda: Buffer(np.array(b"\xff"), "binary")),
        ("int", "bool", lambda: Buffer(1, "int"), lambda: Buffer(True, "bool")),
        ("yaml", "plain", lambda: Buffer("a: 1\n", "yaml"), lambda: Buffer({"a": 1}, "plain")),
        (
            "ipython",
            "python",
            lambda: Buffer("%time 1 + 1", "ipython"),
            lambda: Buffer("_ = get_ipython().run_line_magic('time', '1 + 1')\n", "python"),
        ),
    ],
)
def test_reformat_conversion_rules_create_canonical_buffer(
    source, target, source_buffer, expected_buffer
):
    input_buffer = source_buffer()
    checksum = input_buffer.get_checksum()
    expected = expected_buffer()
    result_checksum, result_buffer = convert_checksum(
        checksum, source, target, lambda: input_buffer
    )
    assert result_buffer is not None
    assert result_buffer.content == expected.content
    assert result_buffer.get_checksum() == result_checksum
    assert result_checksum != checksum


def test_bool_to_int_reformats_without_fetching_input(monkeypatch):
    input_buffer = Buffer(True, "bool")
    checksum = input_buffer.get_checksum()
    from seamless.checksum import expression

    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    result_checksum, result_buffer = convert_checksum(
        checksum, "bool", "int", _raise_if_fetched
    )
    expected = Buffer(1, "int")
    assert result_buffer is not None
    assert result_buffer.content == expected.content
    assert result_checksum == expected.get_checksum()
    assert result_checksum != checksum
    assert choose_expression_evaluation_location(checksum, "", "bool", "int") == "local"


@pytest.mark.parametrize(
    "source,target,buffer",
    [
        ("str", "int", lambda: Buffer("hello", "str")),
        ("plain", "str", lambda: Buffer([1], "plain")),
        ("plain", "bool", lambda: Buffer(42, "plain")),
        ("python", "bool", lambda: Buffer("True", "python")),
    ],
)
def test_failed_conversion_rules_raise(source, target, buffer):
    input_buffer = buffer()
    checksum = input_buffer.get_checksum()
    with pytest.raises(SeamlessConversionError):
        convert_checksum(checksum, source, target, lambda: input_buffer)


@pytest.mark.parametrize("source,target", sorted(conversion_trivial))
def test_every_trivial_conversion_needs_no_buffer(source, target, monkeypatch):
    input_buffer = _source_buffer(source)
    checksum = input_buffer.get_checksum()
    from seamless.checksum import expression

    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    assert not conversion_needs_buffer(checksum, source, target)
    result_checksum, result_buffer = convert_checksum(
        checksum, source, target, _raise_if_fetched
    )
    assert result_checksum == checksum
    assert result_buffer is None
    assert choose_expression_evaluation_location(checksum, "", source, target) == "local"


@pytest.mark.parametrize(
    "source,target,buffer",
    [
        ("plain", "bool", lambda: Buffer(True, "plain")),
        ("bool", "str", lambda: Buffer(True, "bool")),
        ("plain", "int", lambda: Buffer(None, "plain")),
    ],
)
def test_virtual_no_fetch_conversions_are_local(source, target, buffer, monkeypatch):
    input_buffer = buffer()
    checksum = input_buffer.get_checksum()
    from seamless.checksum import expression

    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    assert not conversion_needs_buffer(checksum, source, target)
    result_checksum, result_buffer = convert_checksum(
        checksum, source, target, _raise_if_fetched
    )
    assert result_checksum == checksum
    assert result_buffer is None
    assert choose_expression_evaluation_location(checksum, "", source, target) == "local"


def test_plain_number_to_bool_fails_without_fetch_and_is_local(monkeypatch):
    input_buffer = Buffer(42, "plain")
    checksum = input_buffer.get_checksum()
    from seamless.checksum import expression

    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    assert not conversion_needs_buffer(checksum, "plain", "bool")
    with pytest.raises(SeamlessConversionError):
        convert_checksum(checksum, "plain", "bool", _raise_if_fetched)
    assert choose_expression_evaluation_location(checksum, "", "plain", "bool") == "local"


def test_construct_and_evaluate_expression_sync_without_fetch(monkeypatch):
    input_buffer = Buffer(True, "plain")
    checksum = input_buffer.get_checksum()
    key = ExpressionKey(checksum, "", "plain", "bool")
    assert key.input_checksum == checksum
    assert key.path == ""
    assert key.input_celltype == "plain"
    assert key.celltype == "bool"

    get_expression_cache().clear()
    from seamless.checksum import expression

    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    result = evaluate_expression(
        key.input_checksum,
        key.path,
        key.input_celltype,
        key.celltype,
    )
    assert result == checksum


def test_construct_and_evaluate_expression_async_fetches_once(monkeypatch):
    input_buffer = Buffer("hi", "text")
    checksum = input_buffer.get_checksum()
    key = ExpressionKey(checksum, "", "text", "plain")
    assert key.input_checksum == checksum
    assert key.path == ""
    assert key.input_celltype == "text"
    assert key.celltype == "plain"

    get_expression_cache().clear()
    from seamless.checksum import expression

    original_resolution = Checksum.resolution
    calls = []

    async def resolution(self, celltype=None):
        calls.append((self, celltype))
        return await original_resolution(self, celltype)

    monkeypatch.setattr(Checksum, "resolution", resolution)
    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    result = asyncio.run(
        evaluate_expression_async(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.celltype,
        )
    )
    expected = Buffer("hi", "plain")
    assert calls == [(checksum, None)]
    assert result == expected.get_checksum()
    assert result.resolve("plain") == "hi"


def test_evaluate_expression_async_reads_from_local_buffer_directory(monkeypatch):
    from seamless.checksum import expression
    from seamless.checksum.hash_type import register_hash_type_for_buffer
    from seamless_remote import buffer_remote
    from seamless_remote.buffer_client import BufferClient

    raw = b'"async-buffer-directory-witness"\n'
    checksum = Checksum(hashlib.sha256(raw).digest())
    register_hash_type_for_buffer(checksum, raw)
    key = ExpressionKey(checksum, "", "plain", "str")
    get_expression_cache().clear()

    monkeypatch.setattr(
        expression,
        "_get_local_buffer",
        lambda *args, **kwargs: _raise_if_fetched(),
    )
    monkeypatch.setattr(buffer_remote, "_read_server_clients", [])

    with tempfile.TemporaryDirectory() as directory:
        buffer_path = Path(directory) / checksum.hex()
        buffer_path.write_bytes(raw)
        client = BufferClient(readonly=True)
        client.directory = directory
        reads = []
        original_get_file_buffer = client.get_file_buffer

        async def get_file_buffer(requested_checksum, *args, **kwargs):
            reads.append(requested_checksum)
            return await original_get_file_buffer(
                requested_checksum, *args, **kwargs
            )

        monkeypatch.setattr(client, "get_file_buffer", get_file_buffer)
        monkeypatch.setattr(buffer_remote, "_read_folders_clients", [client])

        result = asyncio.run(
            evaluate_expression_async(
                key.input_checksum,
                key.path,
                key.input_celltype,
                key.celltype,
            )
        )
        assert buffer_path.is_file()
        assert reads == [checksum]
        assert result == checksum
        assert result.resolve("str") == "async-buffer-directory-witness"

    assert not Path(directory).exists()
