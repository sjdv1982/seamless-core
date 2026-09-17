"""Contract tests for scalar reference parsing and virtual checksums."""

import asyncio
import importlib
import sys
import types

import pytest

from seamless import Buffer
import seamless.checksum_class as checksum_class
from seamless.checksum.hash_type import (
    HashType,
    get_hash_type_cache,
    register_hash_type_for_buffer,
)
from seamless.checksum.hash_type_validation import (
    HashTypeValidationError,
    validate_deserializable_as,
)
from seamless.checksum.parse_buffer import _parse_buffer


def _serialized(raw):
    buffer = Buffer(raw)
    return buffer, buffer.get_checksum()


@pytest.mark.parametrize(
    "raw,celltype",
    [
        (b"x =", "python"),
        (b"value: [", "yaml"),
    ],
)
def test_invalid_text_celltypes_raise_hashtype_validation_error(raw, celltype):
    buffer, checksum = _serialized(raw)

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        _parse_buffer(buffer, checksum, celltype)


def test_failed_ipython_validation_raises_hashtype_validation_error(monkeypatch):
    parse_buffer_module = importlib.import_module("seamless.checksum.parse_buffer")
    buffer, checksum = _serialized(b"invalid ipython")

    def reject_ipython(_text):
        raise SyntaxError

    monkeypatch.setattr(parse_buffer_module, "ipython2python", reject_ipython)
    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        _parse_buffer(buffer, checksum, "ipython")


def test_nonhex_checksum_value_raises_hashtype_validation_error():
    buffer, checksum = _serialized(b"z" * 64)

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        _parse_buffer(buffer, checksum, "checksum")


@pytest.mark.parametrize(
    "raw,celltype,expected",
    [
        (b"4.5", "int", 4),
        (b'"4.5"', "int", 4),
        (b"123456789012345678901", "int", 123456789012345683968),
        (b'"12345678901234567890"', "int", 12345678901234567890),
        (b"true", "str", "True"),
        (b"true\n", "bool", True),
        (b"null", "int", None),
        (b"null", "str", None),
        (b"null", "plain", None),
    ],
)
def test_parse_buffer_scalars(raw, celltype, expected):
    buffer, checksum = _serialized(raw)
    assert _parse_buffer(buffer, checksum, celltype) == expected


@pytest.mark.parametrize("raw", [b'"nan"', b'"inf"'])
@pytest.mark.parametrize("celltype", ["int", "float"])
def test_nonfinite_numeric_strings_are_rejected_by_hashtype_and_parser(raw, celltype):
    buffer, checksum = _serialized(raw)
    hash_type = HashType.from_buffer(buffer)
    assert not hash_type.deserializable_as(celltype, checksum=checksum)
    with pytest.raises(HashTypeValidationError):
        validate_deserializable_as(checksum, celltype, buffer=buffer)
    with pytest.raises(ValueError):
        _parse_buffer(buffer, checksum, celltype)


@pytest.mark.parametrize(
    "raw,celltype",
    [
        (b"[1]", "str"),
        (b"42", "bool"),
        (b" true ", "bool"),
        (b" null ", "str"),
    ],
)
def test_parse_buffer_rejects_invalid_scalar_reinterpretations(raw, celltype):
    buffer, checksum = _serialized(raw)
    with pytest.raises(ValueError):
        _parse_buffer(buffer, checksum, celltype)


@pytest.mark.parametrize("known_hash_type", [False, True])
def test_true_as_int_is_rejected_with_and_without_known_hashtype(known_hash_type):
    buffer, checksum = _serialized(b"true")
    if known_hash_type:
        register_hash_type_for_buffer(checksum, buffer)
        with pytest.raises(HashTypeValidationError):
            validate_deserializable_as(checksum, "int", buffer=buffer)
    else:
        get_hash_type_cache().pop(checksum, None)
    with pytest.raises(ValueError):
        _parse_buffer(buffer, checksum, "int")


class _FailingBufferCache:
    def get(self, checksum):
        raise AssertionError("virtual resolution must not read the buffer cache")


async def _failing_remote_get_buffer(*args, **kwargs):
    raise AssertionError("virtual resolution must not fetch a remote buffer")


def _patch_all_fetches_to_fail(monkeypatch):
    monkeypatch.setattr(checksum_class, "get_buffer_cache", lambda: _FailingBufferCache())
    remote_package = types.ModuleType("seamless_remote")
    remote_module = types.ModuleType("seamless_remote.buffer_remote")
    remote_module.get_buffer = _failing_remote_get_buffer
    remote_package.buffer_remote = remote_module
    monkeypatch.setitem(sys.modules, "seamless_remote", remote_package)
    monkeypatch.setitem(sys.modules, "seamless_remote.buffer_remote", remote_module)


@pytest.mark.parametrize(
    "raw,celltype,expected",
    [
        (b"null", "plain", None),
        (b"null\n", "plain", None),
        (b"true", "bool", True),
        (b"true\n", "bool", True),
        (b"false", "bool", False),
        (b"false\n", "bool", False),
    ],
)
def test_virtual_checksum_resolution_never_fetches(raw, celltype, expected, monkeypatch):
    buffer, checksum = _serialized(raw)
    _patch_all_fetches_to_fail(monkeypatch)
    assert checksum.resolve(celltype) == expected
    assert asyncio.run(checksum.resolution(celltype)) == expected


def test_nonbool_checksum_as_bool_is_rejected_without_fetch(monkeypatch):
    buffer, checksum = _serialized(b"42")
    _patch_all_fetches_to_fail(monkeypatch)
    with pytest.raises(ValueError):
        checksum.resolve("bool")
    with pytest.raises(ValueError):
        asyncio.run(checksum.resolution("bool"))
