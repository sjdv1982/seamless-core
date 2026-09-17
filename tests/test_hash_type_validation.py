from __future__ import annotations

import asyncio

import pytest

import seamless.checksum.hash_type_validation as hash_type_validation
from seamless import Buffer, Checksum, Expression
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless.checksum.hash_type import (
    Kind,
    Length,
    get_hash_type_cache,
    get_hash_type_remote,
    pack,
    set_hash_type,
)
from seamless.checksum.hash_type_validation import (
    HashTypeValidationError,
    ensure_hash_type,
    validate_deserializable_as,
)
from seamless.checksum.parse_buffer import _parse_buffer, parse_buffer
from tests.helpers.fake_remotes import install_fake_remotes


def test_buffer_deserialization_rejects_impossible_celltype_before_parse():
    buffer = Buffer(b"\xff\xfe\x00")

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        buffer.get_value("text")


def test_async_buffer_deserialization_rejects_impossible_celltype_before_parse():
    async def main():
        buffer = Buffer(b'{"a": 1}')
        await buffer.get_checksum_async()
        with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
            await buffer.get_value_async("binary")

    asyncio.run(main())


def test_expression_path_capability_rejects_before_materialization():
    checksum = Buffer([1, 2, 3], "plain").get_checksum()
    expression = Expression(checksum, "missing", input_celltype="plain", celltype="plain")

    with pytest.raises(HashTypeValidationError, match="requires MAP capability"):
        expression.compute()


@pytest.mark.parametrize("raw", [b"true", b"false", b"null"])
def test_json_constants_remain_permissive_for_positional_path_validation(raw):
    buffer = Buffer(raw)
    checksum = buffer.get_checksum()

    hash_type = hash_type_validation.validate_expression(
        checksum,
        buffer=buffer,
        source_celltype="plain",
        path_steps=(("item", 0),),
        target_celltype="plain",
    )

    assert hash_type.kind == Kind.JSON_STRING


def test_identity_expression_keeps_validity_gate_for_overlong_numbers():
    checksum = Buffer(b"1" * 1001).get_checksum()
    expression = Expression(checksum, "", input_celltype="float", celltype="float")

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        expression.compute()


@pytest.mark.parametrize(
    "celltype,value",
    [
        ("plain", {"answer": 42}),
        ("str", "hello"),
        ("int", 42),
        ("mixed", {"answer": 42}),
    ],
)
def test_empty_path_expression_can_convert_valid_sources_to_checksum(celltype, value):
    checksum = Buffer(value, celltype).get_checksum()
    expression = Expression(
        checksum, "", input_celltype=celltype, celltype="checksum"
    )

    assert expression.run() == checksum


def test_ensure_hash_type_consults_database(monkeypatch):
    checksum = Checksum("7" * 64)
    word = pack(Kind.UTF8_UNTESTED, Length.SHORT)
    rows = {checksum.hex(): word}
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, hash_type_rows=rows)
    get_hash_type_cache().clear()

    result = ensure_hash_type(checksum)

    assert result.word == word
    assert calls == ["database:get_hash_type"]
    assert get_hash_type_cache()[checksum] == word


def test_ensure_hash_type_in_running_loop_does_not_block(monkeypatch):
    checksum = Checksum("8" * 64)
    word = pack(Kind.UTF8_UNTESTED, Length.SHORT)
    rows = {checksum.hex(): word}
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, hash_type_rows=rows)
    get_hash_type_cache().clear()

    async def main():
        return ensure_hash_type(checksum)

    assert asyncio.run(main()) is None
    assert calls == []


def test_ensure_hash_type_classifies_buffer_in_running_loop(monkeypatch):
    raw = b'{"a": 1}'
    checksum = Checksum(calculate_checksum(raw))
    buffer = Buffer(raw)
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls)
    get_hash_type_cache().pop(checksum, None)

    async def main():
        return ensure_hash_type(checksum, buffer=buffer)

    result = asyncio.run(main())

    assert result.kind == Kind.JSON_OBJECT
    assert get_hash_type_cache()[checksum] == result.word
    assert "database:get_hash_type" not in calls


def test_async_parse_classifies_unregistered_buffer_inside_running_loop():
    raw = b"{}"
    checksum = Checksum(calculate_checksum(raw))
    buffer = Buffer(raw)
    get_hash_type_cache().pop(checksum, None)

    async def main():
        return await parse_buffer(buffer, checksum, "plain", copy=True)

    assert asyncio.run(main()) == {}
    assert checksum in get_hash_type_cache()


def test_parse_buffer_reports_missing_hash_type_as_runtime_error(monkeypatch):
    raw = b"{}"
    buffer = Buffer(raw)
    checksum = Checksum(calculate_checksum(raw))
    monkeypatch.setattr(
        hash_type_validation,
        "validate_deserializable_as",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="did not classify the buffer"):
        _parse_buffer(buffer, checksum, "plain")


def test_validation_with_database_only_hash_type(monkeypatch):
    checksum = Checksum("9" * 64)
    word = pack(Kind.UTF8_UNTESTED, Length.SHORT)
    rows = {checksum.hex(): word}
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, hash_type_rows=rows)
    get_hash_type_cache().clear()

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        validate_deserializable_as(checksum, "binary")

    get_hash_type_cache().clear()
    assert validate_deserializable_as(checksum, "text").word == word


def test_database_word_enters_cache_and_then_conflicts(monkeypatch):
    checksum = Checksum("a" * 64)
    stored = pack(Kind.RAW_TEXT, Length.SHORT)
    contradictory = pack(Kind.RAW_BYTES, Length.SHORT)
    rows = {checksum.hex(): stored}
    install_fake_remotes(monkeypatch, {}, {}, [], hash_type_rows=rows)
    get_hash_type_cache().clear()

    loaded = asyncio.run(get_hash_type_remote(checksum))
    assert loaded.word == stored
    with pytest.raises(ValueError, match="Conflicting HashType"):
        set_hash_type(checksum, contradictory)
    assert get_hash_type_cache()[checksum] == stored
