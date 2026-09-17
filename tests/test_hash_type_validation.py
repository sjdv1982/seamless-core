from __future__ import annotations

import asyncio

import pytest

from seamless import Buffer, Checksum, Expression
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


def test_identity_expression_keeps_validity_gate_for_overlong_numbers():
    checksum = Buffer(b"1" * 1001).get_checksum()
    expression = Expression(checksum, "", input_celltype="float", celltype="float")

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        expression.compute()


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
