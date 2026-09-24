"""Contract tests for contracts/hashtype.md: when it runs, lookup order, upload, envelope."""

from __future__ import annotations

import asyncio

import pytest

from seamless import Buffer, Checksum
from seamless.caching import buffer_writer
from seamless.checksum import hash_type as ht
from seamless.checksum.hash_type import (
    HashType,
    Kind,
    Length,
    from_buffer,
    get_hash_type,
    get_hash_type_cache,
    get_hash_type_remote,
    pack,
)
from seamless.checksum.hash_type_validation import (
    HashTypeValidationError,
    ensure_hash_type,
    ensure_hash_type_async,
)

from helpers.fake_remotes import install_fake_remotes


@pytest.fixture
def remote(monkeypatch):
    rows: dict[str, int] = {}
    calls: list[str] = []
    install_fake_remotes(monkeypatch, {}, {}, calls, hash_type_rows=rows)
    buffer_writer.flush()
    calls.clear()
    get_hash_type_cache().clear()
    yield rows, calls
    buffer_writer.flush()
    get_hash_type_cache().clear()


# --------------------------------------------------------------------------
# When it runs
# --------------------------------------------------------------------------


def test_cached_calculate_checksum_sync_registers_hash_type(remote):
    """§Producer, *When it runs*: cached_calculate_checksum_sync."""
    from seamless.checksum.cached_calculate_checksum import cached_calculate_checksum_sync

    raw = b'[1, 2, "contract-sync"]'
    checksum = cached_calculate_checksum_sync(Buffer(raw))
    assert get_hash_type(checksum) == from_buffer(raw)


def test_cached_calculate_checksum_async_registers_hash_type(remote):
    """§Producer, *When it runs*: cached_calculate_checksum."""
    from seamless.checksum.cached_calculate_checksum import cached_calculate_checksum

    raw = b'{"contract": "async"}'
    checksum = asyncio.run(cached_calculate_checksum(Buffer(raw)))
    assert get_hash_type(checksum) == from_buffer(raw)


def test_registration_queues_an_upload(remote):
    """§Storage, *Upload*: a local change is sent with database_remote.set_hash_type."""
    rows, calls = remote
    raw = b'"contract-upload"'
    checksum = Buffer(raw).get_checksum()
    buffer_writer.flush()
    assert rows[checksum.hex()] == from_buffer(raw).word
    assert calls.count("database:set_hash_type") == 1


# --------------------------------------------------------------------------
# Lookup order
# --------------------------------------------------------------------------


def test_get_hash_type_is_local_cache_only(remote):
    """§Lookup: get_hash_type(cs) -> local cache only."""
    rows, calls = remote
    checksum = Checksum("1" * 64)
    rows[checksum.hex()] = pack(Kind.RAW_TEXT, Length.SHORT)
    assert get_hash_type(checksum) is None
    assert calls == []


def test_ensure_hash_type_sync_with_buffer_classifies_without_database(remote):
    """§Lookup: sync ensure_hash_type with a buffer classifies it; no DB query."""
    rows, calls = remote
    raw = b'{"contract": "ensure-buffer"}'
    buffer = Buffer(raw)
    checksum = Checksum(buffer.get_checksum())
    get_hash_type_cache().clear()
    # A looser, legal DB row that a DB-first lookup would return.
    buffer_writer.flush()
    rows[checksum.hex()] = pack(Kind.JSON_UNTESTED, from_buffer(raw).length)
    calls.clear()

    result = ensure_hash_type(checksum, buffer=buffer)

    assert result == from_buffer(raw)
    assert "database:get_hash_type" not in calls


def test_ensure_hash_type_async_prefers_database_over_buffer(remote):
    """§Lookup: ensure_hash_type_async -> local cache, then database, then buffer."""
    rows, calls = remote
    raw = b'{"contract": "ensure-async"}'
    buffer = Buffer(raw)
    checksum = Checksum(buffer.get_checksum())
    get_hash_type_cache().clear()
    looser = pack(Kind.JSON_UNTESTED, from_buffer(raw).length)
    buffer_writer.flush()
    rows[checksum.hex()] = looser
    calls.clear()

    result = asyncio.run(ensure_hash_type_async(checksum, buffer=buffer))

    assert calls[:1] == ["database:get_hash_type"]
    assert result.word == looser


def test_ensure_hash_type_async_classifies_buffer_when_database_has_nothing(remote):
    rows, calls = remote
    raw = b'{"contract": "ensure-async-miss"}'
    buffer = Buffer(raw)
    checksum = Checksum(buffer.get_checksum())
    get_hash_type_cache().clear()
    buffer_writer.flush()
    rows.pop(checksum.hex(), None)
    calls.clear()

    result = asyncio.run(ensure_hash_type_async(checksum, buffer=buffer))

    assert "database:get_hash_type" in calls
    assert result == from_buffer(raw)
    assert get_hash_type(checksum) == result


def test_ensure_hash_type_local_cache_first(remote):
    rows, calls = remote
    checksum = Checksum("2" * 64)
    word = pack(Kind.RAW_TEXT, Length.SHORT)
    get_hash_type_cache()[checksum] = word
    rows[checksum.hex()] = pack(Kind.UTF8_UNTESTED, Length.SHORT)
    assert ensure_hash_type(checksum).word == word
    assert asyncio.run(ensure_hash_type_async(checksum)).word == word
    assert asyncio.run(get_hash_type_remote(checksum)).word == word
    assert calls == []


def test_get_hash_type_remote_returns_none_when_database_has_nothing(remote):
    _rows, calls = remote
    checksum = Checksum("3" * 64)
    assert asyncio.run(get_hash_type_remote(checksum)) is None
    assert calls == ["database:get_hash_type"]
    assert get_hash_type(checksum) is None


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def test_a_word_loaded_from_the_database_is_not_uploaded_again(remote):
    """§Storage, *Upload*: a word from get_hash_type_remote enters the cache without upload."""
    rows, calls = remote
    checksum = Checksum("4" * 64)
    word = pack(Kind.RAW_TEXT, Length.SHORT)
    rows[checksum.hex()] = word

    assert asyncio.run(get_hash_type_remote(checksum)).word == word
    buffer_writer.flush()

    assert get_hash_type_cache()[checksum] == word
    assert "database:set_hash_type" not in calls


def test_sync_database_lookup_is_not_uploaded_again(remote):
    rows, calls = remote
    checksum = Checksum("5" * 64)
    word = pack(Kind.JSON_UNTESTED, Length.SHORT)
    rows[checksum.hex()] = word

    assert ensure_hash_type(checksum).word == word
    buffer_writer.flush()

    assert "database:set_hash_type" not in calls


def test_tightening_after_a_database_load_uploads_the_tighter_word(remote):
    rows, calls = remote
    checksum = Checksum("6" * 64)
    loose = pack(Kind.JSON_UNTESTED, Length.SHORT)
    tight = pack(Kind.JSON_OBJECT, Length.SHORT)
    rows[checksum.hex()] = loose
    asyncio.run(get_hash_type_remote(checksum))

    ht.set_hash_type(checksum, tight)
    buffer_writer.flush()

    assert rows[checksum.hex()] == tight
    assert calls.count("database:set_hash_type") == 1


# --------------------------------------------------------------------------
# HashTypeValidationError identity
# --------------------------------------------------------------------------


def test_validation_error_is_a_value_error_and_round_trips_its_envelope_kind():
    """§The false-negative property: HashTypeValidationError is a ValueError subclass;
    error-envelope kind "hash_type_validation"."""
    from seamless.error_envelope import decode_error, encode_error, error_kind

    assert issubclass(HashTypeValidationError, ValueError)
    exc = HashTypeValidationError("Cannot deserialize checksum as requested celltype")
    assert error_kind(exc) == "hash_type_validation"
    envelope = encode_error(exc)
    assert envelope["error"]["kind"] == "hash_type_validation"
    decoded = decode_error(envelope)
    assert type(decoded) is HashTypeValidationError
    assert str(decoded) == str(exc)


# --------------------------------------------------------------------------
# Where HashType is consulted: the conversion engine, before fetching
# --------------------------------------------------------------------------


def test_conversion_engine_refuses_before_fetching_the_source_buffer(remote):
    """§Where consulted: conversion engine validates without a buffer, before fetching."""
    from seamless.checksum.conversion import SeamlessConversionError
    from seamless.checksum.convert import convert_checksum

    raw = b"\xff\xfe\x00contract"
    checksum = Buffer(raw).get_checksum()
    assert get_hash_type(checksum).kind == Kind.RAW_BYTES
    fetched = []

    def get_buffer():
        fetched.append(True)
        return Buffer(raw)

    with pytest.raises(SeamlessConversionError, match="Cannot deserialize"):
        convert_checksum(checksum, "text", "plain", get_buffer)
    assert fetched == []


def test_conversion_engine_bytes_to_mixed_keeps_checksum_on_cached_mixed_word(remote):
    """§Where consulted: `bytes->mixed` keeps the checksum when the cached word says mixed."""
    from seamless.checksum.convert import convert_checksum

    raw = b'{"contract": "bytes-to-mixed"}'
    checksum = Buffer(raw).get_checksum()
    fetched = []

    def get_buffer():
        fetched.append(True)
        return Buffer(raw)

    assert convert_checksum(checksum, "bytes", "mixed", get_buffer) == (checksum, None)
    assert fetched == []
