from __future__ import annotations

import pytest

from seamless import Buffer
from seamless.checksum.cached_calculate_checksum import checksum_cache
from seamless.checksum.hash_type import (
    DType,
    Flag,
    HashType,
    Kind,
    Length,
    Rank,
    capabilities,
    deserializable_as,
    get_hash_type,
    get_hash_type_cache,
)

from helpers.expression_hashtype_cases import iter_hashtype_witnesses


WITNESSES = tuple(iter_hashtype_witnesses())


def _witness_id(witness):
    return f"{witness.name}|hash_type={witness.expected_hash_type}"


@pytest.mark.parametrize("witness", WITNESSES, ids=_witness_id)
def test_from_buffer_matches_witness_hash_type(witness):
    hash_type = HashType.from_buffer(witness.raw_buffer)

    assert hash_type.word == witness.expected_hash_type
    assert hash_type.mic == HashType.unpack(witness.expected_hash_type).mic


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checksum": "a" * 64},
        {"value": "ignored"},
        {"celltype": "text"},
        {"semantic": True},
    ],
)
def test_from_buffer_rejects_retired_keywords(kwargs):
    with pytest.raises(TypeError):
        HashType.from_buffer(b"x = 1\n", **kwargs)


def test_checksum_calculation_populates_hash_type_cache():
    get_hash_type_cache().clear()
    witness = next(w for w in WITNESSES if w.name == "json_object_short")

    checksum = Buffer(witness.raw_buffer).get_checksum()

    assert checksum == witness.source_checksum
    assert get_hash_type(checksum).word == witness.expected_hash_type


def test_hash_type_cache_survives_checksum_buffer_lru_eviction():
    get_hash_type_cache().clear()
    witness = next(w for w in WITNESSES if w.name == "numpy_numeric_d1_medium")
    checksum = Buffer(witness.raw_buffer).get_checksum()
    checksum_cache.clear()

    assert checksum_cache.get(checksum) is None
    assert get_hash_type(checksum).word == witness.expected_hash_type


def test_known_checksum_buffer_construction_registers_hash_type():
    get_hash_type_cache().clear()
    witness = next(w for w in WITNESSES if w.name == "mixed_object_medium")

    Buffer(witness.raw_buffer, checksum=witness.source_checksum)

    assert get_hash_type(witness.source_checksum).word == witness.expected_hash_type


def test_pure_json_is_not_mixed_format_mixed():
    hash_type = HashType.from_buffer(b'{"a": 1}')

    assert hash_type.kind == Kind.JSON_OBJECT
    assert hash_type.kind != Kind.MIXED_OBJECT
    assert hash_type.deserializable_as("mixed", checksum="a" * 64)


def test_hash_type_well_formedness_invariants():
    for witness in WITNESSES:
        hash_type = HashType.unpack(witness.expected_hash_type)
        if hash_type.kind != Kind.NUMPY:
            assert hash_type.dtype == DType.NA
            assert hash_type.rank == Rank.SCALAR
        if hash_type.flags & Flag.NUMPY_BYTES:
            assert hash_type.kind == Kind.NUMPY
            assert hash_type.dtype == DType.NONNUMERIC
            assert hash_type.rank == Rank.SCALAR
        if hash_type.flags & Flag.NUMERIC_SCALAR:
            assert hash_type.kind in (Kind.JSON_NUMBER, Kind.JSON_STRING)


def test_query_methods_cover_representative_capabilities():
    checksum = "a" * 64
    text = HashType(Kind.RAW_TEXT, Length.SHORT)
    raw_bytes = HashType(Kind.RAW_BYTES, Length.SHORT)
    json_object = HashType(Kind.JSON_OBJECT, Length.SHORT)
    json_string = HashType(Kind.JSON_STRING, Length.SHORT)
    number = HashType(Kind.JSON_NUMBER, Length.SHORT, flags=Flag.NUMERIC_SCALAR)
    numpy_vector = HashType(Kind.NUMPY, Length.MEDIUM, DType.NUMERIC, Rank.D1)
    structured_scalar = HashType(
        Kind.NUMPY, Length.MEDIUM, DType.STRUCTURED, Rank.SCALAR
    )

    assert deserializable_as(raw_bytes, "bytes", checksum=checksum)
    assert not deserializable_as(raw_bytes, "text", checksum=checksum)
    assert text.deserializable_as("text", checksum=checksum)
    assert json_object.deserializable_as("plain", checksum=checksum)
    assert json_object.deserializable_as("mixed", checksum=checksum)
    assert not json_object.deserializable_as("binary", checksum=checksum)
    assert number.deserializable_as("float", checksum=checksum)
    assert json_string.capabilities("str") == {"SEQ"}
    assert capabilities(json_object, "plain") == {"MAP"}
    assert capabilities(numpy_vector, "binary") == {"SEQ"}
    assert capabilities(structured_scalar, "binary") == {"MAP"}
    assert numpy_vector.has_numeric_items("binary") is True
    assert json_object.has_slicing("plain") is False
