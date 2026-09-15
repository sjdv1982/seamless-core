"""Local information refinement and conservative HashType validation."""

import asyncio

import pytest

from seamless import Checksum
from seamless.checksum import hash_type as ht
from seamless.checksum.hash_type import DType, Flag, HashType, Kind, Length, Rank
from seamless.checksum.hash_type_validation import (
    HashTypeValidationError,
    conversion_feasible,
    validate_deserializable_as,
    validate_deserializable_as_async,
    validate_expression,
)


@pytest.fixture(autouse=True)
def isolated_hash_type_cache(monkeypatch):
    monkeypatch.setattr(ht, "_hash_type_cache", {})


CHECKSUM = Checksum("a" * 64)
UNTESTED = (Kind.UNTESTED, Kind.UTF8_UNTESTED, Kind.JSON_UNTESTED)


@pytest.mark.parametrize("kind,number,utf8,json,mic", [
    (Kind.UNTESTED, 9, False, False, "bytes"),
    (Kind.UTF8_UNTESTED, 10, True, False, "text"),
    (Kind.JSON_UNTESTED, 11, True, True, "plain"),
])
def test_untested_encoding_and_membership(kind, number, utf8, json, mic):
    assert int(kind) == number
    for length in Length:
        value = HashType(kind, length)
        assert ht.is_valid_word(value.word)
        assert HashType.unpack(value.word) == value
        assert value.is_utf8 is utf8
        assert value.is_json is json
        assert value.is_untested
        assert value.mic == mic
        for flag in Flag:
            assert not ht.is_valid_word(HashType(kind, length, flags=flag).word)
        assert not ht.is_valid_word(HashType(kind, length, dtype=DType.NUMERIC).word)
        assert not ht.is_valid_word(HashType(kind, length, rank=Rank.D1).word)


@pytest.mark.parametrize("kind", UNTESTED)
@pytest.mark.parametrize("celltype,outcomes", [
    ("bytes", (True, True, True)),
    ("text", (None, True, True)),
    ("python", (None, True, True)),
    ("ipython", (None, True, True)),
    ("yaml", (None, True, True)),
    ("plain", (None, None, True)),
    ("mixed", (None, None, True)),
    ("str", (None, None, None)),
    ("int", (None, None, None)),
    ("float", (None, None, None)),
    ("binary", (None, False, False)),
    ("checksum", (None, None, False)),
    ("bool", (False, False, False)),
])
def test_three_outcome_deserialization(kind, celltype, outcomes):
    value = HashType(kind, Length.EQ64)
    assert value.deserializable_as(celltype, checksum=CHECKSUM) is outcomes[UNTESTED.index(kind)]


@pytest.mark.parametrize("kind", UNTESTED)
def test_untested_known_length_and_checksum_rejections(kind):
    value = HashType(kind, Length.LONG)
    for celltype in ("int", "float", "checksum"):
        assert value.deserializable_as(celltype, checksum=CHECKSUM) is False
    assert value.deserializable_as("bool", checksum=ht.CHECKSUM_TRUE) is True
    assert value.deserializable_as("binary", checksum=ht.CHECKSUM_NULL) is True


@pytest.mark.parametrize("kind", list(Kind))
def test_all_kind_branches_tighten_from_their_ancestors(kind):
    flags = Flag.NUMERIC_SCALAR if kind == Kind.JSON_NUMBER else Flag(0)
    dtype = DType.NUMERIC if kind == Kind.NUMPY else DType.NA
    concrete = HashType(kind, Length.SHORT, dtype=dtype, flags=flags)
    ancestors = [HashType(Kind.UNTESTED, Length.SHORT)]
    if concrete.is_utf8:
        ancestors.append(HashType(Kind.UTF8_UNTESTED, Length.SHORT))
    if concrete.is_json:
        ancestors.append(HashType(Kind.JSON_UNTESTED, Length.SHORT))
    for value in ancestors + [concrete]:
        ht.set_hash_type(CHECKSUM, value)
        assert ht.get_hash_type(CHECKSUM) == value
    for value in [concrete] + ancestors[::-1]:
        ht.set_hash_type(CHECKSUM, value)
        assert ht.get_hash_type(CHECKSUM) == concrete


@pytest.mark.parametrize("left,right", [
    (HashType(Kind.UNTESTED, Length.SHORT), HashType(Kind.UNTESTED, Length.LONG)),
    (HashType(Kind.JSON_UNTESTED, Length.SHORT), HashType(Kind.RAW_TEXT, Length.SHORT)),
    (HashType(Kind.UTF8_UNTESTED, Length.SHORT), HashType(Kind.RAW_BYTES, Length.SHORT)),
    (HashType(Kind.JSON_OBJECT, Length.SHORT), HashType(Kind.JSON_ARRAY, Length.SHORT)),
    (HashType(Kind.JSON_STRING, Length.SHORT), HashType(Kind.JSON_STRING, Length.SHORT, flags=Flag.NUMERIC_SCALAR)),
    (HashType(Kind.RAW_TEXT, Length.SHORT), HashType(Kind.RAW_TEXT, Length.SHORT, flags=Flag.SEMANTIC)),
    (HashType(Kind.NUMPY, Length.LONG, DType.NUMERIC), HashType(Kind.NUMPY, Length.LONG, DType.NONNUMERIC)),
    (HashType(Kind.NUMPY, Length.LONG, DType.NUMERIC), HashType(Kind.NUMPY, Length.LONG, DType.NUMERIC, Rank.D1)),
])
def test_conflicting_writes_are_logged_and_preserve_first_word(left, right, caplog):
    for stored, incoming in ((left, right), (right, left)):
        ht.get_hash_type_cache().clear()
        ht.set_hash_type(CHECKSUM, stored)
        with pytest.raises(ValueError, match="Conflicting HashType"):
            ht.set_hash_type(CHECKSUM, incoming)
        assert ht.get_hash_type(CHECKSUM) == stored
    assert "Conflicting HashType" in caplog.text


def test_buffer_registration_tightens_numeric_information():
    ht.set_hash_type(CHECKSUM, HashType(Kind.JSON_UNTESTED, Length.SHORT))
    result = ht.register_hash_type_for_buffer(CHECKSUM, b'"12.5"')
    assert result.kind == Kind.JSON_STRING
    assert result.flags & Flag.NUMERIC_SCALAR
    assert ht.get_hash_type(CHECKSUM) == result


@pytest.mark.parametrize("kind", UNTESTED)
def test_local_validation_accepts_unknown_and_checks_proven_negatives(kind):
    value = HashType(kind, Length.SHORT)
    ht.set_hash_type(CHECKSUM, value)
    assert validate_deserializable_as(CHECKSUM, "float") == value
    assert asyncio.run(validate_deserializable_as_async(CHECKSUM, "float")) == value
    assert validate_expression(
        CHECKSUM, buffer=None, source_celltype="mixed",
        path_steps=(("item", "key"), ("slice", (None, None, None))),
        target_celltype="float",
    ) == value
    assert conversion_feasible(value, "plain", "float", checksum=CHECKSUM) is None
    if kind != Kind.UNTESTED:
        with pytest.raises(HashTypeValidationError):
            validate_deserializable_as(CHECKSUM, "binary")


@pytest.mark.parametrize("kind", UNTESTED)
def test_local_validation_rejects_long_numeric_even_when_untested(kind):
    value = HashType(kind, Length.LONG)
    ht.set_hash_type(CHECKSUM, value)
    with pytest.raises(HashTypeValidationError):
        validate_deserializable_as(CHECKSUM, "float")
    with pytest.raises(HashTypeValidationError):
        asyncio.run(validate_deserializable_as_async(CHECKSUM, "float"))
    assert conversion_feasible(value, "plain", "float", checksum=CHECKSUM) is False
