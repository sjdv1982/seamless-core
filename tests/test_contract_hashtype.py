"""Contract tests for contracts/hashtype.md: word, producer, queries, validation.

Complements test_hash_type_contract.py / test_hash_type_tightening.py /
test_hash_type_validation.py with the rules those files leave uncovered or pin
with a single case, and with an oracle-backed check of the central rule
(*The false-negative property*): HashType may answer "unknown" but never
rejects a valid deserialization or conversion.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import itertools
import json
import re

import numpy as np
import orjson
import pytest

from seamless import Buffer, Checksum, Expression
from seamless.checksum import hash_type_validation as htv
from seamless.checksum.hash_type import (
    CHECKSUM_FALSE,
    CHECKSUM_NULL,
    CHECKSUM_TRUE,
    HASH_TYPE_CELLTYPES,
    MAGIC_SEAMLESS_MIXED,
    DType,
    Flag,
    HashType,
    Kind,
    Length,
    Rank,
    deserializable_as,
    from_buffer,
    is_valid_word,
    unpack,
)
from seamless.checksum.hash_type_validation import (
    HashTypeValidationError,
    conversion_feasible,
    ensure_hash_type,
    validate_expression,
    validate_expression_async,
)
from seamless.checksum.conversion import (
    conversion_chain,
    conversion_equivalent,
    conversion_forbidden,
)
from seamless.checksum.deep import DEEP_CELLTYPES

from helpers.expression_hashtype_cases import iter_hashtype_witnesses
from helpers.fake_remotes import install_fake_remotes


CHECKSUM = "a" * 64
THE_13 = tuple(sorted(HASH_TYPE_CELLTYPES))
# "The four structural names": three deep celltypes, plus `module`, which is NOT deep.
THE_THREE_DEEP = ("deepcell", "deepfolder", "folder")
THE_FOUR_STRUCTURAL = THE_THREE_DEEP + ("module",)
# "either null checksum" (canonical null, with and without trailing newline).
BOTH_NULLS = (CHECKSUM_NULL, Checksum(hashlib.sha256(b"null\n").digest()))
# §conversion_feasible: exactly the 16 pairs python/ipython <-> yaml,
# python/ipython -> int/float/bool and int/float/bool -> python/ipython.
FORBIDDEN_16 = sorted(
    {(code, "yaml") for code in ("python", "ipython")}
    | {("yaml", code) for code in ("python", "ipython")}
    | {(code, scalar) for code in ("python", "ipython") for scalar in ("int", "float", "bool")}
    | {(scalar, code) for code in ("python", "ipython") for scalar in ("int", "float", "bool")}
)
LEGAL_PAIRS = [
    (s, t) for s in THE_13 for t in THE_13 if s != t and (s, t) not in FORBIDDEN_16
]


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    # Sync validation outside an event loop may consult the database; keep it fake.
    install_fake_remotes(monkeypatch, {}, {}, [])


# --------------------------------------------------------------------------
# The domain: exactly the 13 celltypes
# --------------------------------------------------------------------------


def test_the_domain_is_exactly_the_13_celltypes():
    """hashtype.md §Queries / *The domain*: the 13, and only the 13."""
    assert set(THE_13) == {
        "binary", "mixed", "text", "python", "ipython", "plain", "yaml",
        "str", "bytes", "int", "float", "bool", "checksum",
    }


@pytest.mark.parametrize("celltype", ("deepcell", "deepfolder", "folder", "module", "nonsense"))
def test_outside_the_domain_is_a_value_error_not_false(celltype):
    """§Queries: a name outside the 13 raises ValueError; never False/None/empty."""
    word = HashType(Kind.RAW_BYTES, Length.SHORT)  # would answer False for most names
    with pytest.raises(ValueError):
        deserializable_as(word, celltype, checksum=CHECKSUM)
    with pytest.raises(ValueError):
        word.capabilities(celltype)
    with pytest.raises(ValueError):
        word.has_numeric_items(celltype)
    with pytest.raises(ValueError):
        word.has_string_items(celltype)


@pytest.mark.parametrize("celltype", THE_FOUR_STRUCTURAL + ("nonsense",))
@pytest.mark.parametrize("position", ("source", "target"))
@pytest.mark.parametrize(
    "checksum", (CHECKSUM,) + BOTH_NULLS, ids=("plain-checksum", "null", "null-nl")
)
def test_conversion_feasible_domain_is_checked_even_for_a_null_checksum(
    celltype, position, checksum
):
    """§Queries: `conversion_feasible` needs source **and** target in the 13; outside
    it is ValueError. The null shortcut (rule 3) never answers a question that was
    not HashType's to ask."""
    source, target = (celltype, "plain") if position == "source" else ("plain", celltype)
    word = HashType(Kind.JSON_OBJECT, Length.SHORT)
    with pytest.raises(ValueError):
        conversion_feasible(word, source, target, checksum=checksum)


# --------------------------------------------------------------------------
# The word
# --------------------------------------------------------------------------


@pytest.mark.parametrize("word", (-1, 1 << 13, 1 << 20))
def test_out_of_range_words_are_invalid(word):
    """§The word: a HashType is a 13-bit integer."""
    assert is_valid_word(word) is False


@pytest.mark.parametrize("word", ("5", 5.0, None))
def test_non_integer_words_are_invalid(word):
    assert is_valid_word(word) is False


@pytest.mark.parametrize("word", ("5", 5.0, None, -1, 1 << 13, 1 << 20))
def test_unpack_raises_for_non_integer_and_out_of_range_words(word):
    """§The word: `unpack` raises for a non-integer and for an integer outside
    [0, 2**13). The exception classes are unspecified, so only "raises" is pinned."""
    with pytest.raises(Exception):
        unpack(word)
    with pytest.raises(Exception):
        HashType.unpack(word)


DERIVED = {
    # kind: (is_utf8, is_json, is_untested, is_numpy, is_mixed, mic)
    Kind.RAW_BYTES: (False, False, False, False, False, "bytes"),
    Kind.NUMPY: (False, False, False, True, False, "binary"),
    Kind.MIXED_OBJECT: (False, False, False, False, True, "mixed"),
    Kind.MIXED_ARRAY: (False, False, False, False, True, "mixed"),
    Kind.RAW_TEXT: (True, False, False, False, False, "text"),
    Kind.JSON_OBJECT: (True, True, False, False, False, "plain"),
    Kind.JSON_ARRAY: (True, True, False, False, False, "plain"),
    Kind.JSON_STRING: (True, True, False, False, False, "str"),
    Kind.JSON_NUMBER: (True, True, False, False, False, "float"),
    Kind.UNTESTED: (False, False, True, False, False, "bytes"),
    Kind.UTF8_UNTESTED: (True, False, True, False, False, "text"),
    Kind.JSON_UNTESTED: (True, True, True, False, False, "plain"),
}


@pytest.mark.parametrize("kind", list(Kind), ids=lambda k: k.name)
def test_derived_properties_of_every_kind(kind):
    """§The word, *Derived properties*."""
    assert set(DERIVED) == set(Kind)
    value = HashType(kind, Length.SHORT)
    assert (
        value.is_utf8, value.is_json, value.is_untested,
        value.is_numpy, value.is_mixed, value.mic,
    ) == DERIVED[kind]


# --------------------------------------------------------------------------
# Producer: from_buffer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,length",
    [(0, Length.SHORT), (63, Length.SHORT), (64, Length.EQ64), (65, Length.MEDIUM),
     (1000, Length.MEDIUM), (1001, Length.LONG)],
)
def test_length_is_always_the_byte_length_bucket(size, length):
    """§Producer: Length is the byte-length bucket (<64, =64, 65-1000, >1000)."""
    for raw in (b"\xff" * size, b"x" * size):
        assert from_buffer(raw).length == length


@pytest.mark.parametrize(
    "raw", (b'"inf"', b'"nan"', b'"-Infinity"', b'"1e400"', b'"abc"')
)
def test_json_string_gets_numeric_scalar_only_if_it_parses_as_a_finite_float(raw):
    """§Producer step 4: string -> JSON_STRING (+NUMERIC_SCALAR if finite float)."""
    word = from_buffer(raw)
    assert word.kind == Kind.JSON_STRING
    assert not (word.flags & Flag.NUMERIC_SCALAR)


@pytest.mark.parametrize("raw", (b'"42"', b'"-1.5e3"'))
def test_finite_numeric_json_string_is_flagged(raw):
    word = from_buffer(raw)
    assert word.kind == Kind.JSON_STRING
    assert word.flags & Flag.NUMERIC_SCALAR


@pytest.mark.parametrize("raw", (b"true", b"false", b"null", b"true\n", b"null\n"))
def test_json_constants_are_unflagged_json_strings(raw):
    """§Producer step 4: true/false/null -> JSON_STRING with no flags."""
    word = from_buffer(raw)
    assert word.kind == Kind.JSON_STRING
    assert word.flags == Flag(0)


@pytest.mark.parametrize("raw", (b"42", b"-0", b"1e308", b"3.5\n"))
def test_json_numbers_are_json_number_with_numeric_scalar(raw):
    word = from_buffer(raw)
    assert word.kind == Kind.JSON_NUMBER
    assert word.flags == Flag.NUMERIC_SCALAR


def _mixed_header(form: dict) -> bytes:
    encoded = json.dumps(form).encode()
    return (
        MAGIC_SEAMLESS_MIXED + bytes([4]) + b"pure"
        + len(encoded).to_bytes(4, "little") + encoded
    )


MALFORMED_MIXED_MAGIC = [
    ("magic-only", MAGIC_SEAMLESS_MIXED),
    ("junk-header", MAGIC_SEAMLESS_MIXED + b"\x05abc"),
    ("other-root-type", _mixed_header({"type": "string"})),
    ("non-object-form", _mixed_header(["object"])),
]


@pytest.mark.parametrize(
    "raw", [c[1] for c in MALFORMED_MIXED_MAGIC], ids=[c[0] for c in MALFORMED_MIXED_MAGIC]
)
def test_malformed_mixed_magic_is_raw_bytes_and_has_a_checksum(raw):
    """§Producer step 2: a Seamless-mixed magic whose header does not give an
    object/array root is RAW_BYTES, as for a malformed .npy; the buffer still
    gets a checksum."""
    assert from_buffer(raw).kind == Kind.RAW_BYTES
    checksum = Buffer(raw).get_checksum()
    assert checksum.hex() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("kind", (Kind.MIXED_OBJECT, Kind.MIXED_ARRAY))
def test_seamless_mixed_word_does_not_prove_mixed(kind):
    """§deserializable_as, concrete `mixed`: the producer read only the header,
    so a Seamless-mixed word answers None, never True."""
    assert deserializable_as(HashType(kind, Length.MEDIUM), "mixed", checksum=CHECKSUM) is None


def test_producer_never_emits_untested_kinds():
    """§The word: from_buffer never emits the untested kinds."""
    for _name, raw in _corpus():
        assert not from_buffer(raw).is_untested


# --------------------------------------------------------------------------
# deserializable_as: the rules not pinned elsewhere
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", (Kind.RAW_TEXT, Kind.JSON_OBJECT, Kind.MIXED_OBJECT))
@pytest.mark.parametrize("checksum", (CHECKSUM_TRUE, CHECKSUM_FALSE, CHECKSUM_NULL))
def test_concrete_str_accepts_null_and_boolean_checksums(kind, checksum):
    """§deserializable_as, concrete `str`: ... or a null/boolean checksum."""
    assert deserializable_as(HashType(kind, Length.SHORT), "str", checksum=checksum) is True


def _any_word(kind: Kind, length: Length = Length.LONG) -> HashType:
    """A well-formed word of `kind` (LONG by default: disproves int/float if reached)."""
    if kind == Kind.NUMPY:
        return HashType(kind, length, DType.NONNUMERIC, Rank.D1)
    if kind == Kind.JSON_NUMBER:
        return HashType(kind, length, flags=Flag.NUMERIC_SCALAR)
    return HashType(kind, length)


@pytest.mark.parametrize("kind", list(Kind), ids=lambda k: k.name)
@pytest.mark.parametrize("null", BOTH_NULLS, ids=("null", "null-nl"))
def test_null_checksum_deserializes_as_all_13_whatever_the_word(kind, null):
    """§deserializable_as step 1: either null checksum -> True for every one of the 13,
    evaluated before every other step, so no word can disprove it."""
    word = _any_word(kind)
    assert is_valid_word(word.word)
    for celltype in THE_13:
        assert word.deserializable_as(celltype, checksum=null) is True, celltype


@pytest.mark.parametrize("source,target", FORBIDDEN_16, ids=[f"{s}->{t}" for s, t in FORBIDDEN_16])
def test_null_deserializability_is_not_convertibility(source, target):
    """§deserializable_as step 1: deserializability, not convertibility. A null checksum
    reads as both ends of every forbidden pair; the pair itself is still forbidden
    (the conversion half is pinned by the corresponding conversion test)."""
    word = HashType(Kind.JSON_STRING, Length.SHORT)
    for null in BOTH_NULLS:
        assert word.deserializable_as(source, checksum=null) is True
        assert word.deserializable_as(target, checksum=null) is True
    assert (source, target) in conversion_forbidden


@pytest.mark.parametrize("kind", (Kind.UNTESTED, Kind.UTF8_UNTESTED, Kind.JSON_UNTESTED))
def test_untested_str_is_true_for_boolean_checksums(kind):
    """§deserializable_as step 4: untested `str` -> True for null/boolean checksums."""
    word = HashType(kind, Length.SHORT)
    assert word.deserializable_as("str", checksum=CHECKSUM_TRUE) is True
    assert word.deserializable_as("str", checksum=CHECKSUM_FALSE) is True
    assert word.deserializable_as("str", checksum=CHECKSUM) is None


@pytest.mark.parametrize("kind", (Kind.UNTESTED, Kind.UTF8_UNTESTED, Kind.JSON_UNTESTED))
@pytest.mark.parametrize("length", (Length.SHORT, Length.MEDIUM, Length.LONG))
def test_untested_checksum_celltype_needs_eq64(kind, length):
    """§deserializable_as step 4: `checksum` -> None if EQ64 else False."""
    word = HashType(kind, length)
    assert word.deserializable_as("checksum", checksum=CHECKSUM) is False
    assert HashType(kind, Length.EQ64).deserializable_as("checksum", checksum=CHECKSUM) is None


@pytest.mark.parametrize("celltype", THE_13)
def test_long_concrete_numbers_are_disproved_only_for_int_and_float(celltype):
    """§deserializable_as, concrete `int`/`float`: False if LONG."""
    word = HashType(Kind.JSON_NUMBER, Length.LONG, flags=Flag.NUMERIC_SCALAR)
    result = word.deserializable_as(celltype, checksum=CHECKSUM)
    if celltype in ("int", "float"):
        assert result is False
    elif celltype in ("bytes", "text", "python", "ipython", "yaml", "plain", "str", "mixed"):
        assert result is True


# --------------------------------------------------------------------------
# conversion_feasible: the rule order and the rule-table branches
# --------------------------------------------------------------------------


def test_null_checksum_is_feasible_even_when_the_word_disproves_the_source():
    """§conversion_feasible: a null checksum -> True (on a legal pair; plain has no
    forbidden target)."""
    word = HashType(Kind.RAW_BYTES, Length.SHORT)
    assert word.deserializable_as("plain", checksum=CHECKSUM) is False
    for target in THE_13:
        assert conversion_feasible(word, "plain", target, checksum=CHECKSUM_NULL) is True


@pytest.mark.parametrize("source,target", LEGAL_PAIRS, ids=[f"{s}->{t}" for s, t in LEGAL_PAIRS])
def test_null_checksum_is_true_on_every_legal_pair(source, target):
    """§conversion_feasible rule 3: a null checksum on a legal pair -> True, whatever
    the word says (RAW_BYTES LONG disproves most sources for a non-null checksum)."""
    for kind in Kind:
        word = _any_word(kind)
        for null in BOTH_NULLS:
            assert conversion_feasible(word, source, target, checksum=null) is True


@pytest.mark.parametrize("null", BOTH_NULLS, ids=("null", "null-nl"))
@pytest.mark.parametrize(
    "source,target", FORBIDDEN_16, ids=[f"{s}->{t}" for s, t in FORBIDDEN_16]
)
def test_null_checksum_does_not_make_a_forbidden_pair_feasible(source, target, null):
    """Rule 2: illegal conversions remain illegal, also for a null checksum."""
    word = HashType(Kind.JSON_STRING, Length.SHORT)
    assert conversion_feasible(word, source, target, checksum=null) is False


@pytest.mark.parametrize("celltype", ("plain", "binary", "int", "text"))
def test_source_equal_target_is_true_before_the_source_check(celltype):
    """§conversion_feasible: `source == target` -> True, ahead of deserializable_as."""
    word = HashType(Kind.RAW_BYTES, Length.SHORT)
    assert word.deserializable_as(celltype, checksum=CHECKSUM) is False
    assert conversion_feasible(word, celltype, celltype, checksum=CHECKSUM) is True


def _valid_source(source: str) -> tuple[HashType, Checksum]:
    """A non-null (word, checksum) that deserializable_as(source) answers True for."""
    if source in ("int", "float"):
        return HashType(Kind.JSON_NUMBER, Length.SHORT, flags=Flag.NUMERIC_SCALAR), Checksum(CHECKSUM)
    if source == "bool":
        return HashType(Kind.JSON_STRING, Length.SHORT), CHECKSUM_TRUE
    return HashType(Kind.RAW_TEXT, Length.SHORT), Checksum(CHECKSUM)


def test_forbidden_pairs_are_exactly_the_16():
    """§conversion_feasible: a pair is forbidden iff it is in conversion_forbidden,
    exactly the 16 pairs python/ipython <-> yaml, python/ipython -> int/float/bool,
    int/float/bool -> python/ipython."""
    assert len(FORBIDDEN_16) == 16
    assert set(conversion_forbidden) == set(FORBIDDEN_16)


def test_no_equivalence_or_chain_resolves_to_a_forbidden_pair():
    """§conversion_feasible: no equivalence or chain resolves to a forbidden pair, so
    every other pair of distinct celltypes is legal."""
    for pair, equivalent in conversion_equivalent.items():
        if pair[0] in HASH_TYPE_CELLTYPES and pair[1] in HASH_TYPE_CELLTYPES:
            assert pair not in conversion_forbidden
            assert tuple(equivalent) not in conversion_forbidden, (pair, equivalent)
    for (source, target), middle in conversion_chain.items():
        assert (source, target) not in conversion_forbidden
        assert (source, middle) not in conversion_forbidden, (source, middle, target)
        assert (middle, target) not in conversion_forbidden, (source, middle, target)


@pytest.mark.parametrize(
    "source,target", FORBIDDEN_16, ids=[f"{s}->{t}" for s, t in FORBIDDEN_16]
)
def test_forbidden_pairs_are_false_even_for_a_valid_source(source, target):
    """§conversion_feasible rule 2: a forbidden pair -> False, even where the source
    reads fine (rule 4 would not have refused it)."""
    word, checksum = _valid_source(source)
    assert word.deserializable_as(source, checksum=checksum) is True
    assert conversion_feasible(word, source, target, checksum=checksum) is False


@pytest.mark.parametrize(
    "word,source,target",
    [
        (HashType(Kind.RAW_TEXT, Length.SHORT), "text", "bytes"),  # trivial
        (HashType(Kind.JSON_OBJECT, Length.SHORT), "plain", "mixed"),  # trivial
        (HashType(Kind.RAW_TEXT, Length.SHORT), "text", "plain"),  # reformat
        (HashType(Kind.JSON_OBJECT, Length.SHORT), "plain", "text"),  # reformat
        (HashType(Kind.RAW_BYTES, Length.SHORT), "bytes", "binary"),  # reformat
    ],
)
def test_trivial_and_reformat_pairs_are_true(word, source, target):
    """§conversion_feasible: conversion_trivial or conversion_reformat -> True."""
    assert conversion_feasible(word, source, target, checksum=CHECKSUM) is True


@pytest.mark.parametrize(
    "kind,expected",
    [(Kind.RAW_BYTES, False), (Kind.RAW_TEXT, True), (Kind.UNTESTED, None)],
)
def test_reinterpret_answers_deserializable_as_target(kind, expected):
    """§conversion_feasible: conversion_reinterpret -> deserializable_as(target)."""
    word = HashType(kind, Length.SHORT)
    assert conversion_feasible(word, "bytes", "text", checksum=CHECKSUM) is expected


@pytest.mark.parametrize(
    "kind,expected",
    [(Kind.JSON_OBJECT, False), (Kind.JSON_ARRAY, False), (Kind.JSON_STRING, True),
     (Kind.JSON_NUMBER, True), (Kind.MIXED_ARRAY, True)],
)
def test_mixed_to_str_rule(kind, expected):
    """§conversion_feasible, conversion_possible: mixed->str False only for JSON object/array."""
    flags = Flag.NUMERIC_SCALAR if kind == Kind.JSON_NUMBER else Flag(0)
    word = HashType(kind, Length.SHORT, flags=flags)
    assert conversion_feasible(word, "mixed", "str", checksum=CHECKSUM) is expected


@pytest.mark.parametrize(
    "source,word",
    [
        ("binary", HashType(Kind.NUMPY, Length.SHORT, DType.NUMERIC)),
        ("mixed", HashType(Kind.NUMPY, Length.SHORT, DType.NUMERIC)),
        ("mixed", HashType(Kind.JSON_NUMBER, Length.SHORT, flags=Flag.NUMERIC_SCALAR)),
        ("mixed", HashType(Kind.JSON_OBJECT, Length.SHORT)),
    ],
)
def test_bool_targets_of_possible_conversions_are_none(source, word):
    """§conversion_feasible, conversion_possible: everything else, incl. bool targets -> None."""
    assert conversion_feasible(word, source, "bool", checksum=CHECKSUM) is None


# --------------------------------------------------------------------------
# Expression validation (evaluation-time half)
# --------------------------------------------------------------------------


def _register(raw: bytes) -> tuple[Checksum, HashType]:
    buffer = Buffer(raw)
    checksum = buffer.get_checksum()
    return checksum, ensure_hash_type(checksum, buffer=buffer)


def test_conversion_is_checked_only_for_an_empty_path():
    """§Where consulted: conversion_feasible is asked only where there is no path."""
    checksum, word = _register(b'{"a": 1}')
    with pytest.raises(HashTypeValidationError):
        validate_expression(
            checksum, buffer=None, source_celltype="plain",
            path_steps=(), target_celltype="binary",
        )
    assert validate_expression(
        checksum, buffer=None, source_celltype="plain",
        path_steps=(("item", "a"),), target_celltype="binary",
    ) == word


def test_capability_is_asked_of_the_input_celltype():
    """§Where consulted: the capability check is asked of the **input** celltype."""
    checksum, word = _register(b'"hello"')
    # A JSON string read as str admits SEQ, whatever the target.
    assert validate_expression(
        checksum, buffer=None, source_celltype="str",
        path_steps=(("item", 0),), target_celltype="int",
    ) == word
    with pytest.raises(HashTypeValidationError):
        validate_expression(
            checksum, buffer=None, source_celltype="str",
            path_steps=(("item", "k"),), target_celltype="plain",
        )


def test_any_step_after_a_bytes_item_is_rejected():
    """§capabilities: any further step after a `bytes` item is rejected (the item is an int)."""
    checksum, word = _register(b"\xff\xfe\x00")
    assert validate_expression(
        checksum, buffer=None, source_celltype="bytes",
        path_steps=(("slice", (1, None, None)), ("item", 0)), target_celltype="int",
    ) == word
    with pytest.raises(HashTypeValidationError):
        validate_expression(
            checksum, buffer=None, source_celltype="bytes",
            path_steps=(("item", 0), ("item", 0)), target_celltype="int",
        )


def test_an_item_step_stops_checking_for_json_containers():
    """§capabilities: an item step stops checking (the child is not typed by the root word)."""
    checksum, word = _register(b'{"a": {"b": 1}}')
    # After ("item", "a") nothing is known: a positional step is not disproved.
    assert validate_expression(
        checksum, buffer=None, source_celltype="plain",
        path_steps=(("item", "a"), ("item", 0), ("item", "zz")), target_celltype="plain",
    ) == word


def _npy(value) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def test_numeric_d3plus_array_stops_checking_after_the_first_item():
    """§capabilities: rank-counting applies only below D3PLUS."""
    checksum, word = _register(_npy(np.zeros((2, 2, 2, 2))))
    assert word.rank == Rank.D3PLUS
    assert validate_expression(
        checksum, buffer=None, source_celltype="binary",
        path_steps=tuple(("item", 0) for _ in range(6)), target_celltype="binary",
    ) == word


def test_nonnumeric_d1_array_stops_checking_after_an_item():
    checksum, word = _register(_npy(np.array(["ab", "cd"])))
    assert word.dtype == DType.NONNUMERIC
    assert validate_expression(
        checksum, buffer=None, source_celltype="binary",
        path_steps=(("item", 0), ("item", 0)), target_celltype="binary",
    ) == word


def _forbid_hashtype_queries(monkeypatch):
    def asked(*args, **kwargs):
        raise AssertionError("HashType was asked about a structural name")

    for name in (
        "validate_deserializable_as", "validate_deserializable_as_async",
        "_validate_path_capability", "conversion_feasible",
    ):
        monkeypatch.setattr(htv, name, asked)
    for name in ("deserializable_as", "capabilities"):
        monkeypatch.setattr(HashType, name, asked)


@pytest.mark.parametrize("structural", THE_FOUR_STRUCTURAL)
@pytest.mark.parametrize("position", ("source", "target"))
@pytest.mark.parametrize("mode", ("sync", "async"))
def test_structural_names_are_never_vetted_by_hashtype(monkeypatch, structural, position, mode):
    """§Where consulted, Expression validation: if either celltype is one of the four
    structural names (the three deep celltypes, and `module`), HashType is not asked
    anything; the checksum is only classified."""
    checksum, word = _register(b"\xff\xfe\x00")  # raw bytes: disproved as plain
    assert word.deserializable_as("plain", checksum=checksum) is False
    source, target = (structural, "plain") if position == "source" else ("plain", structural)
    _forbid_hashtype_queries(monkeypatch)
    kwargs = dict(
        buffer=None, source_celltype=source,
        path_steps=(("item", "x"), ("item", 0)), target_celltype=target,
    )
    if mode == "sync":
        result = validate_expression(checksum, **kwargs)
    else:
        result = asyncio.run(validate_expression_async(checksum, **kwargs))
    assert result == word


@pytest.mark.parametrize("position", ("source", "target"))
@pytest.mark.parametrize("mode", ("sync", "async"))
def test_expression_validation_unknown_celltype_is_a_value_error(position, mode):
    """§Where consulted: an unknown celltype name raises ValueError (it is neither one
    of the 13 nor one of the four structural names)."""
    checksum, _ = _register(b'{"a": 1}')
    source, target = ("nonsense", "plain") if position == "source" else ("plain", "nonsense")
    kwargs = dict(buffer=None, source_celltype=source, path_steps=(), target_celltype=target)
    with pytest.raises(ValueError):
        if mode == "sync":
            validate_expression(checksum, **kwargs)
        else:
            asyncio.run(validate_expression_async(checksum, **kwargs))


def test_module_is_a_structural_name_but_not_a_deep_celltype():
    """Intro / §Queries: the four structural names are the three deep celltypes plus
    `module`, which is NOT deep. Every place that decides deep feasibility (the deep
    registry and the conversion engine's deep table, which Expression construction
    also uses) names exactly the three; HashType's structural carve-out names all four."""
    from seamless.checksum import convert

    assert set(THE_THREE_DEEP) == set(DEEP_CELLTYPES)
    assert "module" not in DEEP_CELLTYPES
    assert set(convert._DEEP_CELLTYPES) == set(THE_THREE_DEEP)
    assert "module" not in convert._DEEP_CELLTYPES
    assert set(htv._STRUCTURAL_CELLTYPES) == set(THE_FOUR_STRUCTURAL)
    assert not set(THE_FOUR_STRUCTURAL) & set(HASH_TYPE_CELLTYPES)


# --------------------------------------------------------------------------
# The false-negative property, against the reference engine as oracle
# --------------------------------------------------------------------------


def _extra_buffers():
    return [
        b'"inf"', b'"nan"', b'"1e400"', b"1e308", b"-0", b'""', b"[]", b"{}",
        b'"abc"', b"3.5", b"17", b"true\n", b"false\n", b"null\n", b'"true"',
        b"x = 1\n", b"a: 1\n", b"  42  ", b"42\n",
        _npy(np.array(True)), _npy(np.array(b"")), _npy(np.array(1 + 2j)),
        _npy(np.array([1.5])), _npy(np.array(np.nan)), _npy(np.array("12")),
        _npy(np.array(b"12")), _npy(np.array(["a"])),
        b"0123456789" * 6 + b"abcd", b"ab" * 32, b"1" * 64 + b"\n",
    ]


def _corpus():
    items = [(w.name, w.raw_buffer) for w in iter_hashtype_witnesses()]
    items += [(f"extra{i}:{raw[:16]!r}", raw) for i, raw in enumerate(_extra_buffers())]
    return items


CORPUS = _corpus()


def _contract_valid(raw: bytes, celltype: str) -> bool:
    """Validity rules the reference parser leaves to HashType
    (celltypes-and-conversion.md, *Reading*): `binary` is an .npy buffer;
    `mixed` is .npy, Seamless-mixed or JSON; `checksum` is a bare 64-hex digest."""
    if celltype == "binary":
        return raw.startswith(b"\x93NUMPY")
    if celltype == "mixed":
        if raw.startswith(b"\x93NUMPY") or raw.startswith(MAGIC_SEAMLESS_MIXED):
            return True
        try:
            orjson.loads(raw)
        except orjson.JSONDecodeError:
            return False
        return True
    if celltype == "checksum":
        return re.fullmatch(rb"[0-9a-f]{64}", raw) is not None
    return True


def _permissive(checksum, celltype, *, buffer=None):
    """validate_deserializable_as with the HashType gate removed."""
    return ensure_hash_type(checksum, buffer=buffer)


def _oracle_parses(monkeypatch, buffer, checksum, celltype, raw) -> bool:
    from seamless.checksum.parse_buffer import _parse_buffer

    with monkeypatch.context() as m:
        m.setattr(htv, "validate_deserializable_as", _permissive)
        try:
            _parse_buffer(buffer, checksum, celltype)
        except Exception:
            return False
    return _contract_valid(raw, celltype)


def _oracle_converts(monkeypatch, buffer, checksum, source, target) -> bool:
    from seamless.checksum.convert import convert_checksum

    with monkeypatch.context() as m:
        m.setattr(htv, "validate_deserializable_as", _permissive)
        try:
            convert_checksum(checksum, source, target, lambda: buffer)
        except Exception:
            return False
    return True


def _words_for(concrete: HashType) -> list[HashType]:
    """The concrete word, plus every untested word it implies (all legal DB states)."""
    words = [concrete, HashType(Kind.UNTESTED, concrete.length)]
    if concrete.is_utf8:
        words.append(HashType(Kind.UTF8_UNTESTED, concrete.length))
    if concrete.is_json:
        words.append(HashType(Kind.JSON_UNTESTED, concrete.length))
    return words


@pytest.mark.parametrize("name,raw", CORPUS, ids=[c[0] for c in CORPUS])
def test_false_negative_property_for_deserialization(monkeypatch, name, raw):
    """§The false-negative property: a False from deserializable_as is a proof."""
    buffer = Buffer(raw)
    checksum = buffer.get_checksum()
    concrete = from_buffer(raw)
    false_rejections = []
    for celltype in THE_13:
        if not _oracle_parses(monkeypatch, buffer, checksum, celltype, raw):
            continue
        for word in _words_for(concrete):
            if word.deserializable_as(celltype, checksum=checksum) is False:
                false_rejections.append((word.kind.name, celltype))
    assert false_rejections == []


@pytest.mark.parametrize("name,raw", CORPUS, ids=[c[0] for c in CORPUS])
def test_false_negative_property_for_conversion(monkeypatch, name, raw):
    """§The false-negative property: a False from conversion_feasible is a proof."""
    buffer = Buffer(raw)
    checksum = buffer.get_checksum()
    concrete = from_buffer(raw)
    valid_sources = [
        ct for ct in THE_13
        if _oracle_parses(monkeypatch, buffer, checksum, ct, raw)
    ]
    false_rejections = []
    for source, target in itertools.product(valid_sources, THE_13):
        for word in _words_for(concrete):
            if conversion_feasible(word, source, target, checksum=checksum) is not False:
                continue
            if _oracle_converts(monkeypatch, buffer, checksum, source, target):
                false_rejections.append((word.kind.name, source, target))
    assert false_rejections == []


MIXED_NPY_PATHS = [
    (np.arange(3), (("item", 0),), "[0]"),
    (np.arange(4), (("slice", (1, 3, None)),), "[1:3]"),
    (np.zeros((2, 2)), (("item", 1),), "[1]"),
    (
        np.array((1, 2.0), dtype=np.dtype([("a", "<i4"), ("b", "<f8")], align=True)),
        (("item", "a"),),
        "a",
    ),
]


@pytest.mark.parametrize(
    "value,steps,path", MIXED_NPY_PATHS, ids=[c[2] for c in MIXED_NPY_PATHS]
)
def test_mixed_over_npy_path_is_not_falsely_rejected(value, steps, path):
    buffer = Buffer(value, "mixed")
    checksum = buffer.get_checksum()
    word = ensure_hash_type(checksum, buffer=buffer)
    assert word.kind == Kind.NUMPY
    # HashType must preserve the capabilities supported by the underlying NumPy value.
    validate_expression(
        checksum, buffer=None, source_celltype="mixed",
        path_steps=steps, target_celltype="mixed",
    )
    Expression(checksum, path, input_celltype="mixed", celltype="mixed").compute()


# --------------------------------------------------------------------------
# §capabilities, `mixed` row: a NUMPY word follows the `binary` rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (DType.NUMERIC, DType.NONNUMERIC, DType.STRUCTURED))
@pytest.mark.parametrize("rank", list(Rank), ids=lambda r: r.name)
def test_mixed_numpy_word_follows_the_binary_capability_row(dtype, rank):
    """§capabilities: `mixed` — `NUMPY` follows the `binary` rule."""
    word = HashType(Kind.NUMPY, Length.MEDIUM, dtype, rank)
    expected = ({"SEQ"} if rank != Rank.SCALAR else set()) | (
        {"MAP"} if dtype == DType.STRUCTURED else set()
    )
    assert word.capabilities("binary") == expected
    assert word.capabilities("mixed") == expected


MIXED_NPY_REJECTIONS = [
    (np.array(3.0), (("item", 0),), "SEQ"),  # scalar: no SEQ
    (np.arange(3), (("item", "a"),), "MAP"),  # unstructured: no MAP
    (np.arange(3.0), (("item", 0), ("item", 0)), "SEQ"),  # rank 1 admits one item
    (np.zeros((2, 3)), (("item", 1), ("item", 2), ("item", 0)), "SEQ"),  # rank 2 admits two
]


@pytest.mark.parametrize(
    "value,steps,needs", MIXED_NPY_REJECTIONS, ids=["scalar", "unstructured", "d1-rank", "d2-rank"]
)
def test_mixed_over_npy_keeps_the_binary_rejections(value, steps, needs):
    """§capabilities: under `mixed`, an .npy word is disproved exactly where `binary` is,
    including the rank count for NUMERIC arrays below D3PLUS."""
    buffer = Buffer(value, "mixed")
    checksum = buffer.get_checksum()
    assert ensure_hash_type(checksum, buffer=buffer).kind == Kind.NUMPY
    for celltype in ("binary", "mixed"):
        with pytest.raises(HashTypeValidationError):
            validate_expression(
                checksum, buffer=None, source_celltype=celltype,
                path_steps=steps, target_celltype=celltype,
            )


MIXED_NPY_RANK_ADMITS = [
    (np.arange(3.0), (("item", 2),), "d1-one-item"),
    (np.zeros((2, 3)), (("item", 1), ("item", 2)), "d2-two-items"),
    (np.zeros((2, 3)), (("slice", (0, 1, None)), ("item", 0), ("item", 2)), "d2-slice-keeps-rank"),
    # D3PLUS: rank counting does not apply; checking stops after the first item.
    (np.zeros((2, 2, 2, 2)), tuple(("item", 0) for _ in range(6)), "d3plus-stops"),
    # NONNUMERIC: rank counting does not apply; checking stops after the first item.
    (np.array(["ab", "cd"]), (("item", 0), ("item", 0), ("item", "x")), "nonnumeric-stops"),
]


@pytest.mark.parametrize(
    "value,steps,case", MIXED_NPY_RANK_ADMITS, ids=[c[2] for c in MIXED_NPY_RANK_ADMITS]
)
def test_mixed_over_npy_admits_exactly_what_binary_admits(value, steps, case):
    """§Path validation: under `binary`, and under `mixed` over a word of kind NUMPY,
    a NUMERIC array below D3PLUS admits as many integer item steps as its rank (a slice
    keeps the rank); otherwise checking stops after an item step."""
    buffer = Buffer(value, "mixed")
    checksum = buffer.get_checksum()
    word = ensure_hash_type(checksum, buffer=buffer)
    assert word.kind == Kind.NUMPY
    for celltype in ("binary", "mixed"):
        assert validate_expression(
            checksum, buffer=None, source_celltype=celltype,
            path_steps=steps, target_celltype=celltype,
        ) == word
