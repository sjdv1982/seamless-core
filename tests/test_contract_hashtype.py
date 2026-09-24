"""Contract tests for contracts/hashtype.md: word, producer, queries, validation.

Complements test_hash_type_contract.py / test_hash_type_tightening.py /
test_hash_type_validation.py with the rules those files leave uncovered or pin
with a single case, and with an oracle-backed check of the central rule
(*The false-negative property*): HashType may answer "unknown" but never
rejects a valid deserialization or conversion.
"""

from __future__ import annotations

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
)

from helpers.expression_hashtype_cases import iter_hashtype_witnesses
from helpers.fake_remotes import install_fake_remotes


CHECKSUM = "a" * 64
THE_13 = tuple(sorted(HASH_TYPE_CELLTYPES))


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


# --------------------------------------------------------------------------
# The word
# --------------------------------------------------------------------------


@pytest.mark.parametrize("word", (-1, 1 << 13, 1 << 20))
def test_out_of_range_words_do_not_unpack_and_are_invalid(word):
    """§The word: a HashType is a 13-bit integer."""
    with pytest.raises(ValueError):
        unpack(word)
    assert is_valid_word(word) is False


@pytest.mark.parametrize("word", ("5", 5.0, None))
def test_non_integer_words_are_invalid(word):
    with pytest.raises(TypeError):
        unpack(word)
    assert is_valid_word(word) is False


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
    "raw", (b'"inf"', b'"nan"', b'"-Infinity"', b'"1e400"', b'"0x10"', b'"abc"')
)
def test_json_string_gets_numeric_scalar_only_if_it_parses_as_a_finite_float(raw):
    """§Producer step 4: string -> JSON_STRING (+NUMERIC_SCALAR if finite float)."""
    word = from_buffer(raw)
    assert word.kind == Kind.JSON_STRING
    assert not (word.flags & Flag.NUMERIC_SCALAR)


@pytest.mark.parametrize("raw", (b'"42"', b'" 42 "', b'"-1.5e3"'))
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


def test_mixed_form_with_other_root_type_raises_value_error():
    """§Producer step 2: any mixed root type other than object/array raises."""
    form = json.dumps({"type": "string"}).encode()
    raw = (
        MAGIC_SEAMLESS_MIXED + bytes([4]) + b"pure"
        + len(form).to_bytes(4, "little") + form
    )
    with pytest.raises(ValueError):
        from_buffer(raw)


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
    """§conversion_feasible: a null checksum -> True (first rule)."""
    word = HashType(Kind.RAW_BYTES, Length.SHORT)
    assert word.deserializable_as("plain", checksum=CHECKSUM) is False
    for target in THE_13:
        assert conversion_feasible(word, "plain", target, checksum=CHECKSUM_NULL) is True


@pytest.mark.parametrize("celltype", ("plain", "binary", "int", "text"))
def test_source_equal_target_is_true_before_the_source_check(celltype):
    """§conversion_feasible: `source == target` -> True, ahead of deserializable_as."""
    word = HashType(Kind.RAW_BYTES, Length.SHORT)
    assert word.deserializable_as(celltype, checksum=CHECKSUM) is False
    assert conversion_feasible(word, celltype, celltype, checksum=CHECKSUM) is True


@pytest.mark.parametrize(
    "source,target", (("python", "yaml"), ("yaml", "python"), ("ipython", "yaml"))
)
def test_forbidden_pairs_are_false_even_for_a_valid_source(source, target):
    """§conversion_feasible: conversion_forbidden -> False."""
    from seamless.checksum.conversion import conversion_forbidden

    assert (source, target) in conversion_forbidden
    word = HashType(Kind.RAW_TEXT, Length.SHORT)
    assert word.deserializable_as(source, checksum=CHECKSUM) is True
    assert conversion_feasible(word, source, target, checksum=CHECKSUM) is False


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
    from seamless.checksum.conversion import conversion_reinterpret

    assert ("bytes", "text") in conversion_reinterpret
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
    with pytest.raises(HashTypeValidationError, match="Cannot convert"):
        validate_expression(
            checksum, buffer=None, source_celltype="plain",
            path_steps=(), target_celltype="binary",
        )
    assert validate_expression(
        checksum, buffer=None, source_celltype="plain",
        path_steps=(("item", "a"),), target_celltype="binary",
    ) == word


def test_source_is_checked_before_path_capability():
    """§Where consulted: source celltype first, then path capability."""
    checksum, _ = _register(b"\xff\xfe\x00")
    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        validate_expression(
            checksum, buffer=None, source_celltype="plain",
            path_steps=(("item", "missing"),), target_celltype="plain",
        )


def test_capability_is_asked_of_the_input_celltype():
    """§Where consulted: the capability check is asked of the **input** celltype."""
    checksum, word = _register(b'"hello"')
    # A JSON string read as str admits SEQ, whatever the target.
    assert validate_expression(
        checksum, buffer=None, source_celltype="str",
        path_steps=(("item", 0),), target_celltype="int",
    ) == word
    with pytest.raises(HashTypeValidationError, match="MAP"):
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
    with pytest.raises(HashTypeValidationError, match="SEQ"):
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


@pytest.mark.parametrize("deep", ("deepcell", "deepfolder", "folder"))
def test_deep_source_is_never_vetted_by_hashtype(deep):
    """§Where consulted / Non-goals: deep feasibility is structural; HashType is not asked."""
    checksum, _ = _register(b"\xff\xfe\x00")  # raw bytes: disproved as plain
    validate_expression(
        checksum, buffer=None, source_celltype=deep,
        path_steps=(("item", "x"),), target_celltype="plain",
    )


def test_unknown_celltype_in_expression_validation_is_a_value_error():
    checksum, _ = _register(b"{}")
    with pytest.raises(ValueError):
        validate_expression(
            checksum, buffer=None, source_celltype="nonsense",
            path_steps=(), target_celltype="plain",
        )


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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "hashtype.md §The false-negative property vs §capabilities: `mixed` over an "
        ".npy word has empty capabilities ('else empty'), so a path the engine "
        "evaluates is rejected with HashTypeValidationError"
    ),
)
@pytest.mark.parametrize(
    "value,steps,path", MIXED_NPY_PATHS, ids=[c[2] for c in MIXED_NPY_PATHS]
)
def test_mixed_over_npy_path_is_not_falsely_rejected(value, steps, path):
    buffer = Buffer(value, "mixed")
    checksum = buffer.get_checksum()
    word = ensure_hash_type(checksum, buffer=buffer)
    assert word.kind == Kind.NUMPY
    # The reference engine evaluates this path when the gate is lifted
    # (verified in the phase-1 probe); HashType must not disprove it.
    validate_expression(
        checksum, buffer=None, source_celltype="mixed",
        path_steps=steps, target_celltype="mixed",
    )
    Expression(checksum, path, input_celltype="mixed", celltype="mixed").compute()
