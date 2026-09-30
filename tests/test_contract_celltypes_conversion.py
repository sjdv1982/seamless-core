"""Gap-filling contract tests for contracts/celltypes-and-conversion.md.

Complements test_celltype_contract.py, test_conversion_contract.py and
test_conversion_engine.py; each test names the doc section it pins.
"""

import hashlib
import math

import numpy as np
import pytest

from seamless import Buffer, Checksum, Expression
from seamless.checksum.celltypes import celltypes
from seamless.checksum.conversion import (
    SeamlessConversionError,
    conversion_forbidden,
    conversion_reinterpret,
    conversion_trivial,
)
from seamless.checksum.convert import conversion_needs_buffer, convert_checksum
from seamless.checksum.hash_type_validation import HashTypeValidationError
from seamless.checksum.null import NULL_BUFFER, NULL_CHECKSUM
from seamless.checksum.parse_buffer import _parse_buffer
from seamless.checksum.serialize import _serialize
from seamless.checksum.virtual import virtual_value
from seamless.util.mixed import MAGIC_NUMPY, MAGIC_SEAMLESS_MIXED

DOC = "celltypes-and-conversion.md"


def _parse(raw, celltype):
    buffer = Buffer(raw)
    return _parse_buffer(buffer, buffer.get_checksum(), celltype)


def _fail_if_fetched():
    raise AssertionError("conversion unexpectedly fetched its source buffer")


def _counting_getter(buffer):
    calls = []

    def get_buffer():
        calls.append(None)
        return buffer

    return get_buffer, calls


def _source(value, celltype):
    return Buffer(value) if celltype == "bytes" else Buffer(value, celltype)


# Valid sample values per celltype (checksum excluded: covered elsewhere).
SAMPLES = {
    "binary": [np.array([1, 2]), np.array(b"42"), np.array(3.5)],
    "mixed": [{"a": 1}, 7, "hello"],
    "text": ["hello", "42", "a: 1"],
    "python": ["x = 1"],
    "ipython": ["%time 1"],
    "plain": [{"a": [1, 2]}, "hi", 42, [1, 2]],
    "yaml": ["a: 1", "3"],
    "str": ["hello", "42"],
    "bytes": [b"hello", b"42", b'{"a":1}'],
    "int": [4],
    "float": [4.5],
    "bool": [True, False],
    "checksum": [],
}
SAMPLED = [name for name in celltypes if SAMPLES[name]]


# --- Celltypes -------------------------------------------------------------


@pytest.mark.parametrize("position", ["source", "target"])
def test_module_is_not_a_conversion_celltype(position):
    """§Celltypes: module is neither in the 13 nor a deep celltype."""
    source, target = ("module", "plain") if position == "source" else ("plain", "module")
    with pytest.raises(TypeError):
        convert_checksum(Checksum("44" * 32), source, target, _fail_if_fetched)


@pytest.mark.parametrize("name", ["module", "not-a-celltype"])
@pytest.mark.parametrize("position", ["source", "target"])
def test_conversion_needs_buffer_rejects_celltypes_outside_the_engine(position, name):
    """§Celltypes: the engine row names conversion_needs_buffer too."""
    source, target = (name, "plain") if position == "source" else ("plain", name)
    with pytest.raises(TypeError):
        conversion_needs_buffer(Checksum("44" * 32), source, target)


@pytest.mark.parametrize("name", ["deepcell", "deepfolder", "folder", "module"])
def test_buffer_layer_maps_deep_names_and_module_to_plain(name):
    """§Where this page sits: Buffer._map_celltype serializes and parses all four
    names as plain."""
    assert Buffer._map_celltype(name) == "plain"
    value = {"a": 1}
    assert Buffer(value, name).content == Buffer(value, "plain").content


# --- The celltype hierarchy is a checksum hierarchy -------------------------


@pytest.mark.parametrize("subtype,supertype", sorted(conversion_trivial))
def test_every_hierarchy_edge_keeps_every_valid_subtype_checksum_valid(subtype, supertype):
    """§Hierarchy: every checksum valid as the subtype is valid, unchanged, as the supertype."""
    for value in SAMPLES[subtype]:
        buffer = _source(value, subtype)
        checksum = buffer.get_checksum()
        _parse_buffer(buffer, checksum, subtype)
        _parse_buffer(buffer, checksum, supertype)
        assert convert_checksum(checksum, subtype, supertype, _fail_if_fetched) == (
            checksum,
            None,
        )


# --- Canonical null ---------------------------------------------------------


def test_null_constants_are_the_documented_literals():
    """§Canonical null: NULL_BUFFER and NULL_CHECKSUM are fixed literals."""
    assert NULL_BUFFER == b"null\n"
    assert NULL_CHECKSUM == (
        "38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed"
    )
    assert hashlib.sha256(NULL_BUFFER).hexdigest() == NULL_CHECKSUM


def test_bytes_content_null_collides_with_empty_bytes():
    """§Canonical null, Collisions: b"null\\n" under bytes is empty bytes."""
    collided = Buffer(b"null\n", "bytes")
    empty = Buffer(b"", "bytes")
    assert collided.content == empty.content == NULL_BUFFER
    assert collided.get_checksum() == empty.get_checksum() == Checksum(NULL_CHECKSUM)
    noncanonical = Buffer(b"null")
    assert noncanonical.get_checksum() != Checksum(NULL_CHECKSUM)
    assert noncanonical.get_value("bytes") == b""
    assert Buffer(b"", "bytes").content != b"null"


# --- Virtual values ---------------------------------------------------------


@pytest.mark.parametrize("raw", [b"1\n", b"0", b"True\n", b" true", b'"true"', b"true\n\n"])
def test_bool_accepts_only_canonical_boolean_checksums(raw):
    """§Virtual values: no content-based bool coercion; raises without fetching."""
    checksum = Buffer(raw).get_checksum()
    with pytest.raises(ValueError, match="boolean"):
        virtual_value(checksum, "bool")
    with pytest.raises(ValueError):
        _parse(raw, "bool")


@pytest.mark.parametrize("raw,value", [(b"true", True), (b"false\n", False), (b"null\n", None)])
def test_bool_accepts_the_canonical_checksums_and_null(raw, value):
    assert _parse(raw, "bool") is value


def test_boolean_checksum_reads_as_json_text_under_text():
    """§Virtual values: other celltypes are NOT_VIRTUAL and parse the buffer."""
    assert _parse(b"true", "text") == "true"
    assert _parse(b"false\n", "yaml") == "false"


# --- Reference parser -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,celltype",
    [
        (b'{"a": 1}', "binary"),
        (b'{"a": 1}', "str"),
        (b"[1]", "str"),
        (b"[1]", "int"),
        (b'{"a": 1}', "float"),
        (b"{", "plain"),
        (b"\xff\xfe", "mixed"),
    ],
)
def test_reference_parser_value_error_rows_guarantee_only_value_error(raw, celltype):
    """§Reference parser, Exception classes: the rows marked ValueError (plain,
    binary, mixed, str, int, float) guarantee only the ValueError family; a
    subclass (e.g. HashTypeValidationError from step 2) is conformant."""
    with pytest.raises(ValueError):
        _parse(raw, celltype)


@pytest.mark.parametrize(
    "raw",
    [b"hello", b'"' + b"ab" * 32 + b'"', b"ab" * 31, b"zz" * 32],
)
def test_checksum_row_failures_raise_hash_type_validation_error(raw):
    """§Reference parser, Exception classes: a failure in the checksum row is
    HashTypeValidationError (contract)."""
    with pytest.raises(HashTypeValidationError):
        _parse(raw, "checksum")


@pytest.mark.parametrize(
    "raw,celltype",
    [
        (b"1\n", "bool"),
        (b"0", "bool"),
        (b'"hello"', "bool"),
        (b"true", "int"),
        (b"false\n", "int"),
        (b"true\n", "float"),
        (b"false", "float"),
    ],
)
def test_step_one_refusals_are_plain_value_error_not_hash_type_error(raw, celltype):
    """§Reference parser, Exception classes: bool over a non-boolean checksum and
    int/float over a boolean checksum are refused at step 1 (virtual_value) with a
    plain ValueError, decided from the checksum without HashType."""
    with pytest.raises(ValueError) as exc_info:
        _parse(raw, celltype)
    assert not isinstance(exc_info.value, HashTypeValidationError)


@pytest.mark.parametrize("celltype", ["plain", "str", "int", "float", "text", "yaml"])
def test_step_two_hash_type_disproof_raises_hash_type_validation_error(celltype):
    """§Reference parser, Exception classes: a HashType disproof at step 2 raises
    HashTypeValidationError whatever the celltype. Non-UTF-8, non-.npy bytes are
    proved non-deserializable as every JSON/text celltype by classification."""
    from seamless.checksum.hash_type_validation import validate_deserializable_as

    buffer = Buffer(b"\xff\xfe\x00\x01")
    checksum = buffer.get_checksum()
    with pytest.raises(HashTypeValidationError):
        validate_deserializable_as(checksum, celltype, buffer=buffer)
    with pytest.raises(HashTypeValidationError):
        _parse_buffer(buffer, checksum, celltype)


@pytest.mark.parametrize(
    "raw,celltype",
    [(b"def (:\n", "python"), (b"a: b: c\n", "yaml"), (b"\xff\xfe", "text")],
)
def test_text_parser_failures_raise_hash_type_validation_error(raw, celltype):
    """§Reference parser, text row: 'Failure raises HashTypeValidationError'."""
    with pytest.raises(HashTypeValidationError):
        _parse(raw, celltype)


@pytest.mark.parametrize("celltype", ["int", "float"])
@pytest.mark.parametrize(
    "accepted,rejected",
    [
        (b" " * 999 + b"1", b" " * 1000 + b"1"),
        (b"1" + b"\n" * 999, b"1" + b"\n" * 1000),
        (b"1." + b"0" * 998, b"1." + b"0" * 999),
    ],
)
def test_numeric_readings_accept_exactly_1000_bytes(celltype, accepted, rejected):
    """§Reference parser: only buffers OVER 1000 bytes are rejected."""
    assert (len(accepted), len(rejected)) == (1000, 1001)
    assert _parse(accepted, celltype) == 1
    with pytest.raises(ValueError):
        _parse(rejected, celltype)


@pytest.mark.parametrize("celltype", ["int", "float"])
@pytest.mark.parametrize("raw", [b'"nan"', b'"inf"', b'"-inf"', b"NaN", b"Infinity"])
def test_numeric_readings_reject_nonfinite(celltype, raw):
    """§Reference parser: only a finite number or finite-float string is accepted."""
    with pytest.raises(ValueError):
        _parse(raw, celltype)


@pytest.mark.parametrize(
    "raw,expected",
    [(b'"4.5"', 4), (b'"-4.5"', -4), (b"4.5\n", 4), (b'"12"', 12)],
)
def test_int_reading_truncates(raw, expected):
    """§Reference parser, int truncates."""
    assert _parse(raw, "int") == expected


def test_large_integers_lose_precision_through_orjson():
    """§Large integers / §Deliberate imprecisions (contract, not a limitation):
    orjson returns a float beyond u64, so int and plain readings are imprecise."""
    big = 2**64 + 1
    raw = str(big).encode()
    assert _parse(raw, "int") != big
    assert isinstance(_parse(raw, "plain"), float)


@pytest.mark.parametrize("container", ["dict", "list"])
def test_mixed_zero_dimensional_leaf_reads_back_as_numpy_scalar(container):
    """§Reference parser, mixed: nested 0-d NumPy leaf is a NumPy scalar."""
    leaf = np.array(2.5, dtype=np.float32)
    value = {"x": leaf} if container == "dict" else [leaf]
    restored = Buffer(value, "mixed").get_value("mixed")
    item = restored["x"] if container == "dict" else restored[0]
    assert isinstance(item, np.generic)
    assert item.dtype == np.float32
    assert item == 2.5
    top = Buffer(leaf, "mixed").get_value("mixed")
    assert isinstance(top, np.float32)


# --- Canonical serialization -----------------------------------------------


@pytest.mark.parametrize(
    "value,celltype,expected",
    [(b"4", "int", b"4\n"), (b"4.5", "float", b"4.5\n"), (b"1", "bool", b"true\n")],
)
def test_scalar_serializers_decode_bytes_first(value, celltype, expected):
    assert _serialize(value, celltype) == expected


@pytest.mark.parametrize("celltype", ["python", "ipython", "yaml"])
def test_text_serializers_perform_no_syntax_check(celltype):
    assert _serialize("def (: : :\n\n", celltype) == b"def (: : :\n"


def test_bytes_serializer_fallbacks():
    """§Canonical serialization, bytes: .tobytes(), then str(value) stripped."""
    array = np.array([1, 2], dtype=np.uint8)
    assert _serialize(array, "bytes") == b"\x01\x02"
    assert _serialize("héllo\n\n", "bytes") == "héllo".encode()
    assert _serialize(12, "bytes") == b"12"


@pytest.mark.parametrize("value", [{"b": 2, "a": [1, "x"]}, [3, 1], "s", 1.5, True])
def test_mixed_serializes_pure_json_like_plain(value):
    assert Buffer(value, "mixed").content == Buffer(value, "plain").content


@pytest.mark.parametrize(
    "array",
    [np.array([1.5, 2.0]), np.array(3.5), np.array(3, dtype=np.int8), np.array([1 + 2j])],
)
def test_mixed_serializes_numpy_arrays_like_binary_keeping_dtype(array):
    mixed = Buffer(array, "mixed")
    assert mixed.content == Buffer(array, "binary").content
    for celltype in ("binary", "mixed"):
        restored = Buffer(array, celltype).get_value(celltype)
        assert restored.dtype == array.dtype
        np.testing.assert_array_equal(restored, array)


def test_binary_serializer_stores_bytes_unchanged():
    assert _serialize(b"not an npy buffer", "binary") == b"not an npy buffer"


@pytest.mark.parametrize(
    "value",
    [np.float32("nan"), np.float16("inf"), np.float64("-inf"), math.inf],
)
def test_mixed_top_level_nonfinite_is_a_zero_dimensional_float64_npy(value):
    """§NaN and infinity: any float precision -> 0-d float64 .npy under mixed."""
    buffer = Buffer(value, "mixed")
    assert buffer.content.startswith(MAGIC_NUMPY)
    restored = buffer.get_value("mixed")
    assert type(restored) is np.float64
    if math.isnan(value):
        assert math.isnan(restored)
    else:
        assert restored == value


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_int_serialization_of_nonfinite_raises(value):
    with pytest.raises((ValueError, OverflowError)):
        _serialize(value, "int")


def test_binary_keeps_nonfinite_elements_in_their_own_dtype():
    array = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    restored = Buffer(array, "binary").get_value("binary")
    assert restored.dtype == np.float32
    assert np.isnan(restored[0]) and restored[1] == np.inf and restored[2] == -np.inf


# --- Conversion engine: rule-table categories -------------------------------


def test_conversion_error_is_a_value_error():
    assert issubclass(SeamlessConversionError, ValueError)


def test_rule_table_category_sizes():
    """§Rule table: 18 T, 14 RI, 10 RF, 7 P, 30 V, 16 X, 32 =, 29 » = 156 pairs."""
    from seamless.checksum import conversion as c

    sizes = {
        "trivial": len(c.conversion_trivial),
        "reinterpret": len(c.conversion_reinterpret),
        "reformat": len(c.conversion_reformat),
        "possible": len(c.conversion_possible),
        "values": len(c.conversion_values),
        "forbidden": len(c.conversion_forbidden),
        "equivalent": len(c.conversion_equivalent),
        "chain": len(c.conversion_chain),
    }
    assert sizes == {
        "trivial": 18,
        "reinterpret": 14,
        "reformat": 10,
        "possible": 7,
        "values": 30,
        "forbidden": 16,
        "equivalent": 32,
        "chain": 29,
    }
    assert sum(sizes.values()) == 13 * 12 == 156


def _check_conversions_with(monkeypatch, **tables):
    from seamless.checksum import conversion as c

    for name, table in tables.items():
        monkeypatch.setattr(c, name, table)
    c.check_conversions()


def test_check_conversions_accepts_the_shipped_table():
    from seamless.checksum.conversion import check_conversions

    check_conversions()


def test_check_conversions_raises_on_a_missing_pair(monkeypatch):
    """§Rule table: check_conversions() raises on a missing pair."""
    from seamless.checksum import conversion as c

    trimmed = set(c.conversion_trivial) - {("int", "plain")}
    with pytest.raises(SeamlessConversionError, match="Missing"):
        _check_conversions_with(monkeypatch, conversion_trivial=trimmed)


def test_check_conversions_raises_on_a_duplicate(monkeypatch):
    """§Rule table: check_conversions() raises on a duplicate."""
    from seamless.checksum import conversion as c

    doubled = set(c.conversion_reformat) | {("int", "plain")}
    with pytest.raises(SeamlessConversionError, match="Duplicate"):
        _check_conversions_with(monkeypatch, conversion_reformat=doubled)


def test_check_conversions_raises_on_a_circular_mapping(monkeypatch):
    """§Rule table: check_conversions() raises on a circular mapping."""
    from seamless.checksum import conversion as c

    a, b = ("int", "plain"), ("plain", "int")
    trivial = set(c.conversion_trivial) - {a}
    reinterpret = set(c.conversion_reinterpret) - {b}
    equivalent = dict(c.conversion_equivalent)
    equivalent[a] = b
    equivalent[b] = a
    with pytest.raises(SeamlessConversionError, match="Circular"):
        _check_conversions_with(
            monkeypatch,
            conversion_trivial=trivial,
            conversion_reinterpret=reinterpret,
            conversion_equivalent=equivalent,
        )


@pytest.mark.parametrize("source,target", sorted(conversion_forbidden))
def test_every_forbidden_pair_always_raises_without_fetching(source, target):
    """§Rule table: X is 'never' -- always raises, from the checksum alone."""
    for value in SAMPLES[source]:
        checksum = _source(value, source).get_checksum()
        with pytest.raises(SeamlessConversionError):
            convert_checksum(checksum, source, target, _fail_if_fetched)
        assert conversion_needs_buffer(checksum, source, target) is False


REINTERPRET_CASES = {
    ("bytes", "plain"): (b'{"a": 1}', b"not json"),
    ("bytes", "text"): (b"hello", b"\xff\xfe"),
    ("mixed", "binary"): (np.array([1, 2]), {"a": 1}),
    ("mixed", "plain"): ({"a": 1}, np.array([1, 2])),
    ("plain", "bool"): (True, 42),
    ("plain", "float"): (4.5, "abc"),
    ("plain", "int"): (4, [1]),
    ("plain", "str"): ("hi", {"a": 1}),
    ("str", "bool"): (True, "true"),
    ("str", "float"): ("4.5", "hello"),
    ("str", "int"): ("42", "hello"),
    ("text", "ipython"): ("%time 1", None),
    ("text", "python"): ("x = 1", "def (:"),
    ("text", "yaml"): ("a: 1", "a: b: c"),
}


def test_reinterpret_case_table_is_exhaustive():
    assert set(REINTERPRET_CASES) == set(conversion_reinterpret)


@pytest.mark.parametrize("source,target", sorted(conversion_reinterpret))
def test_every_reinterpret_pair_keeps_checksum_or_raises(source, target):
    """§Rule table, RI: same checksum, target reading validated and may raise."""
    good, bad = REINTERPRET_CASES[(source, target)]
    if good is not None:
        buffer = _source(good, source)
        checksum = buffer.get_checksum()
        assert convert_checksum(checksum, source, target, lambda: buffer) == (checksum, None)
        _parse_buffer(buffer, checksum, target)
    if bad is not None:
        buffer = _source(bad, source)
        with pytest.raises(SeamlessConversionError):
            convert_checksum(buffer.get_checksum(), source, target, lambda: buffer)


@pytest.mark.parametrize(
    "source,target,value",
    [
        ("bytes", "text", b"proof bytes text"),
        ("bytes", "plain", b'{"proof":"bytes plain"}'),
        ("mixed", "plain", {"proof": "mixed plain"}),
        ("mixed", "binary", np.array([31, 32])),
        ("plain", "int", 127),
        ("plain", "float", 128.5),
        ("str", "int", "129"),
        ("str", "float", "130.5"),
    ],
)
def test_reinterpretation_hash_type_proof_settles_without_fetching(
    source, target, value
):
    """§Executor, RI: a HashType proof keeps the checksum without fetching."""
    buffer = _source(value, source)
    checksum = buffer.get_checksum()

    assert conversion_needs_buffer(checksum, source, target) is False
    assert convert_checksum(checksum, source, target, _fail_if_fetched) == (
        checksum,
        None,
    )


@pytest.mark.parametrize(
    "source,target,value",
    [
        ("bytes", "text", b"no-proof bytes text"),
        ("bytes", "plain", b'{"no-proof":"bytes plain"}'),
        ("mixed", "plain", {"no-proof": "mixed plain"}),
        ("mixed", "binary", np.array([231, 232])),
        ("plain", "int", 223),
        ("plain", "float", 224.5),
        ("str", "int", "225"),
        ("str", "float", "226.5"),
    ],
)
def test_reinterpretation_without_hash_type_fetches_once(source, target, value):
    """§Executor, RI: with no word, parsing fetches the source exactly once."""
    buffer = _source(value, source)
    checksum = Checksum(hashlib.sha256(buffer.content).digest())
    get_buffer, calls = _counting_getter(buffer)

    assert conversion_needs_buffer(checksum, source, target) is True
    assert convert_checksum(checksum, source, target, get_buffer) == (
        checksum,
        None,
    )
    assert calls == [None]


@pytest.mark.parametrize(
    "source,target,raw,parses",
    [
        ("text", "python", b"def\n", False),
        ("text", "ipython", b"def\n", True),
        ("text", "yaml", b"a: [\n", False),
        ("plain", "str", b'"abc"', True),
        ("plain", "str", b" null \n", False),
    ],
)
def test_reinterpretation_without_a_proof_fetches_and_parses(
    source, target, raw, parses
):
    """§Executor, RI: unknown readings fetch; parser errors remain errors."""
    buffer = Buffer(raw)
    checksum = buffer.get_checksum()
    get_buffer, calls = _counting_getter(buffer)

    assert conversion_needs_buffer(checksum, source, target) is True
    if parses:
        assert convert_checksum(checksum, source, target, get_buffer) == (
            checksum,
            None,
        )
    else:
        with pytest.raises(SeamlessConversionError):
            convert_checksum(checksum, source, target, get_buffer)
    assert calls == [None]


def test_reinterpretation_proofs_settle_a_conversion_chain():
    """§Executor, RI: each checksum-preserving step can settle from the word."""
    buffer = Buffer(b"42")
    checksum = buffer.get_checksum()

    assert conversion_needs_buffer(checksum, "bytes", "int") is False
    assert convert_checksum(checksum, "bytes", "int", _fail_if_fetched) == (
        checksum,
        None,
    )


def test_hash_type_validation_error_is_wrapped_as_conversion_error():
    """§Executor, Errors: HashTypeValidationError is wrapped."""
    buffer = Buffer("def (:", "text")
    with pytest.raises(SeamlessConversionError) as exc_info:
        convert_checksum(buffer.get_checksum(), "text", "python", lambda: buffer)
    assert type(exc_info.value) is SeamlessConversionError
    assert f"{buffer.get_checksum().hex()} cannot be converted from text to python" in str(
        exc_info.value
    )


@pytest.mark.parametrize("source", ["text", "yaml"])
def test_to_mixed_goes_through_str_not_plain(source):
    """§Matrix readings: composition is not associative (text->mixed is =text->str)."""
    buffer = _source("123", source)
    checksum = buffer.get_checksum()
    assert convert_checksum(checksum, "text", "plain", lambda: buffer) == (checksum, None)
    result_checksum, result_buffer = convert_checksum(
        checksum, source, "mixed", lambda: buffer
    )
    assert result_buffer is not None
    assert result_buffer.content == b'"123"\n'
    assert result_checksum == Buffer("123", "str").get_checksum()
    assert result_buffer.get_value("mixed") == "123"


def test_equivalence_evaluates_the_mapped_pair():
    """§Resolving an indirection: python->bytes is =python->text (trivial)."""
    buffer = Buffer("x = 1", "python")
    checksum = buffer.get_checksum()
    assert convert_checksum(checksum, "python", "bytes", _fail_if_fetched) == (checksum, None)


# --- Conversion engine: reformat rules as executed --------------------------


@pytest.mark.parametrize("raw", [NULL_BUFFER, b"null", b"true", b"false\n"])
@pytest.mark.parametrize("source,target", [("bytes", "mixed"), ("text", "plain")])
def test_null_and_boolean_checksums_keep_without_fetch(raw, source, target):
    checksum = Buffer(raw).get_checksum()
    assert convert_checksum(checksum, source, target, _fail_if_fetched) == (checksum, None)


def test_bytes_to_mixed_utf8_strips_trailing_newlines():
    buffer = Buffer(b"hello\n\n")
    result_checksum, result_buffer = convert_checksum(
        buffer.get_checksum(), "bytes", "mixed", lambda: buffer
    )
    assert result_buffer is not None
    assert result_buffer.content == Buffer("hello", "str").content
    assert result_checksum == Buffer("hello", "str").get_checksum()


@pytest.mark.parametrize("cached_word", [True, False], ids=["cached-word", "no-word"])
def test_bytes_to_binary_wraps_a_coincidental_npy_magic(cached_word):
    """§Reformat rules, bytes->binary: .npy magic is validated as binary, and a
    corrupt payload behind it is wrapped as a dtype-S .npy, like any other
    bytes, whether or not its (RAW_BYTES) word is cached."""
    raw = MAGIC_NUMPY + (b"garbage" if cached_word else b"rubbish")
    if cached_word:
        checksum = Buffer(raw).get_checksum()
    else:
        checksum = Checksum(hashlib.sha256(raw).digest())
    get_buffer, calls = _counting_getter(raw)
    result_checksum, result_buffer = convert_checksum(checksum, "bytes", "binary", get_buffer)
    expected = Buffer(np.array(raw), "binary")
    assert (result_checksum, result_buffer.content) == (expected.get_checksum(), expected.content)
    assert calls == [None]


def _seamless_mixed_raw():
    raw = Buffer({"a": np.arange(5), "b": "x"}, "mixed").content
    assert raw.startswith(MAGIC_SEAMLESS_MIXED)
    return raw


@pytest.mark.parametrize("cached_word", [True, False], ids=["cached-word", "no-word"])
def test_bytes_to_mixed_wraps_a_truncated_seamless_mixed_buffer(cached_word):
    """§Reformat rules, bytes->mixed: a Seamless-mixed word is not a proof, so a
    corrupt payload behind the magic is never kept; like any non-UTF-8 bytes it
    becomes a dtype-S .npy, whether or not its word is cached."""
    # Distinct truncations, so that the no-word case finds no cached word.
    raw = _seamless_mixed_raw()[: -7 if cached_word else -9]
    if cached_word:
        checksum = Buffer(raw).get_checksum()  # registers the MIXED_OBJECT word
    else:
        checksum = Checksum(hashlib.sha256(raw).digest())
    get_buffer, calls = _counting_getter(raw)
    result_checksum, result_buffer = convert_checksum(checksum, "bytes", "mixed", get_buffer)
    expected = Buffer(np.array(raw), "binary")
    assert (result_checksum, result_buffer.content) == (expected.get_checksum(), expected.content)
    assert calls == [None]


def test_bytes_to_mixed_keeps_a_valid_seamless_mixed_buffer_after_fetching():
    """§Reformat rules, bytes->mixed: a valid Seamless-mixed buffer is kept, but
    its cached word does not skip the fetch."""
    buffer = Buffer(_seamless_mixed_raw())
    checksum = buffer.get_checksum()
    assert conversion_needs_buffer(checksum, "bytes", "mixed") is True
    get_buffer, calls = _counting_getter(buffer)
    assert convert_checksum(checksum, "bytes", "mixed", get_buffer) == (checksum, None)
    assert calls == [None]


def test_truncated_seamless_mixed_buffer_read_as_mixed_is_a_value_error():
    """§Reference parser, exception classes: a corrupt Seamless-mixed payload
    is a ValueError, not the format internals' AssertionError."""
    with pytest.raises(ValueError):
        _parse(_seamless_mixed_raw()[:-11], "mixed")


@pytest.mark.parametrize("source", ["binary", "mixed"])
@pytest.mark.parametrize(
    "value, expected",
    [
        (np.array(b"ab\x00"), b"ab\x00"),
        (np.frombuffer(b"ab\x00", dtype="S1"), b"ab\x00"),
        (np.frombuffer(b"abcd", dtype="S1").reshape(2, 2), b"abcd"),
        (np.array([b"ab", b"c"]), b"abc\x00"),
    ],
)
def test_s_array_to_bytes_is_tobytes_for_any_shape(source, value, expected):
    """§Reformat rules, binary/mixed->bytes: any dtype-S array becomes .tobytes()."""
    buffer = Buffer(value, source)
    result_checksum, result_buffer = convert_checksum(
        buffer.get_checksum(), source, "bytes", lambda: buffer
    )
    assert result_buffer is not None
    assert result_buffer.content == expected
    assert result_checksum == Buffer(expected).get_checksum()


@pytest.mark.parametrize("source", ["binary", "mixed"])
def test_empty_s_array_to_bytes_is_canonical_null(source):
    """§Reformat rules, binary/mixed->bytes: an empty S array is empty bytes."""
    buffer = Buffer(np.frombuffer(b"", dtype="S1"), source)
    result_checksum, result_buffer = convert_checksum(
        buffer.get_checksum(), source, "bytes", lambda: buffer
    )
    assert result_checksum == NULL_CHECKSUM
    assert result_buffer.content == NULL_BUFFER


@pytest.mark.parametrize("source", ["binary", "mixed"])
@pytest.mark.parametrize(
    "array",
    [np.array([1, 2]), np.array(3.5), np.array(["ab"]), np.array([1 + 2j])],
    ids=["int", "0d-float", "unicode-U", "complex"],
)
def test_non_s_array_to_bytes_keeps_the_npy_checksum(source, array):
    """§Reformat rules, binary->bytes (and mixed->bytes on .npy magic): any array
    whose dtype is not S keeps its .npy checksum."""
    buffer = Buffer(array, source)
    assert buffer.content.startswith(MAGIC_NUMPY)
    assert convert_checksum(
        buffer.get_checksum(), source, "bytes", lambda: buffer
    ) == (buffer.get_checksum(), None)


def test_raw_bytes_with_coincidental_npy_magic_have_a_checksum():
    raw = MAGIC_NUMPY + b"garbage"
    buffer = Buffer(raw)
    assert buffer.get_checksum() == Checksum(hashlib.sha256(raw).digest())
    assert buffer.get_value("bytes").content == raw


def test_plain_to_binary_rejects_a_string():
    buffer = Buffer("hello", "plain")
    with pytest.raises(SeamlessConversionError):
        convert_checksum(buffer.get_checksum(), "plain", "binary", lambda: buffer)


# --- Executor ----------------------------------------------------------------


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_get_buffer_may_return_bytes_like(wrap):
    buffer = Buffer(b"hi")
    expected = convert_checksum(buffer.get_checksum(), "text", "plain", lambda: buffer)
    result = convert_checksum(
        buffer.get_checksum(), "text", "plain", lambda: wrap(b"hi")
    )
    assert result[0] == expected[0]
    assert result[1].content == expected[1].content


def test_cached_hash_type_disproof_needs_no_buffer():
    """§Executor: a cached HashType disproof fails from the checksum alone."""
    buffer = Buffer({"a": 1}, "plain")
    checksum = buffer.get_checksum()
    assert conversion_needs_buffer(checksum, "plain", "int") is False
    with pytest.raises(SeamlessConversionError):
        convert_checksum(checksum, "plain", "int", _fail_if_fetched)


@pytest.mark.parametrize("source", SAMPLED)
def test_get_buffer_is_called_at_most_once_and_dry_run_agrees(source):
    """§Executor: get_buffer at most once per call; conversion_needs_buffer is exact."""
    for value in SAMPLES[source]:
        buffer = _source(value, source)
        checksum = buffer.get_checksum()
        for target in celltypes:
            get_buffer, calls = _counting_getter(buffer)
            try:
                convert_checksum(checksum, source, target, get_buffer)
            except SeamlessConversionError:
                pass
            assert len(calls) <= 1, (source, target, value)
            assert conversion_needs_buffer(checksum, source, target) is bool(calls), (
                source,
                target,
                value,
            )


_RESULT_PAIRS = [
    (source, target)
    for source in SAMPLED
    for target in celltypes
    if source != target
]


@pytest.mark.parametrize("source,target", _RESULT_PAIRS)
def test_new_checksum_comes_with_its_buffer_and_reads_as_target(source, target):
    """§Executor, Return value: (checksum, None) only when the checksum is kept."""
    for value in SAMPLES[source]:
        buffer = _source(value, source)
        checksum = buffer.get_checksum()
        try:
            result_checksum, result_buffer = convert_checksum(
                checksum, source, target, lambda: buffer
            )
        except SeamlessConversionError:
            continue
        if result_buffer is None:
            assert result_checksum == checksum, (source, target, value)
            _parse_buffer(buffer, checksum, target)
        else:
            assert result_buffer.get_checksum() == result_checksum
            _parse_buffer(result_buffer, result_checksum, target)


@pytest.mark.parametrize("target", ["python", "ipython", "yaml"])
def test_binary_array_converts_through_plain_and_text(target):
    """binary->python/ipython/yaml resolve via plain and text; [1, 2] is valid in all."""
    buffer = Buffer(np.array([1, 2]), "binary")
    result_checksum, result_buffer = convert_checksum(
        buffer.get_checksum(), "binary", target, lambda: buffer
    )
    expected = Buffer([1, 2], "plain")
    assert result_checksum == expected.get_checksum()
    readable = result_buffer if result_buffer is not None else expected
    _parse_buffer(readable, result_checksum, target)


# --- Empty-path Expression is the engine's only caller ----------------------


@pytest.mark.parametrize("source", list(celltypes))
def test_empty_path_expression_short_circuits_null_for_every_legal_pair(
    source, monkeypatch
):
    """§Executor: null input -> null result, before the executor, for every
    LEGAL pair (clarity ruling: illegal conversions remain illegal for null)."""
    from seamless.checksum import convert as convert_module
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    def engine_called(*args, **kwargs):
        raise AssertionError("executor must not be called for a null input")

    monkeypatch.setattr(convert_module, "convert_checksum", engine_called)
    get_expression_cache().clear()
    null = Checksum(NULL_CHECKSUM)
    for target in celltypes:
        if (source, target) in conversion_forbidden:
            continue
        assert evaluate_expression(null, "", source, target) == null


_NULL_FORMS = [
    pytest.param(Checksum(NULL_CHECKSUM), id="canonical"),
    pytest.param(Checksum(hashlib.sha256(b"null").digest()), id="noncanonical"),
]


@pytest.mark.parametrize("null", _NULL_FORMS)
@pytest.mark.parametrize("source,target", sorted(conversion_forbidden))
def test_engine_refuses_null_on_every_forbidden_pair(source, target, null):
    """Ruling: illegal conversions remain illegal for null (engine level)."""
    with pytest.raises(SeamlessConversionError):
        convert_checksum(null, source, target, _fail_if_fetched)


@pytest.mark.parametrize("source,target", sorted(conversion_forbidden))
def test_expression_construction_refuses_null_on_every_forbidden_pair(source, target):
    with pytest.raises(ValueError):
        Expression(Checksum(NULL_CHECKSUM), input_celltype=source, celltype=target)


@pytest.mark.parametrize("source,target", sorted(conversion_forbidden))
def test_empty_path_evaluation_refuses_null_on_every_forbidden_pair(source, target):
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    get_expression_cache().clear()
    with pytest.raises(ValueError):
        evaluate_expression(Checksum(NULL_CHECKSUM), "", source, target)


_ILLEGAL_DEEP_PAIRS = [
    ("deepfolder", "deepcell"),
    ("folder", "plain"),
    ("plain", "deepcell"),
    ("plain", "deepfolder"),
    ("plain", "folder"),
    ("deepcell", "text"),
    ("text", "folder"),
    ("deepcell", "mixed"),
    ("deepfolder", "mixed"),
    ("checksum", "deepcell"),
]


@pytest.mark.parametrize("null", _NULL_FORMS)
@pytest.mark.parametrize("source,target", _ILLEGAL_DEEP_PAIRS)
def test_engine_refuses_null_on_illegal_deep_pairs(source, target, null):
    with pytest.raises(SeamlessConversionError):
        convert_checksum(null, source, target, _fail_if_fetched)


@pytest.mark.parametrize("source,target", _ILLEGAL_DEEP_PAIRS)
def test_expression_refuses_null_on_illegal_deep_pairs(source, target):
    with pytest.raises(ValueError):
        Expression(Checksum(NULL_CHECKSUM), input_celltype=source, celltype=target)


@pytest.mark.parametrize("source,target", _ILLEGAL_DEEP_PAIRS)
def test_empty_path_evaluation_refuses_null_on_illegal_deep_pairs(source, target):
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    get_expression_cache().clear()
    with pytest.raises(ValueError):
        evaluate_expression(Checksum(NULL_CHECKSUM), "", source, target)


_LEGAL_DEEP_PAIRS = [
    ("deepcell", "deepcell"),
    ("deepfolder", "deepfolder"),
    ("folder", "folder"),
    ("deepcell", "plain"),
    ("deepfolder", "plain"),
    ("folder", "deepfolder"),
    ("deepfolder", "folder"),
    ("deepcell", "deepfolder"),
    ("folder", "mixed"),
]
_LEGAL_ORDINARY_PAIRS = [
    (source, target)
    for source in celltypes
    for target in celltypes
    if (source, target) not in conversion_forbidden
]


@pytest.mark.parametrize("source,target", _LEGAL_ORDINARY_PAIRS + _LEGAL_DEEP_PAIRS)
def test_expression_construction_accepts_null_on_every_legal_pair(source, target):
    """§Null and conversion legality, legal row: construction succeeds."""
    expression = Expression(
        Checksum(NULL_CHECKSUM), input_celltype=source, celltype=target
    )
    assert expression.input_celltype == source


@pytest.mark.parametrize("source,target", _LEGAL_DEEP_PAIRS)
def test_empty_path_evaluation_short_circuits_null_on_legal_deep_pairs(
    source, target, monkeypatch
):
    """§Null and conversion legality, legal row: a legal deep pair yields the
    canonical null without calling the executor."""
    from seamless.checksum import convert as convert_module
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    def engine_called(*args, **kwargs):
        raise AssertionError("executor must not be called for a null input")

    monkeypatch.setattr(convert_module, "convert_checksum", engine_called)
    get_expression_cache().clear()
    null = Checksum(NULL_CHECKSUM)
    assert evaluate_expression(null, "", source, target) == null


def test_empty_bytes_input_is_canonicalized_then_short_circuited(monkeypatch):
    """§Null and conversion legality, 'Which inputs count as null': the key
    canonicalizes empty bytes (input celltype bytes) to NULL_CHECKSUM first."""
    from seamless.checksum import convert as convert_module
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    def engine_called(*args, **kwargs):
        raise AssertionError("executor must not be called for a null input")

    monkeypatch.setattr(convert_module, "convert_checksum", engine_called)
    get_expression_cache().clear()
    empty = Checksum(hashlib.sha256(b"").digest())
    for target in ("plain", "text", "checksum", "int"):
        assert evaluate_expression(empty, "", "bytes", target) == NULL_CHECKSUM


def test_noncanonical_null_takes_the_ordinary_route_through_the_executor(monkeypatch):
    """§Null and conversion legality: sha256(b"null") is not short-circuited; it
    goes through the executor and the pair's own rule (here X->checksum stores the
    source checksum's digest, so the result is not null)."""
    from seamless.checksum import convert as convert_module
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    calls = []
    original = convert_module.convert_checksum

    def spy(*args, **kwargs):
        calls.append(args[1:3])
        return original(*args, **kwargs)

    monkeypatch.setattr(convert_module, "convert_checksum", spy)
    get_expression_cache().clear()
    noncanonical = Checksum(hashlib.sha256(b"null").digest())
    result = evaluate_expression(noncanonical, "", "plain", "checksum")
    assert calls == [("plain", "checksum")]
    assert result != NULL_CHECKSUM
    assert result == Buffer(noncanonical, "checksum").get_checksum()
    get_expression_cache().clear()


def test_executor_converts_null_by_the_pair_rule_on_a_legal_pair():
    """§Null and conversion legality: the executor has no null short-circuit of
    its own for ordinary pairs; contrast with the empty-path Expression."""
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    null = Checksum(NULL_CHECKSUM)
    result_checksum, result_buffer = convert_checksum(
        null, "plain", "checksum", _fail_if_fetched
    )
    assert result_buffer is not None
    assert result_buffer.content == NULL_CHECKSUM.encode()
    get_expression_cache().clear()
    assert evaluate_expression(null, "", "plain", "checksum") == null


# --- Deliberate imprecisions (contract) --------------------------------------


def test_imprecision_checksum_to_x_is_not_validated_against_x():
    """§Deliberate imprecisions: checksum->X dereferences without checking that
    the referenced checksum is deserializable as X."""
    referenced = Buffer(b"\xff\xfe definitely not json").get_checksum()
    source = Buffer(referenced, "checksum")
    for target in ("plain", "int", "python", "bool", "binary"):
        result = convert_checksum(
            source.get_checksum(), "checksum", target, lambda: source
        )
        assert result == (referenced, None)


def test_imprecision_bytes_reading_is_a_buffer_or_empty_bytes_for_null():
    """§Deliberate imprecisions: a non-null bytes reading is the Buffer object
    itself, not Python bytes; the null reading is b""."""
    buffer = Buffer(b"payload")
    value = _parse_buffer(buffer, buffer.get_checksum(), "bytes")
    assert isinstance(value, Buffer)
    assert not isinstance(value, bytes)
    assert value.content == b"payload"
    null = Buffer(NULL_BUFFER)
    assert _parse_buffer(null, null.get_checksum(), "bytes") == b""


def test_imprecision_bool_does_not_round_trip_through_str():
    """§Deliberate imprecisions: reading the true buffer as str gives "True",
    serializing True as str writes b"true\\n"."""
    true_buffer = Buffer(True, "str")
    assert true_buffer.content == b"true\n"
    assert true_buffer.get_value("str") == "True"
    assert Buffer("True", "str").content == b'"True"\n'
    assert Buffer("True", "str").get_checksum() != true_buffer.get_checksum()


def test_trivial_checksums_resolve_without_cache_residency():
    """§Virtual values: TRIVIAL_CHECKSUMS are materialized by Checksum.resolve()
    without a buffer (b"[]" without newline is never written by a serializer)."""
    checksum = Checksum(hashlib.sha256(b"[]").digest())
    assert checksum.resolve("plain") == []
    checksum = Checksum(hashlib.sha256(b"{}").digest())
    assert checksum.resolve("plain") == {}


def test_engine_result_is_recorded_as_the_empty_path_expression(monkeypatch):
    """§Executor: a repeated conversion is an Expression-identity cache hit."""
    from seamless.checksum import convert as convert_module
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    source = Buffer("hi", "text")
    source.tempref()
    checksum = source.get_checksum()
    get_expression_cache().clear()
    first = evaluate_expression(checksum, "", "text", "plain")
    assert first == Buffer("hi", "plain").get_checksum()

    def engine_called(*args, **kwargs):
        raise AssertionError("second conversion must be a cache hit")

    monkeypatch.setattr(convert_module, "convert_checksum", engine_called)
    assert evaluate_expression(checksum, "", "text", "plain") == first
    get_expression_cache().clear()


@pytest.mark.parametrize(
    "source,target",
    [
        ("deepfolder", "deepcell"),
        ("folder", "plain"),
        ("plain", "deepcell"),
        ("plain", "deepfolder"),
        ("plain", "folder"),
        ("deepcell", "text"),
        ("text", "folder"),
        ("deepcell", "mixed"),
        ("deepfolder", "mixed"),
        ("checksum", "deepcell"),
    ],
)
def test_rejected_deep_pairs_raise_conversion_error_not_type_error(source, target):
    """§Celltypes: the engine knows the deep names; an illegal deep pair is a
    SeamlessConversionError decided from the celltypes alone, never TypeError
    (contracts/deep-celltypes.md, Zero-path conversions)."""
    checksum = Checksum("44" * 32)
    with pytest.raises(SeamlessConversionError) as exc_info:
        convert_checksum(checksum, source, target, _fail_if_fetched)
    assert type(exc_info.value) is SeamlessConversionError
    assert conversion_needs_buffer(checksum, source, target) is False
