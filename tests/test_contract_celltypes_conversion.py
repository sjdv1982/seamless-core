"""Gap-filling contract tests for contracts/celltypes-and-conversion.md.

Complements test_celltype_contract.py, test_conversion_contract.py and
test_conversion_engine.py; each test names the doc section it pins.
"""

import hashlib
import math

import numpy as np
import pytest

from seamless import Buffer, Checksum
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
from seamless.util.mixed import MAGIC_NUMPY

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
        (b"def (:\n", "python"),
        (b"a: b: c\n", "yaml"),
        (b"\xff\xfe", "text"),
        (b'{"a": 1}', "binary"),
        (b'{"a": 1}', "str"),
        (b"[1]", "str"),
        (b"hello", "checksum"),
        (b'"' + b"ab" * 32 + b'"', "checksum"),
    ],
)
def test_reference_parser_rejects_invalid_readings(raw, celltype):
    """§Reference parser: syntax/storage failures raise HashTypeValidationError."""
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
    with pytest.raises(HashTypeValidationError):
        _parse(rejected, celltype)


@pytest.mark.parametrize("celltype", ["int", "float"])
@pytest.mark.parametrize("raw", [b'"nan"', b'"inf"', b'"-inf"', b"NaN", b"Infinity"])
def test_numeric_readings_reject_nonfinite(celltype, raw):
    """§Reference parser: only a finite number or finite-float string is accepted."""
    with pytest.raises(HashTypeValidationError):
        _parse(raw, celltype)


@pytest.mark.parametrize(
    "raw,expected",
    [(b'"4.5"', 4), (b'"-4.5"', -4), (b"4.5\n", 4), (b'"12"', 12)],
)
def test_int_reading_truncates(raw, expected):
    """§Reference parser, int truncates."""
    assert _parse(raw, "int") == expected


def test_large_integers_lose_precision_through_orjson():
    """§Large integers (current limitation): orjson returns a float beyond u64."""
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


def test_bytes_to_binary_rejects_a_coincidental_npy_magic():
    """§Reformat rules, bytes->binary: .npy magic is validated as binary."""
    raw = MAGIC_NUMPY + b"garbage"
    checksum = Checksum(hashlib.sha256(raw).digest())
    get_buffer, calls = _counting_getter(raw)
    with pytest.raises(SeamlessConversionError):
        convert_checksum(checksum, "bytes", "binary", get_buffer)
    assert calls == [None]


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §Celltypes/§Reformat rules (bytes->binary): bytes is raw storage, "
    "but checksum calculation raises ValueError for a buffer that starts with the "
    ".npy magic and is not a valid .npy (HashType.from_buffer parses it)",
)
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


# Chains whose first step writes a new buffer and whose last step keeps the
# checksum return (new_checksum, None): the intermediate buffer is dropped.
_CHAIN_DROPS_BUFFER = {
    ("binary", "str"),
    ("binary", "text"),
    ("mixed", "ipython"),
    ("mixed", "python"),
    ("mixed", "yaml"),
    ("plain", "ipython"),
    ("plain", "python"),
}
_RESULT_PAIRS = [
    pytest.param(
        source,
        target,
        marks=pytest.mark.xfail(
            strict=False,
            reason=f"{DOC} §Executor, Return value: a chain whose last step keeps "
            "the checksum drops the intermediate buffer and returns "
            "(new_checksum, None)",
        ),
    )
    if (source, target) in _CHAIN_DROPS_BUFFER
    else (source, target)
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


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §Resolving an indirection: after a chain step that dropped its "
    "buffer, the next step is fed the ORIGINAL source buffer (binary .npy) and fails",
)
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
def test_empty_path_expression_short_circuits_null_for_every_pair(source, monkeypatch):
    """§Executor: null input -> null result for every pair, before the executor."""
    from seamless.checksum import convert as convert_module
    from seamless.checksum.expression import evaluate_expression, get_expression_cache

    def engine_called(*args, **kwargs):
        raise AssertionError("executor must not be called for a null input")

    monkeypatch.setattr(convert_module, "convert_checksum", engine_called)
    get_expression_cache().clear()
    null = Checksum(NULL_CHECKSUM)
    for target in celltypes:
        assert evaluate_expression(null, "", source, target) == null


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
