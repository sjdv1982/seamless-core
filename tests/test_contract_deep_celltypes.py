"""Contract coverage for docs/agent/contracts/deep-celltypes.md (feature 4, deep side).

Complements test_expression_contract.py / test_conversion_contract.py, which already pin
the illegal-pair spot checks, the one-step path, and the S1 child representation.
This file adds the exhaustive zero-path table, the flatness phase table, the
validator's failure shapes, folder -> mixed all-or-nothing / non-sticky, and the
value rule ({key: Checksum}) at the value layers.

Known gaps are non-strict xfails that assert the contract, never the bug.
"""
import numpy as np
import pytest

from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless.checksum.celltypes import celltypes
from seamless.checksum.conversion import SeamlessConversionError
from seamless.checksum.convert import convert_checksum
from seamless.checksum.deep import DEEP_CELLTYPES, validate_deep_structure
from seamless.checksum.expression import ExpressionEvaluationError
from seamless.checksum.hash_type_validation import (
    HashTypeValidationError,
    ensure_hash_type,
)
from seamless.error_envelope import error_kind

DEEP = ("deepcell", "deepfolder", "folder")
DOC = "deep-celltypes.md"

LEGAL_ZERO_PATH = {
    ("deepcell", "deepcell"),
    ("deepfolder", "deepfolder"),
    ("folder", "folder"),
    ("deepcell", "plain"),
    ("deepfolder", "plain"),
    ("folder", "deepfolder"),
    ("deepfolder", "folder"),
    ("deepcell", "deepfolder"),
    ("folder", "mixed"),
}
FREE_ZERO_PATH = sorted(LEGAL_ZERO_PATH - {("folder", "mixed")})


def _held(value, celltype=None):
    buffer = Buffer(value, celltype) if celltype else Buffer(value)
    buffer.tempref()
    return buffer


def _absent_checksum(content: bytes) -> Checksum:
    """Checksum of content that no buffer cache has seen."""
    return Checksum(calculate_checksum(content))


def _all_zero_path_pairs():
    everything = list(DEEP) + list(celltypes)
    return [
        (source, target)
        for source in everything
        for target in everything
        if source in DEEP or target in DEEP
    ]


# --- Zero-path conversions -------------------------------------------------


def test_deep_set_is_exactly_three_and_excludes_module():
    assert set(DEEP_CELLTYPES) == set(DEEP)
    assert "module" not in DEEP_CELLTYPES


@pytest.mark.parametrize("source,target", _all_zero_path_pairs())
def test_zero_path_table_is_exhaustive_at_construction(source, target):
    """'Only these are legal ... everything else: rejected' (Zero-path conversions)."""
    checksum = Checksum("5a" * 32)
    if (source, target) in LEGAL_ZERO_PATH:
        Expression(checksum, input_celltype=source, celltype=target)
    else:
        with pytest.raises(ValueError):
            Expression(checksum, input_celltype=source, celltype=target)


@pytest.mark.parametrize("source,target", FREE_ZERO_PATH)
def test_free_conversions_evaluate_at_expression_level_without_any_buffer(source, target):
    """Free class: the result checksum comes from the rule alone; no buffer fetched."""
    checksum = _absent_checksum(b"deep-celltypes free conversion, never stored " + source.encode() + target.encode())
    expression = Expression(checksum, input_celltype=source, celltype=target)
    assert expression.compute(execution="local") == checksum


@pytest.mark.parametrize(
    "source,target",
    [("deepcell", "deepfolder"), ("deepcell", "plain"), ("folder", "deepfolder"), ("deepfolder", "folder")],
)
def test_false_deep_claim_is_not_detected_by_a_free_conversion(source, target):
    """'A checksum falsely declared deep is not detected by a free conversion, and that is by design.'"""
    not_an_index = _held("this text is not an index", "text")
    expression = Expression(not_an_index.get_checksum(), input_celltype=source, celltype=target)
    assert expression.compute(execution="local") == not_an_index.get_checksum()


@pytest.mark.parametrize("source,target", [("deepcell", "plain"), ("folder", "mixed"), ("deepfolder", "folder")])
def test_null_deep_checksum_short_circuits(source, target):
    null = _held(None, "plain")
    expression = Expression(null.get_checksum(), input_celltype=source, celltype=target)
    assert expression.compute(execution="local") == null.get_checksum()


# --- What a deep buffer is -------------------------------------------------


def test_deep_index_buffers_are_byte_identical_across_the_three_celltypes():
    index = {"dir/a.txt": "ab" * 32, "b.bin": "cd" * 32}
    contents = {Buffer(index, celltype).content for celltype in (*DEEP, "plain")}
    assert len(contents) == 1


def test_checksum_object_index_round_trips_to_the_same_checksum():
    """'Serializing such a dict back at a deep celltype writes the same hex JSON.'"""
    index = {"dir/a.txt": "ab" * 32}
    value = Buffer(index, "deepcell").get_value("deepcell")
    assert all(isinstance(member, Checksum) for member in value.values())
    for celltype in DEEP:
        assert Buffer(value, celltype).get_checksum() == Buffer(index, "plain").get_checksum()


@pytest.mark.parametrize("celltype", DEEP)
def test_checksum_resolve_at_deep_celltype_yields_checksum_members(celltype):
    member = _absent_checksum(b"deep-celltypes resolve member never stored")
    index = _held({"k": member.hex()}, "plain")
    value = index.get_checksum().resolve(celltype)
    assert value == {"k": member}
    assert isinstance(value["k"], Checksum)


# --- Flatness: when it is checked, and the failure -------------------------


def test_constructing_a_deep_expression_does_not_parse_the_index():
    nested = _held({"n": {"x": "aa" * 32}}, "plain")
    # construction checks celltypes and path shape only
    Expression(nested.get_checksum(), path="['n']", input_celltype="deepcell", celltype="mixed")
    Expression(nested.get_checksum(), input_celltype="folder", celltype="mixed")


@pytest.mark.parametrize(
    "bad_index,match",
    [
        ({"n": {"x": "aa" * 32}}, "nested"),
        ({"n": ["aa" * 32]}, "nested"),
        ({"n": "AA" * 32}, "n"),
        ({"n": "aa" * 31}, "n"),
        ({"n": "not a checksum"}, "n"),
        (["aa" * 32], "flat"),
    ],
    ids=["nested-dict", "nested-list", "uppercase", "short", "text", "not-a-dict"],
)
@pytest.mark.parametrize("source,target", [("deepcell", "mixed"), ("deepfolder", "bytes"), ("folder", "checksum")])
def test_one_step_path_fails_a_false_deep_claim_as_expression_evaluation_error(bad_index, match, source, target):
    """One-step path parses the index: validator ValueError surfaces as ExpressionEvaluationError."""
    buffer = _held(bad_index, "plain")
    expression = Expression(buffer.get_checksum(), path="['n']", input_celltype=source, celltype=target)
    with pytest.raises(ExpressionEvaluationError, match=match) as info:
        expression.compute(execution="local")
    assert not isinstance(info.value, HashTypeValidationError)
    assert error_kind(info.value) == "expression_evaluation"


_SYNC_FOLDER_GAP = pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §When flatness is checked: the sync folder -> mixed path "
    "(_evaluate_expression_after_validation) parses the index with the lenient "
    "Buffer.get_value and never calls the shared validator; a non-hex member "
    "crashes with AttributeError instead of ExpressionEvaluationError",
)


@pytest.mark.parametrize(
    "bad_index",
    [
        pytest.param({"n": {"x": "aa" * 32}}, id="nested"),
        pytest.param({"n": "not a checksum"}, id="text", marks=_SYNC_FOLDER_GAP),
        pytest.param({"n": "AA" * 32}, id="uppercase", marks=_SYNC_FOLDER_GAP),
    ],
)
def test_folder_to_mixed_fails_a_false_deep_claim_as_expression_evaluation_error(bad_index):
    buffer = _held(bad_index, "plain")
    expression = Expression(buffer.get_checksum(), input_celltype="folder", celltype="mixed")
    with pytest.raises(ExpressionEvaluationError) as info:
        expression.compute(execution="local")
    assert not isinstance(info.value, HashTypeValidationError)
    assert error_kind(info.value) == "expression_evaluation"


@pytest.mark.parametrize(
    "bad,match",
    [
        ({1: "aa" * 32}, "1"),
        ({"k": {"x": "aa" * 32}}, "k"),
        ({"k": "AA" * 32}, "k"),
        ({"k": "aa" * 31}, "k"),
        ({"k": 5}, "k"),
        ("not a dict", "flat"),
    ],
    ids=["int-key", "nested", "uppercase", "short", "int-member", "not-a-dict"],
)
def test_shared_validator_raises_value_error_naming_the_offence(bad, match):
    with pytest.raises(ValueError, match=match) as info:
        validate_deep_structure(bad)
    assert not isinstance(info.value, HashTypeValidationError)


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §When flatness is checked: reading .value at a deep celltype must reject "
    "a flat dict whose members are not 64-hex checksums; Buffer.get_value falls back to "
    "validate_deep_structure(index=False) and returns it",
)
@pytest.mark.parametrize("celltype", DEEP)
@pytest.mark.parametrize(
    "not_an_index",
    [{"k": "hello"}, {"k": 1}, {"k": "AA" * 32}],
    ids=["text", "int", "uppercase"],
)
def test_reading_value_at_deep_celltype_rejects_a_flat_non_index(celltype, not_an_index):
    buffer = Buffer(not_an_index, "plain")
    with pytest.raises(ValueError):
        buffer.get_value(celltype)


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §When flatness is checked: Cell.value at a deep celltype over a flat "
    "non-index must fail; the code returns the raw dict and records no exception",
)
@pytest.mark.parametrize("celltype", DEEP)
def test_cell_value_at_deep_celltype_rejects_a_flat_non_index(celltype):
    buffer = _held({"k": "hello"}, "plain")
    cell = Cell(celltype, checksum=buffer.get_checksum())
    cell.compute()
    with pytest.raises(Exception):
        cell.value
    assert cell.exception is not None


@pytest.mark.parametrize("celltype", DEEP)
def test_cell_value_at_deep_celltype_rejects_a_nested_index(celltype):
    buffer = _held({"n": {"x": "aa" * 32}}, "plain")
    cell = Cell(celltype, checksum=buffer.get_checksum())
    cell.compute()
    with pytest.raises(Exception, match="nested"):
        cell.value
    assert "nested" in cell.exception


# --- Paths -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["a.b/c.txt", "with space", "x['y']"])
def test_one_step_bracket_form_accepts_opaque_keys(key):
    child = _absent_checksum(b"deep-celltypes opaque-key member never stored")
    index = _held({key: child.hex()}, "plain")
    expression = Expression(index.get_checksum(), path=f"[{key!r}]", input_celltype="deepfolder", celltype="bytes")
    assert expression.compute(execution="local") == child


def test_one_step_on_a_missing_key_is_an_expression_evaluation_error():
    index = _held({"present": "99" * 32}, "plain")
    expression = Expression(index.get_checksum(), path="['absent']", input_celltype="deepfolder", celltype="bytes")
    with pytest.raises(ExpressionEvaluationError):
        expression.compute(execution="local")


def test_one_step_member_run_materializes_the_child_and_checksum_run_returns_a_checksum():
    member = _held({"x": 1}, "mixed")
    index = _held({"k": member.get_checksum().hex()}, "deepcell")
    as_member = Expression(index.get_checksum(), path="['k']", input_celltype="deepcell", celltype="mixed")
    as_checksum = Expression(index.get_checksum(), path="['k']", input_celltype="deepcell", celltype="checksum")
    assert as_member.run() == {"x": 1}
    reference = as_checksum.run()
    assert isinstance(reference, Checksum)
    assert reference == member.get_checksum()


def test_checksum_target_dereferences_back_to_the_child():
    """'The two targets therefore converge' via checksum -> X dereference."""
    member = _held({"x": 2}, "mixed")
    index = _held({"k": member.get_checksum().hex()}, "deepcell")
    reference = Expression(index.get_checksum(), path="['k']", input_celltype="deepcell", celltype="checksum")
    dereferenced = Expression(reference, input_celltype="checksum", celltype="mixed")
    assert dereferenced.compute(execution="local") == member.get_checksum()


def test_one_step_member_validation_uses_the_childs_known_hash_type():
    """Member check stays at checksum level: a child HashType proves not-mixed -> refuse."""
    raw = _held(b"\xff\xfe\x00 not json, not mixed")
    hash_type = ensure_hash_type(raw.get_checksum(), buffer=raw)
    assert hash_type.deserializable_as("mixed", checksum=raw.get_checksum()) is False
    index = _held({"k": raw.get_checksum().hex()}, "deepcell")
    expression = Expression(index.get_checksum(), path="['k']", input_celltype="deepcell", celltype="mixed")
    with pytest.raises(HashTypeValidationError):
        expression.compute(execution="local")


@pytest.mark.parametrize("celltype,member", [("deepcell", "mixed"), ("deepfolder", "bytes"), ("folder", "bytes")])
def test_projected_cell_takes_the_member_celltype(celltype, member):
    """Current-status bug 'A projected Cell inherits the parent's deep celltype' is fixed."""
    index = _held({"k": "99" * 32}, "plain")
    root = Cell(celltype, checksum=index.get_checksum())
    child = root["k"]
    assert child.celltype == member
    assert child.input_celltype == celltype


# --- folder -> mixed -------------------------------------------------------


def test_folder_to_mixed_is_all_or_nothing_and_a_missing_child_is_not_sticky():
    present = _held(b"present child")
    absent_content = b"deep-celltypes all-or-nothing child, stored only later"
    absent = _absent_checksum(absent_content)
    index = _held({"p": present.get_checksum().hex(), "a": absent.hex()}, "folder")

    first = Expression(index.get_checksum(), input_celltype="folder", celltype="mixed")
    with pytest.raises(CacheMissError):
        first.compute(execution="local")

    late = _held(absent_content)
    assert late.get_checksum() == absent
    second = Expression(index.get_checksum(), input_celltype="folder", celltype="mixed")
    value = second.run()
    assert set(value) == {"p", "a"}
    assert value["a"].tobytes() == absent_content
    assert value["p"].dtype == np.dtype("S1")


def test_empty_folder_survives_folder_to_mixed_to_plain():
    empty = _held({}, "folder")
    as_mixed = Expression(empty.get_checksum(), input_celltype="folder", celltype="mixed")
    assert as_mixed.run() == {}
    as_plain = Expression(as_mixed, input_celltype="mixed", celltype="plain")
    assert as_plain.run() == {}
    assert as_plain.compute(execution="local") == as_mixed.compute(execution="local")


@pytest.mark.parametrize("content", [b"all utf-8 text\n", b"\x00\xff binary"], ids=["utf8", "binary"])
def test_non_empty_folder_to_mixed_to_plain_is_rejected(content):
    child = _held(content)
    index = _held({"f": child.get_checksum().hex()}, "folder")
    as_mixed = Expression(index.get_checksum(), input_celltype="folder", celltype="mixed")
    mixed_checksum = as_mixed.compute(execution="local")
    mixed_buffer = mixed_checksum.resolve()
    with pytest.raises(SeamlessConversionError):
        convert_checksum(mixed_checksum, "mixed", "plain", lambda: mixed_buffer)
    as_plain = Expression(as_mixed, input_celltype="mixed", celltype="plain")
    with pytest.raises((SeamlessConversionError, HashTypeValidationError)):
        as_plain.compute(execution="local")
