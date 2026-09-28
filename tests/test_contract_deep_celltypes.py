"""Contract coverage for docs/agent/contracts/deep-celltypes.md (feature 4, deep side).

Complements test_expression_contract.py / test_conversion_contract.py, which already pin
the illegal-pair spot checks, the one-step path, and the S1 child representation.
This file adds the exhaustive zero-path table, the flatness phase table, the
validator's failure shapes, folder -> mixed all-or-nothing / non-sticky, and the
value rule ({key: Checksum}) at the value layers.

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
    """'Only these are legal ... everything else: illegal: refused at construction with ValueError'."""
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


@pytest.mark.parametrize(
    "bad_index",
    [
        pytest.param({"n": {"x": "aa" * 32}}, id="nested"),
        pytest.param({"n": "not a checksum"}, id="text"),
        pytest.param({"n": "AA" * 32}, id="uppercase"),
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


@pytest.mark.parametrize("celltype", DEEP)
def test_cell_value_at_deep_celltype_rejects_a_flat_non_index(celltype):
    buffer = _held({"k": "hello"}, "plain")
    cell = Cell(celltype, checksum=buffer.get_checksum())
    cell.compute()
    # A read that fails to materialize an existing result raises, every time,
    # and records nothing (contracts/cells.md, ruled 2026-09-28).
    for _ in range(2):
        with pytest.raises(Exception):
            cell.value
        assert cell.exception is None
        assert cell.state == "complete"


@pytest.mark.parametrize("celltype", DEEP)
def test_cell_value_at_deep_celltype_rejects_a_nested_index(celltype):
    buffer = _held({"n": {"x": "aa" * 32}}, "plain")
    cell = Cell(celltype, checksum=buffer.get_checksum())
    cell.compute()
    for _ in range(2):
        with pytest.raises(Exception, match="nested"):
            cell.value
        assert cell.exception is None
        assert cell.state == "complete"


@pytest.mark.parametrize("celltype", DEEP)
def test_cell_value_and_buffer_at_deep_celltype_are_the_unresolved_index(celltype):
    """'.value' is {key: Checksum}; '.buffer' returns the index buffer; nothing is resolved."""
    absent = _absent_checksum(b"deep-celltypes cell value member never stored " + celltype.encode())
    index = _held({"k": absent.hex()}, "plain")
    cell = Cell(celltype, checksum=index.get_checksum())
    value = cell.value
    assert value == {"k": absent}
    assert type(value["k"]) is Checksum
    assert cell.buffer.get_checksum() == index.get_checksum()


@pytest.mark.parametrize("celltype", DEEP)
def test_cell_buffer_at_deep_celltype_validates_at_the_mapped_celltype(celltype):
    """'.buffer ... validates against the same mapped celltype as .value' (plain)."""
    not_json = _held(b"\xff\xfe deep-celltypes not json at all")
    cell = Cell(celltype, checksum=not_json.get_checksum())
    with pytest.raises(HashTypeValidationError):
        cell.buffer


@pytest.mark.parametrize("celltype", DEEP)
def test_cell_buffer_at_deep_celltype_rejects_a_nested_index(celltype):
    nested = _held({"n": {"x": "aa" * 32}}, "plain")
    cell = Cell(celltype, checksum=nested.get_checksum())
    with pytest.raises(Exception, match="nested"):
        cell.buffer


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
    """'One step below a deep parent, a projection or handle carries the member celltype' (standalone)."""
    index = _held({"k": "99" * 32}, "plain")
    root = Cell(celltype, checksum=index.get_checksum())
    child = root["k"]
    assert child.celltype == member
    assert child.input_celltype == celltype


@pytest.mark.parametrize("celltype", DEEP)
def test_standalone_handle_reads_the_child_checksum_value_and_reference_form(celltype):
    """d['k'].checksum is the child's checksum, .value the child's value,
    .as_celltype('checksum') the reference form (Handles and writes one step below a deep parent)."""
    if celltype == "deepcell":
        member, member_value = _held({"x": 1}, "mixed"), {"x": 1}
    else:
        member, member_value = _held(b"member bytes"), b"member bytes"
    index = _held({"k": member.get_checksum().hex()}, "plain")
    handle = Cell(celltype, checksum=index.get_checksum())["k"]
    assert handle.checksum == member.get_checksum()
    assert handle.value == member_value
    reference = handle.as_celltype("checksum")
    assert reference.celltype == "checksum"
    assert reference.checksum == Buffer(member.get_checksum(), "checksum").get_checksum()
    assert isinstance(reference.value, Checksum)
    assert reference.value == member.get_checksum()


@pytest.mark.parametrize("celltype", DEEP)
def test_standalone_handle_on_an_absent_member_evaluates_from_the_index_alone(celltype):
    """Both targets are evaluated from the index buffer alone; no member buffer is fetched."""
    absent = _absent_checksum(b"deep-celltypes handle over an absent member " + celltype.encode())
    index = _held({"k": absent.hex()}, "plain")
    handle = Cell(celltype, checksum=index.get_checksum())["k"]
    assert handle.checksum == absent
    assert handle.as_celltype("checksum").checksum == Buffer(absent, "checksum").get_checksum()


@pytest.mark.parametrize("form", ["checksum", "buffer", "set_checksum"])
@pytest.mark.parametrize("celltype", DEEP)
def test_clearing_through_a_deep_handle_is_refused_with_value_error(celltype, form):
    """'Clearing through the handle follows the general sub-path rule of cells.md (ValueError).'"""
    member = _held({"v": 1}, "mixed") if celltype == "deepcell" else _held(b"member")
    index = _held({"k": member.get_checksum().hex()}, "plain")
    root = Cell(celltype, checksum=index.get_checksum())
    with pytest.raises(ValueError):
        if form == "set_checksum":
            root["k"].set_checksum(None)
        else:
            setattr(root["k"], form, None)
    assert root.checksum == index.get_checksum()


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


# --- Writes through a handle one step below a deep parent (cells.md, Writes through a handle) ---

def _member_buffer(celltype):
    if celltype == "deepcell":
        return _held({"v": "new deep member"}, "mixed")
    return _held(b"new folder member bytes")


@pytest.mark.parametrize("form", ["checksum", "set_checksum", "buffer", "set_buffer"])
@pytest.mark.parametrize("celltype", DEEP)
def test_handle_write_below_a_deep_parent_replaces_the_member_checksum(celltype, form):
    old = _held({"v": "old"}, "mixed") if celltype == "deepcell" else _held(b"old bytes")
    index = _held({"k": old.get_checksum().hex(), "other": old.get_checksum().hex()}, "plain")
    new = _member_buffer(celltype)
    root = Cell(celltype, checksum=index.get_checksum())
    handle = root["k"]
    argument = new.get_checksum() if "checksum" in form else new
    if form.startswith("set_"):
        getattr(handle, form)(argument)
    else:
        setattr(handle, form, argument)
    assert root.value == {"k": new.get_checksum(), "other": old.get_checksum()}
    assert handle.checksum == new.get_checksum()


@pytest.mark.parametrize("celltype", DEEP)
def test_handle_checksum_write_below_a_deep_parent_with_absent_buffer_records_nothing(celltype):
    old = _held(b"old bytes") if celltype != "deepcell" else _held({"v": "old"}, "mixed")
    index = _held({"k": old.get_checksum().hex()}, "plain")
    root = Cell(celltype, checksum=index.get_checksum())
    absent = _absent_checksum(b"deep-celltypes handle write, never stored " + celltype.encode())
    with pytest.raises(CacheMissError):
        root["k"].checksum = absent
    assert root.checksum == index.get_checksum()


# --- Null on illegal deep pairs; value writes and writes below k ------------

ILLEGAL_DEEP_PAIRS = [
    ("deepfolder", "deepcell"),
    ("folder", "plain"),
    ("plain", "deepcell"),
    ("plain", "folder"),
    ("deepcell", "mixed"),
    ("deepcell", "int"),
]


@pytest.mark.parametrize("source,target", ILLEGAL_DEEP_PAIRS)
def test_null_checksum_does_not_make_an_illegal_deep_pair_legal_at_construction(source, target):
    null = _held(None, "plain")
    with pytest.raises(ValueError):
        Expression(null.get_checksum(), input_celltype=source, celltype=target)


@pytest.mark.parametrize("source,target", ILLEGAL_DEEP_PAIRS)
def test_null_checksum_does_not_make_an_illegal_deep_pair_legal_in_the_engine(source, target):
    null = _held(None, "plain")

    def fail():
        raise AssertionError("an illegal deep pair must be refused without a fetch")

    with pytest.raises(SeamlessConversionError):
        convert_checksum(null.get_checksum(), source, target, fail)


@pytest.mark.parametrize("celltype", DEEP)
def test_value_write_at_k_inserts_the_member_checksum(celltype):
    old = _held({"v": "old"}, "mixed") if celltype == "deepcell" else _held(b"old bytes")
    index = _held({"k": old.get_checksum().hex(), "other": old.get_checksum().hex()}, "plain")
    root = Cell(celltype, checksum=index.get_checksum())
    new_value = {"v": "new"} if celltype == "deepcell" else b"new bytes"
    expected = Buffer(new_value, "mixed" if celltype == "deepcell" else "bytes").get_checksum()
    root["k"].set(new_value)
    assert root.value == {"k": expected, "other": old.get_checksum()}


@pytest.mark.parametrize("form", ["set", "checksum", "buffer"])
@pytest.mark.parametrize("celltype", DEEP)
def test_write_below_k_is_refused_and_records_nothing(celltype, form):
    """Ruling: writes below a deep parent are illegal below k (exception type unruled)."""
    member = _held({"v": 1}, "mixed")
    index = _held({"k": member.get_checksum().hex()}, "plain")
    root = Cell(celltype, checksum=index.get_checksum())
    target = root["k"]["v"]
    replacement = _held(5, "mixed")
    with pytest.raises(Exception):
        if form == "set":
            target.set(5)
        elif form == "checksum":
            target.checksum = replacement.get_checksum()
        else:
            target.buffer = replacement
    assert root.checksum == index.get_checksum()
