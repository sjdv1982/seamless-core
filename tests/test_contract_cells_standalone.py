"""Standalone Cell rules from contracts/cells.md not pinned by the existing suites.

Companion of test_cells_contract_alignment.py (feature 5 alignment pass,
2026-09-22); this file only adds rules that pass missed. Bound counterparts
live in seamless-workflow/tests/test_contract_cells_bound.py.
"""
import asyncio
import re

import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.cell_errors import ProjectionError
from seamless.checksum.expression import get_expression_cache
from seamless.error_envelope import RunningLoopRefusal
from tests.helpers.fake_remotes import drop_buffer, install_fake_remotes


def _held(value, celltype):
    buffer = Buffer(value, celltype)
    return buffer, buffer.tempref()


# --- The definition -------------------------------------------------------

@pytest.mark.parametrize("where", ["constructor", "setter", "as_celltype"])
def test_unsupported_celltype_raises_type_error_listing_supported(where):
    """cells.md *The definition*: an unsupported name raises TypeError listing the supported celltypes."""
    with pytest.raises(TypeError) as info:
        if where == "constructor":
            Cell("no-such-celltype")
        elif where == "setter":
            cell = Cell("int")
            cell.celltype = "no-such-celltype"
        else:
            Cell("int").as_celltype("no-such-celltype")
    message = str(info.value)
    assert "no-such-celltype" in message
    for supported in ("plain", "text", "bytes", "checksum"):
        assert supported in message


def test_bare_checksum_as_source_names_the_checksum_keyword():
    """cells.md *The definition*: a bare Checksum passed as source= raises TypeError naming checksum=."""
    with pytest.raises(TypeError, match=r"checksum="):
        Cell("int", source=Checksum("ab" * 32))


def test_checksum_and_source_together_is_a_type_error():
    """cells.md *The definition*: checksum= and source= are mutually exclusive (TypeError)."""
    with pytest.raises(TypeError):
        Cell("int", checksum=Checksum("ab" * 32), source=Cell("int"))


def test_conflicting_input_declaration_is_a_value_error():
    """cells.md *The definition*: with a typed source, a different input_celltype raises ValueError."""
    with pytest.raises(ValueError):
        Cell("int", source=Cell("text"), input_celltype="plain")


# --- Failures ---------------------------------------------------------------

def _failed_cell():
    cell = Cell("str")
    cell.set("not an integer")
    cell.celltype = "int"
    assert cell.checksum is None
    assert cell.state == "failed"
    assert isinstance(cell.exception, str) and cell.exception
    return cell


@pytest.mark.parametrize("change", ["celltype", "validator", "validator_language"])
def test_failure_is_reset_when_the_recipe_changes(change):
    """cells.md *Failures*: the standalone failure is reset by a new celltype, validator or validator_language.

    Inspection stays passive afterwards: the new recipe is waiting, not re-evaluated.
    """
    cell = _failed_cell()
    if change == "celltype":
        cell.celltype = "text"
    elif change == "validator":
        cell.validator = Checksum("ab" * 32)
    else:
        cell.validator_language = "python"
    assert cell.exception is None
    assert cell.state == "waiting"


def test_new_cell_on_the_same_recipe_does_not_inherit_a_failure():
    """cells.md *Failures*: a failure lives on the handle, never in the substrate."""
    _failed_cell()
    fresh = Cell("str")
    fresh.set("not an integer")
    fresh.celltype = "int"
    assert fresh.exception is None
    assert fresh.state == "waiting"
    # A deterministic failure simply reproduces when the fresh handle pulls.
    assert fresh.checksum is None
    assert fresh.state == "failed"
    assert isinstance(fresh.exception, str)


def test_compute_records_failure_and_run_reraises():
    """cells.md *Work*, notes: standalone compute() records the failure and returns None; run() re-raises it."""
    cell = Cell("str")
    cell.set("not an integer")
    cell.celltype = "int"
    assert cell.compute() is None
    assert cell.state == "failed"
    message = cell.exception
    assert isinstance(message, str) and message
    with pytest.raises(Exception) as info:
        cell.run()
    assert str(info.value) == message


# --- The input: .checksum table ------------------------------------------------

def test_connected_checksum_follows_the_conversion_class():
    """cells.md *The input*: the connected rows of the .checksum table, standalone."""
    s5 = Cell("str")
    s5.set("5")
    preserving = Cell("int", source=s5)  # str -> int is reinterpret: same checksum
    assert preserving.checksum == s5.checksum
    assert preserving.value == 5

    i42 = Cell("int")
    i42.set(42)
    widened = Cell("float", source=i42)  # int -> float: same checksum
    assert widened.checksum == i42.checksum
    assert widened.value == 42.0

    hello = Cell("str")
    hello.set("hello")
    reformatted = Cell("text", source=hello)  # str -> text reformats
    assert reformatted.checksum != hello.checksum
    assert reformatted.checksum == Buffer("hello", "text").get_checksum()
    assert reformatted.value == "hello"

    unwired = Cell("int")
    downstream = Cell("int", source=unwired)
    assert downstream.source is unwired
    assert downstream.checksum is None


# --- Connecting: the wiring rule -------------------------------------------------

def test_wiring_rule_compares_against_celltype_not_input_celltype():
    """cells.md *Connecting*: the comparison is against celltype, never input_celltype.

    A cell whose declared input is text but whose celltype is plain may be
    rewired to a projected plain source: a divergence may end behind a path.
    """
    text_buffer, text_hold = _held("[1, 2]", "text")
    plain_buffer, plain_hold = _held({"a": [7]}, "plain")
    try:
        declared = Cell("plain", checksum=text_buffer.get_checksum(), input_celltype="text")
        assert declared.input_celltype == "text"
        source = Cell("plain", checksum=plain_buffer.get_checksum())
        rewired = declared.with_input(source["a"])
        assert rewired.celltype == "plain"
        assert rewired.value == [7]
        with pytest.raises(TypeError):
            Cell("text").with_input(source["a"])
    finally:
        text_hold.clear()
        plain_hold.clear()


def test_projected_source_with_matching_celltype_is_legal():
    """cells.md *Connecting*: legal iff the new source carries no path or its celltype equals the cell's."""
    source = Cell("text")
    source.set("[10, 20, 30, 40]")
    consumer = Cell("text", source=source[3])
    assert consumer.value == ","
    assert consumer.state == "complete"


# --- Scratch policy -------------------------------------------------------------

@pytest.mark.parametrize("derive", ["with_input", "with_validator"])
def test_modified_copies_keep_the_scratch_flag(derive):
    """cells.md *Scratch policy*: with_input() and with_validator() keep the flag."""
    cell = Cell("int")
    cell.set(3)
    cell.scratch = True
    if derive == "with_input":
        copy = cell.with_input(Buffer(4, "int").get_checksum())
    else:
        copy = cell.with_validator(Checksum("ab" * 32))
    assert copy is not cell
    assert copy.scratch is True


def test_slice_projection_starts_non_scratch():
    """cells.md *Scratch policy*: a projection is a new Cell that owns its own result."""
    cell = Cell("plain")
    cell.set([1, 2, 3])
    cell.scratch = True
    assert cell[0:2].scratch is False
    assert cell.slice(0, 2).scratch is False


# --- Projections: API-name arbitration ---------------------------------------------

@pytest.mark.parametrize("name", ["mount", "block_reason"])
def test_bound_only_members_never_fall_through_to_projection(name):
    """cells.md *Projections*: a bound-only member raising AttributeError must not project."""
    cell = Cell("plain")
    cell.set({name: 1})
    with pytest.raises(AttributeError):
        getattr(cell, name)
    projected = cell[name]
    assert isinstance(projected, Cell)
    assert projected.path == name
    assert projected.value == 1


def test_prune_is_bound_only_and_underscore_names_never_project():
    """cells.md *Work* / *Projections*: prune() is bound-only; names starting with _ never project."""
    cell = Cell("plain")
    cell.set({"_hidden": 1})
    with pytest.raises(AttributeError):
        cell.prune()
    with pytest.raises(AttributeError):
        cell._hidden
    assert cell["_hidden"].value == 1


# --- Reads: running-loop refusal ---------------------------------------------------

def test_running_loop_refusal_leaves_state_waiting(monkeypatch):
    """cells.md *Reads*: a refusal gives None, .exception None and state waiting (not failed)."""
    get_expression_cache().clear()
    source = Checksum("df" * 32)
    drop_buffer(source)
    result = Buffer("remote cell read, state", "str")
    result_ref = result.tempref()
    calls = []
    key = (source.hex(), "", "plain", "str")
    install_fake_remotes(
        monkeypatch, {}, {key: result.get_checksum().hex()}, calls,
        jobserver_available=True,
    )
    cell = Cell("str", checksum=source, input_celltype="plain")

    async def read_on_running_loop():
        with pytest.raises(RunningLoopRefusal):
            Expression(source, input_celltype="plain", celltype="str").compute()
        assert cell.checksum is None
        assert cell.exception is None
        assert cell.state == "waiting"
        assert "jobserver:run" not in calls

    try:
        asyncio.run(read_on_running_loop())
        assert cell.checksum == result.get_checksum()
        assert cell.state == "complete"
    finally:
        result_ref.clear()


# --- Ruling 8: the wiring refusal text is contract ------------------------------------------

@pytest.mark.xfail(strict=False, reason=(
    "cells.md §Connecting / ruling 8: contract ahead of code: the refusal says 'Cannot implicitly convert "
    "behind a projection; use as_celltype() before or after projecting' instead of the two-spelling message"))
@pytest.mark.parametrize("api", ["constructor", "with_input"])
def test_wiring_refusal_message_names_both_spellings(api):
    """cells.md §Connecting: header line, then the two spellings (projection first), each glossed.

    A standalone Cell has no name, so the unknown part is omitted and each
    spelling starts at the step (the same rendering pins.md's standalone test pins).
    """
    source = Cell("text")
    source.set("[10, 20, 30, 40]")
    with pytest.raises(TypeError) as info:
        if api == "constructor":
            Cell("plain", source=source[3])
        else:
            Cell("plain").with_input(source[3])
    lines = str(info.value).splitlines()
    assert lines[0] == "would convert text -> plain behind a projection."
    assert len(lines) == 3
    projection_first, conversion_first = lines[1], lines[2]
    assert re.match(r'^\s*(…|\.\.\.)\[3\]\.as_celltype\("plain"\)\s+'
                    r'# item 3 of the text \(a character\), as plain$', projection_first)
    assert re.match(r'^\s*(…|\.\.\.)\.as_celltype\("plain"\)\[3\]\s+'
                    r'# item 3 of the parsed list$', conversion_first)


# --- Projections: a Cell is a handle ----------------------------------------------------

def test_list_membership_raises_on_a_non_identical_member_but_set_and_dict_do_not():
    """cells.md §Projections: `x in [cells]` compares by identity first, then with ==, so it raises
    ProjectionError as soon as it meets a non-identical member; hash-based membership does not."""
    a = Cell("int")
    a.set(1)
    b = Cell("int")
    b.set(1)
    assert a in [a, b]  # identity hit before any == comparison
    with pytest.raises(ProjectionError):
        a in [b, a]
    assert a in {b, a}
    assert a in {a: 1}
    assert hash(a) == hash(a) and hash(a) != hash(b)


def test_value_unavailable_error_location():
    """cells.md code-location table and §Projections: ValueUnavailableError lives in
    seamless.cell_errors and is importable as seamless.ValueUnavailableError."""
    import seamless
    from seamless import cell_errors
    assert seamless.ValueUnavailableError is cell_errors.ValueUnavailableError
    assert issubclass(cell_errors.AuthorityError, cell_errors.WorkflowError)
    assert issubclass(ProjectionError, TypeError) and issubclass(ProjectionError, AttributeError)


# --- .set() / .value = take values only -------------------------------------------------

def test_reference_refusal_messages():
    """cells.md §`.set()` and `.value =` take values only: the quoted TypeError texts."""
    target = Cell("int")
    other = Cell("int")
    other.set(3)
    with pytest.raises(TypeError) as info:
        target.set(Checksum("ab" * 32))
    assert str(info.value) == ("A Checksum is not a value: use .set_checksum() "
                               "(a Checksum is a value only for celltype 'checksum')")
    with pytest.raises(TypeError) as info:
        target.value = other
    assert str(info.value) == "A Cell is not a value: connect it by assignment, or with Cell(source=...)"
    assert target.checksum is None


def test_a_64_character_str_is_always_a_value():
    """cells.md §Celltype checksum: a 64-character str is a value, never a checksum;
    Checksum == hex_str is true, but the two hash differently."""
    digest = Checksum("ab" * 32)
    cell = Cell("str")
    cell.set(digest.hex())
    assert cell.value == digest.hex()
    assert cell.checksum == Buffer(digest.hex(), "str").get_checksum()
    assert cell.checksum != digest
    assert digest == digest.hex()
    assert hash(digest) != hash(digest.hex())


# --- Null and None: null only on legal pairs ------------------------------------------------

@pytest.mark.xfail(strict=False, reason=(
    "cells.md §Null and None / clarity ruling (null short-circuits only on legal pairs): contract ahead of "
    "code: a standalone null as_celltype child over an illegal pair reads back a complete NULL"))
@pytest.mark.parametrize("celltype", ["python", "ipython", "deepcell", "deepfolder", "folder"])
def test_null_as_celltype_child_over_an_illegal_pair_fails(celltype):
    parent = Cell(celltype)
    parent.set(None)
    child = parent.as_celltype("int")
    assert child.checksum is None
    assert child.state == "failed"
    assert isinstance(child.exception, str) and child.exception
