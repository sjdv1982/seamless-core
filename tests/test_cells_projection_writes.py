"""Standalone projection writes are pathed writes to the standalone parent.

cells.md, as ruled in register/cells-RULINGS.md round 9 (2026-09-25; option B):
on a standalone Cell `c`, every write through a projection child (`p = c.b`:
the six value/buffer/checksum verbs and `p += 1`), attribute and item
assignment (`c.b = v`, `c["b"] = v`) and root augmented assignment (`c += 1`)
are read-modify-set writes of `c`'s root value. Checksum and buffer forms are
resolved at the child's celltype. `AuthorityError` if `c` has a source; the
clearing forms raise `ValueError`; writes through an `as_celltype` child raise
`AuthorityError`. This supersedes "sub-path writes and augmented assignment are
bound-only". The seamless-workflow file of the same name holds the bound cases.

These tests assert the ruled result directly.
"""
import pytest
from seamless import Buffer, CacheMissError, Cell, Checksum
from seamless.cell_errors import AuthorityError, ValueUnavailableError


FORMS = ["value", "buffer", "checksum"]


def write(cell, form, method, value):
    if method:
        getattr(cell, {"value": "set", "buffer": "set_buffer", "checksum": "set_checksum"}[form])(value)
    else:
        setattr(cell, form, value)


def _root(value=None):
    root = Cell("plain")
    root.set({"a": 1, "other": 2} if value is None else value)
    return root


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("method", [False, True])
def test_projection_write_matrix(form, method):
    root = _root()
    projected = root["a"]
    assert projected.value == 1
    buffer = Buffer(3, "plain")
    hold = buffer.tempref()
    try:
        value = {"value": 3, "buffer": buffer, "checksum": buffer.get_checksum()}[form]
        write(projected, form, method, value)
        assert root.value == {"a": 3, "other": 2}
        assert root.source is None
        # The child's memo is keyed by the parent checksum: the next read re-evaluates.
        assert projected.value == 3
        assert root["a"].value == 3
    finally:
        hold.clear()


@pytest.mark.parametrize("method", [False, True])
def test_nested_projection_write_updates_the_root(method):
    root = _root({"a": {"b": 1}, "other": 2})
    child = root.a.b
    if method:
        child.set(5)
    else:
        child.value = 5
    assert root.value == {"a": {"b": 5}, "other": 2}
    assert child.value == 5


@pytest.mark.parametrize("form", ["checksum", "set_checksum"])
def test_projection_checksum_write_is_resolved_at_the_child_celltype(form):
    root = _root()
    four = Buffer(4, "plain")
    hold = four.tempref()
    try:
        if form == "checksum":
            root.a.checksum = four.get_checksum()
        else:
            root.a.set_checksum(four.get_checksum())
        assert root.value == {"a": 4, "other": 2}
    finally:
        hold.clear()


def test_projection_set_checksum_with_input_celltype_converts_first():
    root = _root()
    five = Buffer("5", "str")
    hold = five.tempref()
    try:
        root.a.set_checksum(five.get_checksum(), input_celltype="str")
        assert root.value == {"a": "5", "other": 2}
    finally:
        hold.clear()


@pytest.mark.parametrize("form", ["checksum", "set_checksum"])
def test_unresolvable_projection_checksum_write_raises_cache_miss(form):
    root = _root()
    missing = Buffer({"never": "stored"}, "plain").get_checksum()
    with pytest.raises(CacheMissError):
        if form == "checksum":
            root.a.checksum = missing
        else:
            root.a.set_checksum(missing)
    assert root.value == {"a": 1, "other": 2}
    assert root.exception is None


@pytest.mark.parametrize("form", ["checksum", "buffer", "set_checksum"],
                         ids=["checksum-None", "buffer-None", "set_checksum-None"])
def test_clearing_a_standalone_sub_path_raises_value_error(form):
    root = _root()
    with pytest.raises(ValueError) as info:
        if form == "set_checksum":
            root.a.set_checksum(None)
        else:
            setattr(root.a, form, None)
    assert "del" in str(info.value)
    assert root.value == {"a": 1, "other": 2}


def test_storing_null_through_a_child_is_a_pathed_write():
    root = _root()
    root.a.set(None)
    assert root.value == {"a": None, "other": 2}
    root.other.value = None
    assert root.value == {"a": None, "other": None}


_UNDER_SOURCE_WRITES = {
    "item": lambda root: root.__setitem__("a", 3),
    "attribute": lambda root: setattr(root, "a", 3),
    "iadd": lambda root: root.a.__iadd__(1),
    "root-iadd": lambda root: root.__iadd__({"b": 2}),
}


def _under_source(form, method):
    return pytest.param(form, method, id=(f"{form}-{'method' if method else 'property'}" if form in FORMS else form),
                        )


@pytest.mark.parametrize("form, method", [
    *[_under_source(form, method) for form in FORMS for method in (False, True)],
    _under_source("item", False),
    _under_source("attribute", False),
    _under_source("iadd", False),
    _under_source("root-iadd", False),
])
def test_projection_write_under_source_is_refused(form, method):
    source = Cell("plain")
    source.set({"a": 1})
    root = Cell("plain", source=source)
    buffer = Buffer(3, "plain")
    hold = buffer.tempref()
    try:
        with pytest.raises(AuthorityError):
            if form in _UNDER_SOURCE_WRITES:
                _UNDER_SOURCE_WRITES[form](root)
            else:
                value = {"value": 3, "buffer": buffer, "checksum": buffer.get_checksum()}[form]
                write(root["a"], form, method, value)
        assert root.value == source.value == {"a": 1}
        assert root.source is source
    finally:
        hold.clear()


def test_subpath_assignment_and_augmented_assignment_are_pathed_root_writes():
    root = _root({"a": 1, "y": [1, 2]})
    root["a"] = 2
    assert root.value == {"a": 2, "y": [1, 2]}
    root.a = 3
    assert root.value == {"a": 3, "y": [1, 2]}
    root.y[0] = 9
    assert root.value == {"a": 3, "y": [9, 2]}
    root.b = {"c": 4}
    assert root.value == {"a": 3, "y": [9, 2], "b": {"c": 4}}
    retained = root.a
    retained += 7
    assert root.value["a"] == 10
    root.a += 1
    assert root.value["a"] == 11
    assert root.source is None


def test_root_augmented_assignment_is_a_read_modify_set():
    cell = Cell("int")
    cell.set(1)
    original = cell
    cell += 1
    assert cell is original
    assert cell.value == 2
    cell *= 5
    assert cell.value == 10


_AS_CELLTYPE_WRITES = {
    "set": lambda x: x.set(3),
    "value": lambda x: setattr(x, "value", 3),
    "set_buffer": lambda x: x.set_buffer(Buffer(3, "mixed")),
    "buffer": lambda x: setattr(x, "buffer", Buffer(3, "mixed")),
    "set_checksum": lambda x: x.set_checksum(Buffer(3, "mixed").get_checksum()),
    "checksum": lambda x: setattr(x, "checksum", Buffer(3, "mixed").get_checksum()),
    "iadd": lambda x: x.__iadd__(1),
}

@pytest.mark.parametrize("form", [
    "set", "set_buffer", "set_checksum", "value", "buffer", "checksum", "iadd",
])
def test_writes_through_a_standalone_as_celltype_child_raise_authority_error(form):
    """Round 9 / round 7 item 1: a conversion has no general inverse."""
    root = _root({"k": 1})
    child = root.as_celltype("mixed")
    with pytest.raises(AuthorityError):
        _AS_CELLTYPE_WRITES[form](child)
    assert root.value == {"k": 1}
    assert child.source is root
    assert child.value == {"k": 1}


# --- Round 10: standalone `del` follows ruling B ------------------------------------------

def test_del_item_removes_the_key_from_the_root_value():
    root = _root()
    del root["a"]
    assert root.value == {"other": 2}
    assert root.source is None
    with pytest.raises(KeyError):
        del root["missing"]
    assert root.value == {"other": 2}


@pytest.mark.parametrize("spelling", ["chained", "retained-child"])
def test_nested_del_through_a_child_is_a_pathed_delete(spelling):
    root = _root({"a": {"b": 1, "c": 2}, "other": 3})
    if spelling == "chained":
        del root["a"]["b"]
    else:
        child = root.a
        del child["b"]
        assert child.value == {"c": 2}
    assert root.value == {"a": {"c": 2}, "other": 3}


@pytest.mark.parametrize("spelling", ["root", "child"])
def test_del_under_source_raises_authority_error(spelling):
    source = Cell("plain")
    source.set({"a": {"b": 1}})
    root = Cell("plain", source=source)
    with pytest.raises(AuthorityError):
        if spelling == "root":
            del root["a"]
        else:
            del root.a["b"]
    assert root.value == source.value == {"a": {"b": 1}}
    assert root.source is source


def test_del_through_a_standalone_as_celltype_child_raises_authority_error():
    root = _root({"k": 1})
    child = root.as_celltype("mixed")
    with pytest.raises(AuthorityError):
        del child["k"]
    assert root.value == {"k": 1}
    assert child.value == {"k": 1}


# --- Remaining clauses of ruling B (rounds 9-10) ------------------------------------------

def test_buffer_writes_through_a_child_are_validated_at_the_child_celltype():
    root = _root()
    root.a.set_buffer(Buffer(b'"x"\n'))
    assert root.value == {"a": "x", "other": 2}
    root.a.buffer = Buffer(b"[1, 2]\n")
    assert root.value == {"a": [1, 2], "other": 2}
    with pytest.raises(ValueError):
        root.a.buffer = Buffer(b"not json")
    assert root.value == {"a": [1, 2], "other": 2}


def test_pathed_write_is_visible_to_the_next_read_and_to_existing_handles():
    root = _root()
    downstream = Cell(source=root)
    sibling = root.a
    unrelated = root.other
    before = root.checksum
    assert (downstream.value, sibling.value, unrelated.value) == ({"a": 1, "other": 2}, 1, 2)
    root.a.set(5)
    assert root.checksum != before
    assert root.checksum == Buffer({"a": 5, "other": 2}, "plain").get_checksum()
    assert downstream.value == {"a": 5, "other": 2}
    assert sibling.value == 5
    assert unrelated.value == 2


@pytest.mark.parametrize("op, operand, expected", [
    ("iadd", 3, 7), ("isub", 3, 1), ("imul", 3, 12), ("itruediv", 2, 2.0),
], ids=["+=", "-=", "*=", "/="])
def test_every_augmented_operator_through_a_child_is_a_pathed_write(op, operand, expected):
    root = _root({"a": 4, "other": 2})
    child = root.a
    result = getattr(child, f"__{op}__")(operand)
    assert result is child
    assert root.value == {"a": expected, "other": 2}
    assert child.value == expected


@pytest.mark.parametrize("op, operand, expected", [
    ("iadd", 3, 7), ("isub", 3, 1), ("imul", 3, 12), ("itruediv", 2, 2.0),
], ids=["+=", "-=", "*=", "/="])
def test_every_augmented_operator_on_a_root_is_a_read_modify_set(op, operand, expected):
    cell = Cell("float")
    cell.set(4)
    assert getattr(cell, f"__{op}__")(operand) is cell
    assert cell.value == expected


@pytest.mark.parametrize("spelling", ["item", "attribute"])
def test_assignment_into_a_standalone_as_celltype_child_raises_authority_error(spelling):
    root = _root({"k": 1})
    child = root.as_celltype("mixed")
    with pytest.raises(AuthorityError):
        if spelling == "item":
            child["k"] = 2
        else:
            child.k = 2
    assert root.value == {"k": 1}


def test_unwired_standalone_cell_bootstraps_a_mapping_for_string_paths_only():
    cell = Cell("plain")
    cell.b.c = 12
    assert cell.value == {"b": {"c": 12}}
    fresh = Cell("plain")
    fresh["k"] = 1
    assert fresh.value == {"k": 1}
    # An integer path has nothing to bootstrap, and an explicitly stored null is not unwired.
    unwired = Cell("plain")
    with pytest.raises(ValueUnavailableError):
        unwired[0] = 1
    assert unwired.state == "unwired"
    null = Cell("plain")
    null.set(None)
    with pytest.raises(Exception):  # the exception type for a stored null is unspecified
        null.b = 1
    assert null.value is None



# --- Writes below a deep parent (clarity rulings, 2026-09-26) ------------------------------

def _deep_root(celltype, members):
    member_celltype = "mixed" if celltype == "deepcell" else "bytes"
    buffers = {key: Buffer(value, member_celltype) for key, value in members.items()}
    holds = [buffer.tempref() for buffer in buffers.values()]
    index = Buffer({key: buffer.get_checksum().hex() for key, buffer in buffers.items()}, "plain")
    holds.append(index.tempref())
    return Cell(celltype, checksum=index.get_checksum()), holds


@pytest.mark.parametrize("form", ["checksum", "set_checksum"])
def test_member_checksum_write_replaces_the_index_entry(form):
    """At the one-step key k of a deep parent, the member checksum replaces index[k]."""
    root, holds = _deep_root("deepcell", {"k": 1, "other": 2})
    replacement = Buffer(5, "mixed")
    holds.append(replacement.tempref())
    try:
        if form == "checksum":
            root["k"].checksum = replacement.get_checksum()
        else:
            root["k"].set_checksum(replacement.get_checksum())
        index = root.value
        assert index["k"] == replacement.get_checksum()
        assert index["other"] == Buffer(2, "mixed").get_checksum()
        assert root["k"].value == 5
    finally:
        for hold in holds:
            hold.clear()


@pytest.mark.parametrize("celltype, value", [
    ("deepcell", {"x": 5}), ("deepfolder", b"file bytes"), ("folder", b"folder bytes"),
])
def test_member_value_write_is_serialized_at_the_member_celltype(celltype, value):
    """A value write at k serializes the value at the member celltype and inserts its checksum."""
    member_celltype = "mixed" if celltype == "deepcell" else "bytes"
    root, holds = _deep_root(celltype, {"k": {"old": 1} if celltype == "deepcell" else b"old"})
    try:
        root["k"].set(value)
        entry = root.value["k"]
        assert isinstance(entry, Checksum)
        assert entry == Buffer(value, member_celltype).get_checksum()
    finally:
        for hold in holds:
            hold.clear()


@pytest.mark.parametrize("form", ["buffer", "set_buffer"])
@pytest.mark.parametrize("celltype, value", [("deepcell", {"x": 5}), ("folder", b"folder bytes")])
def test_member_buffer_write_inserts_the_buffer_checksum(form, celltype, value):
    """A buffer write at k validates the buffer at the member celltype and puts its checksum in index[k]."""
    member_celltype = "mixed" if celltype == "deepcell" else "bytes"
    root, holds = _deep_root(celltype, {"k": {"old": 1} if celltype == "deepcell" else b"old", "other": b"o" if celltype != "deepcell" else 2})
    replacement = Buffer(value, member_celltype)
    holds.append(replacement.tempref())
    try:
        if form == "buffer":
            root["k"].buffer = replacement
        else:
            root["k"].set_buffer(replacement)
        index = root.value
        assert index["k"] == replacement.get_checksum()
        assert set(index) == {"k", "other"}
    finally:
        for hold in holds:
            hold.clear()


def test_writes_below_a_deep_member_are_illegal():
    """Writes below k are illegal; the exception class is unruled."""
    root, holds = _deep_root("deepcell", {"k": {"x": 1}})
    try:
        before = root.value
        with pytest.raises(Exception):
            root["k"]["x"].set(3)
        assert root.value == before
    finally:
        for hold in holds:
            hold.clear()
