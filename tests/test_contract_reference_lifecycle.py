"""Contract tests: contracts/internal/checksum-reference-lifecycle.md (feature 9).

Covers the rules that the existing lifecycle tests leave uncovered or only
partially pin: tempref neutrality, neutral claims, the transfer write, the
"non-scratch incref is the only publisher" rule, Expression ownership
(tempref-only inputs, scratch-neutral result claim), fingertip non-publication,
copy independence and top-level-only deep ownership.

Buffer publication is observed by replacing ``buffer_writer.register``: every
hashserver write of the buffer cache goes through it.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import uuid

import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
import seamless.checksum.expression as expression_mod
from seamless.reference_lifecycle import collect_refholder_claims

DOC = "internal/checksum-reference-lifecycle.md"


def _fresh(label: str) -> Buffer:
    return Buffer(f"{label}-{uuid.uuid4().hex}".encode())


def _count(checksum) -> int:
    return get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0]


@pytest.fixture
def writes(monkeypatch):
    written: list[Checksum] = []
    monkeypatch.setattr(
        buffer_writer, "register", lambda buf: written.append(buf.get_checksum())
    )
    return written


# --- §1 / §5: a bare Checksum is not a refholder -------------------------------


def test_bare_checksum_owns_no_reference():
    checksum = Checksum(bytes.fromhex(uuid.uuid4().hex * 2))
    same = Checksum(checksum.hex())
    assert same == checksum
    assert checksum not in get_buffer_cache().reference_snapshot()


# --- §1 Tempref: never writes, neutral about scratch status ----------------------


def test_tempref_never_writes_and_leaves_scratch_status_alone(writes):
    cache = get_buffer_cache()
    buf = _fresh("tempref-neutral")
    checksum = buf.get_checksum()

    cache.tempref(checksum, buffer=buf)
    assert checksum not in writes
    assert cache.is_scratch_ref(checksum) is False, "a tempref marked scratch"
    assert cache.strong_cache[checksum].remote_registered is False

    cache.mark_scratch(checksum)
    cache.tempref(checksum, buffer=buf)  # refresh
    assert cache.is_scratch_ref(checksum) is True, "a tempref cleared scratch"
    assert checksum not in writes
    assert _count(checksum) == 0, "a tempref is not a refholder claim"


# --- §1 Neutral claim -------------------------------------------------------------


def test_neutral_claim_protects_without_publishing_or_changing_scratch(writes):
    cache = get_buffer_cache()
    marked = _fresh("neutral-marked")
    unmarked = _fresh("neutral-unmarked")
    marked_cs, unmarked_cs = marked.get_checksum(), unmarked.get_checksum()
    cache.register(marked_cs, marked, size=len(marked.content))
    cache.register(unmarked_cs, unmarked, size=len(unmarked.content))
    cache.tempref(marked_cs, buffer=marked)
    cache.mark_scratch(marked_cs)

    cache.incref_refholder(marked_cs, scratch=None)
    cache.incref_refholder(unmarked_cs, scratch=None)
    try:
        assert cache.is_scratch_ref(marked_cs) is True
        assert cache.is_scratch_ref(unmarked_cs) is False
        assert marked_cs not in writes and unmarked_cs not in writes
        assert cache.strong_cache[unmarked_cs].remote_registered is False
        # ... and it protects: neither purge nor eviction removes the buffer.
        assert cache.purge_scratch(marked_cs) == 0
        assert marked_cs in cache.strong_cache
        assert cache.reference_snapshot()[marked_cs] == (1, 0, True)
    finally:
        cache.decref_refholder(marked_cs)
        cache.decref_refholder(unmarked_cs)


# --- §8 Buffer persistence: a non-scratch incref is the only publisher -----------


@pytest.mark.parametrize("api", ["refholder", "manual"])
def test_owner_claim_scratch_decides_publication_and_scratch_status(writes, api):
    cache = get_buffer_cache()
    buf = _fresh(f"owner-claim-{api}")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    incref = cache.incref_refholder if api == "refholder" else cache.incref
    decref = cache.decref_refholder if api == "refholder" else cache.decref

    incref(checksum, scratch=True)
    assert cache.is_scratch_ref(checksum) is True
    assert checksum not in writes, "a scratch claim published"

    incref(checksum, scratch=False)
    assert cache.is_scratch_ref(checksum) is False, "a non-scratch claim kept scratch"
    assert writes.count(checksum) == 1, "a non-scratch claim did not publish once"

    incref(checksum, scratch=False)
    assert writes.count(checksum) == 1, "registration is monotonic, not repeated"
    for _ in range(3):
        decref(checksum)


def test_non_scratch_claim_without_local_buffer_writes_nothing(writes):
    cache = get_buffer_cache()
    checksum = Checksum(bytes.fromhex(uuid.uuid4().hex * 2))
    cache.incref_refholder(checksum)
    try:
        assert writes == []
        assert cache.strong_cache[checksum].remote_registered is True
    finally:
        cache.decref_refholder(checksum)


def test_transfer_write_publishes_clears_scratch_and_is_not_a_claim(writes):
    cache = get_buffer_cache()
    buf = _fresh("transfer-write")
    checksum = buf.get_checksum()
    cache.tempref(checksum, buffer=buf)
    cache.mark_scratch(checksum)

    cache.transfer_write(checksum, buffer=buf)

    assert writes.count(checksum) == 1
    assert cache.is_scratch_ref(checksum) is False
    assert cache.reference_snapshot()[checksum] == (0, 0, False)


# --- §6 Expression: scratch-neutral result claim, reads never publish ------------


def test_expression_evaluation_and_result_reads_never_publish(writes):
    source = Buffer({"value": f"no-publish-{uuid.uuid4().hex}"}, "plain")
    expression = Expression(
        source.get_checksum(), path="value", input_celltype="plain", celltype="str"
    )
    result = expression.compute(execution="local")
    assert expression.checksum == result
    assert expression.result == result
    assert result not in writes
    roles = [role for cs, role in expression._refheld_checksums() if cs == result]
    assert roles == ["result"], "public interest acquires exactly one result role"
    expression._release_refholds()


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §6/§7: Expression inputs are tempref-only; the code takes an "
    "'input' refholder claim (expression_class.py incref_refholder(scratch=True))",
)
def test_expression_input_is_tempref_only():
    source = Buffer({"a": f"input-tempref-{uuid.uuid4().hex}"}, "plain")
    checksum = source.get_checksum()
    expression = Expression(checksum, "a", input_celltype="plain", celltype="str")
    try:
        assert not [
            role for cs, role in expression._refheld_checksums() if cs == checksum
        ]
        assert _count(checksum) == 0
    finally:
        expression._release_refholds()


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §1/§8: an Expression has no scratch policy, so it must not change "
    "scratch status; its input claim uses scratch=True and marks an owned, published "
    "input as scratch",
)
def test_expression_does_not_mark_an_owned_input_scratch():
    cache = get_buffer_cache()
    source = Buffer({"a": f"owned-input-{uuid.uuid4().hex}"}, "plain")
    checksum = source.get_checksum()
    owner = Cell(checksum=checksum, celltype="plain")  # non-scratch owner
    assert cache.is_scratch_ref(checksum) is False
    expression = Expression(checksum, "a", input_celltype="plain", celltype="str")
    try:
        assert cache.is_scratch_ref(checksum) is False
    finally:
        expression._release_refholds()
        owner._release_refholds()


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §6: the Expression 'result' claim is scratch-neutral; the code "
    "claims with scratch=True and re-marks a result a non-scratch owner holds",
)
def test_expression_result_claim_does_not_change_scratch_status():
    cache = get_buffer_cache()
    source = Buffer({"a": f"owned-result-{uuid.uuid4().hex}"}, "plain")
    checksum = source.get_checksum()
    first = Expression(checksum, "a", input_celltype="plain", celltype="str")
    result = first._evaluate_internal(execution="local")
    owner = Cell(checksum=result, celltype="str")  # non-scratch owner claim
    assert cache.is_scratch_ref(result) is False
    second = Expression(checksum, "a", input_celltype="plain", celltype="str")
    try:
        assert second.compute(execution="local") == result
        assert any(
            cs == result and role == "result" for cs, role in second._refheld_checksums()
        )
        assert cache.is_scratch_ref(result) is False, "Expression claim marked scratch"
    finally:
        for holder in (first, second, owner):
            holder._release_refholds()


# --- §8 Scratch: a fingertip never publishes ------------------------------------


def _drop_buffer(checksum):
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
        cache._scratch_refs.discard(checksum)
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)


def test_fingertip_never_publishes_and_leaves_the_buffer_temprefed_scratch(writes):
    cache = get_buffer_cache()
    source = Buffer({"a": {"b": f"fingertip-{uuid.uuid4().hex}"}}, "plain")
    first = Expression(source.get_checksum(), "a", input_celltype="plain", celltype="plain")
    first_result = first._evaluate_internal(execution="local")
    second = Expression(first_result, "b", input_celltype="plain", celltype="str")
    second_result = second._evaluate_internal(execution="local")
    # Release the builders first: a fingertip works from the recorded identity,
    # not from live objects, and must not be helped by their claims.
    first._release_refholds()
    second._release_refholds()
    _drop_buffer(first_result)
    _drop_buffer(second_result)
    writes.clear()

    value = asyncio.run(second_result.fingertip("str"))

    assert value.startswith("fingertip-")
    for recovered in (first_result, second_result):
        assert recovered not in writes, "a fingertip chain published a buffer"
        entry = cache.strong_cache.get(recovered)
        assert entry is not None and entry.tempref is not None
        assert cache.is_scratch_ref(recovered) is True
        assert _count(recovered) == 0, "a fingertip is not an owner"


# --- §5 Copying a refholding object creates an independent reference ------------


def _cell_holder():
    checksum = _fresh("copy-cell").get_checksum()
    return Cell(checksum=checksum), checksum


def _expression_holder():
    source = Buffer({"a": f"copy-expression-{uuid.uuid4().hex}"}, "plain")
    expression = Expression(source.get_checksum(), "a", input_celltype="plain", celltype="str")
    return expression, expression.compute(execution="local")


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §5: copying a refholding object must create an independent "
    "reference; copy.copy of a Cell/Expression duplicates the claim without "
    "acquiring it, so releasing the copy releases the original's reference",
)
@pytest.mark.parametrize("make", [_cell_holder, _expression_holder], ids=["cell", "expression"])
@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
def test_copied_refholder_owns_an_independent_reference(make, copier):
    original, checksum = make()
    before = _count(checksum)
    duplicate = copier(original)
    holders = list({id(h): h for h in (original, duplicate)}.values())
    try:
        claims = collect_refholder_claims(holders).get(checksum, [])
        assert len(claims) == _count(checksum), "claims and count disagree after copy"
        if duplicate is not original:
            duplicate._release_refholds()
            del duplicate
            gc.collect()
            assert _count(checksum) == before, "releasing the copy touched the original"
            assert any(h is original for h, _ in collect_refholder_claims([original])[checksum])
    finally:
        original._release_refholds()


# --- §8 Deep checksums: only the top-level checksum is owned --------------------


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
def test_deep_input_owns_only_the_top_level_checksum(celltype):
    leaf_celltype = "mixed" if celltype == "deepcell" else "bytes"
    leaf = Buffer(f"deep-leaf-{uuid.uuid4().hex}".encode(), leaf_celltype)
    leaf_checksum = leaf.get_checksum()
    get_buffer_cache().tempref(leaf_checksum, buffer=leaf)
    index = Buffer({"member": leaf_checksum.hex()}, celltype)
    index_checksum = index.get_checksum()
    cell = Cell(checksum=index_checksum, celltype=celltype)
    try:
        assert _count(index_checksum) == 1
        assert _count(leaf_checksum) == 0
        cell.value  # reading the index acquires nothing on the members
        assert _count(leaf_checksum) == 0
        assert leaf_checksum not in collect_refholder_claims()
    finally:
        cell._release_refholds()
