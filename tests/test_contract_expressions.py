"""Coverage for contracts/expressions.md rules not pinned elsewhere.

Each test names the section of seamless/docs/agent/contracts/expressions.md it
pins. Existing coverage lives in test_expression_contract.py and siblings; this
file only adds the rules found UNCOVERED or WEAK in the 2026-09-24 audit.
"""

from __future__ import annotations

import asyncio
import gc
import sys
from types import ModuleType

import pytest

from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.checksum import expression as expression_mod
from seamless.checksum.expression import (
    ExpressionEvaluationError,
    _active_expressions,
    evaluate_expression,
    evaluate_expression_async,
    evaluate_expression_remote,
    get_expression_cache,
)
from seamless.checksum.null import NULL_CHECKSUM, is_null

from tests.helpers.fake_remotes import drop_buffer, install_fake_remotes


@pytest.fixture(autouse=True)
def clean_expression_state():
    gc.collect()
    get_expression_cache().clear()
    _active_expressions.clear()
    yield
    get_expression_cache().clear()
    _active_expressions.clear()
    gc.collect()


_ANY = Checksum(bytes.fromhex("4c" * 32))

# Step shapes of "The structural path rules": string item, positional item,
# slice, and the two-step continuations that the "after a step" column rules on.
_SHAPES = {
    "string": ".name",
    "string-bracket": "['name']",
    "positional": "[0]",
    "slice": "[1:3]",
    "string-string": ".a.b",
    "positional-positional": "[0][0]",
    "slice-positional": "[1:3][0]",
    "slice-string": "[1:3].a",
}

_ALLOWED = {
    # plain/mixed/binary: every kind allowed, the element type after a step is data.
    **{
        ct: set(_SHAPES)
        for ct in ("plain", "mixed", "binary")
    },
    # text-like: string items refused, positional/slice allowed, row repeats.
    **{
        ct: {"positional", "slice", "positional-positional", "slice-positional"}
        for ct in ("text", "str", "python", "ipython", "yaml")
    },
    # bytes: a positional item yields an int, so the path must end there.
    "bytes": {"positional", "slice", "slice-positional"},
    # scalars and checksum: no members.
    **{ct: set() for ct in ("int", "float", "bool", "checksum")},
    # deep: exactly one string item, as the whole path.
    **{
        ct: {"string", "string-bracket"}
        for ct in ("deepcell", "deepfolder", "folder")
    },
}

_MEMBER_TARGET = {"deepcell": "mixed", "deepfolder": "bytes", "folder": "bytes"}


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("input_celltype", list(_ALLOWED))
def test_structural_path_rules_table_is_decided_at_construction(input_celltype, shape):
    """expressions.md, The structural path rules: the full table, every row x kind."""
    path = _SHAPES[shape]
    celltype = _MEMBER_TARGET.get(input_celltype, input_celltype)
    if shape in _ALLOWED[input_celltype]:
        expression = Expression(
            _ANY, path=path, input_celltype=input_celltype, celltype=celltype
        )
        assert expression.input_celltype == input_celltype
    else:
        with pytest.raises(ValueError):
            Expression(_ANY, path=path, input_celltype=input_celltype, celltype=celltype)


def test_construction_refusal_names_the_offending_step():
    """expressions.md, Which refusal happens where: ValueError naming the step."""
    with pytest.raises(ValueError) as info:
        Expression(_ANY, path="[0].name", input_celltype="str", celltype="str")
    assert "name" in str(info.value)
    assert "[0]" in str(info.value)


def test_construction_refusal_consults_no_hashtype(monkeypatch):
    """expressions.md, Which refusal happens where: no checksum, so no HashType."""
    from seamless.checksum import hash_type as hash_type_mod

    def forbidden(*args, **kwargs):
        pytest.fail("a construction refusal must not consult HashType")

    monkeypatch.setattr(hash_type_mod, "deserializable_as", forbidden)
    with pytest.raises(ValueError):
        Expression(_ANY, path=".name", input_celltype="int", celltype="int")


def test_standalone_cell_records_a_construction_refusal_as_its_exception():
    """expressions.md, Which refusal happens where: a Cell records, not raises."""
    source = Buffer(5, "int").get_checksum()
    projected = Cell("int", checksum=source)[0]

    assert projected.checksum is None
    assert isinstance(projected.exception, str)
    assert "[0]" in projected.exception


def test_hashtype_is_never_asked_about_a_deep_celltype():
    """expressions.md, When an Expression is vetted: deserializable_as raises ValueError."""
    from seamless.checksum.hash_type import deserializable_as

    for celltype in ("deepcell", "deepfolder", "folder"):
        with pytest.raises(ValueError):
            deserializable_as(0, celltype, checksum=_ANY)


def test_unresolved_source_identity_is_the_object_and_has_no_database_key():
    """expressions.md, Identity: ("object", ...) for an unresolved source."""
    first = Expression(Cell("plain"), path="a", input_celltype="plain")

    assert first.identity_key[0][0] == "object"
    assert first == first
    with pytest.raises(ValueError):
        first.database_key


@pytest.mark.xfail(
    strict=False,
    reason="expressions.md, Identity: an Expression over an unresolved source is "
    "identical only to itself; an empty Cell builds to a dummy over None, the dummy "
    "collapses, and every such Expression keys on id(None) and compares equal",
)
def test_expressions_over_two_unresolved_sources_are_distinct():
    first = Expression(Cell("plain"), path="a", input_celltype="plain")
    second = Expression(Cell("plain"), path="a", input_celltype="plain")

    assert first != second
    assert first.identity_key != second.identity_key


def test_validator_source_text_is_not_accepted():
    """expressions.md, Validators are deferred: a validator must be a Checksum."""
    with pytest.raises((ValueError, TypeError)):
        Expression(_ANY, input_celltype="plain", validator="def validate(x): pass")


def test_validator_refusal_is_total_even_for_a_cached_identity(monkeypatch):
    """expressions.md, Validators are deferred: refused at entry, before any cache.

    This is the one test the doc singles out as worth writing: the identity
    tuple's result is already in the process cache.
    """
    source = Buffer({"value": 1}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    plain = Expression(source_checksum, path="value", input_celltype="plain", celltype="plain")
    assert plain.compute(execution="local") is not None
    assert plain.database_key in get_expression_cache()

    validator = Checksum(bytes.fromhex("33" * 32))
    guarded = Expression(
        source_checksum,
        path="value",
        input_celltype="plain",
        celltype="plain",
        validator=validator,
    )
    entry_points = {
        "compute-local": lambda: guarded.compute(execution="local"),
        "compute-auto": lambda: guarded.compute(),
        "compute_async": lambda: asyncio.run(guarded.compute_async(execution="local")),
        "run": lambda: guarded.run(),
        "evaluate_expression": lambda: evaluate_expression(
            source_checksum, "value", "plain", "plain", validator=validator
        ),
        "evaluate_expression_async": lambda: asyncio.run(
            evaluate_expression_async(
                source_checksum, "value", "plain", "plain", validator=validator
            )
        ),
        "evaluate_expression_remote": lambda: asyncio.run(
            evaluate_expression_remote(
                source_checksum, "value", "plain", "plain", validator_language="python"
            )
        ),
    }
    for name, call in entry_points.items():
        with pytest.raises(NotImplementedError):
            call()


_ORDINARY = [
    "plain", "mixed", "binary", "text", "str", "python", "ipython", "yaml",
    "bytes", "int", "float", "bool", "checksum",
]


@pytest.mark.parametrize("celltype", _ORDINARY)
@pytest.mark.parametrize("input_celltype", _ORDINARY)
def test_empty_path_over_null_yields_null_for_every_pair(
    monkeypatch, input_celltype, celltype
):
    """expressions.md, The dummy Expression: null short-circuits for every pair."""

    def fail_if_fetched(*args, **kwargs):
        pytest.fail("a null input must not fetch a buffer")

    monkeypatch.setattr(expression_mod, "_get_local_buffer", fail_if_fetched)
    expression = Expression(
        Checksum(NULL_CHECKSUM), input_celltype=input_celltype, celltype=celltype
    )
    assert is_null(expression.compute(execution="local"))


def test_new_buffer_conversion_then_path_does_not_fuse():
    """expressions.md, Fusion: a new-buffer conversion is a genuine input."""
    source = Buffer({"k": [1, 2]}, "plain").get_checksum()
    converted = Expression(source, "", input_celltype="plain", celltype="text")
    projected = Expression(converted, "[0]", input_celltype="text", celltype="text")

    assert projected.identity_key[0] == ("expression", converted.identity_key)
    assert projected.run() == "{"


def test_deep_step_is_a_fusion_barrier():
    """expressions.md, Fusion: a deep step ends the run."""
    child = Buffer({"x": 5}, "mixed").get_checksum()
    index = Buffer({"member": child.hex()}, "deepcell").get_checksum()
    member = Expression(index, "member", input_celltype="deepcell", celltype="mixed")
    inside = Expression(member, "x", input_celltype="mixed", celltype="int")

    assert inside.identity_key[0] == ("expression", member.identity_key)
    assert inside.path == "x"
    assert inside.input_celltype == "mixed"


def test_fused_chain_agrees_with_the_unfused_evaluation():
    """expressions.md, Fusion: where both succeed they agree; identities differ."""
    source = Buffer({"a": [10, {"b": "leaf"}]}, "plain").get_checksum()
    inner = Expression(source, "a", input_celltype="plain", celltype="plain")
    fused = Expression(inner, "[1].b", input_celltype="plain", celltype="str")
    assert fused.input_checksum == source

    intermediate = inner.compute(execution="local")
    unfused = Expression(intermediate, "[1].b", input_celltype="plain", celltype="str")

    assert fused.identity_key != unfused.identity_key
    assert fused.compute(execution="local") == unfused.compute(execution="local")
    assert fused.run() == "leaf"


def test_free_expression_never_enters_a_waiting_set(monkeypatch):
    """expressions.md, Cancellation: free class has no cancellable window."""
    source = Buffer(True, "bool").get_checksum()
    drop_buffer(source)
    dummy = Expression(source, input_celltype="bool", celltype="bool")

    async def main():
        task = asyncio.create_task(dummy.compute_async(execution="local"))
        await asyncio.sleep(0)
        assert dummy.softcancel() is False
        assert not _active_expressions
        return await task

    assert asyncio.run(main()) == source
    assert dummy.softcancel() is False


@pytest.mark.xfail(
    strict=False,
    reason="expressions.md, Cancellation > API: Expression.cancel is retired and "
    "must raise pointing at softcancel(); code keeps it as a softcancel alias",
)
def test_expression_cancel_is_retired():
    expression = Expression(_ANY, input_celltype="plain")
    with pytest.raises(Exception) as info:
        expression.cancel()
    assert "softcancel" in str(info.value)


def test_deduplicated_failure_reaches_every_member_and_is_not_kept(monkeypatch):
    """expressions.md, Expression failures are not cached / Deduplication."""
    source = Buffer({"present": 1}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    original_resolution = Checksum.resolution
    original_apply_step = expression_mod._apply_step
    applied = 0

    def count_apply_step(value, step):
        nonlocal applied
        applied += 1
        return original_apply_step(value, step)

    monkeypatch.setattr(expression_mod, "_apply_step", count_apply_step)

    async def main():
        gate = asyncio.Event()
        entered = asyncio.Event()

        async def gated(self, *args, **kwargs):
            entered.set()
            await gate.wait()
            return await original_resolution(self, *args, **kwargs)

        monkeypatch.setattr(Checksum, "resolution", gated)
        first = asyncio.create_task(
            evaluate_expression_async(source_checksum, "missing", "plain", "plain")
        )
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(
            evaluate_expression_async(source_checksum, "missing", "plain", "plain")
        )
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        monkeypatch.setattr(Checksum, "resolution", original_resolution)
        return results

    results = asyncio.run(main())
    assert all(isinstance(r, ExpressionEvaluationError) for r in results), results
    assert applied == 1
    assert not _active_expressions
    assert (source_checksum.hex(), "missing", "plain", "plain") not in get_expression_cache()

    with pytest.raises(ExpressionEvaluationError):
        evaluate_expression(source_checksum, "missing", "plain", "plain")
    assert applied == 2


def test_explicit_remote_without_seamless_remote_raises_expression_error(monkeypatch):
    """expressions.md, Placement rules: "remote" never falls back."""
    monkeypatch.setitem(sys.modules, "seamless_remote", None)
    with pytest.raises(ExpressionEvaluationError):
        asyncio.run(
            evaluate_expression_remote(
                Checksum("d" * 64), "a", "plain", "str", execution="remote"
            )
        )


def test_client_without_jobserver_dispatches_to_the_daskserver(monkeypatch):
    """expressions.md, Placement rules: client-side dispatch target."""
    source_checksum = Checksum("6" * 64)
    result_checksum = Buffer("dask", "str").get_checksum()
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, jobserver_available=False)
    daskserver_remote = ModuleType("seamless_remote.daskserver_remote")
    sent = []

    async def run_expression(input_checksum, path, input_celltype, celltype, *, scratch):
        sent.append((Checksum(input_checksum), path, input_celltype, celltype, scratch))
        return result_checksum

    daskserver_remote.has_daskserver = lambda: True
    daskserver_remote.run_expression = run_expression
    monkeypatch.setitem(sys.modules, "seamless_remote.daskserver_remote", daskserver_remote)
    monkeypatch.setattr(
        sys.modules["seamless_remote"], "daskserver_remote", daskserver_remote, raising=False
    )

    result = asyncio.run(
        evaluate_expression_remote(source_checksum, "a", "plain", "str", execution="auto")
    )
    assert result == result_checksum
    assert sent == [(source_checksum, "a", "plain", "str", True)]
    assert "jobserver:run" not in calls


def test_failed_fingertip_reports_a_bare_cache_miss_on_the_wanted_checksum(monkeypatch):
    """expressions.md, What a failed fingertip reports: CacheMissError(wanted) only."""
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, jobserver_available=False)
    wanted = Buffer("never produced", "str").get_checksum()
    drop_buffer(wanted)

    # Candidate 1: its input is reachable nowhere (the reason would be a
    # CacheMissError on the *input*). Candidate 2: evaluating it raises
    # ExpressionEvaluationError. Neither reason may reach the caller.
    unreachable_input = Checksum("7" * 64)
    present = Buffer({"present": 1}, "plain")
    present.tempref()
    cache = get_expression_cache()
    cache[(unreachable_input.hex(), "a", "plain", "str")] = wanted
    cache[(present.get_checksum().hex(), "missing", "plain", "str")] = wanted

    with pytest.raises(CacheMissError) as info:
        asyncio.run(wanted.fingertip("str"))
    assert type(info.value) is CacheMissError
    assert info.value.args == (wanted,)


def test_fingertip_chain_is_never_dispatched(monkeypatch):
    """expressions.md, Fingertipping is exempt from "where the data is"."""
    source = Buffer({"a": "recovered here"}, "plain")
    source.tempref()
    expression = Expression(
        source.get_checksum(), "a", input_celltype="plain", celltype="str"
    )
    result = expression.compute(execution="local")
    drop_buffer(result)

    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, jobserver_available=True)
    jobserver_remote = sys.modules["seamless_remote.jobserver_remote"]

    async def forbidden(*args, **kwargs):
        pytest.fail("a fingertip chain must run locally, never be dispatched")

    monkeypatch.setattr(jobserver_remote, "run_expression", forbidden)

    assert asyncio.run(result.fingertip("str")) == "recovered here"
    assert "jobserver:run" not in calls
