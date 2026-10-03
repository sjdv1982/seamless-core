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
    evaluate_expression_local,
    evaluate_expression_local_async,
    evaluate_expression_placed,
    get_expression_cache,
)
from seamless.checksum.conversion import conversion_forbidden
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
        "evaluate_expression_local": lambda: evaluate_expression_local(
            source_checksum, "value", "plain", "plain", validator=validator
        ),
        "evaluate_expression_local_async": lambda: asyncio.run(
            evaluate_expression_local_async(
                source_checksum, "value", "plain", "plain", validator=validator
            )
        ),
        "evaluate_expression_placed": lambda: asyncio.run(
            evaluate_expression_placed(
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


_LEGAL_PAIRS = [
    (source, target)
    for source in _ORDINARY
    for target in _ORDINARY
    if (source, target) not in conversion_forbidden
]


@pytest.mark.parametrize("input_celltype,celltype", _LEGAL_PAIRS)
def test_empty_path_over_null_yields_null_for_every_legal_pair(
    monkeypatch, input_celltype, celltype
):
    """expressions.md, The dummy Expression: a null input short-circuits to null
    without fetching, for every LEGAL pair. Clarity ruling (2026-09-26): the null
    short-circuit applies only on legal conversion pairs; illegal conversions stay
    illegal for null. The forbidden pairs are pinned in
    test_contract_celltypes_conversion.py and are not duplicated here."""

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


def test_path_conversion_path_does_not_fuse():
    """expressions.md, Fusion: a conversion fuses into a following path only when
    it is pathless; a conversion that follows a path inside its own Expression
    closes the run, even when it is checksum-preserving."""
    source = Buffer({"a": {"b": 3}}, "plain").get_checksum()
    inner = Expression(source, "a", input_celltype="plain", celltype="mixed")
    outer = Expression(inner, "b", input_celltype="mixed", celltype="mixed")

    assert outer.identity_key[0] == ("expression", inner.identity_key)
    assert outer.path == "b"
    assert outer.run() == 3


@pytest.mark.parametrize("path", [".a.__doc__", ".a.upper", ".a.__class__.__name__", ".n.append", ".d.keys"])
def test_string_item_never_reads_an_attribute(path):
    """expressions.md, Path syntax: a string item key indexes a dict or a
    structured array and nothing else; it never falls back to getattr."""
    source = Buffer({"a": "hello", "n": [1, 2], "d": {"k": 1}}, "plain").get_checksum()
    expression = Expression(source, path, input_celltype="plain", celltype="plain")
    with pytest.raises(ExpressionEvaluationError):
        expression.compute(execution="local")


def test_string_item_indexes_a_structured_array_field():
    """expressions.md, Path syntax: a structured array's field, and a record's
    field, remain string-keyed members."""
    import numpy as np

    array = np.zeros(2, dtype=np.dtype([("x", "<f8"), ("y", "<i8")], align=True))
    array["x"] = [1.5, 2.5]
    source = Buffer(array, "mixed").get_checksum()

    assert list(Expression(source, ".x", input_celltype="mixed", celltype="mixed").run()) == [1.5, 2.5]
    assert Expression(source, "[1].x", input_celltype="mixed", celltype="mixed").run() == 2.5


def test_deep_step_is_a_fusion_barrier():
    """expressions.md, Fusion: a deep step is a barrier and forms no pair; the run
    ends at it, and what follows is a separate Expression over the child."""
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


def test_free_expression_never_enters_the_member_set(monkeypatch):
    """expressions.md, Deduplication / Cancellation > API: a free Expression never
    joins the member set keyed by Expression identity, so softcancel() is False."""
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


def test_expression_cancel_is_retired():
    """expressions.md, Cancellation > API: cancel is retired, raises, names softcancel."""
    expression = Expression(_ANY, input_celltype="plain")
    with pytest.raises(Exception) as info:
        expression.cancel()
    message = str(info.value)
    assert "softcancel" in message
    assert "no hard cancel" in message


def test_module_softcancel_without_member_id_is_a_noop_returning_false():
    """expressions.md, Cancellation > API: softcancel_expression(key, None) is a
    no-op returning False, even while an evaluation of that key is in flight."""
    source = Buffer({"a": "noop witness"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    key = (source_checksum.hex(), "a", "plain", "str")
    expression = Expression(source_checksum, "a", input_celltype="plain", celltype="str")
    original_resolution = Checksum.resolution

    async def main():
        started = asyncio.Event()
        release = asyncio.Event()

        async def gated(checksum, *args, **kwargs):
            if checksum == source_checksum:
                started.set()
                await release.wait()
            return await original_resolution(checksum, *args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Checksum, "resolution", gated)
            task = asyncio.create_task(expression.compute_async(execution="local"))
            try:
                await asyncio.wait_for(started.wait(), 5)
                assert key in _active_expressions
                assert expression_mod.softcancel_expression(key, None) is False
                assert key in _active_expressions
            finally:
                release.set()
            return await asyncio.wait_for(task, 5)

    assert asyncio.run(main()) == Buffer("noop witness", "str").get_checksum()


def _gated_local_resolution(source_checksum):
    """Hold the shared local evaluation at its input fetch until released."""
    original_resolution = Checksum.resolution
    state = {
        "started": asyncio.Event(),
        "release": asyncio.Event(),
        "interrupted": asyncio.Event(),
    }

    async def gated(checksum, *args, **kwargs):
        if checksum == source_checksum:
            state["started"].set()
            try:
                await state["release"].wait()
            except asyncio.CancelledError:
                state["interrupted"].set()
                raise
        return await original_resolution(checksum, *args, **kwargs)

    return gated, state


def test_softcancel_leaves_the_member_set_and_the_peer_keeps_the_evaluation(monkeypatch):
    """expressions.md, Cancellation: softcancel() leaves the member set keyed by
    Expression identity and returns True; the instance's own pending call ends
    with asyncio.CancelledError; the remaining member keeps the one evaluation."""
    source = Buffer({"a": "peer survives softcancel"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    first = Expression(source_checksum, "a", input_celltype="plain", celltype="str")
    second = Expression(source_checksum, "a", input_celltype="plain", celltype="str")
    original_apply_step = expression_mod._apply_step
    applied = 0

    def count_apply_step(value, step):
        nonlocal applied
        applied += 1
        return original_apply_step(value, step)

    monkeypatch.setattr(expression_mod, "_apply_step", count_apply_step)

    async def main():
        gated, state = _gated_local_resolution(source_checksum)
        monkeypatch.setattr(Checksum, "resolution", gated)
        task1 = asyncio.create_task(first.compute_async(execution="local"))
        task2 = asyncio.create_task(second.compute_async(execution="local"))
        try:
            await asyncio.wait_for(state["started"].wait(), 5)
            await asyncio.sleep(0)
            assert first.softcancel() is True
            assert first.softcancel() is False  # idempotent
            with pytest.raises(asyncio.CancelledError):
                await task1
            assert not state["interrupted"].is_set()
            state["release"].set()
            return await asyncio.wait_for(task2, 5)
        finally:
            state["release"].set()
            await asyncio.gather(task1, task2, return_exceptions=True)

    result = asyncio.run(main())
    assert result == Buffer("peer survives softcancel", "str").get_checksum()
    assert applied == 1
    assert second.softcancel() is False  # completed: no longer a member


def test_sync_local_evaluation_joins_member_set_between_threads(monkeypatch):
    """Synchronous callers in separate threads share expression work."""
    import concurrent.futures
    import threading

    source = Buffer({"a": "threaded local evaluation"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    original_apply_step = expression_mod._apply_step
    started = threading.Event()
    second_started = threading.Event()
    duplicate_apply = threading.Event()
    release = threading.Event()
    applied = 0
    applied_lock = threading.Lock()

    def count_apply_step(value, step):
        nonlocal applied
        with applied_lock:
            applied += 1
            is_duplicate = applied > 1
        if is_duplicate:
            duplicate_apply.set()
        else:
            started.set()
            assert release.wait(5)
        return original_apply_step(value, step)

    monkeypatch.setattr(expression_mod, "_apply_step", count_apply_step)

    def evaluate():
        return expression_mod.evaluate_expression_local(
            source_checksum, "a", "plain", "str"
        )

    def evaluate_second():
        second_started.set()
        return evaluate()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(evaluate)
        assert started.wait(5)
        second = executor.submit(evaluate_second)
        assert second_started.wait(5)
        duplicated = duplicate_apply.wait(0.1)
        release.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert duplicated is False
    assert applied == 1
    assert first_result == second_result
    assert first_result == Buffer("threaded local evaluation", "str").get_checksum()


def test_sync_local_evaluation_inside_its_active_loop_runs_unshared(monkeypatch):
    source = Buffer({"a": "same loop"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    original = expression_mod._evaluate_expression_async
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(expression_mod, "_evaluate_expression_async", delayed)

    async def main():
        task = asyncio.create_task(expression_mod.evaluate_expression_local_async(
            source_checksum, "a", "plain", "str",
        ))
        await started.wait()
        try:
            direct = expression_mod.evaluate_expression_local(
                source_checksum, "a", "plain", "str",
            )
        finally:
            release.set()
        return direct, await task

    direct, shared = asyncio.run(main())
    assert direct == shared == Buffer("same loop", "str").get_checksum()


def test_sync_local_evaluation_joins_async_from_another_thread(monkeypatch):
    import concurrent.futures
    import threading

    source = Buffer({"a": "cross thread"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    original = expression_mod._evaluate_expression_async
    started, release = threading.Event(), threading.Event()
    applied = 0

    async def delayed(*args, **kwargs):
        nonlocal applied
        applied += 1
        started.set()
        await asyncio.to_thread(release.wait)
        return await original(*args, **kwargs)

    monkeypatch.setattr(expression_mod, "_evaluate_expression_async", delayed)

    def asynchronous():
        return asyncio.run(expression_mod.evaluate_expression_local_async(
            source_checksum, "a", "plain", "str",
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asynchronous)
        assert started.wait(5)
        threading.Timer(0.05, release.set).start()
        joined = expression_mod.evaluate_expression_local(
            source_checksum, "a", "plain", "str",
        )
        assert joined == future.result(timeout=5)
    assert applied == 1


def test_sync_local_evaluation_joins_dispatch(monkeypatch):
    import concurrent.futures
    import threading
    from tests.helpers.fake_remotes import install_fake_remotes

    source = Buffer({"a": "dispatched"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    result_checksum = Buffer("dispatched", "str").get_checksum()
    key = (source_checksum.hex(), "a", "plain", "str")
    calls = []
    started, release = threading.Event(), threading.Event()
    install_fake_remotes(
        monkeypatch, {}, {key: result_checksum}, calls,
        run_expression_gate=release, run_expression_started=started,
    )

    def dispatched():
        return asyncio.run(expression_mod.evaluate_expression_placed(
            source_checksum, "a", "plain", "str", execution="remote",
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(dispatched)
        assert started.wait(5)
        threading.Timer(0.05, release.set).start()
        joined = expression_mod.evaluate_expression_local(
            source_checksum, "a", "plain", "str",
        )
        assert joined == future.result(timeout=5) == result_checksum
    assert calls.count("jobserver:run") == 1


def _dispatched_scratch_evaluation_setup(monkeypatch, value):
    """An Expression whose result a gated scratch dispatch will answer without
    leaving the buffer in this process."""
    source = Buffer({"a": value}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    result_checksum = Buffer(value, "str").get_checksum()
    drop_buffer(result_checksum)
    calls = []
    jobserver_results = {(source_checksum.hex(), "a", "plain", "str"): result_checksum}
    return source_checksum, result_checksum, calls, jobserver_results


@pytest.mark.parametrize("form", ["async", "sync"])
def test_materializing_local_caller_that_joins_a_dispatch_evaluates_here(monkeypatch, form):
    """expressions.md, The materialize mode: a materializing caller may join an
    in-flight evaluation, but when the joined result's buffer is not in this
    process it evaluates here, without joining again, and never re-dispatches."""
    source_checksum, result_checksum, calls, jobserver_results = (
        _dispatched_scratch_evaluation_setup(monkeypatch, f"joined by {form}")
    )
    cache_key = (source_checksum.hex(), "a", "plain", "str")

    async def main():
        gate = asyncio.Event()
        started = asyncio.Event()
        install_fake_remotes(
            monkeypatch,
            {},
            jobserver_results,
            calls,
            run_expression_gate=gate,
            run_expression_started=started,
        )
        dispatched = asyncio.create_task(
            evaluate_expression_placed(
                source_checksum, "a", "plain", "str", execution="remote", scratch=True
            )
        )
        await asyncio.wait_for(started.wait(), 5)
        if form == "async":
            joiner = asyncio.create_task(
                evaluate_expression_local_async(
                    source_checksum, "a", "plain", "str", materialize=True
                )
            )
        else:
            joiner = asyncio.create_task(
                asyncio.to_thread(
                    evaluate_expression_local,
                    source_checksum,
                    "a",
                    "plain",
                    "str",
                    materialize=True,
                )
            )
        for _ in range(500):
            active = _active_expressions.get(cache_key)
            if active is not None and len(active.members) == 2:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the local caller did not join the dispatch")
        gate.set()
        return await dispatched, await asyncio.wait_for(joiner, 5)

    dispatched_result, joined_result = asyncio.run(main())

    assert dispatched_result == joined_result == result_checksum
    assert calls.count("jobserver:run") == 1
    assert expression_mod._has_local_buffer(joined_result)


def test_cancelling_a_callers_task_softcancels_only_that_membership(monkeypatch):
    """expressions.md, Cancellation: cancelling a caller's asyncio task unwinds that
    caller's wait; the evaluator's cleanup softcancels its membership, so work
    continues for the other member."""
    source = Buffer({"a": "task cancel is soft"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    first = Expression(source_checksum, "a", input_celltype="plain", celltype="str")
    second = Expression(source_checksum, "a", input_celltype="plain", celltype="str")

    async def main():
        gated, state = _gated_local_resolution(source_checksum)
        monkeypatch.setattr(Checksum, "resolution", gated)
        task1 = asyncio.create_task(first.compute_async(execution="local"))
        task2 = asyncio.create_task(second.compute_async(execution="local"))
        try:
            await asyncio.wait_for(state["started"].wait(), 5)
            await asyncio.sleep(0)
            task1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task1
            # The cleanup already removed the cancelled caller from the set.
            assert first.softcancel() is False
            await asyncio.sleep(0.05)
            assert not state["interrupted"].is_set()
            state["release"].set()
            return await asyncio.wait_for(task2, 5)
        finally:
            state["release"].set()
            await asyncio.gather(task1, task2, return_exceptions=True)

    assert asyncio.run(main()) == Buffer("task cancel is soft", "str").get_checksum()


class _ConflictResponse:
    def __init__(self, status, text):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text


class _ConflictSession:
    """A database answering every Expression PUT as a key conflict (HTTP 409)."""

    def __init__(self):
        self.expression_puts = []

    def put(self, url, json=None, **kwargs):
        if json["type"] == "expression":
            self.expression_puts.append(json)
            return _ConflictResponse(
                409, "ERROR: Expression already exists with different result"
            )
        return _ConflictResponse(200, "OK")

    def get(self, url, json=None, **kwargs):
        return _ConflictResponse(404, "")


def test_database_key_conflict_raises_nothing_and_the_evaluation_keeps_its_result(
    monkeypatch,
):
    """expressions.md, Identity: on a 409 key conflict the client raises nothing;
    DatabaseClient.set_expression_result returns False, database_remote reports
    the write as not done, and the evaluation still returns its own result, which
    it has already recorded in the process-local Expression cache."""
    pytest.importorskip("seamless_remote")
    from seamless_remote import database_remote
    from seamless_remote.database_client import DatabaseClient

    session = _ConflictSession()
    client = DatabaseClient(readonly=False)
    client.url = "http://database.invalid"
    client._initialized = True
    client._get_session = lambda: session
    monkeypatch.setattr(database_remote, "_read_database_clients", [])
    monkeypatch.setattr(database_remote, "_write_database_clients", [client])

    reported = []
    original_set = database_remote.set_expression_result

    async def recording_set(*args, **kwargs):
        written = await original_set(*args, **kwargs)
        reported.append(written)
        return written

    monkeypatch.setattr(database_remote, "set_expression_result", recording_set)

    source = Buffer({"a": "mine"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    expression = Expression(source_checksum, "a", input_celltype="plain", celltype="str")
    expected = Buffer("mine", "str").get_checksum()

    # "auto" with the input in process memory places the evaluation locally,
    # through the dispatching entry point, which records identity in the database.
    assert expression.compute(execution="auto") == expected
    from seamless.caching import buffer_writer

    buffer_writer.flush()
    assert len(session.expression_puts) >= 1
    assert reported and all(written is False for written in reported)
    assert get_expression_cache()[(source_checksum.hex(), "a", "plain", "str")] == expected

    direct = asyncio.run(
        client.set_expression_result(source_checksum, "a", "plain", "str", expected)
    )
    assert direct is False


def test_explicit_local_evaluation_also_records_identity_in_the_database(monkeypatch):
    """expressions.md, Results and caching: successful results are recorded in the
    process cache and, when seamless_remote is importable, in the database
    `expression` table, by the evaluating process, wherever it was placed."""
    pytest.importorskip("seamless_remote")
    from seamless_remote import database_remote
    from seamless_remote.database_client import DatabaseClient

    session = _ConflictSession()  # records every Expression PUT it receives
    client = DatabaseClient(readonly=False)
    client.url = "http://database.invalid"
    client._initialized = True
    client._get_session = lambda: session
    monkeypatch.setattr(database_remote, "_read_database_clients", [])
    monkeypatch.setattr(database_remote, "_write_database_clients", [client])

    source = Buffer({"a": "recorded locally"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    expression = Expression(source_checksum, "a", input_celltype="plain", celltype="str")

    assert expression.compute(execution="local") is not None
    from seamless.caching import buffer_writer

    buffer_writer.flush()
    assert [
        (put["checksum"], put["path"], put["input_celltype"], put["celltype"])
        for put in session.expression_puts
    ] == [(source_checksum.hex(), "a", "plain", "str")]


def test_explicit_local_async_evaluation_also_records_identity_in_the_database(
    monkeypatch,
):
    """The async local evaluator records the expression identity it produces."""
    pytest.importorskip("seamless_remote")
    from seamless_remote import database_remote
    from seamless_remote.database_client import DatabaseClient

    session = _ConflictSession()
    client = DatabaseClient(readonly=False)
    client.url = "http://database.invalid"
    client._initialized = True
    client._get_session = lambda: session
    monkeypatch.setattr(database_remote, "_read_database_clients", [])
    monkeypatch.setattr(database_remote, "_write_database_clients", [client])

    source = Buffer({"a": "recorded asynchronously"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    expression = Expression(source_checksum, "a", input_celltype="plain", celltype="str")

    async def compute():
        return await expression.compute_async(execution="local")

    assert asyncio.run(compute()) is not None
    from seamless.caching import buffer_writer

    buffer_writer.flush()
    assert [
        (put["checksum"], put["path"], put["input_celltype"], put["celltype"])
        for put in session.expression_puts
    ] == [(source_checksum.hex(), "a", "plain", "str")]


def test_expression_result_registration_failures_are_visible(monkeypatch):
    from seamless.caching import buffer_writer

    def fail_registration(*args, **kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(buffer_writer, "register_expression_result", fail_registration)
    key = expression_mod.ExpressionKey(_ANY, "a", "plain", "str")
    cache_key = (key.input_checksum.hex(), key.path, key.input_celltype, key.celltype)

    with pytest.raises(RuntimeError, match="writer failed"):
        expression_mod._record_expression_result(key, cache_key, _ANY)


def test_worker_dispatch_expression_records_result_on_the_executing_side(
    monkeypatch,
):
    pytest.importorskip("seamless_transformer")
    from seamless.caching import buffer_writer
    transformer_client = pytest.importorskip("seamless_dask.transformer_client")
    from seamless_transformer import worker

    expression_rows = {}
    calls = []
    source = Buffer({"a": "recorded by worker"}, "plain")
    source.tempref()
    source_checksum = source.get_checksum()
    install_fake_remotes(
        monkeypatch,
        expression_rows,
        {},
        calls,
        jobserver_available=False,
    )
    monkeypatch.setattr(
        transformer_client, "get_seamless_dask_client", lambda: None
    )

    result = asyncio.run(
        worker.dispatch_expression(
            source_checksum, "a", "plain", "str", scratch=True
        )
    )
    buffer_writer.flush()

    assert expression_rows[(source_checksum.hex(), "a", "plain", "str")] == result
    assert calls.count("database:set") == 1


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
            evaluate_expression_local_async(source_checksum, "missing", "plain", "plain")
        )
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(
            evaluate_expression_local_async(source_checksum, "missing", "plain", "plain")
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
        evaluate_expression_local(source_checksum, "missing", "plain", "plain")
    assert applied == 2


def test_explicit_remote_without_seamless_remote_raises_expression_error(monkeypatch):
    """expressions.md, Placement rules: "remote" never falls back."""
    monkeypatch.setitem(sys.modules, "seamless_remote", None)
    with pytest.raises(ExpressionEvaluationError):
        asyncio.run(
            evaluate_expression_placed(
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
        evaluate_expression_placed(source_checksum, "a", "plain", "str", execution="auto")
    )
    assert result == result_checksum
    # The scratch flag is the requester's decision; this direct call names no
    # requester, so only the identity fields are contract here.
    assert [entry[:4] for entry in sent] == [(source_checksum, "a", "plain", "str")]
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
