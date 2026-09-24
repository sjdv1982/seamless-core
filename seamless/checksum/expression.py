"""Local expression evaluation over concrete checksums."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
import ast
import asyncio
import concurrent.futures
import threading
import weakref
from typing import Any

from seamless.buffer_class import Buffer
from seamless.checksum_class import Checksum


class ExpressionEvaluationError(ValueError):
    """Raised when an expression cannot be evaluated locally."""


@dataclass(frozen=True, slots=True)
class ExpressionKey:
    input_checksum: Checksum
    path: str
    input_celltype: str
    celltype: str

    def __post_init__(self):
        from .null import canonicalize_checksum
        object.__setattr__(self, "input_checksum",
            canonicalize_checksum(self.input_checksum, self.input_celltype))


_expression_cache: dict[tuple[str, str, str, str], Checksum] = {}
_expression_result_buffers: weakref.WeakValueDictionary[Checksum, Buffer] = (
    weakref.WeakValueDictionary()
)
_active_expression_lock = threading.RLock()
_active_expressions: dict[tuple[str, str, str, str], "_ActiveExpression"] = {}


@dataclass
class _ActiveExpression:
    result_future: concurrent.futures.Future
    task: asyncio.Task | None
    members: set[object]
    canceled: bool = False
    waiters: dict = field(default_factory=dict)
    linger: asyncio.TimerHandle | None = None
    scratch: bool = True
    input_claim: Checksum | None = None

    def hold_input(self, checksum):
        from seamless.reference_lifecycle import register_refholder

        checksum.incref_refholder(scratch=True)
        self.input_claim = checksum
        register_refholder(self)

    def _refheld_checksums(self):
        return (
            ()
            if self.input_claim is None
            else ((self.input_claim, "expression materialization"),)
        )

    def _release_refholds(self):
        checksum = self.input_claim
        self.input_claim = None
        if checksum is not None:
            checksum.decref_refholder()


def get_expression_cache() -> dict[tuple[str, str, str, str], Checksum]:
    return _expression_cache


def wait_for_active_expression(
    input_checksum,
    path,
    input_celltype,
    celltype,
    *,
    validator=None,
    validator_language=None,
):
    """Return an already-running expression's result, without starting one."""
    cache_key = (Checksum(input_checksum).hex(), path, input_celltype, celltype)
    with _active_expression_lock:
        active = _active_expressions.get(cache_key) or _lingering_expressions.get(
            cache_key
        )
        future = None if active is None else active.result_future
    if future is None:
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return Checksum(future.result())
    return None


def choose_expression_evaluation_location(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    celltype: str,
) -> str:
    """Choose local when no buffer is needed or it is in process memory.

    Remote means absent from process memory, independently of backend availability.
    """

    from .convert import conversion_needs_buffer

    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, celltype)
    if not path and not conversion_needs_buffer(
        key.input_checksum, key.input_celltype, key.celltype
    ):
        return "local"

    from seamless import CacheMissError

    try:
        _get_local_buffer(key.input_checksum)
    except CacheMissError:
        return "remote"
    return "local"


def evaluate_expression(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    celltype: str,
    *,
    validator: Checksum | str | bytes | None = None,
    validator_language: str | None = None,
) -> Checksum:
    """Evaluate an expression locally and return the result checksum."""

    if validator is not None or validator_language is not None:
        # TODO validators: reject-only gate, excluded from expression identity.
        raise NotImplementedError("Expression validators are not implemented yet")

    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, celltype)
    parse_path(key.path)
    from .null import is_null

    if key.path or not is_null(key.input_checksum):
        validate_expression_shape(key.path, key.input_celltype, key.celltype)
    cache_key = _cache_key(key)
    key.input_checksum.tempref()
    cached = _expression_cache.get(cache_key)
    if cached is not None:
        _tempref_expression_result(cached)
        return cached

    steps = parse_path(key.path)
    get_buffer = _local_buffer_getter(key.input_checksum)
    input_buffer = get_buffer() if steps else None
    from .hash_type_validation import validate_expression

    validate_expression(
        key.input_checksum,
        buffer=input_buffer,
        source_celltype=key.input_celltype,
        path_steps=steps,
        target_celltype=key.celltype,
    )
    return _evaluate_expression_after_validation(
        key, input_buffer, steps, cache_key, get_buffer
    )


async def evaluate_expression_async(
    input_checksum,
    path,
    input_celltype,
    celltype,
    *,
    validator=None,
    validator_language=None,
    member_id=None,
):
    """Share one local evaluator task for each complete Expression identity."""
    if validator is not None or validator_language is not None:
        raise NotImplementedError("Expression validators are not implemented yet")
    expression_key = ExpressionKey(
        Checksum(input_checksum), path, input_celltype, celltype
    )
    from .null import is_null

    if expression_key.path or not is_null(expression_key.input_checksum):
        validate_expression_shape(
            expression_key.path,
            expression_key.input_celltype,
            expression_key.celltype,
        )
    key = _cache_key(expression_key)
    from .convert import conversion_needs_buffer

    if key in _expression_cache or (
        not path
        and not conversion_needs_buffer(
            expression_key.input_checksum, input_celltype, celltype
        )
    ):
        return await _evaluate_expression_async(
            input_checksum, path, input_celltype, celltype
        )
    member = object() if member_id is None else member_id
    with _active_expression_lock:
        active = _active_expressions.get(key) or _lingering_expressions.pop(key, None)
        if active is None or active.result_future.done():
            active = _ActiveExpression(concurrent.futures.Future(), None, set())
            active.hold_input(expression_key.input_checksum)
            _active_expressions[key] = active

            async def execute():
                try:
                    result = await _evaluate_expression_async(
                        input_checksum,
                        path,
                        input_celltype,
                        celltype,
                        validator=validator,
                        validator_language=validator_language,
                    )
                except BaseException as exc:
                    if not active.result_future.done():
                        active.result_future.set_exception(exc)
                else:
                    if not active.result_future.done():
                        active.result_future.set_result(result)
                finally:
                    with _active_expression_lock:
                        if _active_expressions.get(key) is active:
                            _active_expressions.pop(key, None)

            active.task = asyncio.create_task(execute())
            active.task.add_done_callback(
                lambda task: _discard_active_expression(key, active)
            )
        _active_expressions[key] = active
        waiter = _join_expression(active, member)
    try:
        return await asyncio.shield(waiter)
    finally:
        softcancel_expression(key, member)


_expression_evaluations = 0
_EXPRESSION_LINGER = 3.0
_lingering_expressions = {}


async def _evaluate_expression_async(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    celltype: str,
    *,
    validator: Checksum | str | bytes | None = None,
    validator_language: str | None = None,
) -> Checksum:
    if validator is not None or validator_language is not None:
        # TODO validators: reject-only gate, excluded from expression identity.
        raise NotImplementedError("Expression validators are not implemented yet")

    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, celltype)
    cache_key = _cache_key(key)
    key.input_checksum.tempref()
    cached = _expression_cache.get(cache_key)
    if cached is not None:
        _tempref_expression_result(cached)
        return cached

    global _expression_evaluations
    _expression_evaluations += 1
    steps = parse_path(key.path)
    from .convert import conversion_needs_buffer
    from .hash_type_validation import validate_expression_async
    from .null import canonicalize_checksum, is_null

    needs_buffer = bool(steps)
    if (
        not steps
        and key.input_celltype != key.celltype
        and not is_null(canonicalize_checksum(key.input_checksum, key.celltype))
    ):
        needs_buffer = conversion_needs_buffer(
            key.input_checksum, key.input_celltype, key.celltype
        )
    input_buffer = await key.input_checksum.resolution() if needs_buffer else None

    def get_buffer() -> Buffer:
        if input_buffer is None:
            raise ExpressionEvaluationError(
                "Conversion unexpectedly requested an unresolved input buffer"
            )
        return input_buffer

    await validate_expression_async(
        key.input_checksum,
        buffer=input_buffer,
        source_celltype=key.input_celltype,
        path_steps=steps,
        target_celltype=key.celltype,
    )
    member_buffers = None
    if not steps and (key.input_celltype, key.celltype) == ("folder", "mixed"):
        from .null import is_null

        if not is_null(key.input_checksum):
            try:
                index = get_buffer().get_value("folder")
                from .deep import validate_deep_structure

                index = validate_deep_structure(index)
            except ValueError as exc:
                raise ExpressionEvaluationError(f"Invalid folder index: {exc}") from exc
            member_buffers = {}
            for child in index.values():
                member_buffers[child] = await child.resolution()
    return _evaluate_expression_after_validation(
        key, input_buffer, steps, cache_key, get_buffer, member_buffers=member_buffers
    )


def _has_daskserver():
    try:
        from seamless_remote.daskserver_remote import has_daskserver
    except ImportError:
        return False
    return has_daskserver()


async def evaluate_expression_remote(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    celltype: str,
    *,
    validator: Checksum | str | bytes | None = None,
    validator_language: str | None = None,
    execution: str = "auto",
    member_id: object | None = None,
    scratch: bool = True,
) -> Checksum:
    """Evaluate an expression with remote cache lookup and optional jobserver dispatch."""

    if validator is not None or validator_language is not None:
        # TODO validators: reject-only gate, excluded from expression identity.
        raise NotImplementedError("Expression validators are not implemented yet")
    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, celltype)
    cache_key = _cache_key(key)
    key.input_checksum.tempref()
    cached = _expression_cache.get(cache_key)
    if cached is not None:
        _tempref_expression_result(cached)
        return cached

    try:
        from seamless_remote import database_remote
    except ImportError:
        database_remote = None
    if database_remote is not None:
        result = await database_remote.get_expression_result(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.celltype,
        )
        if result is not None:
            _expression_cache[cache_key] = result
            _tempref_expression_result(result)
            return result

    location = execution
    if location == "auto":
        location = choose_expression_evaluation_location(
            key.input_checksum, key.path, key.input_celltype, key.celltype
        )
        if location == "remote":
            try:
                from seamless_remote import buffer_remote
            except ImportError:
                pass
            else:
                for client in getattr(buffer_remote, "_read_folders_clients", ()):
                    buffer = await client.get_file_buffer(key.input_checksum)
                    if buffer is not None:
                        _tempref_expression_result(key.input_checksum, buffer=buffer)
                        location = "local"
                        break
        if location == "remote":
            try:
                from seamless_remote import jobserver_remote
            except ImportError:
                location = "local"
            else:
                if not jobserver_remote.has_jobserver() and not _has_daskserver():
                    location = "local"
    if location == "local":
        result = await evaluate_expression_async(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.celltype,
            member_id=member_id,
        )
    elif location == "remote":
        return await _run_active_remote_expression(
            key,
            cache_key,
            database_remote=database_remote,
            scratch=scratch,
            member_id=member_id,
        )
    else:
        raise ValueError(f"Unknown expression execution location: {location!r}")

    result = Checksum(result)
    _expression_cache[cache_key] = result
    _tempref_expression_result(result)
    if database_remote is not None:
        await database_remote.set_expression_result(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.celltype,
            result,
        )
    return result


async def _run_active_remote_expression(
    key: ExpressionKey,
    cache_key: tuple[str, str, str, str],
    *,
    database_remote,
    member_id: object | None,
    scratch: bool,
) -> Checksum:
    member = object() if member_id is None else member_id
    with _active_expression_lock:
        active = _active_expressions.get(cache_key) or _lingering_expressions.pop(
            cache_key, None
        )
        if active is not None and active.result_future.done():
            _active_expressions.pop(cache_key, None)
            active = None
        if active is None:
            active = _ActiveExpression(
                result_future=concurrent.futures.Future(),
                task=None,
                members=set(),
            )
            active.hold_input(key.input_checksum)
            active.task = asyncio.create_task(
                _execute_remote_expression(key, cache_key, active, database_remote)
            )
            _active_expressions[cache_key] = active
            active.task.add_done_callback(
                lambda task: _discard_active_expression(cache_key, active)
            )
        _active_expressions[cache_key] = active
        active.scratch = active.scratch and scratch
        waiter = _join_expression(active, member)
    try:
        return await asyncio.shield(waiter)
    finally:
        softcancel_expression(cache_key, member)


async def _execute_remote_expression(
    key: ExpressionKey,
    cache_key: tuple[str, str, str, str],
    active: _ActiveExpression,
    database_remote,
) -> None:
    try:
        try:
            from seamless_remote import jobserver_remote
        except ImportError as exc:
            raise ExpressionEvaluationError(
                "Remote expression evaluation requires seamless_remote"
            ) from exc
        dispatch = jobserver_remote.run_expression
        if not jobserver_remote.has_jobserver() and _has_daskserver():
            from seamless_remote import daskserver_remote

            dispatch = daskserver_remote.run_expression
        # Always pass scratch explicitly; do not rely on the receiver's
        # default (seamless-transformer, seamless-remote and seamless-jobserver
        # all default their own `scratch` parameter to False deliberately).
        kwargs = {"scratch": bool(active.scratch)}
        result = await dispatch(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.celltype,
            **kwargs,
        )
        result = Checksum(result)
        _expression_cache[cache_key] = result
        _tempref_expression_result(result)
        if database_remote is not None:
            await database_remote.set_expression_result(
                key.input_checksum,
                key.path,
                key.input_celltype,
                key.celltype,
                result,
            )
    except asyncio.CancelledError as exc:
        active.canceled = True
        if not active.result_future.done():
            active.result_future.set_exception(exc)
        raise
    except BaseException as exc:
        if not active.result_future.done():
            active.result_future.set_exception(exc)
    else:
        if not active.result_future.done():
            active.result_future.set_result(result)
    finally:
        with _active_expression_lock:
            if _active_expressions.get(cache_key) is active:
                _active_expressions.pop(cache_key, None)


def _join_expression(active, member):
    """Give each waiter a future whose cancellation cannot cancel shared work."""
    if active.linger is not None:
        active.linger.cancel()
        active.linger = None
    active.members.add(member)
    waiter = asyncio.get_running_loop().create_future()
    loop = waiter.get_loop()

    def completed(future):
        def deliver():
            if waiter.done():
                return
            try:
                waiter.set_result(future.result())
            except BaseException as exc:
                waiter.set_exception(exc)

        if not loop.is_closed():
            loop.call_soon_threadsafe(deliver)

    active.result_future.add_done_callback(completed)
    active.waiters[member] = waiter
    return waiter


def _discard_active_expression(key, active):
    with _active_expression_lock:
        if _active_expressions.get(key) is active:
            _active_expressions.pop(key, None)
        if _lingering_expressions.get(key) is active:
            _lingering_expressions.pop(key, None)
        if active.linger is not None:
            active.linger.cancel()
    active._release_refholds()
    if not active.result_future.done():
        active.result_future.set_exception(asyncio.CancelledError())


def softcancel_expression(cache_key, member_id) -> bool:
    """Deregister a waiter; shared evaluation survives a bounded linger."""
    if member_id is None:
        return False
    with _active_expression_lock:
        active = _active_expressions.get(cache_key)
        if (
            active is None
            or member_id not in active.members
            or active.result_future.done()
        ):
            return False
        active.members.remove(member_id)
        waiter = active.waiters.pop(member_id, None)
        if waiter is not None:
            waiter.get_loop().call_soon_threadsafe(waiter.cancel)
        if not active.members:
            _active_expressions.pop(cache_key, None)
            _lingering_expressions[cache_key] = active

            def expire():
                with _active_expression_lock:
                    if _lingering_expressions.get(cache_key) is not active:
                        return
                    _lingering_expressions.pop(cache_key, None)
                    if not active.members and active.task is not None:
                        active.task.cancel()

            loop = active.task.get_loop()

            def schedule():
                if not active.members and not active.result_future.done():
                    active.linger = loop.call_later(_EXPRESSION_LINGER, expire)

            loop.call_soon_threadsafe(schedule)
        return True


def _evaluate_expression_after_validation(
    key: ExpressionKey,
    input_buffer: Buffer | None,
    steps: tuple[tuple[str, Any], ...],
    cache_key: tuple[str, str, str, str],
    get_buffer: Callable[[], Buffer],
    *,
    member_buffers: dict[Checksum, Buffer] | None = None,
) -> Checksum:
    from .null import canonicalize_checksum, is_null, NULL_BUFFER

    if not steps and is_null(canonicalize_checksum(key.input_checksum, key.celltype)):
        result_buffer = Buffer(NULL_BUFFER)
        result = result_buffer.get_checksum()
        _expression_cache[cache_key] = result
        _tempref_expression_result(result, buffer=result_buffer, produced=True)
        return result
    if key.path == "" and key.input_celltype == key.celltype:
        # Validation rejects known structural incompatibilities without requiring
        # source content for an identity expression.
        _expression_cache[cache_key] = key.input_checksum
        _tempref_expression_result(key.input_checksum, buffer=input_buffer)
        return key.input_checksum

    if not steps:
        from .convert import convert_checksum

        if (key.input_celltype, key.celltype) == ("folder", "mixed"):
            import numpy as np
            from seamless import CacheMissError

            try:
                index = get_buffer().get_value("folder")
            except ValueError as exc:
                raise ExpressionEvaluationError(f"Invalid folder index: {exc}") from exc
            children = {}
            for name, child in index.items():
                try:
                    buffer = (
                        member_buffers[child]
                        if member_buffers is not None
                        else _get_local_buffer(child)
                    )
                except CacheMissError:
                    buffer = child.resolve()
                children[name] = np.frombuffer(buffer.content, dtype="S1")
            result_buffer = Buffer(children, "mixed")
            result_checksum = result_buffer.get_checksum()
        else:
            result_checksum, result_buffer = convert_checksum(
                key.input_checksum, key.input_celltype, key.celltype, get_buffer
            )
        if result_buffer is not None:
            if result_buffer.get_checksum() != result_checksum:
                raise ExpressionEvaluationError("Conversion result checksum mismatch")
            _expression_result_buffers[result_checksum] = result_buffer
        _expression_cache[cache_key] = result_checksum
        _tempref_expression_result(result_checksum, buffer=result_buffer, produced=True)
        return result_checksum

    assert input_buffer is not None
    try:
        value = _deserialize_for_expression(input_buffer, key.input_celltype)
    except ValueError as exc:
        raise ExpressionEvaluationError(f"Path {key.path!r}: {exc}") from exc
    from .convert import _DEEP_CELLTYPES

    if key.input_celltype in _DEEP_CELLTYPES:
        child = _apply_step(value, steps[0])
        if key.celltype != "checksum":
            from .hash_type_validation import validate_deserializable_as

            validate_deserializable_as(child, key.celltype)
            _expression_cache[cache_key] = child
            _tempref_expression_result(child)
            return child
    if key.input_celltype == "binary" and steps:
        ndim = getattr(value, "ndim", None)
        fields = getattr(getattr(value, "dtype", None), "fields", None)
        map_only = fields and all(
            kind == "item" and isinstance(payload, str) for kind, payload in steps
        )
        if (ndim is None or ndim == 0) and not map_only:
            raise ExpressionEvaluationError("Cannot apply a path to a binary scalar")

    for step in steps:
        value = _apply_step(value, step)

    result_buffer = _serialize_expression_result(value, key.celltype)
    result_checksum = result_buffer.get_checksum()
    _expression_cache[cache_key] = result_checksum
    _expression_result_buffers[result_checksum] = result_buffer
    _tempref_expression_result(result_checksum, buffer=result_buffer, produced=True)
    return result_checksum


def _tempref_expression_result(
    checksum: Checksum, *, buffer: Buffer | None = None, produced: bool = False
) -> None:
    """Keep successful results in memory without requesting publication.

    ``produced`` marks a buffer this evaluation created as scratch: nobody has
    decided to hold it. A checksum that already existed (the input itself, a
    deep member, a cached result) keeps whatever scratch status it had.
    """

    checksum = Checksum(checksum)
    if buffer is None:
        checksum.tempref()
    else:
        from seamless.caching.buffer_cache import get_buffer_cache

        from .hash_type import register_hash_type_for_buffer

        register_hash_type_for_buffer(checksum, buffer)
        get_buffer_cache().tempref(checksum, buffer=buffer)
    if produced:
        checksum.mark_scratch()


def resolve_expression_value(
    result_checksum: Checksum | str | bytes,
    celltype: str,
) -> Any:
    """Materialize a result like ``.value``: local buffers, then ``Checksum.resolve()``.

    ``resolve()`` also asks the hashserver, so a result computed elsewhere
    materializes; it raises ``CacheMissError`` when no buffer is found.
    """
    from seamless import CacheMissError

    checksum = Checksum(result_checksum)
    try:
        buffer = _get_local_buffer(checksum)
    except CacheMissError:
        buffer = checksum.resolve()
    return _deserialize_for_expression(buffer, celltype)


def validate_expression_shape(path, source, target):
    """Refuse recipes whose declared shape can never be evaluated."""
    from .convert import _DEEP_CELLTYPES, _DEEP_FREE_CONVERSIONS
    from .celltypes import celltypes
    from .conversion import conversion_forbidden

    if (
        source not in set(celltypes) | _DEEP_CELLTYPES
        or target not in set(celltypes) | _DEEP_CELLTYPES
    ):
        raise ValueError(f"Unknown expression celltypes: {source!r}, {target!r}")
    try:
        steps = parse_path(path)
    except ExpressionEvaluationError:
        raise
    except (SyntaxError, TypeError) as exc:
        raise ValueError(f"Invalid path {path!r}: {exc}") from exc
    deep = source in _DEEP_CELLTYPES or target in _DEEP_CELLTYPES
    if not steps:
        if deep:
            legal = (
                source == target
                or (source, target) in _DEEP_FREE_CONVERSIONS
                or (source, target) == ("folder", "mixed")
            )
        else:
            legal = (source, target) not in conversion_forbidden
        if not legal:
            raise ValueError(f"Illegal expression conversion: {source} -> {target}")
        return
    if deep:
        member = "mixed" if source == "deepcell" else "bytes"
        if (
            source not in _DEEP_CELLTYPES
            or len(steps) != 1
            or steps[0][0] != "item"
            or not isinstance(steps[0][1], str)
            or target not in (member, "checksum")
        ):
            raise ValueError(f"Illegal deep path {path!r}: {source} -> {target}")
        return
    current = source
    for step in steps:
        kind, payload = step
        if current in ("plain", "mixed", "binary"):
            break
        if current in ("int", "float", "bool", "checksum") or (
            kind == "item" and isinstance(payload, str)
        ):
            raise ValueError(f"Illegal path step {step!r} in {path!r} for {current}")
        if current == "bytes" and kind == "item":
            current = "int"

def parse_path(path: str) -> tuple[tuple[str, Any], ...]:
    if not path:
        return ()
    steps: list[tuple[str, Any]] = []
    index = 0
    while index < len(path):
        char = path[index]
        if char == ".":
            index += 1
            start = index
            while index < len(path) and (
                path[index] == "_" or path[index].isalnum()
            ):
                index += 1
            name = path[start:index]
            if not name or not name.isidentifier():
                raise ExpressionEvaluationError(f"Invalid path segment at {start}")
            steps.append(("item", name))
        elif char == "[":
            stop = _find_closing_bracket(path, index)
            token = path[index + 1 : stop]
            steps.append(_parse_bracket_token(token))
            index = stop + 1
        else:
            start = index
            while index < len(path) and (
                path[index] == "_" or path[index].isalnum()
            ):
                index += 1
            name = path[start:index]
            if not name or not name.isidentifier():
                raise ExpressionEvaluationError(f"Invalid path segment at {start}")
            steps.append(("item", name))
    return tuple(steps)


def _cache_key(key: ExpressionKey) -> tuple[str, str, str, str]:
    return (
        key.input_checksum.hex(),
        key.path,
        key.input_celltype,
        key.celltype,
    )


def _local_buffer_getter(checksum: Checksum) -> Callable[[], Buffer]:
    """Keep one local buffer lookup lazy and retain its result for evaluation."""

    from seamless import CacheMissError

    buffer: Buffer | None = None
    error: CacheMissError | None = None

    def get_buffer() -> Buffer:
        nonlocal buffer, error
        if error is not None:
            raise error
        if buffer is None:
            try:
                buffer = _get_local_buffer(checksum)
            except CacheMissError as exc:
                error = exc
                raise
        return buffer

    return get_buffer


def _get_local_buffer(checksum: Checksum) -> Buffer:
    from seamless.caching.buffer_cache import get_buffer_cache
    from seamless.checksum.cached_calculate_checksum import checksum_cache

    from .calculate_checksum import TRIVIAL_CHECKSUMS

    trivial = TRIVIAL_CHECKSUMS.get(checksum.hex())
    if trivial is not None:
        return Buffer(trivial)
    buffer = get_buffer_cache().get(checksum)
    if buffer is not None:
        return buffer

    buffer = _expression_result_buffers.get(checksum)
    if buffer is not None:
        return buffer

    raw = checksum_cache.get(checksum)
    if raw is not None:
        return Buffer(raw, checksum=checksum)

    try:
        from seamless_remote import buffer_remote
    except ImportError:
        pass
    else:
        from pathlib import Path

        for client in getattr(buffer_remote, "_read_folders_clients", ()):
            directory = getattr(client, "directory", None)
            if directory is None:
                continue
            for path in (
                Path(directory) / checksum.hex(),
                Path(directory) / checksum.hex()[:2] / checksum.hex(),
            ):
                try:
                    buffer = Buffer(path.read_bytes())
                except (FileNotFoundError, IsADirectoryError):
                    continue
                if buffer.get_checksum() == checksum:
                    return buffer
    from seamless import CacheMissError

    raise CacheMissError(checksum)


def _deserialize_for_expression(buffer: Buffer, input_celltype: str) -> Any:
    if input_celltype == "bytes":
        from .null import NULL_BUFFER
        return b"" if buffer.content == NULL_BUFFER else buffer.content
    value = buffer.get_value(input_celltype)
    from .deep import DEEP_CELLTYPES, validate_deep_structure

    return (
        validate_deep_structure(value)
        if input_celltype in DEEP_CELLTYPES
        else value
    )


def _serialize_expression_result(value: Any, celltype: str) -> Buffer:
    if celltype == "binary" and isinstance(value, bytes):
        import numpy as np

        return Buffer(np.array(value), "binary")
    result = Buffer(value, celltype)
    if celltype == "binary":
        from seamless.util.mixed import MAGIC_NUMPY

        if not result.content.startswith(MAGIC_NUMPY):
            raise ExpressionEvaluationError(
                "Expression result is not serializable as binary"
            )
    return result


def _find_closing_bracket(path: str, start: int) -> int:
    quote: str | None = None
    escape = False
    for index in range(start + 1, len(path)):
        char = path[index]
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "]":
            return index
    raise ExpressionEvaluationError(f"Unclosed path bracket at {start}")


def _parse_bracket_token(token: str) -> tuple[str, Any]:
    if ":" in token:
        return ("slice", _parse_slice(token))
    try:
        key = ast.literal_eval(token)
    except (SyntaxError, ValueError) as exc:
        raise ExpressionEvaluationError(f"Invalid item path token [{token}]") from exc
    return ("item", key)


def _parse_slice(token: str) -> slice:
    parts = token.split(":")
    if len(parts) > 3:
        raise ExpressionEvaluationError(f"Invalid slice path token [{token}]")
    values = []
    for part in parts:
        if part == "":
            values.append(None)
        else:
            try:
                values.append(ast.literal_eval(part))
            except (SyntaxError, ValueError) as exc:
                raise ExpressionEvaluationError(
                    f"Invalid slice path token [{token}]"
                ) from exc
    while len(values) < 3:
        values.append(None)
    return slice(values[0], values[1], values[2])


def _apply_step(value: Any, step: tuple[str, Any]) -> Any:
    kind, payload = step
    try:
        if kind == "slice":
            return value[payload]
        if kind == "item":
            if isinstance(payload, str):
                if isinstance(value, dict):
                    return value[payload]
                if hasattr(value, "dtype") and getattr(value.dtype, "fields", None):
                    return value[payload]
                return getattr(value, payload)
            return value[payload]
    except Exception as exc:
        raise ExpressionEvaluationError(
            f"Cannot apply expression path step {payload!r}"
        ) from exc
    raise ExpressionEvaluationError(f"Unknown expression path step {kind!r}")


__all__ = [
    "ExpressionEvaluationError",
    "choose_expression_evaluation_location",
    "evaluate_expression",
    "evaluate_expression_async",
    "evaluate_expression_remote",
    "get_expression_cache",
    "parse_path",
    "resolve_expression_value",
]
