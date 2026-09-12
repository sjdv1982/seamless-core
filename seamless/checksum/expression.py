"""Local expression evaluation over concrete checksums."""

from __future__ import annotations

from dataclasses import dataclass
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
    target_celltype: str


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


def get_expression_cache() -> dict[tuple[str, str, str, str], Checksum]:
    return _expression_cache


def choose_expression_evaluation_location(
    input_checksum: Checksum | str | bytes,
) -> str:
    """Return "local" when expression input data is available locally, else "remote"."""

    try:
        _get_local_buffer(Checksum(input_checksum))
    except ExpressionEvaluationError:
        return "remote"
    return "local"


def evaluate_expression(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    target_celltype: str,
    *,
    validator: Checksum | str | bytes | None = None,
    validator_language: str | None = None,
) -> Checksum:
    """Evaluate an expression locally and return the result checksum."""

    if validator is not None or validator_language is not None:
        # TODO validators: reject-only gate, excluded from expression identity.
        raise NotImplementedError("Expression validators are not implemented yet")

    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, target_celltype)
    cache_key = _cache_key(key)
    key.input_checksum.tempref()
    cached = _expression_cache.get(cache_key)
    if cached is not None:
        _publish_expression_result(cached)
        return cached

    input_buffer = _get_local_buffer(key.input_checksum)
    steps = parse_path(key.path)
    from .hash_type_validation import validate_expression

    validate_expression(
        key.input_checksum,
        buffer=input_buffer,
        source_celltype=key.input_celltype,
        path_steps=steps,
        target_celltype=key.target_celltype,
    )
    return _evaluate_expression_after_validation(key, input_buffer, steps, cache_key)


async def evaluate_expression_async(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    target_celltype: str,
    *,
    validator: Checksum | str | bytes | None = None,
    validator_language: str | None = None,
) -> Checksum:
    if validator is not None or validator_language is not None:
        # TODO validators: reject-only gate, excluded from expression identity.
        raise NotImplementedError("Expression validators are not implemented yet")

    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, target_celltype)
    cache_key = _cache_key(key)
    key.input_checksum.tempref()
    cached = _expression_cache.get(cache_key)
    if cached is not None:
        _publish_expression_result(cached)
        return cached

    input_buffer = _get_local_buffer(key.input_checksum)
    steps = parse_path(key.path)
    from .hash_type_validation import validate_expression_async

    await validate_expression_async(
        key.input_checksum,
        buffer=input_buffer,
        source_celltype=key.input_celltype,
        path_steps=steps,
        target_celltype=key.target_celltype,
    )
    return _evaluate_expression_after_validation(key, input_buffer, steps, cache_key)


async def evaluate_expression_remote(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    target_celltype: str,
    *,
    validator: Checksum | str | bytes | None = None,
    validator_language: str | None = None,
    execution: str = "auto",
    member_id: object | None = None,
) -> Checksum:
    """Evaluate an expression with remote cache lookup and optional jobserver dispatch."""

    if validator is not None or validator_language is not None:
        # TODO validators: reject-only gate, excluded from expression identity.
        raise NotImplementedError("Expression validators are not implemented yet")
    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, target_celltype)
    cache_key = _cache_key(key)
    key.input_checksum.tempref()
    cached = _expression_cache.get(cache_key)
    if cached is not None:
        _publish_expression_result(cached)
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
            key.target_celltype,
        )
        if result is not None:
            _expression_cache[cache_key] = result
            _publish_expression_result(result)
            return result

    location = execution
    if location == "auto":
        location = choose_expression_evaluation_location(key.input_checksum)
    if location == "local":
        result = await evaluate_expression_async(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.target_celltype,
        )
    elif location == "remote":
        return await _run_active_remote_expression(
            key,
            cache_key,
            database_remote=database_remote,
            member_id=member_id,
        )
    else:
        raise ValueError(f"Unknown expression execution location: {location!r}")

    result = Checksum(result)
    _expression_cache[cache_key] = result
    _publish_expression_result(result)
    if database_remote is not None:
        await database_remote.set_expression_result(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.target_celltype,
            result,
        )
    return result


async def _run_active_remote_expression(
    key: ExpressionKey,
    cache_key: tuple[str, str, str, str],
    *,
    database_remote,
    member_id: object | None,
) -> Checksum:
    member = object() if member_id is None else member_id
    with _active_expression_lock:
        active = _active_expressions.get(cache_key)
        if active is not None and active.result_future.done():
            _active_expressions.pop(cache_key, None)
            active = None
        if active is None:
            active = _ActiveExpression(
                result_future=concurrent.futures.Future(),
                task=None,
                members=set(),
            )
            active.task = asyncio.create_task(
                _execute_remote_expression(key, cache_key, active, database_remote)
            )
            _active_expressions[cache_key] = active
        active.members.add(member)
    try:
        return await asyncio.shield(asyncio.wrap_future(active.result_future))
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
        result = await jobserver_remote.run_expression(
            key.input_checksum,
            key.path,
            key.input_celltype,
            key.target_celltype,
        )
        result = Checksum(result)
        _expression_cache[cache_key] = result
        _publish_expression_result(result)
        if database_remote is not None:
            await database_remote.set_expression_result(
                key.input_checksum,
                key.path,
                key.input_celltype,
                key.target_celltype,
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


def softcancel_expression(
    cache_key: tuple[str, str, str, str], member_id: object | None
) -> bool:
    if member_id is None:
        return False
    with _active_expression_lock:
        active = _active_expressions.get(cache_key)
        if active is None or member_id not in active.members:
            return False
        active.members.remove(member_id)
        should_cancel = not active.members and not active.result_future.done()
        if should_cancel:
            active.canceled = True
    if not should_cancel:
        return True
    if active.task is not None and not active.task.done():
        active.task.cancel()
    if not active.result_future.done():
        active.result_future.set_exception(asyncio.CancelledError())
    return True


def cancel_expression(
    input_checksum: Checksum | str | bytes,
    path: str,
    input_celltype: str,
    target_celltype: str,
    *,
    member_id: object | None = None,
) -> bool:
    key = ExpressionKey(Checksum(input_checksum), path, input_celltype, target_celltype)
    return softcancel_expression(_cache_key(key), member_id)


def _evaluate_expression_after_validation(
    key: ExpressionKey,
    input_buffer: Buffer,
    steps: tuple[tuple[str, Any], ...],
    cache_key: tuple[str, str, str, str],
) -> Checksum:
    if key.path == "" and key.input_celltype == key.target_celltype:
        # HashType validation above has already proved the source input_celltype is
        # structurally compatible. This is the intended skipped source
        # deserialization path for identity expressions.
        _expression_cache[cache_key] = key.input_checksum
        _publish_expression_result(key.input_checksum, buffer=input_buffer)
        return key.input_checksum

    value = _deserialize_for_expression(input_buffer, key.input_celltype)
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

    result_buffer = _serialize_expression_result(value, key.target_celltype)
    result_checksum = result_buffer.get_checksum()
    _expression_cache[cache_key] = result_checksum
    _expression_result_buffers[result_checksum] = result_buffer
    _publish_expression_result(result_checksum, buffer=result_buffer)
    return result_checksum


def _publish_expression_result(
    checksum: Checksum, *, buffer: Buffer | None = None
) -> None:
    """Publish bounded cache interest for every successful result path."""

    checksum = Checksum(checksum)
    if buffer is None:
        checksum.tempref()
    else:
        from seamless.caching.buffer_cache import get_buffer_cache

        get_buffer_cache().tempref(checksum, buffer=buffer)


def resolve_expression_value(
    result_checksum: Checksum | str | bytes,
    target_celltype: str,
) -> Any:
    buffer = _get_local_buffer(Checksum(result_checksum))
    return _deserialize_for_expression(buffer, target_celltype)


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
        key.target_celltype,
    )


def _get_local_buffer(checksum: Checksum) -> Buffer:
    from seamless.caching.buffer_cache import get_buffer_cache
    from seamless.checksum.cached_calculate_checksum import checksum_cache

    buffer = get_buffer_cache().get(checksum)
    if buffer is not None:
        return buffer

    buffer = _expression_result_buffers.get(checksum)
    if buffer is not None:
        return buffer

    raw = checksum_cache.get(checksum)
    if raw is not None:
        return Buffer(raw, checksum=checksum)

    raise ExpressionEvaluationError(
        f"Buffer for checksum {checksum.hex()} is not available locally"
    )


def _deserialize_for_expression(buffer: Buffer, input_celltype: str) -> Any:
    if input_celltype == "bytes":
        return buffer.content
    return buffer.get_value(input_celltype)


def _serialize_expression_result(value: Any, target_celltype: str) -> Buffer:
    if target_celltype == "binary" and isinstance(value, bytes):
        import numpy as np

        return Buffer(np.array(value), "binary")
    result = Buffer(value, target_celltype)
    if target_celltype == "binary":
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
