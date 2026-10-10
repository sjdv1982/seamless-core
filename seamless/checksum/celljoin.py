"""Canonical definitions and pure evaluation for cell-level joins."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import index as integer_index

from seamless import Buffer
from seamless.checksum_class import Checksum


CELLJOIN_CELLTYPES = ("mixed", "plain", "deepcell", "deepfolder")
_DEEP_CELLTYPES = frozenset(("deepcell", "deepfolder"))
_HEX_CHECKSUM = re.compile(r"[0-9a-f]{64}\Z")
_NUMERIC_KEY = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_RESERVED_KEYS = frozenset(("<root>", "<numeric>"))


@dataclass(frozen=True, slots=True)
class CellJoinSpec:
    """Parsed, immutable celljoin definition and its evaluation celltype."""

    checksum: Checksum
    celltype: str
    root: Checksum | None
    numeric: bool
    members: tuple[tuple[str, Checksum], ...]
    buffer: Buffer


def build_celljoin(
    root: Checksum | None, members: Mapping[str | int, Checksum]
) -> dict[str, str | None]:
    """Build the canonical mapping of input checksums for a celljoin."""
    if root is not None and not isinstance(root, Checksum):
        raise TypeError("Celljoin root must be a Checksum or None")
    if not isinstance(members, Mapping):
        raise TypeError("Celljoin members must be a mapping")

    member_kind = None
    normalized_members: dict[str, str] = {}
    for key, checksum in members.items():
        if isinstance(key, bool):
            raise TypeError("Boolean celljoin member keys are not allowed")
        if isinstance(key, str):
            if key in _RESERVED_KEYS:
                raise ValueError(f"Reserved celljoin member key: {key!r}")
            normalized_key = key
            this_kind = "string"
        elif isinstance(key, int):
            # The index protocol is broader than the celljoin key contract:
            # only integer keys themselves are accepted, not arbitrary objects
            # that happen to implement __index__.
            integer_key = integer_index(key)
            if integer_key < 0:
                raise ValueError("Celljoin member keys must be non-negative")
            normalized_key = str(integer_key)
            this_kind = "integer"
        else:
            raise TypeError(
                "Celljoin member keys must be strings or non-negative integers"
            )

        if member_kind is None:
            member_kind = this_kind
        elif member_kind != this_kind:
            raise TypeError("Celljoin member keys cannot mix strings and integers")

        if normalized_key in normalized_members:
            raise ValueError(f"Duplicate normalized celljoin member key: {normalized_key!r}")
        if not isinstance(checksum, Checksum):
            raise TypeError("Celljoin member values must be Checksum objects")
        normalized_members[normalized_key] = checksum.hex()

    result: dict[str, str | None] = {}
    if member_kind == "integer":
        result["<numeric>"] = None
    if root is not None:
        result["<root>"] = root.hex()
    result.update(normalized_members)
    return result


def celljoin_buffer(celljoin: dict[str, str | None]) -> Buffer:
    """Serialize a celljoin mapping with Seamless plain canonicalization."""
    return Buffer(celljoin, "plain")


def definition_store_available() -> bool:
    """Return whether both the hashserver and database can accept writes."""
    try:
        from seamless_remote import buffer_remote, database_remote
    except ImportError:
        return False
    return buffer_remote.has_write_server() and database_remote.has_write_server()


def publish_celljoin_definition(spec: CellJoinSpec) -> None:
    """Queue the canonical celljoin definition on the hashserver."""
    if definition_store_available():
        spec.buffer.transfer_write()


def _load_definition(
    definition: Buffer | bytes | str | Mapping,
) -> tuple[dict, Buffer | None]:
    if isinstance(definition, Buffer):
        buffer = definition
        celljoin = buffer.get_value("plain")
    elif isinstance(definition, bytes):
        buffer = Buffer(definition)
        celljoin = buffer.get_value("plain")
    elif isinstance(definition, str):
        buffer = Buffer(definition.encode("utf-8"))
        celljoin = buffer.get_value("plain")
    elif isinstance(definition, Mapping):
        celljoin = dict(definition)
        buffer = None
    else:
        raise TypeError("Celljoin definition must be a Buffer, bytes, string, or mapping")

    if not isinstance(celljoin, dict):
        raise TypeError("Celljoin definition must be a JSON object")
    return celljoin, buffer


def _parse_checksum(value, field: str) -> Checksum:
    if not isinstance(value, str):
        raise TypeError(f"Celljoin {field} checksum must be a string")
    if _HEX_CHECKSUM.fullmatch(value) is None:
        raise ValueError(f"Celljoin {field} checksum must be 64 lowercase hex characters")
    return Checksum(value)


def parse_celljoin(
    definition: Buffer | bytes | str | Mapping, celltype: str
) -> CellJoinSpec:
    """Validate and parse a celljoin definition."""
    if celltype not in CELLJOIN_CELLTYPES:
        raise ValueError(f"Unsupported celljoin celltype: {celltype!r}")

    celljoin, buffer = _load_definition(definition)
    for key in celljoin:
        if not isinstance(key, str):
            raise TypeError("Celljoin definition keys must be strings")

    numeric = "<numeric>" in celljoin
    if numeric and celljoin["<numeric>"] is not None:
        raise ValueError('Celljoin "<numeric>" marker must be null')
    if numeric and celltype in _DEEP_CELLTYPES:
        raise ValueError('Deep celljoins cannot contain a "<numeric>" marker')

    root = None
    if "<root>" in celljoin:
        root = _parse_checksum(celljoin["<root>"], "root")

    members = []
    for key, value in celljoin.items():
        if key in _RESERVED_KEYS:
            continue
        if numeric and _NUMERIC_KEY.fullmatch(key) is None:
            raise ValueError(
                'Celljoin member keys with a "<numeric>" marker must be canonical non-negative integers'
            )
        members.append((key, _parse_checksum(value, f"member {key!r}")))

    if numeric and not members:
        raise ValueError('Celljoin "<numeric>" marker requires at least one member')

    if buffer is None:
        buffer = celljoin_buffer(celljoin)
    return CellJoinSpec(
        checksum=buffer.get_checksum(),
        celltype=celltype,
        root=root,
        numeric=numeric,
        members=tuple(members),
        buffer=buffer,
    )


def celljoin_cache_key(checksum: Checksum | str | bytes, celltype: str):
    """Return the process-wide identity key for a celljoin."""
    if not isinstance(checksum, Checksum):
        checksum = Checksum(checksum)
    return ("celljoin", checksum.hex(), celltype)


def required_buffers(spec: CellJoinSpec) -> tuple[Checksum, ...]:
    """Return the distinct input buffers needed to evaluate ``spec``."""
    checksums = []
    seen = set()

    if spec.root is not None:
        checksums.append(spec.root)
        seen.add(spec.root)
    if spec.celltype not in _DEEP_CELLTYPES:
        for _, checksum in spec.members:
            if checksum not in seen:
                checksums.append(checksum)
                seen.add(checksum)
    return tuple(checksums)


def _resolve_value(
    checksum: Checksum,
    celltype: str,
    get_buffer: Callable[[Checksum], Buffer],
):
    from .virtual import NOT_VIRTUAL, virtual_value

    value = virtual_value(checksum, celltype)
    if value is not NOT_VIRTUAL:
        return value
    return get_buffer(checksum).get_value(celltype)


def evaluate_celljoin(
    spec: CellJoinSpec, get_buffer: Callable[[Checksum], Buffer]
) -> Buffer:
    """Assemble and serialize a celljoin using only the supplied buffer getter."""
    if spec.celltype in _DEEP_CELLTYPES:
        index = (
            {}
            if spec.root is None
            else _resolve_value(spec.root, spec.celltype, get_buffer)
        )
        for key, checksum in spec.members:
            index[key] = checksum
        return Buffer(index, spec.celltype)

    value = (
        {}
        if spec.root is None
        else _resolve_value(spec.root, spec.celltype, get_buffer)
    )
    if spec.numeric and not isinstance(value, (list, tuple)):
        raise TypeError("Integer Cell connection targets require an existing sequence")

    if spec.numeric:
        for key, checksum in spec.members:
            item_index = int(key)
            if item_index >= len(value):
                raise IndexError("list assignment index out of range")
            value[item_index] = _resolve_value(checksum, spec.celltype, get_buffer)
    else:
        for key, checksum in spec.members:
            value[key] = _resolve_value(checksum, spec.celltype, get_buffer)
    return Buffer(value, spec.celltype)


_CELLJOIN_FETCH_CONCURRENCY = 32


async def _gather_celljoin_buffers(required, preloaded=None):
    """Collect the required input buffers, using provided buffers first."""
    import asyncio

    from .expression import _local_input_buffer_async

    preloaded = {} if preloaded is None else preloaded
    semaphore = asyncio.Semaphore(_CELLJOIN_FETCH_CONCURRENCY)

    async def get_one(checksum):
        if checksum in preloaded:
            return checksum, preloaded[checksum]
        async with semaphore:
            buffer = await _local_input_buffer_async(checksum)
            if buffer is None:
                buffer = await checksum.resolution()
        return checksum, buffer

    tasks = [asyncio.create_task(get_one(checksum)) for checksum in required]
    try:
        pairs = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return dict(pairs)


def _record_celljoin_result(
    spec: CellJoinSpec, result: Checksum, buffer: Buffer
) -> bool:
    """Keep every produced buffer, and report whether its mapping was new."""
    from .expression import (
        _expression_result_buffers,
        _insert_expression_result,
        _tempref_expression_result,
    )

    cache_key = celljoin_cache_key(spec.checksum, spec.celltype)
    inserted = _insert_expression_result(cache_key, result)
    _expression_result_buffers[Checksum(result)] = buffer
    _tempref_expression_result(result, buffer=buffer, produced=True)
    if inserted:
        from seamless.caching import buffer_writer

        buffer_writer.register_celljoin_result(
            spec.checksum.hex(), spec.celltype, result
        )
    return inserted


async def _evaluate_celljoin_with_buffers(spec: CellJoinSpec, buffers):
    import asyncio

    buffer = await asyncio.to_thread(evaluate_celljoin, spec, buffers.__getitem__)
    result = buffer.get_checksum()
    _record_celljoin_result(spec, result, buffer)
    return result


async def evaluate_celljoin_local_async(
    spec: CellJoinSpec,
    *,
    member_id=None,
    materialize=False,
    buffers=None,
) -> Checksum:
    """Evaluate one celljoin locally, sharing active work by its identity."""
    import asyncio
    import concurrent.futures

    from .expression import (
        _active_expression_lock,
        _active_expressions,
        _discard_active_expression,
        _expression_cache,
        _has_local_buffer,
        _join_expression,
        _lingering_expressions,
        _tempref_expression_result,
        _ActiveExpression,
        softcancel_expression,
    )

    key = celljoin_cache_key(spec.checksum, spec.celltype)
    cached = _expression_cache.get(key)
    if cached is not None and (not materialize or _has_local_buffer(cached)):
        _tempref_expression_result(cached)
        return cached

    required = required_buffers(spec)
    if not required:
        gathered = await _gather_celljoin_buffers(required, buffers)
        return await _evaluate_celljoin_with_buffers(spec, gathered)

    member = object() if member_id is None else member_id
    with _active_expression_lock:
        active = _active_expressions.get(key) or _lingering_expressions.pop(key, None)
        if active is None or active.result_future.done():
            active = _ActiveExpression(concurrent.futures.Future(), None, set())
            active.hold_inputs(required)
            _active_expressions[key] = active

            async def execute():
                try:
                    gathered = await _gather_celljoin_buffers(required, buffers)
                    result = await _evaluate_celljoin_with_buffers(spec, gathered)
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
        result = await asyncio.shield(waiter)
    finally:
        softcancel_expression(key, member)

    if materialize and not _has_local_buffer(result):
        gathered = await _gather_celljoin_buffers(required, buffers)
        return await _evaluate_celljoin_with_buffers(spec, gathered)
    return result


async def evaluate_celljoin_placed(
    spec: CellJoinSpec,
    *,
    execution="auto",
    member_id=None,
    scratch=True,
) -> Checksum:
    """Resolve a recorded result or evaluate the celljoin according to placement."""
    from .expression import (
        _expression_cache,
        _local_input_buffer_async,
        _tempref_expression_result,
        _has_daskserver,
    )

    key = celljoin_cache_key(spec.checksum, spec.celltype)
    result = _expression_cache.get(key)
    if result is not None:
        _tempref_expression_result(result)
        return result
    try:
        from seamless_remote import database_remote
    except ImportError:
        database_remote = None
    if database_remote is not None:
        result = await database_remote.get_celljoin_result(
            spec.checksum, spec.celltype
        )
        if result is not None:
            from .expression import _insert_expression_result

            _insert_expression_result(key, result)
            result = _expression_cache.get(key, Checksum(result))
            _tempref_expression_result(result)
            return result

    if execution not in ("auto", "local", "remote"):
        raise ValueError(f"Unknown celljoin execution location: {execution!r}")

    if spec.celltype in _DEEP_CELLTYPES:
        return await evaluate_celljoin_local_async(spec, member_id=member_id)

    required = required_buffers(spec)
    if execution == "local":
        return await evaluate_celljoin_local_async(spec, member_id=member_id)

    try:
        from seamless_remote import buffer_remote
    except ImportError:
        buffer_remote = None

    if execution == "remote":
        if not definition_store_available():
            raise RuntimeError(
                "Remote celljoin evaluation requires a hashserver and a database"
            )
        present = (
            await buffer_remote.has_buffers(required)
            if buffer_remote is not None
            else [False] * len(required)
        )
        missing = [
            checksum
            for checksum, is_present in zip(required, present)
            if not is_present
        ]
        if missing:
            from seamless import CacheMissError

            raise CacheMissError(missing[0])
        raise NotImplementedError("Remote celljoin dispatch is not implemented yet")

    local_missing = []
    for checksum in required:
        if await _local_input_buffer_async(checksum) is None:
            local_missing.append(checksum)
    if not local_missing:
        return await evaluate_celljoin_local_async(spec, member_id=member_id)

    present = (
        await buffer_remote.has_buffers(required)
        if buffer_remote is not None
        else [False] * len(required)
    )
    try:
        from seamless_remote import jobserver_remote
    except ImportError:
        backend_available = False
    else:
        backend_available = jobserver_remote.has_jobserver() or _has_daskserver()

    if all(present) and backend_available and definition_store_available():
        raise NotImplementedError("Remote celljoin dispatch is not implemented yet")

    remote_missing = [
        checksum
        for checksum, is_present in zip(required, present)
        if checksum in local_missing and not is_present
    ]
    if remote_missing:
        from seamless import CacheMissError

        raise CacheMissError(remote_missing[0])
    return await evaluate_celljoin_local_async(spec, member_id=member_id)
