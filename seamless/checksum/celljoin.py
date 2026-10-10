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
