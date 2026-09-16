"""HashType-based structural validation helpers."""

from __future__ import annotations

from typing import Any

from seamless.buffer_class import Buffer
from seamless.checksum_class import Checksum

from .conversion import (
    conversion_chain,
    conversion_equivalent,
    conversion_forbidden,
    conversion_possible,
    conversion_reformat,
    conversion_reinterpret,
    conversion_trivial,
    conversion_values,
)
from .hash_type import (
    DType,
    HashType,
    Rank,
    get_hash_type,
    get_hash_type_remote,
    register_hash_type_for_buffer,
    register_hash_type_for_buffer_async,
)


class HashTypeValidationError(ValueError):
    """Raised when HashType proves a checksum/celltype operation impossible."""


def _deserialization_error(
    checksum: Checksum,
    hash_type: HashType,
    celltype: str,
) -> HashTypeValidationError:
    return HashTypeValidationError(
        _message(
            "Cannot deserialize checksum as requested celltype",
            checksum,
            hash_type,
            celltype=celltype,
        )
    )


def ensure_hash_type(
    checksum: Checksum | str | bytes,
    *,
    buffer: Buffer | bytes | bytearray | memoryview | None = None,
) -> HashType | None:
    """Consult memory, then the database outside a running loop, then a buffer."""

    checksum = Checksum(checksum)
    hash_type = get_hash_type(checksum)
    if hash_type is not None:
        return hash_type
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        hash_type = asyncio.run(get_hash_type_remote(checksum))
        if hash_type is not None:
            return hash_type
    else:
        return None
    if buffer is None:
        return None
    return register_hash_type_for_buffer(checksum, buffer)


async def ensure_hash_type_async(
    checksum: Checksum | str | bytes,
    *,
    buffer: Buffer | bytes | bytearray | memoryview | None = None,
) -> HashType | None:
    """Return HashType from local/remote cache, computing from a buffer if available."""

    checksum = Checksum(checksum)
    hash_type = await get_hash_type_remote(checksum)
    if hash_type is not None:
        return hash_type
    if buffer is None:
        return None
    return await register_hash_type_for_buffer_async(checksum, buffer)


def validate_deserializable_as(
    checksum: Checksum | str | bytes,
    celltype: str,
    *,
    buffer: Buffer | bytes | bytearray | memoryview | None = None,
) -> HashType | None:
    """Reject when HashType proves checksum cannot deserialize as celltype."""

    checksum = Checksum(checksum)
    hash_type = ensure_hash_type(checksum, buffer=buffer)
    if hash_type is None:
        return None
    if hash_type.deserializable_as(celltype, checksum=checksum) is False:
        raise _deserialization_error(checksum, hash_type, celltype)
    return hash_type


async def validate_deserializable_as_async(
    checksum: Checksum | str | bytes,
    celltype: str,
    *,
    buffer: Buffer | bytes | bytearray | memoryview | None = None,
) -> HashType | None:
    """Async variant that can consult configured remote HashType databases."""

    checksum = Checksum(checksum)
    hash_type = await ensure_hash_type_async(checksum, buffer=buffer)
    if hash_type is None:
        return None
    if hash_type.deserializable_as(celltype, checksum=checksum) is False:
        raise HashTypeValidationError(
            _message(
                "Cannot deserialize checksum as requested celltype",
                checksum,
                hash_type,
                celltype=celltype,
            )
        )
    return hash_type


def validate_expression(
    checksum: Checksum | str | bytes,
    *,
    buffer: Buffer | bytes | bytearray | memoryview | None,
    source_celltype: str,
    path_steps: tuple[tuple[str, Any], ...],
    target_celltype: str,
) -> HashType | None:
    """Validate expression source, path capability, and empty-path conversion."""

    checksum = Checksum(checksum)
    hash_type = validate_deserializable_as(
        checksum, source_celltype, buffer=buffer
    )
    if hash_type is None:
        return None

    _validate_path_capability(checksum, hash_type, source_celltype, path_steps)
    if not path_steps:
        feasible = conversion_feasible(
            hash_type,
            source_celltype,
            target_celltype,
            checksum=checksum,
        )
        if feasible is False:
            raise HashTypeValidationError(
                _message(
                    "Cannot convert expression source to target celltype",
                    checksum,
                    hash_type,
                    celltype=source_celltype,
                    target_celltype=target_celltype,
                )
            )
    return hash_type


async def validate_expression_async(
    checksum: Checksum | str | bytes,
    *,
    buffer: Buffer | bytes | bytearray | memoryview | None,
    source_celltype: str,
    path_steps: tuple[tuple[str, Any], ...],
    target_celltype: str,
) -> HashType | None:
    """Async expression validation that can consult remote HashType databases."""

    checksum = Checksum(checksum)
    hash_type = await validate_deserializable_as_async(
        checksum, source_celltype, buffer=buffer
    )
    if hash_type is None:
        return None

    _validate_path_capability(checksum, hash_type, source_celltype, path_steps)
    if not path_steps:
        feasible = conversion_feasible(
            hash_type,
            source_celltype,
            target_celltype,
            checksum=checksum,
        )
        if feasible is False:
            raise HashTypeValidationError(
                _message(
                    "Cannot convert expression source to target celltype",
                    checksum,
                    hash_type,
                    celltype=source_celltype,
                    target_celltype=target_celltype,
                )
            )
    return hash_type


def conversion_feasible(
    hash_type: HashType | int,
    source_celltype: str,
    target_celltype: str,
    *,
    checksum: Checksum | str | bytes | None = None,
) -> bool | None:
    """Return False only when HashType proves conversion impossible."""

    from .null import is_null_value
    if is_null_value(checksum):
        return True
    hash_type = hash_type if isinstance(hash_type, HashType) else HashType.unpack(hash_type)
    if source_celltype == target_celltype:
        return True
    if hash_type.deserializable_as(source_celltype, checksum=checksum) is False:
        return False
    conv = (source_celltype, target_celltype)
    conv = conversion_equivalent.get(conv, conv)
    if conv in conversion_chain:
        return None
    if conv in conversion_forbidden:
        return False
    if conv in conversion_trivial or conv in conversion_reformat:
        return True
    if conv in conversion_reinterpret:
        return hash_type.deserializable_as(target_celltype, checksum=checksum)
    if conv in conversion_possible:
        if hash_type.is_untested:
            if target_celltype in ("int", "float") and hash_type.length.name == "LONG":
                return False
            return None
        return _possible_conversion_feasible(hash_type, source_celltype, target_celltype)
    if conv in conversion_values:
        if hash_type.is_untested:
            return None
        return _value_conversion_feasible(hash_type, source_celltype, target_celltype)
    return None


def _validate_path_capability(
    checksum: Checksum,
    hash_type: HashType,
    source_celltype: str,
    path_steps: tuple[tuple[str, Any], ...],
) -> None:
    """Reject a path step only where the root's HashType proves it impossible.

    HashType describes the root value.  A slice preserves the structure it is
    taken from, so the next step is checked against the same capabilities.  An
    item step descends into a child whose bits are unknown, so checking stops,
    except where the root's bits still type the child (type_bits_design §7.2):
    a ``bytes`` item is an int, and a numeric numpy array admits as many
    positional item steps as its rank.
    """
    if hash_type.is_untested:
        return
    caps = hash_type.capabilities(source_celltype)
    rank = int(hash_type.rank)
    for kind, payload in path_steps:
        needs = "SEQ"
        if kind == "item" and isinstance(payload, str):
            needs = "MAP"
        if needs not in caps:
            raise HashTypeValidationError(
                _message(
                    f"Expression path requires {needs} capability",
                    checksum,
                    hash_type,
                    celltype=source_celltype,
                    path_step=(kind, payload),
                )
            )
        if kind == "slice":
            continue
        if source_celltype == "bytes":
            caps = set()
        elif (
            source_celltype == "binary"
            and hash_type.dtype == DType.NUMERIC
            and type(payload) is int
            and rank < Rank.D3PLUS
        ):
            rank -= 1
            caps = {"SEQ"} if rank > Rank.SCALAR else set()
        else:
            return


def _possible_conversion_feasible(
    hash_type: HashType,
    source_celltype: str,
    target_celltype: str,
) -> bool | None:
    if target_celltype in ("int", "float"):
        if hash_type.length.name == "LONG":
            return False
        if source_celltype == "binary":
            return hash_type.dtype.name == "NUMERIC" and hash_type.rank.name == "SCALAR"
        return True if hash_type.is_json_numeric_scalar else False
    if target_celltype == "str" and source_celltype == "mixed":
        if hash_type.kind.name in ("JSON_OBJECT", "JSON_ARRAY"):
            return False
        return True
    return None


def _value_conversion_feasible(
    hash_type: HashType,
    source_celltype: str,
    target_celltype: str,
) -> bool | None:
    if target_celltype == "checksum":
        return hash_type.kind.name == "RAW_TEXT" and hash_type.length.name == "EQ64"
    if source_celltype == "checksum":
        return None
    if source_celltype == "plain" and target_celltype == "binary":
        if hash_type.kind.name == "JSON_OBJECT":
            return False
    if source_celltype == "binary" and target_celltype == "plain":
        if hash_type.dtype.name in ("NUMERIC", "STRUCTURED"):
            return True
    return None


def _message(
    reason: str,
    checksum: Checksum,
    hash_type: HashType,
    **details,
) -> str:
    detail = ", ".join(f"{key}={value!r}" for key, value in details.items())
    suffix = f", {detail}" if detail else ""
    return (
        f"{reason}: checksum={checksum.hex()}, hash_type={hash_type.word} "
        f"({hash_type.kind.name}/{hash_type.length.name}/"
        f"{hash_type.dtype.name}/{hash_type.rank.name}, flags={int(hash_type.flags)})"
        f"{suffix}"
    )


__all__ = [
    "HashTypeValidationError",
    "conversion_feasible",
    "ensure_hash_type",
    "ensure_hash_type_async",
    "validate_deserializable_as",
    "validate_deserializable_as_async",
    "validate_expression",
    "validate_expression_async",
]
