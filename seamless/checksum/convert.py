"""Lazy, checksum-first conversion of concrete cell buffers.

``convert_checksum`` deliberately knows nothing about a buffer cache or a
remote server.  Its caller supplies a zero-argument ``get_buffer`` callback;
the callback is invoked only for a rule that needs actual buffer content.
This makes the cheap part of empty-path Expression evaluation available before
choosing where an expression should run.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import orjson
import yaml

from seamless import CacheMissError
from seamless.buffer_class import Buffer
from seamless.checksum_class import Checksum

from .celltypes import celltypes
from .conversion import (
    SeamlessConversionError,
    conversion_chain,
    conversion_equivalent,
    conversion_forbidden,
    conversion_possible,
    conversion_reformat,
    conversion_reinterpret,
    conversion_trivial,
    conversion_values,
)


class _NeedsBuffer(Exception):
    """Private dry-run signal raised when a conversion needs source content."""


class _LazyBuffer:
    """Memoize a conversion's buffer callback without invoking it eagerly."""

    def __init__(self, get_buffer: Callable[[], Buffer | bytes | bytearray | memoryview]):
        self._get_buffer = get_buffer
        self._loaded = False
        self._buffer: Buffer | None = None

    def __call__(self) -> Buffer:
        if not self._loaded:
            raw_buffer = self._get_buffer()
            self._buffer = _coerce_buffer(raw_buffer)
            self._loaded = True
        assert self._buffer is not None
        return self._buffer


def convert_checksum(
    checksum: Checksum | str | bytes,
    source: str,
    target: str,
    get_buffer: Callable[[], Buffer | bytes | bytearray | memoryview],
) -> tuple[Checksum, Buffer | None]:
    """Convert ``checksum`` from ``source`` to ``target`` lazily.

    A checksum-preserving conversion returns ``(checksum, None)``.  A
    conversion that serializes new content returns the checksum of that content
    together with the newly-created :class:`~seamless.Buffer`.
    """

    checksum = Checksum(checksum)
    if source not in celltypes:
        raise TypeError(source)
    if target not in celltypes:
        raise TypeError(target)
    if not callable(get_buffer):
        raise TypeError("get_buffer must be callable")

    try:
        return _convert(checksum, source, target, _LazyBuffer(get_buffer))
    except (_NeedsBuffer, SeamlessConversionError, CacheMissError):
        # A missing buffer is not a failed conversion: it keeps its type.
        raise
    except Exception as exc:
        raise _conversion_error(checksum, source, target, exc) from None


def conversion_needs_buffer(
    checksum: Checksum | str | bytes,
    source: str,
    target: str,
) -> bool:
    """Return whether this conversion must read its source buffer.

    A conversion which can already be rejected from virtual/checksum metadata
    is not a buffer-requiring conversion.  Its failure is intentionally
    treated as local by expression location selection.
    """

    def needs_buffer() -> Buffer:
        raise _NeedsBuffer

    try:
        convert_checksum(checksum, source, target, needs_buffer)
    except _NeedsBuffer:
        return True
    except Exception:
        return False
    return False


def _convert(
    checksum: Checksum,
    source: str,
    target: str,
    get_buffer: Callable[[], Buffer],
) -> tuple[Checksum, Buffer | None]:
    if source == target:
        return checksum, None

    conv = (source, target)
    equivalent = conversion_equivalent.get(conv)
    if equivalent is not None:
        return _convert(checksum, equivalent[0], equivalent[1], get_buffer)

    intermediate = conversion_chain.get(conv)
    if intermediate is not None:
        intermediate_checksum, intermediate_buffer = _convert(
            checksum, source, intermediate, get_buffer
        )
        if intermediate_buffer is None:
            next_get_buffer = get_buffer
        else:
            next_get_buffer = _LazyBuffer(lambda: intermediate_buffer)
        return _convert(
            intermediate_checksum, intermediate, target, next_get_buffer
        )

    if conv in conversion_forbidden:
        raise SeamlessConversionError(
            f"{checksum.hex()} cannot be converted from {source} to {target}"
        )
    if conv in conversion_trivial:
        return checksum, None
    if conv in conversion_reinterpret:
        _value_of(checksum, target, get_buffer)
        return checksum, None
    if conv in conversion_reformat:
        return _convert_reformat(checksum, source, target, get_buffer)
    if conv in conversion_possible:
        return _convert_possible(checksum, source, target, get_buffer)
    if conv in conversion_values:
        return _convert_values(checksum, source, target, get_buffer)
    raise AssertionError(f"Unclassified conversion: {conv!r}")


def _value_of(
    checksum: Checksum,
    celltype: str,
    get_buffer: Callable[[], Buffer],
) -> Any:
    """Resolve one value, using virtual values before reading a buffer."""

    from .hash_type_validation import validate_deserializable_as
    from .parse_buffer import _parse_buffer
    from .virtual import NOT_VIRTUAL, virtual_value

    value = virtual_value(checksum, celltype)
    if value is not NOT_VIRTUAL:
        return value

    # If cached HashType already disproves the interpretation, fail before a
    # local or remote getter is touched.  With no cached type this is a no-op;
    # _parse_buffer registers and validates after obtaining the buffer.
    validate_deserializable_as(checksum, celltype)
    return _parse_buffer(get_buffer(), checksum, celltype)


def _convert_reformat(
    checksum: Checksum,
    source: str,
    target: str,
    get_buffer: Callable[[], Buffer],
) -> tuple[Checksum, Buffer | None]:
    conv = (source, target)
    if conv == ("bytes", "binary"):
        return _bytes_to_binary(checksum, get_buffer)
    if conv == ("bytes", "mixed"):
        return _bytes_to_mixed(checksum, get_buffer)
    if conv == ("binary", "bytes"):
        return _binary_to_bytes(checksum, get_buffer)
    if conv == ("mixed", "bytes"):
        return _mixed_to_bytes(checksum, get_buffer)
    if conv == ("plain", "text"):
        value = _value_of(checksum, "plain", get_buffer)
        if isinstance(value, str):
            return _buffer_result(Buffer(value, "text"))
        return checksum, None
    if conv == ("text", "plain"):
        return _text_to_plain(checksum, get_buffer)
    if conv in (("text", "str"), ("str", "text")):
        value = _value_of(checksum, source, get_buffer)
        return _buffer_result(Buffer(value, target))
    if conv == ("yaml", "plain"):
        text = _value_of(checksum, "yaml", get_buffer)
        return _buffer_result(Buffer(yaml.safe_load(text), "plain"))
    if conv == ("ipython", "python"):
        from seamless.util.ipython import ipython2python

        text = _value_of(checksum, "ipython", get_buffer)
        return _buffer_result(Buffer(ipython2python(text), "python"))
    raise AssertionError(conv)


def _bytes_to_binary(
    checksum: Checksum, get_buffer: Callable[[], Buffer]
) -> tuple[Checksum, Buffer | None]:
    # Retaining the checksum depends on the actual NPY bytes, so this rule is
    # deliberately content-gated even when HashType metadata is available.
    buffer = get_buffer()
    from seamless.util.mixed import MAGIC_NUMPY

    if buffer.content.startswith(MAGIC_NUMPY):
        # Do not accept a coincidental magic prefix: parse/validate it first.
        _value_of(checksum, "binary", get_buffer)
        return checksum, None

    import numpy as np

    return _buffer_result(Buffer(np.array(buffer.content), "binary"))


def _bytes_to_mixed(
    checksum: Checksum, get_buffer: Callable[[], Buffer]
) -> tuple[Checksum, Buffer | None]:
    from .virtual import NOT_VIRTUAL, virtual_value

    if virtual_value(checksum, "mixed") is not NOT_VIRTUAL:
        return checksum, None

    hash_type = _cached_hash_type(checksum)
    if hash_type is not None and hash_type.deserializable_as(
        "mixed", checksum=checksum
    ):
        return checksum, None

    buffer = get_buffer()
    from .hash_type import HashType

    inspected_type = HashType.from_buffer(buffer, checksum=checksum)
    if inspected_type.deserializable_as("mixed", checksum=checksum):
        # Ensure a malformed mixed/NPY payload cannot be retained merely from
        # its prefix or JSON classification.
        _value_of(checksum, "mixed", get_buffer)
        return checksum, None

    try:
        text = buffer.content.decode().rstrip("\n")
    except UnicodeDecodeError:
        import numpy as np

        return _buffer_result(Buffer(np.array(buffer.content), "binary"))
    return _buffer_result(Buffer(text, "str"))


def _binary_to_bytes(
    checksum: Checksum, get_buffer: Callable[[], Buffer]
) -> tuple[Checksum, Buffer | None]:
    # The concrete array, rather than only its cached type, determines whether
    # it is the exceptional zero-dimensional dtype-S representation.
    value = _value_of(checksum, "binary", get_buffer)
    if getattr(value, "ndim", None) == 0 and getattr(value.dtype, "kind", None) == "S":
        return _buffer_result(Buffer(value.tobytes()))
    return checksum, None


def _mixed_to_bytes(
    checksum: Checksum, get_buffer: Callable[[], Buffer]
) -> tuple[Checksum, Buffer | None]:
    # The NPY magic branch is a property of the concrete source buffer; a
    # mixed-to-bytes conversion therefore cannot be decided from its output
    # checksum alone.
    buffer = get_buffer()
    from seamless.util.mixed import MAGIC_NUMPY

    if buffer.content.startswith(MAGIC_NUMPY):
        return _binary_to_bytes(checksum, get_buffer)

    # A non-NPY mixed buffer is retained, but still has to be a valid mixed
    # value when its type was not known beforehand.
    _value_of(checksum, "mixed", get_buffer)
    return checksum, None


def _text_to_plain(
    checksum: Checksum, get_buffer: Callable[[], Buffer]
) -> tuple[Checksum, Buffer | None]:
    from .virtual import NOT_VIRTUAL, virtual_value

    # Canonical null/booleans are valid JSON plain values by checksum alone.
    if virtual_value(checksum, "plain") is not NOT_VIRTUAL:
        return checksum, None

    text = _value_of(checksum, "text", get_buffer)
    try:
        orjson.loads(text)
    except orjson.JSONDecodeError:
        return _buffer_result(Buffer(text, "plain"))
    return checksum, None


def _convert_possible(
    checksum: Checksum,
    source: str,
    target: str,
    get_buffer: Callable[[], Buffer],
) -> tuple[Checksum, Buffer | None]:
    value = _value_of(checksum, source, get_buffer)
    if isinstance(value, (dict, list)):
        raise TypeError(type(value))
    try:
        import numpy as np

        if isinstance(value, np.ndarray) and value.ndim:
            raise TypeError((type(value), value.ndim))
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    target_value = getattr(builtins, target)(value)
    return _buffer_result(Buffer(target_value, target))


def _convert_values(
    checksum: Checksum,
    source: str,
    target: str,
    get_buffer: Callable[[], Buffer],
) -> tuple[Checksum, Buffer | None]:
    if target == "checksum":
        # A checksum cell stores the checksum value of its source buffer.
        return _buffer_result(Buffer(checksum.hex().encode()))

    if source == "checksum":
        # A checksum value refers to another buffer; the conversion dereferences it.
        return Checksum(_value_of(checksum, "checksum", get_buffer)), None

    value = _value_of(checksum, source, get_buffer)
    conv = (source, target)
    if conv == ("binary", "plain"):
        # Convert numpy objects to ordinary JSON values, then use the normal
        # canonical plain serializer rather than retaining compact JSON bytes.
        encoded = orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)
        return _buffer_result(Buffer(orjson.loads(encoded), "plain"))
    if conv == ("plain", "binary"):
        import numpy as np

        if not isinstance(value, (int, float, bool, list)):
            raise ValueError(value)
        target_value = np.array(value)
        if target_value.dtype == object:
            raise ValueError(value)
        return _buffer_result(Buffer(target_value, "binary"))

    # The remaining value rules are bool <-> int/float conversions.  Source
    # booleans are resolved by _value_of without fetching their canonical
    # buffer.
    target_value = getattr(builtins, target)(value)
    return _buffer_result(Buffer(target_value, target))


def _cached_hash_type(checksum: Checksum):
    from .hash_type import get_hash_type

    return get_hash_type(checksum)


def _coerce_buffer(value: Buffer | bytes | bytearray | memoryview) -> Buffer:
    if isinstance(value, Buffer):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes()
    elif isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise TypeError(f"get_buffer returned {type(value)!r}, not a Buffer or bytes")
    return Buffer(value)


def _buffer_result(buffer: Buffer) -> tuple[Checksum, Buffer]:
    checksum = buffer.get_checksum()
    return checksum, buffer


def _conversion_error(
    checksum: Checksum, source: str, target: str, exc: Exception
) -> SeamlessConversionError:
    message = f"{checksum.hex()} cannot be converted from {source} to {target}"
    detail = str(exc)
    if detail:
        message += f"\n\nOriginal exception:\n\n{detail}"
    return SeamlessConversionError(message)


__all__ = ["conversion_needs_buffer", "convert_checksum"]
