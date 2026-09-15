"""Packed checksum HashType words and local producers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import hashlib
import io
import json
import math
from typing import Any

import orjson

from seamless.checksum_class import Checksum


class Kind(IntEnum):
    RAW_BYTES = 0
    NUMPY = 1
    MIXED_OBJECT = 2
    MIXED_ARRAY = 3
    RAW_TEXT = 4
    JSON_OBJECT = 5
    JSON_ARRAY = 6
    JSON_STRING = 7
    JSON_NUMBER = 8
    UNTESTED = 9
    UTF8_UNTESTED = 10
    JSON_UNTESTED = 11


class Length(IntEnum):
    SHORT = 0
    EQ64 = 1
    MEDIUM = 2
    LONG = 3


class DType(IntEnum):
    NA = 0
    NUMERIC = 1
    NONNUMERIC = 2
    STRUCTURED = 3


class Rank(IntEnum):
    SCALAR = 0
    D1 = 1
    D2 = 2
    D3PLUS = 3


class Flag(IntFlag):
    NUMERIC_SCALAR = 1 << 0
    NUMPY_BYTES = 1 << 1
    SEMANTIC = 1 << 2


_KIND_SHIFT = 0
_LENGTH_SHIFT = 4
_DTYPE_SHIFT = 6
_RANK_SHIFT = 8
_FLAG_SHIFT = 10
_MAX_WORD = 1 << 13


@dataclass(frozen=True, slots=True)
class HashType:
    """Decoded representation of a packed HashType word."""

    kind: Kind
    length: Length
    dtype: DType = DType.NA
    rank: Rank = Rank.SCALAR
    flags: Flag = Flag(0)

    @property
    def word(self) -> int:
        return pack(self.kind, self.length, self.dtype, self.rank, self.flags)

    @property
    def is_utf8(self) -> bool:
        return self.kind in (
            Kind.RAW_TEXT, Kind.JSON_OBJECT, Kind.JSON_ARRAY,
            Kind.JSON_STRING, Kind.JSON_NUMBER,
            Kind.UTF8_UNTESTED, Kind.JSON_UNTESTED,
        )

    @property
    def is_json(self) -> bool:
        return self.kind in (
            Kind.JSON_OBJECT, Kind.JSON_ARRAY, Kind.JSON_STRING,
            Kind.JSON_NUMBER, Kind.JSON_UNTESTED,
        )

    @property
    def is_untested(self) -> bool:
        return self.kind in (Kind.UNTESTED, Kind.UTF8_UNTESTED, Kind.JSON_UNTESTED)

    @property
    def mic(self) -> str:
        return MIC_BY_KIND[self.kind]

    @property
    def is_numpy(self) -> bool:
        return self.kind == Kind.NUMPY

    @property
    def is_mixed(self) -> bool:
        return self.kind in (Kind.MIXED_OBJECT, Kind.MIXED_ARRAY)

    @property
    def is_json_numeric_scalar(self) -> bool:
        return bool(self.flags & Flag.NUMERIC_SCALAR)

    def deserializable_as(
        self, celltype: str, *, checksum: Checksum | str | bytes | None = None
    ) -> bool | None:
        return deserializable_as(self, celltype, checksum=checksum)

    def capabilities(self, source_celltype: str) -> set[str]:
        return capabilities(self, source_celltype)

    def has_slicing(self, source_celltype: str) -> bool:
        return "SEQ" in self.capabilities(source_celltype)

    def has_numeric_items(self, source_celltype: str) -> bool | None:
        return has_numeric_items(self, source_celltype)

    def has_string_items(self, source_celltype: str) -> bool | None:
        return has_string_items(self, source_celltype)

    @classmethod
    def from_buffer(
        cls,
        buffer: bytes | bytearray | memoryview | Any,
        *,
        checksum: Checksum | str | bytes | None = None,
        value: Any = None,
        celltype: str | None = None,
        semantic: bool = False,
    ) -> "HashType":
        return from_buffer(
            buffer,
            checksum=checksum,
            value=value,
            celltype=celltype,
            semantic=semantic,
        )

    @classmethod
    def unpack(cls, word: int) -> "HashType":
        return unpack(word)

    @staticmethod
    def is_valid_word(word: int) -> bool:
        return is_valid_word(word)


MIC_BY_KIND = {
    Kind.RAW_BYTES: "bytes",
    Kind.RAW_TEXT: "text",
    Kind.NUMPY: "binary",
    Kind.MIXED_OBJECT: "mixed",
    Kind.MIXED_ARRAY: "mixed",
    Kind.JSON_OBJECT: "plain",
    Kind.JSON_ARRAY: "plain",
    Kind.JSON_STRING: "str",
    Kind.JSON_NUMBER: "float",
    Kind.UNTESTED: "bytes",
    Kind.UTF8_UNTESTED: "text",
    Kind.JSON_UNTESTED: "plain",
}

FLAT_SEQ_CELLTYPES = {"text", "str", "python", "ipython", "yaml"}
CHECKSUM_TRUE = Checksum(hashlib.sha256(b"true").digest())
CHECKSUM_FALSE = Checksum(hashlib.sha256(b"false").digest())
CHECKSUM_NULL = Checksum(hashlib.sha256(b"null").digest())
_CHECKSUM_TRUE_NL = Checksum(hashlib.sha256(b"true\n").digest())
_CHECKSUM_FALSE_NL = Checksum(hashlib.sha256(b"false\n").digest())
_CHECKSUM_NULL_NL = Checksum(hashlib.sha256(b"null\n").digest())
_BOOL_CHECKSUMS = {
    CHECKSUM_TRUE,
    CHECKSUM_FALSE,
    _CHECKSUM_TRUE_NL,
    _CHECKSUM_FALSE_NL,
}
_SCALAR_CONST_CHECKSUMS = {
    CHECKSUM_TRUE,
    CHECKSUM_FALSE,
    CHECKSUM_NULL,
    _CHECKSUM_TRUE_NL,
    _CHECKSUM_FALSE_NL,
    _CHECKSUM_NULL_NL,
}
MAGIC_NUMPY = b"\x93NUMPY"
MAGIC_SEAMLESS_MIXED = b"\x94SEAMLESS-MIXED"
_hash_type_cache: dict[Checksum, int] = {}


def pack(
    kind: Kind,
    length: Length,
    dtype: DType = DType.NA,
    rank: Rank = Rank.SCALAR,
    flags: Flag = Flag(0),
) -> int:
    """Pack HashType fields into the 13-bit integer representation."""

    kind = Kind(kind)
    length = Length(length)
    dtype = DType(dtype)
    rank = Rank(rank)
    flags = Flag(flags)
    return (
        (kind << _KIND_SHIFT)
        | (length << _LENGTH_SHIFT)
        | (dtype << _DTYPE_SHIFT)
        | (rank << _RANK_SHIFT)
        | (flags << _FLAG_SHIFT)
    )


def unpack(word: int) -> HashType:
    """Decode a packed HashType word."""

    if not isinstance(word, int):
        raise TypeError(type(word))
    if word < 0 or word >= _MAX_WORD:
        raise ValueError(word)
    kind = Kind((word >> _KIND_SHIFT) & 0xF)
    length = Length((word >> _LENGTH_SHIFT) & 0x3)
    dtype = DType((word >> _DTYPE_SHIFT) & 0x3)
    rank = Rank((word >> _RANK_SHIFT) & 0x3)
    flags = Flag((word >> _FLAG_SHIFT) & 0x7)
    return HashType(kind, length, dtype, rank, flags)


def is_valid_word(word: int) -> bool:
    """Return whether a word satisfies the HashType well-formedness rules."""

    try:
        decoded = unpack(word)
    except (TypeError, ValueError):
        return False

    kind = decoded.kind
    dtype = decoded.dtype
    rank = decoded.rank
    flags = decoded.flags

    if (dtype != DType.NA) != (kind == Kind.NUMPY):
        return False
    if rank != Rank.SCALAR and kind != Kind.NUMPY:
        return False
    if flags & Flag.NUMPY_BYTES:
        if not (
            kind == Kind.NUMPY
            and dtype == DType.NONNUMERIC
            and rank == Rank.SCALAR
        ):
            return False
    if kind == Kind.JSON_NUMBER and not (flags & Flag.NUMERIC_SCALAR):
        return False
    if flags & Flag.NUMERIC_SCALAR:
        if kind not in (Kind.JSON_NUMBER, Kind.JSON_STRING):
            return False
    if flags & Flag.SEMANTIC:
        if kind != Kind.RAW_TEXT:
            return False
    return True


def get_hash_type_cache() -> dict[Checksum, int]:
    """Return the process-local checksum-to-HashType cache."""

    return _hash_type_cache


def set_hash_type(checksum: Checksum | str | bytes, hash_type: HashType | int) -> None:
    """Tighten local knowledge; ignore looser writes and reject contradictions."""

    checksum = Checksum(checksum)
    word = hash_type.word if isinstance(hash_type, HashType) else int(hash_type)
    if not is_valid_word(word):
        raise ValueError(f"Invalid HashType word: {word!r}")
    stored_word = _hash_type_cache.get(checksum)
    if stored_word is not None:
        if stored_word == word:
            return
        stored = unpack(stored_word)
        incoming = unpack(word)
        if _hash_type_implies(stored, incoming):
            return
        if not _hash_type_implies(incoming, stored):
            import logging

            message = (
                f"Conflicting HashType for {checksum.hex()}: "
                f"stored={stored_word}, incoming={word}"
            )
            logging.getLogger(__name__).error(message)
            raise ValueError(message)
    _hash_type_cache[checksum] = word


def _hash_type_implies(tighter: HashType, looser: HashType) -> bool:
    """Whether tighter contains all tested facts in looser.

    Length is always tested. At untested levels all flags are placeholders;
    concrete words must agree on every field, including NUMERIC_SCALAR.
    """
    if tighter.length != looser.length:
        return False
    if looser.kind == Kind.UNTESTED:
        return True
    if looser.kind == Kind.UTF8_UNTESTED:
        return tighter.is_utf8
    if looser.kind == Kind.JSON_UNTESTED:
        return tighter.is_json
    return tighter == looser


def get_hash_type(checksum: Checksum | str | bytes) -> HashType | None:
    """Return the cached HashType for a checksum, if present."""

    word = _hash_type_cache.get(Checksum(checksum))
    if word is None:
        return None
    return HashType.unpack(word)


async def get_hash_type_remote(checksum: Checksum | str | bytes) -> HashType | None:
    """Return HashType from the local cache or configured remote database."""

    checksum = Checksum(checksum)
    hash_type = get_hash_type(checksum)
    if hash_type is not None:
        return hash_type
    try:
        from seamless_remote import database_remote
    except ImportError:
        return None
    word = await database_remote.get_hash_type(checksum)
    if word is None:
        return None
    set_hash_type(checksum, word)
    return HashType.unpack(word)


async def set_hash_type_remote(
    checksum: Checksum | str | bytes, hash_type: HashType | int
) -> bool:
    """Store HashType locally and in configured remote write databases."""

    checksum = Checksum(checksum)
    set_hash_type(checksum, hash_type)
    word = get_hash_type(checksum).word
    try:
        from seamless_remote import database_remote
    except ImportError:
        return False
    return await database_remote.set_hash_type(checksum, word)


def register_hash_type_for_buffer(
    checksum: Checksum | str | bytes,
    buffer: bytes | bytearray | memoryview | Any,
    *,
    value: Any = None,
    celltype: str | None = None,
    semantic: bool = False,
) -> HashType:
    """Compute and cache HashType for a known checksum and buffer."""

    hash_type = from_buffer(
        buffer,
        checksum=checksum,
        value=value,
        celltype=celltype,
        semantic=semantic,
    )
    set_hash_type(checksum, hash_type)
    return hash_type


async def register_hash_type_for_buffer_async(
    checksum: Checksum | str | bytes,
    buffer: bytes | bytearray | memoryview | Any,
    *,
    value: Any = None,
    celltype: str | None = None,
    semantic: bool = False,
) -> HashType:
    """Compute and cache HashType locally and remotely for a known buffer."""

    hash_type = register_hash_type_for_buffer(
        checksum,
        buffer,
        value=value,
        celltype=celltype,
        semantic=semantic,
    )
    await set_hash_type_remote(checksum, hash_type)
    return hash_type


def from_buffer(
    buffer: bytes | bytearray | memoryview | Any,
    *,
    checksum: Checksum | str | bytes | None = None,
    value: Any = None,
    celltype: str | None = None,
    semantic: bool = False,
) -> HashType:
    """Compute a HashType from buffer bytes.

    `mixed` is only emitted for Seamless mixed-format buffers. Pure JSON or raw
    binary buffers are classified by their actual storage format.
    """

    del checksum, value, celltype  # Phase 5 producer is byte-authoritative.
    raw = _as_bytes(buffer)
    length = _length_bucket(len(raw))
    if raw.startswith(_magic_numpy()):
        dtype, rank, flags = _numpy_type(raw)
        return HashType(Kind.NUMPY, length, dtype, rank, flags)
    if raw.startswith(_magic_seamless_mixed()):
        return HashType(_mixed_kind(raw), length)

    try:
        text = raw.decode()
    except UnicodeDecodeError:
        return HashType(Kind.RAW_BYTES, length)

    kind, flags = _json_kind_and_flags(text)
    if kind is None:
        flags = Flag.SEMANTIC if semantic else Flag(0)
        return HashType(Kind.RAW_TEXT, length, flags=flags)
    return HashType(kind, length, flags=flags)


def deserializable_as(
    hash_type: HashType | int,
    celltype: str,
    *,
    checksum: Checksum | str | bytes | None = None,
) -> bool | None:
    """Return True (known), False (disproved), or None (requires parsing)."""

    ti = _coerce(hash_type)
    kind = ti.kind
    checksum_obj = None if checksum is None else Checksum(checksum)
    if checksum_obj is not None and checksum_obj in (CHECKSUM_NULL, _CHECKSUM_NULL_NL):
        return True
    if celltype == "bytes":
        return True
    if celltype == "bool":
        return checksum_obj in _BOOL_CHECKSUMS
    if ti.is_untested:
        if celltype in ("text", "yaml", "ipython", "python"):
            return True if ti.is_utf8 else None
        if celltype in ("plain", "mixed"):
            return True if ti.is_json else None
        if celltype == "str":
            return True if checksum_obj in _SCALAR_CONST_CHECKSUMS else None
        if celltype in ("int", "float"):
            return False if ti.length == Length.LONG else None
        if celltype == "binary":
            return None if kind == Kind.UNTESTED else False
        if celltype == "checksum":
            if ti.length != Length.EQ64 or kind == Kind.JSON_UNTESTED:
                return False
            return None
        return False
    if celltype in ("text", "yaml", "ipython", "python"):
        return ti.is_utf8
    if celltype == "plain":
        return ti.is_json
    if celltype == "str":
        return kind in (Kind.JSON_STRING, Kind.JSON_NUMBER) or (
            checksum_obj in _SCALAR_CONST_CHECKSUMS
        )
    if celltype in ("int", "float"):
        if ti.length == Length.LONG:
            return False
        return bool(ti.flags & Flag.NUMERIC_SCALAR)
    if celltype == "binary":
        return kind == Kind.NUMPY
    if celltype == "mixed":
        return kind not in (Kind.RAW_BYTES, Kind.RAW_TEXT)
    if celltype == "checksum":
        # A digest of only decimal digits is also a JSON number.
        return kind in (Kind.RAW_TEXT, Kind.JSON_NUMBER) and ti.length == Length.EQ64
    return False


def capabilities(hash_type: HashType | int, source_celltype: str) -> set[str]:
    """Return expression capabilities relative to a source celltype."""

    ti = _coerce(hash_type)
    if source_celltype == "bytes":
        return {"SEQ"}
    if source_celltype in FLAT_SEQ_CELLTYPES:
        return {"SEQ"}
    if source_celltype == "binary":
        caps = {"SEQ"} if ti.rank != Rank.SCALAR else set()
        if ti.dtype == DType.STRUCTURED:
            caps.add("MAP")
        return caps
    if source_celltype in ("plain", "mixed"):
        if ti.kind in (Kind.JSON_OBJECT, Kind.MIXED_OBJECT):
            return {"MAP"}
        if ti.kind in (Kind.JSON_ARRAY, Kind.MIXED_ARRAY):
            return {"SEQ"}
        if ti.kind == Kind.JSON_STRING:
            return {"SEQ"}
    return set()


def has_numeric_items(hash_type: HashType | int, source_celltype: str) -> bool | None:
    ti = _coerce(hash_type)
    if source_celltype == "bytes":
        return True
    if source_celltype == "binary":
        return ti.dtype == DType.NUMERIC and ti.rank != Rank.SCALAR
    if source_celltype in ("plain", "mixed") and ti.kind in (
        Kind.JSON_ARRAY,
        Kind.MIXED_ARRAY,
    ):
        return None
    return False


def has_string_items(hash_type: HashType | int, source_celltype: str) -> bool | None:
    ti = _coerce(hash_type)
    if source_celltype in FLAT_SEQ_CELLTYPES:
        return True
    if source_celltype in ("plain", "mixed"):
        if ti.kind == Kind.JSON_STRING:
            return True
        if ti.kind in (Kind.JSON_ARRAY, Kind.MIXED_ARRAY):
            return None
    if source_celltype == "binary":
        return None if ti.dtype == DType.NONNUMERIC else False
    return False


def _as_bytes(buffer: bytes | bytearray | memoryview | Any) -> bytes:
    if hasattr(buffer, "content"):
        buffer = buffer.content
    if isinstance(buffer, memoryview):
        return buffer.tobytes()
    if isinstance(buffer, bytearray):
        return bytes(buffer)
    if not isinstance(buffer, bytes):
        raise TypeError(type(buffer))
    return buffer


def _length_bucket(length: int) -> Length:
    if length < 64:
        return Length.SHORT
    if length == 64:
        return Length.EQ64
    if length <= 1000:
        return Length.MEDIUM
    return Length.LONG


def _magic_numpy() -> bytes:
    return MAGIC_NUMPY


def _magic_seamless_mixed() -> bytes:
    return MAGIC_SEAMLESS_MIXED


def _numpy_type(raw: bytes) -> tuple[DType, Rank, Flag]:
    import numpy as np

    array = np.load(io.BytesIO(raw), allow_pickle=False)
    dtype = array.dtype
    if dtype.fields is not None:
        hash_dtype = DType.STRUCTURED
    elif dtype.kind in "biufc":
        hash_dtype = DType.NUMERIC
    else:
        hash_dtype = DType.NONNUMERIC
    if array.ndim == 0:
        rank = Rank.SCALAR
    elif array.ndim == 1:
        rank = Rank.D1
    elif array.ndim == 2:
        rank = Rank.D2
    else:
        rank = Rank.D3PLUS
    flags = (
        Flag.NUMPY_BYTES
        if hash_dtype == DType.NONNUMERIC
        and rank == Rank.SCALAR
        and dtype.kind == "S"
        else Flag(0)
    )
    return hash_dtype, rank, flags


def _mixed_kind(raw: bytes) -> Kind:
    offset = len(_magic_seamless_mixed())
    storage_len = raw[offset]
    offset += 1 + storage_len
    form_len = int.from_bytes(raw[offset : offset + 4], "little")
    offset += 4
    form = json.loads(raw[offset : offset + form_len].decode())
    top_type = form.get("type")
    if top_type == "object":
        return Kind.MIXED_OBJECT
    if top_type == "array":
        return Kind.MIXED_ARRAY
    raise ValueError(f"Unexpected mixed root type: {top_type!r}")


def _json_kind_and_flags(text: str) -> tuple[Kind | None, Flag]:
    try:
        value = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None, Flag(0)
    if isinstance(value, dict):
        return Kind.JSON_OBJECT, Flag(0)
    if isinstance(value, list):
        return Kind.JSON_ARRAY, Flag(0)
    if isinstance(value, str):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            flags = Flag(0)
        else:
            flags = (
                Flag.NUMERIC_SCALAR
                if math.isfinite(numeric_value)
                else Flag(0)
            )
        return Kind.JSON_STRING, flags
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Kind.JSON_NUMBER, Flag.NUMERIC_SCALAR
    if value is True or value is False or value is None:
        return Kind.JSON_STRING, Flag(0)
    return None, Flag(0)


def _coerce(hash_type: HashType | int) -> HashType:
    return hash_type if isinstance(hash_type, HashType) else HashType.unpack(hash_type)


__all__ = [
    "DType",
    "Flag",
    "HashType",
    "Kind",
    "Length",
    "MIC_BY_KIND",
    "Rank",
    "capabilities",
    "deserializable_as",
    "from_buffer",
    "get_hash_type",
    "get_hash_type_cache",
    "get_hash_type_remote",
    "has_numeric_items",
    "has_string_items",
    "is_valid_word",
    "pack",
    "register_hash_type_for_buffer",
    "register_hash_type_for_buffer_async",
    "set_hash_type",
    "set_hash_type_remote",
    "unpack",
]
