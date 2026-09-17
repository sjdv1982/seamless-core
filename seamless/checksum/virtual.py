"""Checksum-only values for canonical JSON scalar buffers.

Some values have a small, well-known representation whose value can be
determined from the checksum alone.  Keeping these values virtual avoids a
cache or remote-buffer lookup when resolving a checksum.
"""

from __future__ import annotations

from hashlib import sha256


def _checksums(*buffers: bytes) -> frozenset[str]:
    return frozenset(sha256(buffer).hexdigest() for buffer in buffers)


NULL_CHECKSUMS = _checksums(b"null", b"null\n")
TRUE_CHECKSUMS = _checksums(b"true", b"true\n")
FALSE_CHECKSUMS = _checksums(b"false", b"false\n")


# A sentinel is needed because ``None`` is itself a virtual value.
NOT_VIRTUAL = object()


def _checksum_hex(checksum) -> str:
    """Normalize a checksum-like argument to its hexadecimal digest."""

    # Import lazily: checksum_class imports the public ``seamless`` package,
    # and virtual values are also used from checksum_class itself.
    from seamless.checksum_class import Checksum

    # ``Checksum`` normally receives raw 32-byte digests.  Accept the ASCII
    # representation as well, matching the historical null helper's API.
    if isinstance(checksum, bytes) and len(checksum) == 64:
        checksum = checksum.decode("ascii")
    return Checksum(checksum).hex()


def virtual_value(checksum, celltype: str):
    """Return a value encoded by ``checksum``, or :data:`NOT_VIRTUAL`.

    Null and boolean checksums are interpreted without reading their buffers.
    A boolean request is deliberately strict: only the four canonical boolean
    checksums are valid, and a non-boolean checksum raises ``ValueError``.
    """

    checksum_hex = _checksum_hex(checksum)

    if checksum_hex in NULL_CHECKSUMS:
        return b"" if celltype == "bytes" else None

    if checksum_hex in TRUE_CHECKSUMS:
        value = True
    elif checksum_hex in FALSE_CHECKSUMS:
        value = False
    else:
        if celltype == "bool":
            raise ValueError("checksum does not encode a boolean value")
        return NOT_VIRTUAL

    if celltype in ("bool", "plain", "mixed"):
        return value
    if celltype == "str":
        return str(value)
    if celltype in ("int", "float"):
        raise ValueError("boolean checksum cannot be interpreted as a number")
    return NOT_VIRTUAL


__all__ = [
    "NULL_CHECKSUMS",
    "TRUE_CHECKSUMS",
    "FALSE_CHECKSUMS",
    "NOT_VIRTUAL",
    "virtual_value",
]
