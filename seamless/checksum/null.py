"""Canonical null is a value, distinct from the absence of a checksum."""
from hashlib import sha256

from .virtual import NULL_CHECKSUMS

NULL_BUFFER = b"null\n"
NULL_CHECKSUM = sha256(NULL_BUFFER).hexdigest()


def is_null(checksum):
    if isinstance(checksum, bytes):
        return checksum in (bytes.fromhex(NULL_CHECKSUM), NULL_CHECKSUM.encode())
    if checksum is None:
        return False
    from seamless.checksum_class import Checksum

    try:
        return Checksum(checksum).hex() == NULL_CHECKSUM
    except (TypeError, ValueError):
        return False


def is_null_value(checksum):
    """Return whether ``checksum`` represents either virtual null buffer."""

    if checksum is None:
        return False
    try:
        if isinstance(checksum, bytes):
            if len(checksum) == 32:
                checksum_hex = checksum.hex()
            elif len(checksum) == 64:
                checksum_hex = checksum.decode("ascii")
            else:
                return False
        else:
            from seamless.checksum_class import Checksum

            checksum_hex = Checksum(checksum).hex()
    except (TypeError, UnicodeDecodeError, ValueError):
        return False
    return checksum_hex in NULL_CHECKSUMS


def canonicalize_checksum(checksum, celltype):
    """Treat the legacy empty bytes buffer as the canonical null value."""
    if checksum is not None and celltype == "bytes":
        from seamless.checksum_class import Checksum
        if Checksum(checksum).hex() == sha256(b"").hexdigest():
            return Checksum(NULL_CHECKSUM)
    return checksum
