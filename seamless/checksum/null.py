"""Canonical null is a value, distinct from the absence of a checksum."""
from hashlib import sha256

NULL_BUFFER = b"null\n"
NULL_CHECKSUM = sha256(NULL_BUFFER).hexdigest()


def is_null(checksum):
    if isinstance(checksum, bytes):
        return checksum in (bytes.fromhex(NULL_CHECKSUM), NULL_CHECKSUM.encode())
    return checksum is not None and str(checksum) == NULL_CHECKSUM
