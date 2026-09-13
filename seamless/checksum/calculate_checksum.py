"""Calculate SHA-256 checksums"""

from hashlib import sha256


def calculate_checksum(content: bytes) -> str:  # pylint: disable=redefined-builtin
    """Calculate a SHA-256 checksum.
    Return it as hexidecimal string"""
    if isinstance(content, str):
        content = content.encode()
    if not isinstance(content, bytes):
        raise TypeError(type(content))
    hasher = sha256(content)
    result = hasher.digest()
    return result.hex()


def calculate_file_checksum(filename: str) -> str | None:
    """Calculate a file checksum"""
    if filename in ("/dev/stdout", "/dev/stderr"):
        return None
    blocksize = 2**16
    with open(filename, "rb") as f:
        hasher = sha256()
        while 1:
            block = f.read(blocksize)
            if not block:
                break
            hasher.update(block)
    checksum = hasher.digest().hex()
    return checksum


def calculate_dict_checksum(d: dict) -> str:  # pylint: disable=redefined-builtin
    """This function is compatible with the checksum of a "plain" cell"""
    from .json_ import json_dumps_bytes

    content = json_dumps_bytes(d) + b"\n"
    return calculate_checksum(content)


TRIVIAL_CHECKSUMS = {
    calculate_checksum(b): b for b in (b"null\n", b"", b"{}", b"{}\n", b"[]", b"[]\n")
}
