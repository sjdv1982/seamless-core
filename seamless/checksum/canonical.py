"""Canonical file bytes, using the value-assignment serialization contract."""
from .serialize import serialize_sync

FILE_CELLTYPES = frozenset(('text', 'python', 'ipython', 'yaml', 'plain', 'str',
                           'int', 'float', 'bool', 'bytes', 'binary', 'mixed'))
DIRECTORY_CELLTYPES = frozenset(('folder', 'deepfolder'))


def canon_T(content: bytes, celltype: str) -> bytes:
    if celltype not in FILE_CELLTYPES:
        raise TypeError(f'Celltype {celltype!r} cannot be mounted as a file')
    from .null import NULL_BUFFER
    if content in (b'', NULL_BUFFER):
        return NULL_BUFFER
    if celltype == 'bytes':
        return bytes(content)
    # Code assignment serializes text without executing or syntax-checking it.
    if celltype in {'text', 'python', 'ipython', 'yaml'}:
        value = content.decode('utf-8')
    else:
        from seamless import Buffer, Checksum
        from .parse_buffer import parse_buffer_sync
        buffer = Buffer(content)
        value = parse_buffer_sync(buffer, buffer.get_checksum(), celltype, copy=True)
    return serialize_sync(value, celltype, use_cache=False)
