"""Functions to parse a buffer into a value"""

import builtins
import logging
from copy import deepcopy
import json
import ast
import orjson
import yaml
from seamless import Buffer, Checksum
from seamless.util.get_event_loop import get_event_loop

from seamless.util.ipython import ipython2python
from ..util import lrucache2
from .celltypes import celltypes, text_types2

from .serialize import serialize_cache

logger = logging.getLogger(__name__)

parse_buffer_cache = lrucache2(10)


def _parse_buffer_plain(buffer):
    """"""
    """
    value, storage = mixed_parse_buffer(buffer)
    if storage != "pure-plain":
        raise TypeError
    """
    s = buffer.decode()
    s = s.rstrip("\n")
    try:
        value = orjson.loads(s)
    except json.JSONDecodeError:
        msg = s
        if len(msg) > 1000:
            msg = s[:920] + "..." + s[-50:]
        raise ValueError(msg) from None
    return value


def validate_text(text: str, celltype: str, code_filename):
    """Validate that 'text' is a valid value of 'celltype'.
    A 'code_filename' can be provided for code buffers, to mark them with a
    temporary source code filename.
    """
    try:
        if text is None:
            return
        if celltype == "python":
            ast.parse(text, filename=code_filename)
        elif celltype == "ipython":
            ipython2python(text)
        elif celltype == "yaml":
            yaml.safe_load(text)
    except Exception:
        msg = text
        if len(text) > 1000:
            msg = text[:920] + "..." + text[-50:]
        raise ValueError(msg) from None


text_validation_celltype_cache = set()


def validate_text_celltype(text, checksum: Checksum, celltype: str):
    """Validate that 'text' is a valid value of 'celltype'.
    The checksum is provided for caching purposes."""
    assert celltype in text_types2
    checksum = Checksum(checksum)
    if checksum:
        if (checksum, celltype) in text_validation_celltype_cache:
            return
    validate_text(text, celltype, "value_conversion")
    if checksum:
        text_validation_celltype_cache.add((checksum, celltype))


def _parse_buffer(buffer: Buffer, checksum: Checksum, celltype: str):
    from seamless.checksum_class import validate_checksum
    from .hash_type_validation import validate_deserializable_as

    if celltype not in celltypes:
        raise TypeError(celltype)
    checksum = Checksum(checksum)
    from .null import is_null
    if is_null(checksum):
        return b"" if celltype == "bytes" else None
    logger.debug(
        "DESERIALIZE: buffer of length {}, checksum {}".format(len(buffer), checksum)
    )
    validate_deserializable_as(checksum, celltype, buffer=buffer)
    if celltype in text_types2:
        s = buffer.decode()
        value = s.rstrip("\n")
        if checksum and validate_text_celltype is not None:
            validate_text_celltype(value, checksum, celltype)
    elif celltype == "plain":
        value = _parse_buffer_plain(buffer.content)
    elif celltype == "binary":
        from ..util.mixed.io import deserialize as mixed_deserialize

        value, storage = mixed_deserialize(buffer.content)
        if storage != "pure-binary":
            raise TypeError
    elif celltype == "mixed":
        from ..util.mixed.io import deserialize as mixed_deserialize

        value, _ = mixed_deserialize(buffer.content)
    elif celltype == "bytes":
        value = buffer
    elif celltype in ("str", "int", "float", "bool"):
        value = _parse_buffer_plain(buffer)
        if not isinstance(value, getattr(builtins, celltype)):
            value = getattr(builtins, celltype)(value)
    elif celltype == "checksum":
        try:
            value = buffer.decode()
            if validate_checksum is not None:
                validate_checksum(value)
        except (ValueError, UnicodeDecodeError):
            from ..util.mixed.io import deserialize as mixed_deserialize

            value, storage = mixed_deserialize(buffer)
            if storage != "pure-plain":
                raise TypeError from None
            if validate_checksum is not None:
                validate_checksum(value)
    else:
        raise NotImplementedError(celltype)

    return value


async def parse_buffer(buffer: Buffer, checksum: Checksum, celltype: str, copy: bool):
    """Deserializes a buffer into a value
    The celltype must be one of the allowed celltypes.

    First, it is attempted to retrieve the value from cache.
    In case of a cache hit, a copy is returned only if copy=True
    In case of a cache miss, deserialization is performed in a subprocess
     (and copy is irrelevant).
    """
    from seamless.diagnostics import record
    record("deserialize", checksum, celltype, len(buffer))
    if buffer is None:
        return None
    value = parse_buffer_cache.get((checksum, celltype))
    if value is not None and not copy:
        return value

    """
    # TODO:
    # ProcessPool is too slow, but ThreadPool works... do experiment with later
    loop = get_event_loop()
    with ThreadPoolExecutor() as executor:
        value = await loop.run_in_executor(
            executor, _parse_buffer, buffer, checksum, celltype
        )
    """
    value = _parse_buffer(buffer, checksum, celltype)

    if celltype not in text_types2 and not copy:
        parse_buffer_cache[checksum, celltype] = value

    if not copy:
        id_value = id(value)
        serialize_cache[id_value, celltype] = buffer, value
    return value


def parse_buffer_sync(buffer: Buffer, checksum: Checksum, celltype: str, copy):
    """Deserializes a buffer into a value
    The celltype must be one of the allowed celltypes.

    First, it is attempted to retrieve the value from cache.
    In case of a cache hit, a copy is returned only if copy=True
    In case of a cache miss, deserialization is performed
    (and copy is irrelevant).

    This function can be executed if the asyncio event loop is already running"""
    from seamless.diagnostics import record
    record("deserialize", checksum, celltype, len(buffer))

    value = None
    if checksum:
        value = parse_buffer_cache.get((checksum, celltype))
    if value is not None:
        if copy:
            newvalue = deepcopy(value)
            return newvalue
        else:
            return value

    value = _parse_buffer(buffer, checksum, celltype)
    if celltype not in text_types2 and not copy:
        parse_buffer_cache[checksum, celltype] = value
    if not copy:
        id_value = id(value)
        serialize_cache[id_value, celltype] = buffer, value
    return value
