"""Functions to serialize a value into a buffer"""

import logging

from ..util import lrucache2
from ..util import unchecksum
from .celltypes import celltypes, text_types

# serialize_cache: maps id(value),celltype to (buffer, value).
# Need to store (a ref to) value,
#  because id(value) is only unique while value does not die!!!
serialize_cache = lrucache2(10)

logger = logging.getLogger(__name__)


def _normalize_mixed_special_values(value):
    """Move values without a JSON representation into NumPy storage."""
    import math
    import numpy as np

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return np.float64(value)
    if isinstance(value, list):
        return [_normalize_mixed_special_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_mixed_special_values(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _normalize_mixed_special_values(item) for key, item in value.items()
        }
    return value


def _serialize(value, celltype: str):
    from seamless import Checksum
    from .json_ import json_dumps_bytes

    if celltype not in celltypes:
        if value is None and celltype in {"deepcell", "deepfolder", "folder", "module"}:
            from .null import NULL_BUFFER

            return NULL_BUFFER
        raise TypeError(celltype)
    if value is None:
        from .null import NULL_BUFFER
        return NULL_BUFFER
    if isinstance(value, Checksum):
        value = value.hex()
    if celltype == "str":
        if not isinstance(value, bool):
            value = str(value)
        buffer = json_dumps_bytes(value) + b"\n"
    elif celltype in text_types:
        if isinstance(value, bytes):
            value = value.decode()
        if celltype in ("int", "float", "bool"):
            if celltype == "int":
                value = int(value)
            elif celltype == "float":
                value = float(value)
            elif celltype == "bool":
                value = bool(value)
            if celltype == "float":
                import math

                if not math.isfinite(value):
                    raise ValueError("cannot serialize a non-finite float")
            buffer = json_dumps_bytes(value) + b"\n"
        else:
            buffer = (str(value).rstrip("\n") + "\n").encode()
    elif celltype == "plain":
        import math
        import numpy as np

        def check_finite(item):
            if isinstance(item, (float, np.floating)) and not math.isfinite(item):
                raise ValueError("cannot serialize a non-finite float as plain")
            if isinstance(item, dict):
                for child in item.values():
                    check_finite(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    check_finite(child)
            elif isinstance(item, np.ndarray) and item.dtype.kind in "fc":
                if not np.isfinite(item).all():
                    raise ValueError("cannot serialize a non-finite array as plain")

        value = unchecksum(value)
        check_finite(value)
        buffer = json_dumps_bytes(value) + b"\n"
    elif celltype == "bytes":
        buffer = None
        if isinstance(value, bytes):
            buffer = value
        elif buffer is None:
            try:
                buffer = value.tobytes()  # type: ignore
            except Exception:
                pass
        if buffer is None:
            buffer = (str(value).rstrip("\n")).encode()
    elif celltype == "checksum":
        # The bare 64-character hex digest, without a newline.
        if not isinstance(value, str):
            raise TypeError(
                f"A checksum value must be a Checksum or a hex string, not {type(value).__name__}"
            )
        buffer = Checksum(value).hex().encode()
    else:
        if celltype == "mixed":
            from ..util.mixed.io import serialize as mixed_serialize
            import math
            import numpy as np

            value = unchecksum(value)
            value = _normalize_mixed_special_values(value)
            if isinstance(value, (complex, np.complexfloating)):
                value = np.asarray(value)
                buffer = mixed_serialize(value, storage="pure-binary", form={})
            elif isinstance(value, np.floating) and not math.isfinite(value):
                # JSON has no representation for these values. Store them as a
                # zero-dimensional float64 array, consistently with NumPy data.
                value = np.float64(value)
                buffer = mixed_serialize(value)
            else:
                buffer = mixed_serialize(value)
        elif celltype == "binary":
            if isinstance(value, bytes):
                buffer = value
            else:
                import numpy as np  # delayed import, since it takes ~1 sec in user time
                from ..util.mixed.io import serialize as mixed_serialize

                value = np.array(value)
                buffer = mixed_serialize(value, storage="pure-binary", form={})
        else:
            raise TypeError(celltype)
    if celltype == "bytes" and buffer == b"":
        from .null import NULL_BUFFER
        buffer = NULL_BUFFER
    logger.debug("SERIALIZE: buffer of length {}".format(len(buffer)))
    return buffer


async def serialize(value, celltype: str, use_cache=True) -> bytes:
    """Serializes a value into a buffer
    The celltype must be one of the allowed celltypes.
    """
    from seamless.diagnostics import record
    record("serialize", celltype=celltype)

    assert value is not None
    if use_cache:
        id_value = id(value)
        buffer, _ = serialize_cache.get((id_value, celltype), (None, None))  # type: ignore
        if buffer is not None:
            return buffer

    """
    # what can we do to make this async??
    # ThreadPool doesn't work, and ProcessPool is slow
    # (seems to make a copy of the Python structure)

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        buffer = await loop.run_in_executor(
            executor,
            _serialize,
            value, celltype
        )
    return buffer
    """

    buffer = _serialize(value, celltype)  ### for now...
    if use_cache:
        serialize_cache[id_value, celltype] = buffer, value
    return buffer


def serialize_sync(value, celltype: str, use_cache: bool = True) -> bytes:
    """Serializes a value into a buffer
    The celltype must be one of the allowed celltypes.
    This function can be executed if the asyncio event loop is already running"""
    from seamless.diagnostics import record
    record("serialize", celltype=celltype)
    if use_cache:
        id_value = id(value)
        buffer, _ = serialize_cache.get((id_value, celltype), (None, None))  # type: ignore
        if buffer is not None:
            return buffer

    buffer = _serialize(value, celltype)  ### for now...
    if use_cache:
        serialize_cache[id_value, celltype] = buffer, value
    return buffer
