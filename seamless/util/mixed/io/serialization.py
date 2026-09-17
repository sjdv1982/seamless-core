import json
import numpy as np
from .to_stream import to_stream
from .from_stream import from_stream
from .. import MAGIC_SEAMLESS_MIXED


def _as_aligned_struct_dtype(dt):
    """Rebuild a structured dtype as aligned, dropping anonymous padding fields."""

    if dt.fields is None:
        return dt

    fields = []
    for name in dt.names:
        field_dtype = dt.fields[name][0]
        subdtype = field_dtype.subdtype
        if subdtype is None:
            fields.append((name, _as_aligned_struct_dtype(field_dtype)))
        else:
            base_dtype, shape = subdtype
            fields.append((name, _as_aligned_struct_dtype(base_dtype), shape))
    return np.dtype(fields, align=True)


def serialize(data, *, storage=None, form=None):
    from ..get_form import get_form

    if storage is None or form is None:
        storage, form = get_form(data)
    content = to_stream(data, storage, form)
    if storage in ("pure-plain", "pure-binary"):
        return content
    result = MAGIC_SEAMLESS_MIXED
    h1 = storage.encode()
    result += np.uint8(len(h1)).tobytes()
    result += h1
    h2 = json.dumps(form).encode()
    result += np.uint32(len(h2)).tobytes()
    result += h2
    result += content
    return result


def deserialize(data):
    from .. import MAGIC_SEAMLESS, MAGIC_NUMPY
    from ..get_form import get_form, dt_builtins, is_np_str

    pure_plain, pure_binary = False, False
    if isinstance(data, str):
        pure_plain = True
    else:
        if not isinstance(data, bytes):
            raise TypeError(type(data))
        if data.startswith(MAGIC_NUMPY):
            pure_binary = True
        elif not data.startswith(MAGIC_SEAMLESS_MIXED):
            pure_plain = True
    if pure_plain or pure_binary:
        mode = "pure-plain" if pure_plain else "pure-binary"
        if pure_plain and not isinstance(data, str):
            assert not data.startswith(MAGIC_SEAMLESS)
            buffer = data
            data = buffer.decode()
            try:
                value = from_stream(data, mode, None)
            except:
                value = np.array(buffer)
                mode = "pure-binary"
        else:
            value = from_stream(data, mode, None)
        if mode == "pure-binary":
            dt = value.dtype
            if dt.base.isbuiltin or is_np_str(dt) or dt in dt_builtins:
                pass
            elif not dt.isalignedstruct:
                dt2 = _as_aligned_struct_dtype(dt)
                value = value.astype(dt2)
        return value, mode

    offset = len(MAGIC_SEAMLESS_MIXED)
    lh1 = np.frombuffer(data[offset : offset + 1], np.uint8)[0]
    offset += 1
    h1 = data[offset : offset + lh1]
    offset += lh1
    lh2 = np.frombuffer(data[offset : offset + 4], np.uint32)[0]
    offset += 4
    h2 = data[offset : offset + lh2]
    offset += lh2
    storage = h1.decode()
    form = json.loads(h2.decode())
    return from_stream(data[offset:], storage, form), storage
