"""Shared helpers for Seamless mixed-form data."""

import numpy as np

MAGIC_NUMPY = b"\x93NUMPY"
MAGIC_SEAMLESS = b"\x93SEAMLESS"
MAGIC_SEAMLESS_MIXED = b"\x94SEAMLESS-MIXED"

_integer_types = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_unsigned_types = (np.uint8, np.uint16, np.uint32, np.uint64)
_float_types = (float, np.float16, np.float32, np.float64, np.float128)
_array_types = (list, tuple, np.ndarray)
_string_types = (str, bytes)

Scalar = (type(None), bool, str, bytes) + _integer_types + _float_types
_allowed_types = Scalar + _array_types + (np.void, dict, complex, np.complexfloating)

__all__ = [
    "MAGIC_NUMPY",
    "MAGIC_SEAMLESS",
    "MAGIC_SEAMLESS_MIXED",
    "Scalar",
    "_array_types",
    "_integer_types",
    "_float_types",
    "_string_types",
    "_unsigned_types",
    "_allowed_types",
]
