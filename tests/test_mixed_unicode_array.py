"""Top-level numpy unicode arrays serialize as .npy, not as their first element."""

import numpy as np
import pytest

from seamless import Buffer, Expression
from seamless.checksum.hash_type import DType, HashType, Kind
from seamless.util.mixed import MAGIC_NUMPY
from seamless.util.mixed.io import deserialize, serialize

ARRAYS = [
    np.array(["ab", "cd"]),
    np.array([["a", "bc"], ["def", ""]]),
    np.array("abc"),
]


@pytest.mark.parametrize("array", ARRAYS, ids=lambda a: f"U-ndim{a.ndim}")
def test_unicode_array_roundtrip(array):
    restored, storage = deserialize(serialize(array))
    assert storage == "pure-binary"
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array)


@pytest.mark.parametrize("celltype", ["binary", "mixed"])
@pytest.mark.parametrize("array", ARRAYS, ids=lambda a: f"U-ndim{a.ndim}")
def test_buffer_unicode_array_is_numpy(array, celltype):
    buffer = Buffer(array, celltype)
    assert buffer.content.startswith(MAGIC_NUMPY)
    hash_type = HashType.from_buffer(buffer.content)
    assert (hash_type.kind, hash_type.dtype) == (Kind.NUMPY, DType.NONNUMERIC)
    np.testing.assert_array_equal(buffer.get_value(celltype), array)


@pytest.mark.parametrize(
    "path,target_celltype,expected",
    [("[1]", "str", "cd"), ("[1][0]", "str", "c"), ("[0:1]", "binary", np.array(["ab"]))],
)
def test_unicode_array_expression(path, target_celltype, expected):
    buffer = Buffer(np.array(["ab", "cd"]), "binary")
    buffer.tempref()
    expression = Expression(
        buffer.get_checksum(),
        path=path,
        input_celltype="binary",
        target_celltype=target_celltype,
    )
    np.testing.assert_array_equal(expression.run(), expected)
