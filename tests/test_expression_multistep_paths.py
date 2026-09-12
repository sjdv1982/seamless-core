"""Multi-step expression paths: HashType types the root, not the children.

Static validation rejects a step only where the root's bits prove it
impossible (type_bits_design §7.2); every other step is left to evaluation.
"""

import numpy as np
import pytest

from seamless import Buffer, Expression
from seamless.checksum.hash_type_validation import HashTypeValidationError


def _expression(value, celltype, path, target_celltype=None):
    buffer = Buffer(value, celltype)
    buffer.tempref()
    return Expression(
        buffer.get_checksum(),
        path=path,
        input_celltype=celltype,
        target_celltype=target_celltype or celltype,
    )


D2 = np.arange(6.0).reshape(2, 3)

VALID = [
    ({"a": 1, "b": [5, 6]}, "plain", "b[1]", None, 6),
    ([{"x": 3}], "plain", "[0].x", None, 3),
    ([[1, 2], [3, 4]], "mixed", "[1][0]", None, 3),
    ({"a": {"b": {"c": 7}}}, "mixed", "a.b.c", None, 7),
    ([7, 8, 9], "plain", "[1:][0]", None, 8),
    ("abc", "text", "[1][0]", None, "b"),
    ("abcdef", "text", "[1:5][::2]", None, "bd"),
    (b"abc", "bytes", "[1:][0]", "int", 98),
    (D2, "binary", "[1][2]", None, 5.0),
    (D2, "binary", "[0:1][0][1]", None, 1.0),
]

INVALID = [
    ([7, 8], "plain", "[0:1].x"),  # a slice of a list is still a list
    (b"ab", "bytes", "[0][0]"),  # a bytes item is an int
    (np.arange(3.0), "binary", "[0][0]"),  # rank 1 admits one positional item
    (D2, "binary", "[1][2][0]"),  # rank 2 admits two
    (np.arange(3.0), "binary", "[0][0:1]"),  # ... and nothing after them
]


@pytest.mark.parametrize(
    "value,celltype,path,target_celltype,expected",
    VALID,
    ids=[f"{case[1]}:{case[2]}" for case in VALID],
)
def test_valid_multistep_path_evaluates(value, celltype, path, target_celltype, expected):
    assert _expression(value, celltype, path, target_celltype).run() == expected


@pytest.mark.parametrize(
    "value,celltype,path", INVALID, ids=[f"{case[1]}:{case[2]}" for case in INVALID]
)
def test_invalid_multistep_path_is_rejected_statically(value, celltype, path):
    with pytest.raises(HashTypeValidationError):
        _expression(value, celltype, path).compute()
