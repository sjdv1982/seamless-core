import pytest

from seamless.checksum.hash_type import DType, Flag, HashType, Kind, Length, Rank
from seamless.checksum.hash_type_validation import conversion_feasible


CHECKSUM = "a" * 64

NUMPY_NUMERIC_VECTOR = HashType(
    Kind.NUMPY, Length.SHORT, DType.NUMERIC, Rank.D1
)
NUMPY_NUMERIC_SCALAR = HashType(
    Kind.NUMPY, Length.SHORT, DType.NUMERIC, Rank.SCALAR
)
NUMPY_NONNUMERIC_SCALAR = HashType(
    Kind.NUMPY, Length.SHORT, DType.NONNUMERIC, Rank.SCALAR
)
JSON_NUMBER = HashType(
    Kind.JSON_NUMBER, Length.SHORT, flags=Flag.NUMERIC_SCALAR
)
LONG_JSON_NUMBER = HashType(
    Kind.JSON_NUMBER, Length.LONG, flags=Flag.NUMERIC_SCALAR
)
NUMERIC_JSON_STRING = HashType(
    Kind.JSON_STRING, Length.SHORT, flags=Flag.NUMERIC_SCALAR
)
LONG_NUMERIC_JSON_STRING = HashType(
    Kind.JSON_STRING, Length.LONG, flags=Flag.NUMERIC_SCALAR
)
UNFLAGGED_JSON_STRING = HashType(Kind.JSON_STRING, Length.SHORT)
LONG_UNFLAGGED_JSON_STRING = HashType(Kind.JSON_STRING, Length.LONG)


@pytest.mark.parametrize(
    "source,target,hash_type,expected",
    [
        ("binary", "int", NUMPY_NUMERIC_VECTOR, False),
        ("mixed", "float", NUMPY_NUMERIC_VECTOR, False),
        ("binary", "float", NUMPY_NUMERIC_SCALAR, True),
        ("mixed", "int", NUMPY_NUMERIC_SCALAR, True),
        ("binary", "int", NUMPY_NONNUMERIC_SCALAR, None),
        ("mixed", "float", NUMPY_NONNUMERIC_SCALAR, None),
        ("mixed", "int", HashType(Kind.JSON_OBJECT, Length.SHORT), False),
        ("mixed", "float", HashType(Kind.JSON_ARRAY, Length.SHORT), False),
        ("mixed", "int", HashType(Kind.MIXED_OBJECT, Length.SHORT), False),
        ("mixed", "float", HashType(Kind.MIXED_ARRAY, Length.SHORT), False),
        ("mixed", "int", JSON_NUMBER, True),
        ("mixed", "float", LONG_JSON_NUMBER, True),
        ("mixed", "int", NUMERIC_JSON_STRING, True),
        ("mixed", "float", LONG_NUMERIC_JSON_STRING, True),
        ("mixed", "float", UNFLAGGED_JSON_STRING, None),
        ("mixed", "int", LONG_UNFLAGGED_JSON_STRING, None),
        ("mixed", "int", HashType(Kind.UNTESTED, Length.LONG), None),
        ("mixed", "float", HashType(Kind.JSON_UNTESTED, Length.LONG), None),
    ],
)
def test_conversion_possible_int_float_rules_follow_hash_type_evidence(
    source, target, hash_type, expected
):
    assert conversion_feasible(
        hash_type, source, target, checksum=CHECKSUM
    ) is expected


def test_long_still_disproves_numeric_reinterpretation():
    assert conversion_feasible(
        LONG_JSON_NUMBER, "plain", "float", checksum=CHECKSUM
    ) is False
