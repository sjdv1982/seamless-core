import pytest

from seamless.checksum.hash_type import HashType, Kind, Length
from seamless.checksum.hash_type_validation import conversion_feasible


CHECKSUM = "a" * 64


@pytest.mark.parametrize(
    "target", ("str", "int", "float", "bool", "yaml", "python", "ipython")
)
def test_bytes_conversion_chains_reject_raw_bytes(target):
    value = HashType(Kind.RAW_BYTES, Length.SHORT)

    assert conversion_feasible(value, "bytes", target, checksum=CHECKSUM) is False


def test_bytes_to_int_chain_rejects_non_json_text():
    value = HashType(Kind.RAW_TEXT, Length.SHORT)

    assert conversion_feasible(value, "bytes", "int", checksum=CHECKSUM) is False


def test_bytes_to_int_chain_checks_second_step_on_unchanged_checksum():
    value = HashType(Kind.JSON_OBJECT, Length.SHORT)

    assert conversion_feasible(value, "bytes", "int", checksum=CHECKSUM) is False


@pytest.mark.parametrize("target", ("text", "yaml", "python", "ipython"))
def test_mixed_conversion_chains_reject_non_json_mixed_values(target):
    value = HashType(Kind.MIXED_OBJECT, Length.SHORT)

    assert conversion_feasible(value, "mixed", target, checksum=CHECKSUM) is False


def test_chain_can_follow_checksum_preserving_step_into_reformat():
    value = HashType(Kind.JSON_OBJECT, Length.SHORT)

    assert conversion_feasible(value, "mixed", "text", checksum=CHECKSUM) is True


def test_chain_does_not_predict_through_buffer_reformatting():
    value = HashType(Kind.JSON_STRING, Length.SHORT)

    assert conversion_feasible(value, "text", "int", checksum=CHECKSUM) is None
