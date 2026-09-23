"""Storage prerequisites for the compiled scalar ABI."""

import numpy as np
import pytest

from seamless import Buffer


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), np.float32("nan")]
)
def test_nonfinite_mixed_numpy_values(value):
    for source in [value, [value], {"x": value}]:
        result = Buffer(source, "mixed").get_value("mixed")
        if isinstance(result, list):
            result = result[0]
        if isinstance(result, dict):
            result = result["x"]
        assert isinstance(result, np.float64)
        if np.isnan(value):
            assert np.isnan(result)
        else:
            assert result == value
    for ct in ("plain", "float"):
        with pytest.raises(ValueError):
            Buffer(value, ct)


@pytest.mark.parametrize("ct", ["mixed", "binary"])
@pytest.mark.parametrize(
    "value",
    [complex(1, 2), np.complex64(1 + 2j), np.array([1 + 2j], dtype="complex128")],
)
def test_complex_roundtrip(ct, value):
    result = Buffer(value, ct).get_value(ct)
    assert np.asarray(result).dtype == np.asarray(value).dtype
    np.testing.assert_array_equal(result, value)


def test_numpy_boolean_mixed():
    assert Buffer(np.bool_(True), "mixed").get_value("mixed") is True


def test_nested_complex_numpy_value():
    value = {"z": np.complex64(1 + 2j)}
    result = Buffer(value, "mixed").get_value("mixed")
    assert isinstance(result["z"], np.complex64)
    assert result["z"] == value["z"]
