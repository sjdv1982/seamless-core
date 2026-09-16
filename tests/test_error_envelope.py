import copy
import pytest
from seamless import Checksum, CacheMissError
from seamless.error_envelope import (
    encode_error,
    decode_error,
    execution_error,
    WorkflowExecutionError,
)


def test_cache_miss_round_trip_preserves_checksum_argument():
    checksum = Checksum("a" * 64)
    for argument in (checksum, checksum.hex(), bytes.fromhex(checksum.hex())):
        envelope = encode_error(CacheMissError(argument))
        for _ in range(3):
            error = decode_error(envelope)
            assert isinstance(error, CacheMissError)
            assert isinstance(error.args[0], Checksum)
            assert error.args[0] == checksum
            assert error.__traceback__ is None
            envelope = encode_error(error)


def test_argumentless_cache_miss():
    envelope = encode_error(CacheMissError())
    assert "checksum" not in envelope["error"]
    assert decode_error(envelope).args == ()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"error": None},
        {"error": {"kind": "execution"}},
        {"error": {"kind": "cache_miss", "message": "", "checksum": "bad"}},
    ],
)
def test_malformed_envelopes(payload):
    with pytest.raises(ValueError):
        decode_error(payload)


def test_unknown_kind_and_failure_identity():
    error = decode_error(
        {"error": {"kind": "future_kind", "message": "future message"}}
    )
    assert isinstance(error, WorkflowExecutionError)
    assert error.kind == "future_kind"
    assert str(error) == "future message"
    assert copy.deepcopy(error).failure_id == error.failure_id
    assert copy.deepcopy(error).kind == error.kind
    assert execution_error(error) is error
