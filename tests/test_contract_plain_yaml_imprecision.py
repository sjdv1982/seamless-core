"""celltypes-and-conversion.md, *Deliberate imprecisions*: `plain -> yaml` is
trivial, although PyYAML (YAML 1.1) refuses some valid `plain` buffers."""

import pytest

from seamless import Buffer
from seamless.checksum.convert import conversion_needs_buffer, convert_checksum
from seamless.checksum.hash_type_validation import HashTypeValidationError
from seamless.checksum.parse_buffer import _parse_buffer


def _fail_if_fetched():
    raise AssertionError("conversion unexpectedly fetched its source buffer")


PLAIN_BUT_NOT_YAML = [
    pytest.param(lambda: Buffer(b'{\t"a": 1}'), id="tab-whitespace"),
    pytest.param(lambda: Buffer(b'{"a"\n: 1}'), id="break-before-colon"),
    # Canonical plain serialization produces the remaining three.
    pytest.param(lambda: Buffer({"k" * 1100: 1}, "plain"), id="long-key"),
    pytest.param(lambda: Buffer({"note": "a\x7fb"}, "plain"), id="raw-del"),
    pytest.param(lambda: Buffer({"note": "a\u0086b"}, "plain"), id="raw-c1"),
    pytest.param(lambda: Buffer({"note": "a￾b"}, "plain"), id="noncharacter"),
]


@pytest.mark.parametrize("make", PLAIN_BUT_NOT_YAML)
def test_plain_to_yaml_is_trivial_although_the_yaml_reading_fails(make):
    buffer = make()
    checksum = buffer.get_checksum()
    _parse_buffer(buffer, checksum, "plain")
    assert conversion_needs_buffer(checksum, "plain", "yaml") is False
    assert convert_checksum(checksum, "plain", "yaml", _fail_if_fetched) == (
        checksum,
        None,
    )
    with pytest.raises(HashTypeValidationError):
        _parse_buffer(buffer, checksum, "yaml")


@pytest.mark.parametrize("source", ["int", "float"])
def test_int_and_float_to_yaml_inherit_the_gap(source):
    buffer = Buffer(b"\t5")
    checksum = buffer.get_checksum()
    _parse_buffer(buffer, checksum, source)
    assert convert_checksum(checksum, source, "yaml", _fail_if_fetched) == (
        checksum,
        None,
    )
    with pytest.raises(HashTypeValidationError):
        _parse_buffer(buffer, checksum, "yaml")
