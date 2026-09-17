import pytest

from seamless import Checksum
from seamless.checksum.null import NULL_CHECKSUM, is_null


def test_null_checksum_display_is_not_a_wire_format():
    checksum = Checksum(NULL_CHECKSUM)

    assert str(checksum) == "NULL"
    assert repr(checksum) == repr(NULL_CHECKSUM)
    assert checksum.hex() == NULL_CHECKSUM
    assert is_null(checksum)
    assert is_null(NULL_CHECKSUM)
    with pytest.raises(ValueError):
        Checksum("NULL")
