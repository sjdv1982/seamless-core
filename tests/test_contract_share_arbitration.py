"""Core half of shares.md *The API* and cells.md *API-name arbitration*."""
import pytest

from seamless import Cell


def test_share_is_a_class_property_with_deleter():
    """The API: share is claimed on Cell even before workflow is imported."""
    assert isinstance(Cell.share, property)
    assert Cell.share.fget is not None and Cell.share.fdel is not None


@pytest.mark.parametrize("celltype", ["plain", "mixed"])
def test_standalone_share_never_falls_through_to_value_projection(celltype):
    """Errors; API-name arbitration: bound-only getter cannot project share."""
    cell = Cell(celltype=celltype)
    cell.set({"share": 23})
    with pytest.raises(AttributeError, match="^share is only available for bound workflow cells$"):
        cell.share
    assert cell["share"].value == 23
    with pytest.raises(AttributeError, match="^share is only available for bound workflow cells$"):
        cell.share()
    assert cell.value == {"share": 23}


def test_standalone_share_deleter_never_deletes_value_key():
    """The API; Errors: the deleter owns the slot and cannot remove a value key."""
    cell = Cell(celltype="plain")
    cell.set({"share": 23})
    with pytest.raises(AttributeError, match="bound workflow cells"):
        del cell.share
    assert cell.value == {"share": 23}
