import pytest
from seamless import Cell, CellBase, Expression


class PinLike(CellBase):
    __slots__ = ()


def test_cell_base_has_no_source_or_projection_protocol():
    assert issubclass(Cell, CellBase)
    pin = PinLike()
    assert not isinstance(pin, Cell)
    for name in ('_workflow_endpoint', '_workflow_capture_source', 'item', 'slice',
                 'mount', 'validator', 'with_input', 'as_celltype', 'prune'):
        assert not hasattr(pin, name)


@pytest.mark.parametrize('make', [lambda pin: Cell(source=pin),
                                 lambda pin: Expression(pin),
                                 lambda pin: Cell().with_input(pin)])
def test_non_cell_base_cannot_be_a_source(make):
    with pytest.raises(TypeError, match="Pin can't be a source.*pin.source"):
        make(PinLike())
