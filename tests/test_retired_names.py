"""Removed API names cannot silently become paths, including on bound handles."""
import pytest

from seamless import Cell, Expression
from seamless.retired_names import RETIRED_NAMES


class Backend:
    path = "root"
    path_python = "root"

    def assign(self, *args):
        raise AssertionError("a retired API name must not write a value key")

    def delete(self, *args):
        raise AssertionError("a retired API name must not delete a value key")


@pytest.fixture
def retired_name(monkeypatch):
    # Actual names remain live until their respective rename phases.
    monkeypatch.setitem(RETIRED_NAMES, "retired_api", "replacement_api")
    return "retired_api"


@pytest.mark.parametrize("kind", ["standalone", "projection", "bound"])
def test_cell_retired_name_never_projects(retired_name, kind):
    cell = Cell()
    if kind == "projection":
        cell = cell["x"]
    elif kind == "bound":
        cell = Cell._from_backend(Backend())
    for operation in (
        lambda: getattr(cell, retired_name),
        lambda: setattr(cell, retired_name, 1),
        lambda: delattr(cell, retired_name),
    ):
        with pytest.raises(AttributeError, match="retired.*replacement_api"):
            operation()


def test_retired_name_remains_a_value_key(retired_name):
    assert Cell()[retired_name].path == retired_name
    assert Expression(None)[retired_name].path == retired_name


def test_expression_retired_name_never_projects(retired_name):
    with pytest.raises(AttributeError, match="retired.*replacement_api"):
        getattr(Expression(None), retired_name)


def test_expression_types_are_keyword_only():
    with pytest.raises(TypeError):
        Expression(None, "", "plain", "str")
    expression = Expression(None, "", input_celltype="plain", celltype="str")
    assert expression.input_celltype == "plain"
    assert expression.celltype == "str"


@pytest.mark.parametrize("make", [Cell, lambda: Cell()["x"], lambda: Expression(None)])
def test_output_rename_blocks_old_standalone_name(make):
    with pytest.raises(AttributeError, match="target_celltype.*celltype"):
        make().target_celltype


def test_input_rename_preserves_bound_celltype():
    backend = Backend()
    backend.celltype = "plain"
    cell = Cell._from_backend(backend)
    assert cell.celltype == "plain"
    cell.celltype = "str"
    assert backend.celltype == "str"


@pytest.mark.parametrize("celltype", ["int", "str", "text", "mixed"])
def test_unwired_cell_has_no_input_type(celltype):
    cell = Cell(celltype)
    assert cell.input_celltype is None
    assert cell.celltype == celltype
