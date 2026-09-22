"""Coverage for the Buffer boundary in the Feature-1 repo footprint."""

import pytest

from seamless import Buffer
from seamless.checksum.celltypes import celltypes


STRUCTURAL_CELLTYPES = ("deepcell", "deepfolder", "folder", "module")


@pytest.mark.parametrize("celltype", STRUCTURAL_CELLTYPES)
def test_structural_celltypes_are_plain_at_the_buffer_boundary(celltype):
    value = {"number": 1, "text": "value"}
    plain = Buffer(value, "plain")
    structural = Buffer(value, celltype)

    assert celltype not in celltypes
    assert Buffer._map_celltype(celltype) == "plain"
    assert structural.content == plain.content
    assert structural.get_value(celltype) == value


def test_buffer_boundary_rejects_unknown_celltypes():
    with pytest.raises(TypeError):
        Buffer._map_celltype("not-a-celltype")
    with pytest.raises(TypeError):
        Buffer({}, "not-a-celltype")
