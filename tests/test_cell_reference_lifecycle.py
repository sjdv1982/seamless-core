import gc

from seamless import Buffer, Cell
from seamless.reference_lifecycle import clear_refholder_registry_for_tests


def test_checksum_cell_acquires_replaces_and_releases_input():
    source = Buffer(b"cell lifecycle source")
    first = source.get_checksum()
    second = Buffer(b"cell lifecycle replacement").get_checksum()
    cache = __import__("seamless.caching.buffer_cache", fromlist=["get_buffer_cache"]).get_buffer_cache()

    cell = Cell(input_ref=first)
    assert cache.reference_snapshot()[first][0] == 1
    cell.input_ref = second
    assert cache.reference_snapshot().get(first, (0, 0, False))[0] == 0
    assert cache.reference_snapshot()[second][0] == 1

    cell._release_refholds()
    cell._release_refholds()
    assert cache.reference_snapshot().get(second, (0, 0, False))[0] == 0


def test_derived_cell_has_independent_input_hold():
    buffer = Buffer(b"derived cell")
    checksum = buffer.get_checksum()
    cell = Cell(input_ref=checksum)
    derived = cell.item("field")
    cache = __import__("seamless.caching.buffer_cache", fromlist=["get_buffer_cache"]).get_buffer_cache()
    assert cache.reference_snapshot()[checksum][0] == 2
    cell._release_refholds()
    assert cache.reference_snapshot()[checksum][0] == 1
    derived._release_refholds()
    assert cache.reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_cell_does_not_hold_expression_result():
    source = Buffer({"x": "cell expression result"}, "plain")
    checksum = source.get_checksum()
    cell = Cell(input_ref=checksum, input_celltype="plain").item("x").as_celltype("str")
    expression = cell.build()
    result = expression.compute()
    cache = __import__("seamless.caching.buffer_cache", fromlist=["get_buffer_cache"]).get_buffer_cache()
    assert cache.reference_snapshot().get(result, (0, 0, False))[0] == 1
    expression._release_refholds()
    cell._release_refholds()
    clear_refholder_registry_for_tests()
    gc.collect()
