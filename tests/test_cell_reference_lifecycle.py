import gc

import pytest

from seamless import Buffer, Cell
from seamless.reference_lifecycle import clear_refholder_registry_for_tests


def test_checksum_cell_acquires_replaces_and_releases_input():
    source = Buffer(b"cell lifecycle source")
    first = source.get_checksum()
    second = Buffer(b"cell lifecycle replacement").get_checksum()
    cache = __import__("seamless.caching.buffer_cache", fromlist=["get_buffer_cache"]).get_buffer_cache()

    cell = Cell(checksum=first)
    assert cache.reference_snapshot()[first][0] == 1
    cell.checksum = second
    assert cache.reference_snapshot().get(first, (0, 0, False))[0] == 0
    assert cache.reference_snapshot()[second][0] == 1

    cell._release_refholds()
    cell._release_refholds()
    assert cache.reference_snapshot().get(second, (0, 0, False))[0] == 0


def test_derived_cell_keeps_parent_input_alive():
    buffer = Buffer(b"derived cell")
    checksum = buffer.get_checksum()
    cell = Cell(checksum=checksum)
    derived = cell.item("field")
    cache = __import__("seamless.caching.buffer_cache", fromlist=["get_buffer_cache"]).get_buffer_cache()
    assert derived.source is cell
    del cell
    gc.collect()
    assert cache.reference_snapshot()[checksum][0] >= 1
    del derived
    gc.collect()
    assert cache.reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_cell_does_not_hold_expression_result():
    source = Buffer({"x": "cell expression result"}, "plain")
    checksum = source.get_checksum()
    cell = Cell(checksum=checksum, celltype="plain").item("x").as_celltype("str")
    expression = cell.build()
    result = expression.compute()
    cache = __import__("seamless.caching.buffer_cache", fromlist=["get_buffer_cache"]).get_buffer_cache()
    assert cache.reference_snapshot().get(result, (0, 0, False))[0] == 1
    expression._release_refholds()
    cell._release_refholds()
    clear_refholder_registry_for_tests()
    gc.collect()


def test_cell_owns_computed_result_and_releases_it_on_recipe_change():
    from seamless.caching.buffer_cache import get_buffer_cache
    from seamless.reference_lifecycle import collect_refholder_claims

    root = Cell("plain")
    root.set({"x": "a computed Cell result with its own lifetime"})
    child = root["x"].as_celltype("text")
    checksum = child.checksum
    claims = collect_refholder_claims([child])
    assert any(owner is child and role == "result" for owner, role in claims[checksum])
    root.set({"x": "replacement result"})
    assert child.value == "replacement result"
    assert checksum not in collect_refholder_claims([child])
    current = child.checksum
    child._release_refholds()
    assert get_buffer_cache().reference_snapshot().get(current, (0, 0, False))[0] == 0
