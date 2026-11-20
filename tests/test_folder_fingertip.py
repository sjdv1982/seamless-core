import asyncio

import pytest

from seamless import Buffer, CacheMissError, Checksum, FingertipCategory
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
from seamless.checksum import expression


def _drop_buffer(checksum):
    checksum = Checksum(checksum)
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression._expression_result_buffers.pop(checksum, None)


def _folder_with_missing_leaf(payload):
    leaf = Buffer(payload, "bytes").get_checksum()
    root = Buffer({"leaf": leaf.hex()}, "folder")
    root.tempref()
    return root.get_checksum(), leaf


def test_folder_fingertip_does_not_run_when_every_leaf_is_local(monkeypatch):
    payload = b"folder fingertip complete leaf"
    root, leaf = _folder_with_missing_leaf(payload)
    leaf_buffer = Buffer(payload, "bytes")
    leaf_buffer.tempref()

    async def unexpected(*args, **kwargs):
        raise AssertionError("a complete folder must not run a producer")

    monkeypatch.setattr(expression, "evaluate_expression_local_async", unexpected)

    assert asyncio.run(root.fingertip_folder()) == {"leaf": leaf}


def test_folder_fingertip_forces_matching_expression_and_checks_leaf_bytes(monkeypatch):
    payload = b"folder fingertip recovered leaf"
    root, leaf = _folder_with_missing_leaf(payload)
    expression.get_expression_cache().clear()
    key = (root.hex(), "", "folder", "folder")
    expression.get_expression_cache()[key] = root
    calls = []

    async def evaluate(input_checksum, path, source_celltype, target_celltype, *, materialize, force):
        calls.append((input_checksum, path, source_celltype, target_celltype, materialize, force))
        Buffer(payload, "bytes").tempref()
        return root

    monkeypatch.setattr(expression, "evaluate_expression_local_async", evaluate)

    assert asyncio.run(root.fingertip_folder()) == {"leaf": leaf}
    assert len(calls) == 1
    assert calls[0][-2:] == (True, True)
    assert leaf.resolve().content == payload


def test_folder_projection_that_only_recreates_index_is_a_failed_fingertip(monkeypatch):
    root, leaf = _folder_with_missing_leaf(b"folder fingertip absent leaf")
    expression.get_expression_cache().clear()
    key = (root.hex(), "", "folder", "folder")
    expression.get_expression_cache()[key] = root

    async def project_index(input_checksum, path, source_celltype, target_celltype, *, materialize, force):
        assert force and materialize
        # This candidate recreates the expected index checksum, but no leaf bytes.
        return root

    monkeypatch.setattr(expression, "evaluate_expression_local_async", project_index)

    with pytest.raises(CacheMissError) as caught:
        asyncio.run(root.fingertip_folder())

    assert caught.value.checksum == root
    assert caught.value.fingertip_category == FingertipCategory.MATERIALIZATION
    with pytest.raises(CacheMissError):
        leaf.resolve("bytes")
