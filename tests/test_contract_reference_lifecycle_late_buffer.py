"""Contract tests: contracts/internal/checksum-reference-lifecycle.md §8, *Buffer
persistence* — the "reverse ordering": a non-scratch claim is held while no
buffer exists, and the buffer arrives later (as when a Cell fingertips a result
it already holds). This is settled contract (§8, *Buffer after claim*):
``BufferCache.register`` writes a buffer that arrives under an existing
non-scratch claim, and not one that arrives under a scratch claim only; so
``Cell.fingertip()`` persists the recovered buffer only on a non-scratch Cell.
The companion order (*claim after buffer*) is tested in
``test_contract_reference_lifecycle.py``.
"""

from __future__ import annotations

import uuid

import pytest

from seamless import Buffer, Cell, Checksum
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
import seamless.checksum.expression as expression_mod


@pytest.fixture
def writes(monkeypatch):
    written: list[Checksum] = []
    monkeypatch.setattr(
        buffer_writer, "register", lambda buf: written.append(buf.get_checksum())
    )
    return written


def _forget_buffer(checksum):
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        entry = cache.strong_cache.get(checksum)
        if entry is not None:
            entry.buffer = None
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)


def test_buffer_arriving_under_a_non_scratch_claim_is_published(writes):
    cache = get_buffer_cache()
    buf = Buffer(f"late-buffer-{uuid.uuid4().hex}".encode())
    checksum = buf.get_checksum()
    _forget_buffer(checksum)

    cache.incref_refholder(checksum)  # non-scratch claim, no buffer here yet
    try:
        assert checksum not in writes
        cache.register(checksum, buf, size=len(buf.content))
        assert writes.count(checksum) == 1
    finally:
        cache.decref_refholder(checksum)


def test_buffer_arriving_under_a_scratch_claim_is_not_published(writes):
    cache = get_buffer_cache()
    buf = Buffer(f"late-scratch-buffer-{uuid.uuid4().hex}".encode())
    checksum = buf.get_checksum()
    _forget_buffer(checksum)

    cache.incref_refholder(checksum, scratch=True)
    try:
        cache.register(checksum, buf, size=len(buf.content))
        assert checksum not in writes
    finally:
        cache.decref_refholder(checksum)


@pytest.mark.parametrize("scratch", [False, True], ids=["non-scratch", "scratch"])
def test_cell_fingertipping_a_held_result_persists_it_only_if_non_scratch(writes, scratch):
    root = Cell("plain")
    root.set({"x": f"late-fingertip-{uuid.uuid4().hex}"})
    child = root["x"].as_celltype("text")
    child.scratch = scratch
    checksum = child.checksum
    assert checksum is not None
    _forget_buffer(checksum)
    writes.clear()
    try:
        assert child.fingertip() is not None
        assert (checksum in writes) is (not scratch)
    finally:
        child._release_refholds()
        root._release_refholds()
