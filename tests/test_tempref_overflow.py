"""Regression: stale tempref (>~2048 s with defaults) must not raise OverflowError(34)."""

import time

import pytest

from seamless import Buffer
from seamless.caching import buffer_cache as bc


def _cs(data: bytes):
    buf = Buffer(data)
    return buf, buf.get_checksum()


@pytest.fixture(params=[2100.0, 1e6], ids=["2100s", "1e6s"])
def shift(request, monkeypatch):
    """Return a function that jumps the clock forward by the parametrized amount."""
    real = time.time

    def jump():
        monkeypatch.setattr(time, "time", lambda: real() + request.param)

    return jump


def _held_with_tempref(cache, data):
    buf, cs = _cs(data)
    cache.incref_refholder(cs, buffer=buf, scratch=True)
    cache.tempref(cs, buffer=buf)
    return buf, cs


def test_current_interest_clamps(shift):
    ref = bc.TempRef(interest=128.0)
    assert ref.current_interest() > 100
    shift()
    assert ref.current_interest() == 0.0


def test_decref_refholder_after_long_idle(shift):
    cache = bc.BufferCache()
    buf, cs = _held_with_tempref(cache, b"decref-long-idle")
    shift()
    assert cache.decref_refholder(cs) is True
    assert cache.refholder_counts == {}
    assert cs not in cache.strong_cache  # demoted (tempref expired)
    assert cache.reference_snapshot() == {}


def test_manual_decref_after_long_idle(shift):
    cache = bc.BufferCache()
    buf, cs = _cs(b"manual-long-idle")
    cache.incref(cs, buffer=buf, scratch=True)
    cache.tempref(cs, buffer=buf)
    shift()
    assert cache.decref(cs) is True
    assert cs not in cache.strong_cache


def test_force_clear_after_long_idle_clears_everything(shift):
    cache = bc.BufferCache()
    held = [_held_with_tempref(cache, b"force-clear-%d" % i) for i in range(5)]
    shift()
    cache.force_clear_reference_accounting()  # must not raise or abort partway
    assert cache.refholder_counts == {}
    assert cache.strong_cache == {}
    del held


def test_run_eviction_once_after_long_idle(shift):
    cache = bc.BufferCache()
    buf, cs = _cs(b"evict-long-idle")
    cache.tempref(cs, buffer=buf)  # tempref only, no refs
    assert cs in cache.strong_cache
    shift()
    before, after = cache.run_eviction_once()
    assert before == after
    assert cs not in cache.strong_cache


def test_multi_release_stays_consistent(shift):
    """Releasing several checksums, one with a stale tempref, leaves no residue."""
    cache = bc.BufferCache()
    items = [_held_with_tempref(cache, b"multi-%d" % i) for i in range(4)]
    shift()
    for _buf, cs in items:
        cache.decref_refholder(cs)
    assert cache.refholder_counts == {}
    assert all(cs not in cache.strong_cache for _b, cs in items)
    snap = cache.reference_snapshot()
    assert all(count == 0 and not bridge for count, _m, bridge in snap.values())
