"""Release loops must continue past a failing decref (core-only: no upward imports)."""

import concurrent.futures

import pytest

from seamless import Buffer
from seamless.caching import buffer_cache as bc
from seamless.checksum import expression as ex


def _checksums(n, tag):
    out = []
    for i in range(n):
        buf = Buffer(b"%s-%d" % (tag, i))
        out.append((buf, buf.get_checksum()))
    return out


@pytest.fixture
def cache():
    c = bc.get_buffer_cache()
    snap = set(c.reference_snapshot())
    yield c
    # leave nothing behind from this test
    for cs in list(c.refholder_counts):
        if cs not in snap:
            while c.refholder_counts.get(cs, 0):
                c.decref_refholder(cs)


def _first_decref_raises_after_decrement(monkeypatch):
    """Mimic the real failure: the count is decremented, then demotion raises."""
    real = bc.BufferCache.decref_refholder
    calls = {"n": 0}

    def wrapper(self, checksum):
        result = real(self, checksum)
        calls["n"] += 1
        if calls["n"] == 1:
            raise OverflowError(34, "Numerical result out of range")
        return result

    monkeypatch.setattr(bc.BufferCache, "decref_refholder", wrapper)
    return calls


def test_multi_checksum_release_continues_after_first_error(cache, monkeypatch):
    items = _checksums(4, b"multi-release")
    active = ex._ActiveExpression(
        result_future=concurrent.futures.Future(), task=None, members=set()
    )
    active.hold_inputs([cs for _b, cs in items])
    for _b, cs in items:
        assert cache.refholder_counts[cs] == 1
    _first_decref_raises_after_decrement(monkeypatch)
    with pytest.raises(OverflowError):
        active._release_refholds()
    # claims are gone (input_claims == ()), so every count must be gone too:
    assert active._refheld_checksums() == ()
    for _b, cs in items:
        assert cache.refholder_counts.get(cs, 0) == 0


def test_discard_active_expression_wakes_awaiters_when_release_fails(cache, monkeypatch):
    items = _checksums(2, b"discard")
    fut = concurrent.futures.Future()
    active = ex._ActiveExpression(result_future=fut, task=None, members=set())
    active.hold_inputs([cs for _b, cs in items])
    _first_decref_raises_after_decrement(monkeypatch)
    ex._discard_active_expression(("no", "such", "key"), active)  # must not raise
    assert fut.done()
    with pytest.raises(BaseException):  # CancelledError is a BaseException
        fut.result()
    for _b, cs in items:
        assert cache.refholder_counts.get(cs, 0) == 0
