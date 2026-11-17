"""Draft tests for the phase-2 periodic eviction thread."""

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from seamless import Buffer
from seamless.caching import buffer_cache as bc


_BASELINE: set = set()


def _threads():
    """Eviction threads started by the test (the process-wide cache has its own)."""
    return [
        t
        for t in threading.enumerate()
        if t.name == "seamless-eviction" and t not in _BASELINE
    ]


@pytest.fixture(autouse=True)
def _baseline():
    bc.get_buffer_cache()  # make sure the process-wide cache (and its thread) exists
    _BASELINE.clear()
    _BASELINE.update(t for t in threading.enumerate() if t.name == "seamless-eviction")
    yield


@pytest.fixture
def cache():
    c = bc.BufferCache()
    yield c
    c.stop_eviction_thread()


def _tempref_only(c, data):
    buf = Buffer(data)
    cs = buf.get_checksum()
    c.tempref(cs, buffer=buf, fade_interval=0.001)  # expires almost at once
    return buf, cs


def test_start_stop_evicts_expired_tempref(cache):
    buf, cs = _tempref_only(cache, b"loop-evicts")
    assert cs in cache.strong_cache
    assert cache.start_eviction_thread(0.02) is True
    deadline = time.time() + 3
    while cs in cache.strong_cache and time.time() < deadline:
        time.sleep(0.02)
    assert cs not in cache.strong_cache  # expired tempref demoted by the loop
    cache.stop_eviction_thread()
    assert cache._eviction_thread is None
    assert not _threads()


def test_start_is_idempotent(cache):
    assert cache.start_eviction_thread(0.05) is True
    assert cache.start_eviction_thread(0.05) is False
    assert len(_threads()) == 1
    cache.stop_eviction_thread()
    assert not _threads()
    assert cache.start_eviction_thread(0.05) is True  # restartable
    cache.stop_eviction_thread()


def test_live_refheld_entry_survives_loop(cache):
    buf, cs = _tempref_only(cache, b"refheld")
    cache.incref_refholder(cs, buffer=buf, scratch=True)
    cache.start_eviction_thread(0.02)
    time.sleep(0.3)
    assert cs in cache.strong_cache and cache.refholder_counts[cs] == 1


def test_not_started_in_workers(monkeypatch):
    monkeypatch.setattr(bc, "_cache_instance", None)
    monkeypatch.setattr(bc, "is_worker", lambda: True)
    with pytest.raises(RuntimeError):
        bc.get_buffer_cache()
    assert bc._cache_instance is None
    assert not _threads()


@pytest.mark.parametrize("value", [None, ""])
def test_on_by_default_when_unset_or_empty(monkeypatch, value):
    monkeypatch.setattr(bc, "_cache_instance", None)
    if value is None:
        monkeypatch.delenv("SEAMLESS_EVICTION_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("SEAMLESS_EVICTION_INTERVAL", value)
    assert bc.eviction_interval_from_env() == 5.0
    c = bc.get_buffer_cache()
    try:
        assert c._eviction_thread is not None and c._eviction_thread.is_alive()
    finally:
        c.stop_eviction_thread()


def test_zero_turns_the_loop_off(monkeypatch):
    monkeypatch.setattr(bc, "_cache_instance", None)
    monkeypatch.setenv("SEAMLESS_EVICTION_INTERVAL", "0")
    assert bc.eviction_interval_from_env() == 0.0
    c = bc.get_buffer_cache()
    assert c._eviction_thread is None and not _threads()


def test_positive_interval_is_used_as_given(monkeypatch):
    monkeypatch.setattr(bc, "_cache_instance", None)
    monkeypatch.setenv("SEAMLESS_EVICTION_INTERVAL", "7.5")
    assert bc.eviction_interval_from_env() == 7.5
    c = bc.get_buffer_cache()
    try:
        assert c._eviction_thread is not None and c._eviction_thread.is_alive()
    finally:
        c.stop_eviction_thread()


def test_shutdown_stops_loop_before_release_and_force_clear():
    script = textwrap.dedent(
        """
        from seamless import Buffer, close
        from seamless.caching import buffer_cache as bc
        from seamless.reference_lifecycle import register_refholder
        order = []
        cache = bc.get_buffer_cache()
        assert cache._eviction_thread is not None
        real_stop = bc.BufferCache.stop_eviction_thread
        def stop(self, *a, **k):
            order.append("stop"); return real_stop(self, *a, **k)
        bc.BufferCache.stop_eviction_thread = stop
        real_clear = bc.BufferCache.force_clear_reference_accounting
        def clear(self):
            order.append("force_clear"); return real_clear(self)
        bc.BufferCache.force_clear_reference_accounting = clear
        buf = Buffer(b"order"); cs = buf.get_checksum()
        cache.incref_refholder(cs, buffer=buf, scratch=True)
        class Holder:
            def _refheld_checksums(self): return ((cs, "h"),)
            def _release_refholds(self):
                order.append("holder"); cs.decref_refholder()
        h = Holder(); register_refholder(h)
        close()
        assert order[:3] == ["stop", "holder", "force_clear"], order
        assert cache._eviction_thread is None
        """
    )
    import os

    env = dict(os.environ, SEAMLESS_EVICTION_INTERVAL="5")  # explicit, although 5 is the default
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr


def test_pressure_drops_expired_but_keeps_live_tempref():
    c = bc.BufferCache(soft_cap=1, hard_cap=2)
    live_buf, live_cs = Buffer(b"live-tempref-buffer"), None
    live_cs = live_buf.get_checksum()
    from seamless.caching import eviction_cost

    eviction_cost.set_download_profile(live_cs, "read_buffer")  # finite eviction cost
    c.register(live_cs, live_buf, size=1000)  # sizes must be known for scoring
    c.tempref(live_cs, buffer=live_buf)  # default: live for ~28 s
    exp_buf = Buffer(b"expired-tempref-buffer")
    exp_cs = exp_buf.get_checksum()
    c.register(exp_cs, exp_buf, size=1000)
    c.tempref(exp_cs, buffer=exp_buf, fade_interval=0.001)  # expires almost at once
    time.sleep(0.05)  # fade_interval=0.001 -> long expired
    c.run_eviction_once()
    assert live_cs in c.strong_cache  # live tempref survives memory pressure
    assert exp_cs not in c.strong_cache  # expired tempref is dropped
    # contract forced expiry is an explicit, test-only parameter:
    c.run_eviction_once(force_expire_temprefs=True)
    assert live_cs not in c.strong_cache


def test_only_one_eviction_thread(monkeypatch):
    """One setting (SEAMLESS_EVICTION_INTERVAL), one thread; tempref() starts nothing."""
    monkeypatch.setenv("SEAMLESS_EVICTION_INTERVAL", "0.3")
    assert bc.eviction_interval_from_env() == 0.3
    # the old debug name is gone and has no effect (the default applies)
    monkeypatch.delenv("SEAMLESS_EVICTION_INTERVAL")
    monkeypatch.setenv("SEAMLESS_DEBUG_EVICTION_LOOP", "0.3")
    assert bc.eviction_interval_from_env() == 5.0
    c = bc.BufferCache()
    try:
        assert c.start_eviction_thread(0.3) is True
        c.tempref(_cs_of(b"x1"), buffer=Buffer(b"x1"))  # tempref() starts nothing
        assert len(_threads()) == 1
    finally:
        c.stop_eviction_thread()


def _cs_of(data):
    return Buffer(data).get_checksum()
