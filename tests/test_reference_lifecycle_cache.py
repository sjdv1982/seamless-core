import importlib.util
import os
import sys

import pytest

from seamless import Buffer
from seamless.checksum_class import Checksum
from seamless.reference_lifecycle import audit_reference_accounting


def load_buffer_cache_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "seamless", "caching", "buffer_cache.py"
    )
    path = os.path.abspath(path)
    name = "seamless_caching_buffer_cache_lifecycle_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_cache():
    return load_buffer_cache_module().BufferCache(soft_cap=0, hard_cap=0)


def test_many_refholders_share_one_bridge_and_interest():
    cache = make_cache()
    buf = Buffer(b"refholder-bridge")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))

    cache.incref_refholder(checksum, buffer=buf)
    one_interest = cache.strong_cache[checksum].interest()
    cache.incref_refholder(checksum)
    cache.incref_refholder(checksum)

    entry = cache.strong_cache[checksum]
    assert cache.refholder_counts[checksum] == 3
    assert entry.has_refholder_bridge is True
    assert entry.interest() == one_interest

    assert cache.decref_refholder(checksum) is True
    assert cache.decref_refholder(checksum) is True
    assert cache.refholder_counts[checksum] == 1
    assert cache.strong_cache[checksum].has_refholder_bridge is True
    assert cache.decref_refholder(checksum) is True
    assert checksum not in cache.refholder_counts
    assert checksum not in cache.strong_cache


def test_manual_refs_and_refholder_bridge_are_independent():
    cache = make_cache()
    buf = Buffer(b"manual-and-refholder")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))

    cache.incref(checksum)
    cache.incref_refholder(checksum)
    assert cache.reference_snapshot()[checksum] == (1, 1, True)

    assert cache.decref_refholder(checksum) is True
    assert cache.reference_snapshot()[checksum] == (0, 1, False)
    assert checksum in cache.strong_cache
    assert cache.decref(checksum) is True
    assert checksum not in cache.strong_cache


def test_protected_entries_survive_zero_cap_and_unprotected_entries_demote():
    cache = make_cache()
    buf = Buffer(b"protected-by-bridge")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    cache.incref_refholder(checksum)

    before, after = cache.run_eviction_once()
    assert before == after == len(buf.content)
    assert checksum in cache.strong_cache

    cache.decref_refholder(checksum)
    assert checksum not in cache.strong_cache
    assert checksum in cache.weak_cache


def test_equal_checksum_acquire_before_release_keeps_bridge():
    cache = make_cache()
    buf = Buffer(b"same-checksum-replacement")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    cache.incref_refholder(checksum)

    # This is the observable state during an acquire-before-release replace.
    cache.incref_refholder(checksum)
    assert cache.reference_snapshot()[checksum] == (2, 0, True)
    cache.decref_refholder(checksum)
    assert cache.reference_snapshot()[checksum] == (1, 0, True)
    cache.decref_refholder(checksum)


def test_absent_buffer_refholder_balances():
    cache = make_cache()
    checksum = Checksum("a" * 64)

    cache.incref_refholder(checksum)
    assert cache.reference_snapshot()[checksum] == (1, 0, True)
    assert cache.decref_refholder(checksum) is True
    assert checksum not in cache.reference_snapshot()


def test_both_decrement_underflows_warn(caplog):
    cache = make_cache()
    checksum = Checksum("b" * 64)
    with caplog.at_level("WARNING", logger="seamless.references"):
        assert cache.decref(checksum) is False
        assert cache.decref_refholder(checksum) is False
    messages = [record.message for record in caplog.records]
    assert any("Manual decref ignored" in message for message in messages)
    assert any("Refholder decref ignored" in message for message in messages)


def test_forced_cleanup_clears_manual_refs_and_refholder_bridges():
    cache = make_cache()
    buffer = Buffer(b"forced-accounting-cleanup")
    checksum = buffer.get_checksum()
    cache.register(checksum, buffer, size=len(buffer.content))
    cache.incref(checksum)
    cache.incref(checksum)
    cache.incref_refholder(checksum)
    cache.force_clear_reference_accounting()
    assert cache.reference_snapshot() == {}


@pytest.mark.parametrize("manual_refs", [0, 1, 3])
@pytest.mark.parametrize("holder_refs", [0, 1, 3])
def test_manual_and_refholder_count_matrix(manual_refs, holder_refs):
    cache = make_cache()
    buf = Buffer(f"matrix-{manual_refs}-{holder_refs}".encode())
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    for _ in range(manual_refs):
        cache.incref(checksum)
    for _ in range(holder_refs):
        cache.incref_refholder(checksum)

    assert cache.reference_snapshot().get(checksum, (0, 0, False)) == (
        holder_refs,
        manual_refs,
        holder_refs > 0,
    )
    for _ in range(holder_refs):
        assert cache.decref_refholder(checksum)
    for _ in range(manual_refs):
        assert cache.decref(checksum)
    assert cache.reference_snapshot().get(checksum, (0, 0, False)) == (
        0,
        0,
        False,
    )


def test_live_and_expired_temprefs_are_independent_on_both_decrement_paths():
    for decrement in ("manual", "holder"):
        cache = make_cache()
        buf = Buffer(f"tempref-{decrement}".encode())
        checksum = buf.get_checksum()
        cache.register(checksum, buf, size=len(buf.content))
        cache.tempref(checksum, interest=128, fade_interval=10_000)
        if decrement == "manual":
            cache.incref(checksum)
            assert cache.decref(checksum)
        else:
            cache.incref_refholder(checksum)
            assert cache.decref_refholder(checksum)
        assert checksum in cache.strong_cache

        cache.strong_cache[checksum].tempref.clear()
        cache.run_eviction_once()
        assert checksum not in cache.strong_cache


@pytest.mark.parametrize("protection", ["none", "bridge", "manual", "tempref"])
def test_scratch_purge_respects_each_protection_kind(protection):
    cache = make_cache()
    buf = Buffer(f"scratch-purge-{protection}".encode())
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    cache.tempref(checksum, scratch=True)
    if protection == "bridge":
        cache.incref_refholder(checksum, scratch=True)
    elif protection == "manual":
        cache.incref(checksum, scratch=True)
    elif protection == "tempref":
        cache.tempref(checksum, scratch=False)

    purged = cache.purge_scratch(checksum)
    if protection == "none":
        assert purged == 1
        assert checksum not in cache.strong_cache
    else:
        assert purged == 0
        assert checksum in cache.strong_cache


def test_scratch_first_then_non_scratch_acquisition_registers_monotonically():
    cache = make_cache()
    buf = Buffer(b"scratch-upgrade")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    cache.tempref(checksum, scratch=True)
    assert cache.strong_cache[checksum].remote_registered is False
    cache.incref_refholder(checksum, scratch=False)
    assert cache.strong_cache[checksum].remote_registered is True
    assert cache.is_scratch_ref(checksum) is False
    cache.decref_refholder(checksum)


def test_protected_entries_are_not_eviction_candidates():
    cache = make_cache()
    protected = Buffer(b"candidate-protected")
    unprotected = Buffer(b"candidate-unprotected")
    protected_cs = protected.get_checksum()
    unprotected_cs = unprotected.get_checksum()
    cache.register(protected_cs, protected, size=len(protected.content))
    cache.register(unprotected_cs, unprotected, size=len(unprotected.content))
    cache.incref_refholder(protected_cs)
    cache.run_eviction_once()
    assert protected_cs in cache.strong_cache
    assert unprotected_cs not in cache.strong_cache
    cache.decref_refholder(protected_cs)


def test_memory_pressure_warning_is_rate_limited_and_reemits_after_change(caplog):
    cache = make_cache()
    first = Buffer(b"pressure-first")
    first_cs = first.get_checksum()
    cache.register(first_cs, first, size=len(first.content))
    cache.incref_refholder(first_cs)
    with caplog.at_level("WARNING", logger="seamless.references"):
        cache.run_eviction_once()
        cache.run_eviction_once()
        assert caplog.text.count("Memory pressure remains above cap") == 1

        second = Buffer(b"pressure-second")
        second_cs = second.get_checksum()
        cache.register(second_cs, second, size=len(second.content))
        cache.incref_refholder(second_cs)
        cache.run_eviction_once()
        assert caplog.text.count("Memory pressure remains above cap") == 2
    cache.decref_refholder(first_cs)
    cache.decref_refholder(second_cs)


def test_forced_cleanup_preserves_a_live_tempref_snapshot():
    cache = make_cache()
    buf = Buffer(b"cleanup-live-tempref")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    cache.tempref(checksum, interest=128, fade_interval=10_000)
    cache.incref(checksum)
    cache.incref_refholder(checksum)
    cache.force_clear_reference_accounting()
    assert cache.reference_snapshot()[checksum] == (0, 0, False)
    assert checksum in cache.strong_cache


def test_corrupt_count_and_bridge_are_audited_before_forced_cleanup(caplog):
    cache = make_cache()
    buf = Buffer(b"corrupt-accounting")
    checksum = buf.get_checksum()
    cache.register(checksum, buf, size=len(buf.content))
    cache.incref_refholder(checksum)
    cache.refholder_counts[checksum] = 2
    cache.strong_cache[checksum].has_refholder_bridge = False
    with caplog.at_level("WARNING", logger="seamless.references"):
        audit_reference_accounting(cache=cache, holders=[])
    assert "only 0 live claims" in caplog.text
    assert "eviction bridge is absent" in caplog.text
    assert cache.reference_snapshot()[checksum] == (2, 0, False)
    cache.force_clear_reference_accounting()
