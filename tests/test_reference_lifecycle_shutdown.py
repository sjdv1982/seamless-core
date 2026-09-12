from __future__ import annotations

import subprocess
import sys
import textwrap


def _script(body: str) -> str:
    body = textwrap.dedent(body).strip()
    return "import logging\nlogging.basicConfig(level=logging.WARNING)\n" + body + "\n"


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_balanced_explicit_close_is_quiet_on_reference_logger():
    result = _run(
        """
import logging
from seamless import Cell, Checksum, close
logging.basicConfig(level=logging.WARNING)
cell = Cell(input_ref=Checksum('00' * 32))
close()
"""
    )
    assert result.returncode == 0, result.stderr
    assert "seamless.references" not in result.stderr


def test_explicit_close_reports_manual_leak_once_and_is_idempotent():
    result = _run(
        """
import logging
from seamless import Checksum, close
logging.basicConfig(level=logging.WARNING)
checksum = Checksum('11' * 32)
checksum.incref()
close()
close()
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("unmatched manual references") == 1


def test_close_forces_manual_accounting_to_zero_after_warning():
    result = _run(
        """
from seamless import Checksum, close
from seamless.caching.buffer_cache import get_buffer_cache
checksum = Checksum('22' * 32)
checksum.incref()
close()
assert get_buffer_cache().reference_snapshot() == {}
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("unmatched manual references") == 1


def test_balanced_atexit_only_close_is_quiet_on_reference_logger():
    result = _run(
        _script(
            """
            import seamless
            from seamless import Cell, Checksum
            Cell(input_ref=Checksum('33' * 32))
            seamless.ensure_open('shutdown test')
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "seamless.references" not in result.stderr
    assert "called via atexit" in result.stderr


def test_explicit_close_then_atexit_audits_only_once():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            checksum = Checksum('44' * 32)
            checksum.incref()
            close()
            close()
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("unmatched manual references") == 1


def test_live_claims_greater_than_count_are_reported_before_cleanup():
    result = _run(
        _script(
            """
            from seamless import Cell, Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            checksum = Checksum('55' * 32)
            cell = Cell(input_ref=checksum)
            checksum.decref_refholder()
            close()
            assert get_buffer_cache().reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "live claims" in result.stderr


def test_dead_holder_excess_is_attributed_as_unmatched():
    result = _run(
        _script(
            """
            import gc
            from seamless import Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            from seamless.reference_lifecycle import register_refholder
            class Dead:
                def _refheld_checksums(self):
                    return ()
            checksum = Checksum('66' * 32)
            checksum.incref_refholder()
            holder = Dead()
            register_refholder(holder)
            del holder
            gc.collect(); gc.collect()
            close()
            assert get_buffer_cache().reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "only 0 live claims" in result.stderr


def test_missing_bridge_with_positive_count_is_reported():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            checksum = Checksum('77' * 32)
            cache = get_buffer_cache()
            cache.incref_refholder(checksum)
            cache.strong_cache[checksum].has_refholder_bridge = False
            close()
            assert cache.reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "eviction bridge is absent" in result.stderr


def test_present_bridge_with_zero_count_is_reported():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            checksum = Checksum('88' * 32)
            cache = get_buffer_cache()
            cache.incref(checksum)
            cache.strong_cache[checksum].has_refholder_bridge = True
            cache.decref(checksum)
            close()
            assert cache.reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "eviction bridge is present" in result.stderr


def test_holder_cleanup_manual_leak_is_detected_post_cleanup():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            from seamless.reference_lifecycle import register_refholder
            checksum = Checksum('99' * 32)
            class Leaky:
                def _refheld_checksums(self):
                    return ()
                def _release_refholds(self):
                    checksum.incref()
            holder = Leaky()
            register_refholder(holder)
            close()
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "unmatched manual references" in result.stderr


def test_decrement_underflows_are_logged_without_abort():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            checksum = Checksum('aa' * 32)
            checksum.decref()
            checksum.decref_refholder()
            close()
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "Manual decref ignored" in result.stderr
    assert "Refholder decref ignored" in result.stderr


def test_claim_collection_exception_does_not_hide_other_holders():
    result = _run(
        _script(
            """
            from seamless import Cell, Checksum, close
            from seamless.reference_lifecycle import register_refholder
            class Broken:
                def _refheld_checksums(self):
                    raise RuntimeError('claim boom')
            checksum = Checksum('bb' * 32)
            Cell(input_ref=checksum)
            broken = Broken()
            register_refholder(broken)
            close()
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "claim collection raised: claim boom" in result.stderr


def test_cleanup_exception_isolated_from_other_holder_cleanup():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            from seamless.reference_lifecycle import register_refholder
            checksum = Checksum('cc' * 32)
            cache = get_buffer_cache()
            cache.incref_refholder(checksum)
            class Broken:
                def _refheld_checksums(self):
                    return ()
                def _release_refholds(self):
                    raise RuntimeError('cleanup boom')
            class Good:
                def _refheld_checksums(self):
                    return ((checksum, 'good'),)
                def _release_refholds(self):
                    checksum.decref_refholder()
            broken = Broken()
            good = Good()
            register_refholder(broken)
            register_refholder(good)
            close()
            assert cache.reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "cleanup boom" in result.stderr


def test_finalizer_cycle_gets_two_gc_passes_before_audit():
    result = _run(
        _script(
            """
            import gc
            from seamless import Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            from seamless.reference_lifecycle import register_refholder, safe_release_refholder
            checksum = Checksum('dd' * 32)
            checksum.incref_refholder()
            class Cyclic:
                def __init__(self):
                    self.cycle = self
                def _refheld_checksums(self):
                    return ((checksum, 'cycle'),)
                def _release_refholds(self):
                    checksum.decref_refholder()
                def __del__(self):
                    safe_release_refholder(self)
            holder = Cyclic()
            register_refholder(holder)
            del holder
            gc.collect(); gc.collect()
            close()
            assert get_buffer_cache().reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr


def test_writer_flush_precedes_holder_cleanup_and_accounting_is_empty():
    result = _run(
        _script(
            """
            from seamless import Buffer, close
            from seamless.caching import buffer_writer
            from seamless.caching.buffer_cache import get_buffer_cache
            from seamless.reference_lifecycle import register_refholder
            order = []
            buffer = Buffer(b'flush-order')
            checksum = buffer.get_checksum()
            cache = get_buffer_cache()
            cache.incref_refholder(checksum, buffer=buffer)
            original_flush = buffer_writer.flush
            def flush(*args, **kwargs):
                order.append('flush')
                buffer_writer._entries.clear()
            buffer_writer.flush = flush
            class Holder:
                def _refheld_checksums(self):
                    return ((checksum, 'flush-holder'),)
                def _release_refholds(self):
                    order.append('holder')
                    checksum.decref_refholder()
            holder = Holder()
            register_refholder(holder)
            close()
            assert order[:2] == ['flush', 'holder'], order
            assert cache.reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr


def test_residual_count_bridge_and_manual_are_reported_before_forced_clear():
    result = _run(
        _script(
            """
            from seamless import Checksum, close
            from seamless.caching.buffer_cache import get_buffer_cache
            checksum = Checksum('ee' * 32)
            cache = get_buffer_cache()
            cache.incref(checksum)
            cache.incref_refholder(checksum)
            close()
            assert cache.reference_snapshot() == {}
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "unmatched manual references" in result.stderr
    assert "only 0 live claims" in result.stderr
