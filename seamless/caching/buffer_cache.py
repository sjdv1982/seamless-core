"""Buffer cache implementation.

This module implements a simplified but functional central buffer cache following the
design notes in caching.txt. It provides:
- a weak cache (WeakValueDictionary) for general registrations
- a strong cache (dict) for buffers that have refs (interest)
- normal refs (incref/decref) and one tempref per checksum with decaying interest
- an eviction procedure that moves buffers from strong to weak based on a cost-per-GB
  ordering and configured soft/hard memory caps

The implementation focuses on the main behaviors and provides hooks for cost and
subsystem integrations.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import weakref
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from seamless.caching import eviction_cost
from seamless.caching import buffer_writer
from seamless import is_worker

if TYPE_CHECKING:
    from seamless.checksum_class import Checksum
    from seamless.buffer_class import Buffer

DEFAULT_SOFT_CAP = 5 * 1024**3
DEFAULT_HARD_CAP = 50 * 1024**3
DEFAULT_BENEFIT_PER_GB = 2.0
TEMPREF_MINIMAL_INTEREST = 1 / 128
_MEMORY_PRESSURE_WARNING_INTERVAL = 60.0
_references_logger = logging.getLogger("seamless.references")


@dataclass
class TempRef:
    interest: float
    fade_factor: float = 2.0
    fade_interval: float = 2.0
    created: float = field(default_factory=time.time)
    last_refreshed: float = field(default_factory=time.time)

    def refresh(self):
        self.last_refreshed = time.time()

    def clear(self):
        """Marks the tempref to be removed by setting the interest to zero"""
        self.interest = 0

    def current_interest(self) -> float:
        """Calculate decayed interest based on elapsed time since last_refresh."""
        elapsed = time.time() - self.last_refreshed
        if elapsed <= 0 or self.fade_interval <= 0:
            return self.interest
        # number of fade steps
        steps = elapsed / self.fade_interval
        return self.interest / (self.fade_factor**steps)


@dataclass
class StrongEntry:
    buffer: Optional[Buffer] = None
    size: Optional[int] = None  # bytes
    manual_refs: int = 0
    has_refholder_bridge: bool = False
    tempref: Optional[TempRef] = None
    tempref_scratch: bool = False
    remote_registered: bool = False

    def interest(self) -> float:
        i = float(self.manual_refs + int(self.has_refholder_bridge))
        if self.tempref is not None:
            i += self.tempref.current_interest()
        return i


class BufferCache:
    """Central buffer cache with weak and strong caches.

    Typical usage:
      cache = BufferCache()
      cache.register(buffer, checksum, size=1234)
      cache.incref(checksum)
      cache.run_eviction_once()

    The implementation intentionally keeps dependencies and subsystem hooks minimal.
    """

    def __init__(
        self,
        soft_cap: int = DEFAULT_SOFT_CAP,
        hard_cap: int = DEFAULT_HARD_CAP,
        benefit_per_gb: float = DEFAULT_BENEFIT_PER_GB,
    ) -> None:
        self.weak_cache: weakref.WeakValueDictionary[Checksum, Buffer] = (
            weakref.WeakValueDictionary()
        )
        self.strong_cache: Dict[Checksum, StrongEntry] = {}
        self.lock = threading.RLock()

        self.soft_cap = soft_cap
        self.hard_cap = hard_cap
        self.benefit_per_gb = benefit_per_gb
        # eviction background task controls
        self._eviction_task = None
        self._eviction_interval = None
        # track sizes for checksums so we can recreate entries after demotion
        self._sizes: Dict[Checksum, Optional[int]] = {}
        # track checksums that only have scratch refs
        self._scratch_refs: set[Checksum] = set()
        self.refholder_counts: Dict[Checksum, int] = {}
        self._memory_pressure_warning_at = 0.0
        self._memory_pressure_warning_state = None

    # --- registration & lookup ---
    def register(
        self,
        checksum: Checksum,
        buffer: Buffer,
        size: Optional[int] = None,
    ) -> None:
        """Register a buffer with the weak cache and move to strong if the checksum has refs.

        - checksum: unique identifier
        - buffer: object to store (can be any Python object)
        - size: optional length in bytes. If None, treated as unknown (infinite cost)
        """
        write_buffer = None
        with self.lock:
            if size is None:
                size = getattr(buffer, "length", None)
            if size is None:
                size = self._sizes.get(checksum)
            if size is not None:
                self._sizes[checksum] = size
                eviction_cost.register_buffer_length(checksum, size)
            # store in weak cache (Buffer is weakable)
            self.weak_cache[checksum] = buffer

            # If there is already a strong entry (refs exist), move/update it
            entry = self.strong_cache.get(checksum)
            if entry is not None:
                if buffer is not None and entry.buffer is None:
                    if entry.remote_registered:
                        write_buffer = buffer
                entry.buffer = buffer
                entry.size = size
        if write_buffer is not None:
            buffer_writer.register(write_buffer)

    def get(self, checksum: Checksum) -> Optional[Buffer]:
        """Return buffer if present in strong or weak caches (promotes to strong if refs exist)."""
        with self.lock:
            entry = self.strong_cache.get(checksum)
            if entry is not None and entry.buffer is not None:
                return entry.buffer
            # try weak cache
            buf = self.weak_cache.get(checksum)
            if buf is None:
                return None
            # if this checksum currently has interest, promote to strong
            if checksum in self.strong_cache:
                self.strong_cache[checksum].buffer = buf
                return buf
            return buf

    # --- refs management ---
    def _ensure_entry_locked(
        self,
        checksum: Checksum,
        *,
        buffer: Optional[Buffer] = None,
        scratch: bool = False,
    ) -> tuple[StrongEntry, Optional[Buffer]]:
        """Return the entry and any buffer to register after releasing the lock.

        Writer registration may import modules being imported by a thread whose
        GC finalizers need this lock. Complete reference accounting first, then
        register outside the lock to avoid that lock cycle.
        """

        write_buffer = None
        write_remote = not scratch
        if buffer is None:
            buffer = self.weak_cache.get(checksum)
        if buffer is not None:
            buffer.get_checksum()
            assert buffer.checksum == checksum
        entry = self.strong_cache.get(checksum)
        if entry is None:
            size = self._sizes.get(checksum)
            if size is None and buffer is not None:
                size = getattr(buffer, "length", None)
            if size is not None:
                self._sizes[checksum] = size
                eviction_cost.register_buffer_length(checksum, size)
            entry = StrongEntry(buffer=buffer, size=size)
            entry.remote_registered = write_remote
            if buffer is not None and write_remote:
                write_buffer = buffer
            self.strong_cache[checksum] = entry
            eviction_cost.add_interest(checksum)
        elif entry.buffer is None and buffer is not None:
            entry.buffer = buffer
            if write_remote:
                entry.remote_registered = True
                write_buffer = buffer
        if write_remote and not entry.remote_registered:
            entry.remote_registered = True
            if entry.buffer is not None:
                write_buffer = entry.buffer
        return entry, write_buffer

    def _has_live_tempref_locked(self, entry: StrongEntry) -> bool:
        if entry.tempref is None:
            return False
        return entry.tempref.current_interest() >= TEMPREF_MINIMAL_INTEREST

    def _can_demote_locked(self, entry: StrongEntry) -> bool:
        return (
            entry.manual_refs == 0
            and not entry.has_refholder_bridge
            and not self._has_live_tempref_locked(entry)
        )

    def _eviction_protected_locked(self, entry: StrongEntry) -> bool:
        """Return whether an entry is protected from memory-pressure eviction."""

        return entry.manual_refs != 0 or entry.has_refholder_bridge

    def _demote_locked(self, checksum: Checksum, entry: StrongEntry) -> bool:
        if not self._can_demote_locked(entry):
            return False
        buf = entry.buffer
        if buf is not None:
            self.weak_cache[checksum] = buf
        if self.strong_cache.get(checksum) is entry:
            del self.strong_cache[checksum]
            eviction_cost.remove_interest(checksum)
        return True

    def incref(
        self,
        checksum: Checksum,
        *,
        buffer: Optional[Buffer] = None,
        scratch: bool = False,
    ) -> None:
        """Increment normal refcount for checksum. Creates a strong entry if needed.

        If scratch is True, keep the ref scratch-only (no remote registration).
        """
        with self.lock:
            if scratch:
                self._scratch_refs.add(checksum)
            else:
                self._scratch_refs.discard(checksum)
            entry, write_buffer = self._ensure_entry_locked(
                checksum, buffer=buffer, scratch=scratch
            )
            entry.manual_refs += 1
        if write_buffer is not None:
            buffer_writer.register(write_buffer)

    def decref(self, checksum: Checksum) -> bool:
        """Decrement a normal refcount and report whether one was held."""
        with self.lock:
            entry = self.strong_cache.get(checksum)
            if entry is None or entry.manual_refs == 0:
                _references_logger.warning(
                    "Manual decref ignored for checksum %s: manual refcount is already zero",
                    checksum.hex(),
                )
                return False
            entry.manual_refs -= 1
            self._demote_locked(checksum, entry)
            return True

    def incref_refholder(
        self,
        checksum: Checksum,
        *,
        buffer: Optional[Buffer] = None,
        scratch: bool = False,
    ) -> None:
        """Acquire one logical lifecycle reference for ``checksum``."""

        with self.lock:
            if scratch:
                self._scratch_refs.add(checksum)
            else:
                self._scratch_refs.discard(checksum)
            entry, write_buffer = self._ensure_entry_locked(
                checksum, buffer=buffer, scratch=scratch
            )
            count = self.refholder_counts.get(checksum, 0)
            if count == 0:
                entry.has_refholder_bridge = True
            self.refholder_counts[checksum] = count + 1
        if write_buffer is not None:
            buffer_writer.register(write_buffer)

    def decref_refholder(self, checksum: Checksum) -> bool:
        """Release one logical lifecycle reference for ``checksum``."""

        with self.lock:
            count = self.refholder_counts.get(checksum, 0)
            entry = self.strong_cache.get(checksum)
            if count == 0:
                _references_logger.warning(
                    "Refholder decref ignored for checksum %s: refholder count is already zero",
                    checksum.hex(),
                )
                return False
            if count == 1:
                del self.refholder_counts[checksum]
                if entry is not None:
                    entry.has_refholder_bridge = False
                    self._demote_locked(checksum, entry)
            else:
                self.refholder_counts[checksum] = count - 1
            return True

    def reference_snapshot(self) -> dict[Checksum, tuple[int, int, bool]]:
        """Return lifecycle accounting copied while holding the cache lock."""

        with self.lock:
            checksums = set(self.refholder_counts)
            checksums.update(self.strong_cache)
            return {
                checksum: (
                    self.refholder_counts.get(checksum, 0),
                    self.strong_cache.get(checksum).manual_refs
                    if checksum in self.strong_cache
                    else 0,
                    self.strong_cache.get(checksum).has_refholder_bridge
                    if checksum in self.strong_cache
                    else False,
                )
                for checksum in checksums
            }

    def force_clear_reference_accounting(self) -> None:
        """Clear residual lifecycle accounting during final shutdown cleanup."""

        with self.lock:
            self.refholder_counts.clear()
            for checksum, entry in list(self.strong_cache.items()):
                # Manual references are deliberately part of the final forced
                # cleanup.  They are warned about by the shutdown audit before
                # this method is called, but must not keep the cache alive.
                entry.manual_refs = 0
                entry.has_refholder_bridge = False
                self._demote_locked(checksum, entry)

    def tempref(
        self,
        checksum: Checksum,
        *,
        buffer: Optional[Buffer] = None,
        interest: float = 128.0,
        fade_factor: float = 2.0,
        fade_interval: float = 2.0,
        scratch: bool = False,
    ) -> TempRef:
        """Add or refresh a single tempref for checksum. Only one tempref allowed per checksum.

        If scratch is True, keep the tempref scratch-only (no remote registration).
        """
        with self.lock:
            if scratch:
                self._scratch_refs.add(checksum)
            else:
                self._scratch_refs.discard(checksum)
            entry, write_buffer = self._ensure_entry_locked(
                checksum, buffer=buffer, scratch=scratch
            )
            if entry.tempref is None:
                entry.tempref = TempRef(
                    interest=interest,
                    fade_factor=fade_factor,
                    fade_interval=fade_interval,
                )
                entry.tempref_scratch = scratch
            else:
                eref = entry.tempref
                if (
                    interest > eref.interest
                    or fade_factor / fade_interval
                    > eref.fade_factor / eref.fade_interval
                ):
                    entry.tempref.interest = interest
                    entry.tempref.fade_factor = fade_factor
                    entry.tempref.fade_interval = fade_interval
                entry.tempref.refresh()
                entry.tempref_scratch = scratch
        if write_buffer is not None:
            buffer_writer.register(write_buffer)
        return entry.tempref

    def purge_scratch(self, checksum: Checksum | None = None) -> int:
        """Drop strong/weak cache entries held only by scratch temprefs.

        Returns the number of entries purged.
        """
        purged = 0
        with self.lock:
            if checksum is not None:
                checksums = [checksum]
            else:
                checksums = list(self._scratch_refs)
            for cs in checksums:
                entry = self.strong_cache.get(cs)
                if entry is None:
                    if cs in self.weak_cache:
                        try:
                            del self.weak_cache[cs]
                        except KeyError:
                            pass
                        purged += 1
                    continue
                if entry.manual_refs != 0 or entry.has_refholder_bridge:
                    continue
                if entry.tempref is None or not entry.tempref_scratch:
                    continue
                if cs in self.weak_cache:
                    try:
                        del self.weak_cache[cs]
                    except KeyError:
                        pass
                del self.strong_cache[cs]
                eviction_cost.remove_interest(cs)
                purged += 1
        return purged

    def is_scratch_ref(self, checksum: Checksum) -> bool:
        """Return True if checksum is tracked as scratch-only."""
        return checksum in self._scratch_refs

    def refresh_tempref(self, checksum: Checksum) -> None:
        with self.lock:
            entry = self.strong_cache.get(checksum)
            if entry is not None and entry.tempref is not None:
                entry.tempref.refresh()

    # --- eviction ---
    def _strong_memory_usage(self) -> int:
        total = 0
        for entry in self.strong_cache.values():
            if entry.size:
                total += int(entry.size)
        return total

    def _candidate_score(self, checksum: Checksum, entry: StrongEntry) -> float:
        """Return cost-per-GB score used to pick eviction candidates.

        Lower scores are evicted first. If size unknown or zero, return +inf to avoid eviction.
        """
        if entry.size is None or entry.size <= 0:
            return float("inf")
        size_gb = entry.size / (1024**3)
        if size_gb <= 0:
            return float("inf")
        cost = eviction_cost.get_cost(checksum, buffer_length=entry.size)
        if cost == float("inf"):
            return float("inf")
        loss = cost * entry.interest()
        # cost-per-GB
        return loss / size_gb

    def run_eviction_once(self) -> Tuple[int, int]:
        """Run a single eviction pass.

        Returns (before_bytes, after_bytes) strong-cache totals.
        """
        with self.lock:
            for k, e in list(self.strong_cache.items()):
                if e.tempref is None:
                    continue
                if e.tempref.current_interest() < TEMPREF_MINIMAL_INTEREST:
                    e.tempref = None
                    self._demote_locked(k, e)

            before = self._strong_memory_usage()
            if before <= self.soft_cap:
                return before, before

            # Build candidate list (checksum, score). Manual and refholder
            # protection never becomes a candidate; a tempref is bounded cache
            # interest and can be dropped under memory pressure.
            candidates = []
            for k, e in self.strong_cache.items():
                if self._eviction_protected_locked(e):
                    continue
                score = self._candidate_score(k, e)
                candidates.append((score, k, e))

            # Sort by score ascending (lowest cost-per-GB first)
            candidates.sort(key=lambda x: x[0])

            # Evict until under caps as required
            current = before
            i = 0
            while current > self.soft_cap and i < len(candidates):
                score, k, e = candidates[i]
                i += 1
                # If score is +inf, skip (unknown sizes)
                if score == float("inf"):
                    continue
                # Eviction discards bounded tempref interest before applying
                # the common demotion predicate.
                e.tempref = None
                # subtract size
                if e.size:
                    current -= int(e.size)
                self._demote_locked(k, e)
                # If we are above the hard cap, keep evicting aggressively (loop continues)

            if current > self.soft_cap:
                protected = sum(
                    self._eviction_protected_locked(entry)
                    for entry in self.strong_cache.values()
                )
                state = (before, current, self.soft_cap, protected)
                now = time.time()
                if (
                    state != self._memory_pressure_warning_state
                    or now - self._memory_pressure_warning_at
                    >= _MEMORY_PRESSURE_WARNING_INTERVAL
                ):
                    _references_logger.warning(
                        "Memory pressure remains above cap: %d -> %d bytes, cap %d, %d protected entries",
                        before,
                        current,
                        self.soft_cap,
                        protected,
                    )
                    self._memory_pressure_warning_state = state
                    self._memory_pressure_warning_at = now

            after = max(0, current)
            return before, after

    # --- background eviction loop ---
    async def _eviction_worker(self, interval: float) -> None:
        """Background coroutine that periodically runs eviction."""
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    self.run_eviction_once()
                except Exception:
                    # swallow exceptions to keep the worker alive
                    pass
        except asyncio.CancelledError:
            return

    async def start_eviction_loop(self, interval: float = 5.0) -> None:
        """Start a background asyncio Task that runs eviction every `interval` seconds.

        If already running, this is a no-op.
        """
        with self.lock:
            if self._eviction_task is not None and not self._eviction_task.done():
                return
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._eviction_worker(interval))
            self._eviction_task = task
            self._eviction_interval = interval

    async def stop_eviction_loop(self) -> None:
        """Stop the background eviction task."""
        task = None
        with self.lock:
            task = self._eviction_task
            self._eviction_task = None
            self._eviction_interval = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # --- utilities & hooks ---
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            strong = self._strong_memory_usage()
            weak_count = len(self.weak_cache)
            strong_count = len(self.strong_cache)
            return {
                "strong_bytes": strong,
                "strong_count": strong_count,
                "weak_count": weak_count,
                "soft_cap": self.soft_cap,
                "hard_cap": self.hard_cap,
            }


# Module-level cache instance
_cache_instance = None


def get_buffer_cache() -> BufferCache:
    """Get or create the global buffer cache instance."""
    if is_worker():
        raise RuntimeError(
            "Buffer cache is not available inside a Seamless worker process"
        )
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = BufferCache()
    return _cache_instance


__all__ = ["BufferCache", "StrongEntry", "TempRef", "get_buffer_cache"]
