"""Reference-holder registration and shutdown-audit primitives.

This module intentionally contains procedural helpers rather than a holder
base class or an audit report model.  Live objects expose semantic claims via
``_refheld_checksums``; cache accounting remains the independent source of
acquisition counts.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections import defaultdict
from typing import Any, Iterable

from . import is_worker
from .checksum_class import Checksum


_refholders: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()
_registry_lock = threading.RLock()
_logger = logging.getLogger("seamless.references")


def register_refholder(obj: object) -> None:
    """Register a weak-referenceable lifecycle owner by object identity."""

    if is_worker():
        return
    # Assignment is deliberately allowed to raise TypeError.  A holder that
    # cannot be audited must not silently disappear from the registry.
    with _registry_lock:
        _refholders[id(obj)] = obj


def registered_refholders() -> list[object]:
    """Return a strong snapshot of currently live registered holders."""

    if is_worker():
        return []
    with _registry_lock:
        return list(_refholders.values())


def safe_release_refholder(holder: object) -> None:
    """Release a holder during finalization and report, but suppress, failures.

    Finalizers run while Python may already be tearing modules down.  Cleanup
    must therefore be best effort, but a swallowed exception is still useful
    shutdown evidence.  Normal idempotent releases do not produce a warning.
    """

    try:
        release = getattr(holder, "_release_refholds", None)
        if release is not None:
            release()
    except Exception as exc:
        try:
            _logger.warning(
                "Refholder %s 0x%x cleanup raised: %s",
                type(holder).__name__,
                id(holder),
                exc,
            )
        except Exception:
            # Logging itself may be unavailable during interpreter teardown.
            pass


def _claim_sort_key(item: tuple[object, str]) -> tuple[str, str, int]:
    holder, role = item
    return (type(holder).__name__, role, id(holder))


def collect_refholder_claims(
    holders: Iterable[object] | None = None,
) -> dict[Checksum, list[tuple[object, str]]]:
    """Build a checksum reverse index from semantic holder state.

    Claim failures are isolated and warned about so one damaged finalizer or
    partially initialized object cannot suppress the rest of shutdown audit.
    """

    if holders is None:
        holders = registered_refholders()
    claims: dict[Checksum, list[tuple[object, str]]] = defaultdict(list)
    for holder in holders:
        try:
            entries = holder._refheld_checksums()
            for checksum, role in entries:
                if not isinstance(checksum, Checksum):
                    raise TypeError(
                        f"claim checksum must be Checksum, got {type(checksum)!r}"
                    )
                claims[checksum].append((holder, str(role)))
        except Exception as exc:
            _logger.warning(
                "Refholder %s 0x%x claim collection raised: %s",
                type(holder).__name__,
                id(holder),
                exc,
            )
    for checksum in claims:
        claims[checksum].sort(key=_claim_sort_key)
    return dict(claims)


def _claim_lines(claims: Iterable[tuple[object, str]]) -> list[str]:
    return [
        f"  {type(holder).__name__} 0x{id(holder):x} {role}"
        for holder, role in sorted(claims, key=_claim_sort_key)
    ]


def audit_reference_accounting(
    *,
    cache: Any = None,
    holders: Iterable[object] | None = None,
    warn_manual: bool = True,
    warned_manual: set[Checksum] | None = None,
) -> None:
    """Compare cache refholder accounting with live semantic claims.

    The audit is deliberately side-effect free with respect to holder state;
    callers perform cleanup in a separate phase.  It returns no report object.
    """

    if is_worker():
        return
    if cache is None:
        from .caching.buffer_cache import get_buffer_cache

        cache = get_buffer_cache()

    claim_index = collect_refholder_claims(holders)
    accounting = cache.reference_snapshot()
    checksums = set(accounting)
    checksums.update(claim_index)
    for checksum in sorted(checksums, key=lambda value: value.hex()):
        count, manual_refs, bridge = accounting.get(checksum, (0, 0, False))
        checksum_claims = claim_index.get(checksum, [])
        claim_count = len(checksum_claims)
        if count > claim_count:
            _logger.warning(
                "Checksum %s has refholder count %d but only %d live claims; %d is unattributed\n%s",
                checksum.hex(),
                count,
                claim_count,
                count - claim_count,
                "\n".join(_claim_lines(checksum_claims)),
            )
        elif claim_count > count:
            lines = "\n".join(_claim_lines(checksum_claims))
            _logger.warning(
                "Checksum %s has refholder count %d but %d live claims\n%s\n"
                "  A reference was not acquired or was released by code that did not own it",
                checksum.hex(),
                count,
                claim_count,
                lines,
            )
        if bridge != (count > 0):
            _logger.warning(
                "Checksum %s has refholder count %d but its eviction bridge is %s",
                checksum.hex(),
                count,
                "present" if bridge else "absent",
            )
        if warn_manual and manual_refs > 0 and (
            warned_manual is None or checksum not in warned_manual
        ):
            _logger.warning(
                "Checksum %s has %d unmatched manual references at shutdown",
                checksum.hex(),
                manual_refs,
            )
            if warned_manual is not None:
                warned_manual.add(checksum)


def clear_refholder_registry_for_tests() -> None:
    """Clear the weak registry for isolated unit tests."""

    with _registry_lock:
        _refholders.clear()


__all__ = [
    "audit_reference_accounting",
    "clear_refholder_registry_for_tests",
    "collect_refholder_claims",
    "register_refholder",
    "registered_refholders",
    "safe_release_refholder",
]
