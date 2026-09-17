import logging

import pytest

from seamless.checksum_class import Checksum
from seamless.caching.buffer_cache import BufferCache
from seamless.reference_lifecycle import (
    audit_reference_accounting,
    clear_refholder_registry_for_tests,
    collect_refholder_claims,
    register_refholder,
)


class SemanticHolder:
    def __init__(self, *checksums):
        self.checksums = list(checksums)
        self.released = False

    def _refheld_checksums(self):
        if self.released:
            return []
        return [(checksum, f"role:{index}") for index, checksum in enumerate(self.checksums)]

    def _release_refholds(self):
        self.released = True


class FailingHolder:
    def _refheld_checksums(self):
        raise RuntimeError("broken claim state")


@pytest.fixture(autouse=True)
def clean_registry():
    clear_refholder_registry_for_tests()
    yield
    clear_refholder_registry_for_tests()


def test_registry_is_weak_and_identity_safe():
    checksum = Checksum("1" * 64)
    first = SemanticHolder(checksum)
    second = SemanticHolder(checksum)
    register_refholder(first)
    register_refholder(second)
    assert len(collect_refholder_claims()) == 1
    assert len(collect_refholder_claims()[checksum]) == 2
    del first
    import gc

    gc.collect()
    assert len(collect_refholder_claims()[checksum]) == 1


def test_non_weak_referenceable_holder_fails_loudly():
    with pytest.raises(TypeError):
        register_refholder([])


def test_audit_distinguishes_count_excess_and_claim_excess(caplog):
    checksum = Checksum("2" * 64)
    cache = BufferCache()
    holder = SemanticHolder(checksum)
    register_refholder(holder)
    cache.incref_refholder(checksum)

    with caplog.at_level(logging.WARNING, logger="seamless.references"):
        audit_reference_accounting(cache=cache, holders=[])
    assert "only 0 live claims" in caplog.text
    caplog.clear()

    cache.decref_refholder(checksum)
    cache.incref_refholder(checksum)
    second = SemanticHolder(checksum)
    register_refholder(second)
    with caplog.at_level(logging.WARNING, logger="seamless.references"):
        audit_reference_accounting(cache=cache)
    assert "but 2 live claims" in caplog.text


def test_failing_claim_does_not_abort_other_holders(caplog):
    checksum = Checksum("3" * 64)
    good = SemanticHolder(checksum)
    broken = FailingHolder()
    register_refholder(good)
    register_refholder(broken)
    with caplog.at_level(logging.WARNING, logger="seamless.references"):
        claims = collect_refholder_claims()
    assert claims[checksum] == [(good, "role:0")]
    assert "claim collection raised" in caplog.text
