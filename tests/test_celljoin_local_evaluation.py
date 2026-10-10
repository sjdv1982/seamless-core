"""Shared local execution, member claims and celljoin/Expression cache coexistence."""
import asyncio
import gc
from threading import Event

import pytest
from seamless import Buffer, Checksum, Expression
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
from seamless.checksum import celljoin as celljoin_mod
from seamless.checksum import expression as expression_mod


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    expression_mod.get_expression_cache().clear()
    monkeypatch.setattr(expression_mod, '_EXPRESSION_LINGER', 0.15)
    yield
    expression_mod.get_expression_cache().clear()


def make_spec(*, deep=False, root=True):
    celltype = 'deepcell' if deep else 'plain'
    root_buffer = Buffer({'kept': '22' * 32} if deep else {'kept': 1}, celltype)
    member = Buffer(7, 'mixed' if deep else celltype)
    root_buffer.tempref()
    member.tempref()
    spec = celljoin_mod.parse_celljoin(
        celljoin_mod.build_celljoin(root_buffer.get_checksum() if root else None,
                                   {'a': member.get_checksum()}), celltype)
    return spec, [root_buffer, member]


def key(spec):
    return celljoin_mod.celljoin_cache_key(spec.checksum, spec.celltype)


def drop_buffer(checksum):
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)


def test_cache_hit_skips_evaluator_and_placement(monkeypatch):
    spec, held = make_spec()
    result = asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec))
    def forbidden(*args, **kwargs):
        raise AssertionError('cached joins must not evaluate again')
    monkeypatch.setattr(celljoin_mod, 'evaluate_celljoin', forbidden)
    assert asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec)) == result
    assert asyncio.run(celljoin_mod.evaluate_celljoin_placed(spec)) == result
    assert expression_mod.get_expression_cache()[key(spec)] == result


def test_two_callers_share_evaluation_softcancel_and_input_claims(monkeypatch):
    spec, held = make_spec()
    original = celljoin_mod.evaluate_celljoin
    started, release = Event(), Event()
    calls = []
    def gated(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(5), 'test gate timed out'
        return original(*args, **kwargs)
    monkeypatch.setattr(celljoin_mod, 'evaluate_celljoin', gated)
    async def main():
        one = asyncio.create_task(celljoin_mod.evaluate_celljoin_local_async(spec, member_id=101))
        two = asyncio.create_task(celljoin_mod.evaluate_celljoin_local_async(spec, member_id=202))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await asyncio.sleep(0)
            active = expression_mod._active_expressions[key(spec)]
            assert active.members == {101, 202}
            expected = {(checksum, 'expression materialization') for checksum in celljoin_mod.required_buffers(spec)}
            assert set(active._refheld_checksums()) == expected
            assert expression_mod.softcancel_expression(key(spec), 101) is True
            with pytest.raises(asyncio.CancelledError):
                await one
            assert active.members == {202}
            release.set()
            result = await asyncio.wait_for(two, 5)
            await asyncio.sleep(0)
            assert active._refheld_checksums() == ()
            assert key(spec) not in expression_mod._active_expressions
            assert key(spec) not in expression_mod._lingering_expressions
            return result
        finally:
            release.set()
    assert asyncio.run(main()).resolve('plain') == {'kept': 1, 'a': 7}
    assert len(calls) == 1


def test_last_member_leaves_then_rejoins_lingering_work(monkeypatch):
    spec, held = make_spec()
    original = celljoin_mod._gather_celljoin_buffers
    async def main():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def gated(*args, **kwargs):
            calls.append(1)
            started.set()
            await release.wait()
            return await original(*args, **kwargs)
        monkeypatch.setattr(celljoin_mod, '_gather_celljoin_buffers', gated)
        first = asyncio.create_task(celljoin_mod.evaluate_celljoin_local_async(spec, member_id=101))
        await asyncio.wait_for(started.wait(), 5)
        active = expression_mod._active_expressions[key(spec)]
        assert expression_mod.softcancel_expression(key(spec), 101)
        with pytest.raises(asyncio.CancelledError):
            await first
        assert expression_mod._lingering_expressions[key(spec)] is active
        second = asyncio.create_task(celljoin_mod.evaluate_celljoin_local_async(spec, member_id=202))
        await asyncio.sleep(0.3)
        assert not active.task.done(), 'rejoining must cancel the linger expiry'
        assert expression_mod._active_expressions[key(spec)] is active
        release.set()
        result = await asyncio.wait_for(second, 5)
        assert len(calls) == 1
        return result
    assert asyncio.run(main()).resolve('plain') == {'kept': 1, 'a': 7}


def test_linger_expiry_cancels_work_releases_claims_and_records_nothing(monkeypatch):
    spec, held = make_spec()
    async def main():
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def blocked(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        monkeypatch.setattr(celljoin_mod, '_gather_celljoin_buffers', blocked)
        caller = asyncio.create_task(celljoin_mod.evaluate_celljoin_local_async(spec, member_id=101))
        await asyncio.wait_for(started.wait(), 5)
        active = expression_mod._active_expressions[key(spec)]
        assert expression_mod.softcancel_expression(key(spec), 101)
        with pytest.raises(asyncio.CancelledError):
            await caller
        await asyncio.wait_for(cancelled.wait(), 5)
        await asyncio.sleep(0)
        assert active._refheld_checksums() == ()
        assert key(spec) not in expression_mod.get_expression_cache()
        assert key(spec) not in expression_mod._lingering_expressions
    asyncio.run(main())


def test_failure_is_not_cached_and_retry_evaluates_again(monkeypatch):
    spec, held = make_spec()
    original = celljoin_mod.evaluate_celljoin
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('temporary failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(celljoin_mod, 'evaluate_celljoin', fail_once)
    async def main():
        with pytest.raises(ValueError, match='temporary failure'):
            await celljoin_mod.evaluate_celljoin_local_async(spec)
        assert key(spec) not in expression_mod.get_expression_cache()
        return await celljoin_mod.evaluate_celljoin_local_async(spec)
    assert asyncio.run(main()).resolve('plain')['a'] == 7
    assert len(calls) == 2


def test_rootless_deep_is_free_and_never_joins_member_set(monkeypatch):
    spec, held = make_spec(deep=True, root=False)
    original = celljoin_mod.evaluate_celljoin
    def check_free(*args, **kwargs):
        assert key(spec) not in expression_mod._active_expressions
        assert key(spec) not in expression_mod._lingering_expressions
        return original(*args, **kwargs)
    monkeypatch.setattr(celljoin_mod, 'evaluate_celljoin', check_free)
    result = asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec))
    assert result.resolve('deepcell') == {'a': held[1].get_checksum()}
    assert expression_mod.get_expression_cache()[key(spec)] == result


def test_materialize_recomputes_cached_result_missing_its_buffer(monkeypatch):
    spec, held = make_spec()
    result = asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec))
    drop_buffer(result)
    assert not expression_mod._has_local_buffer(result)
    original = celljoin_mod.evaluate_celljoin
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(celljoin_mod, 'evaluate_celljoin', counted)
    assert asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec)) == result
    assert calls == []
    assert asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec, materialize=True)) == result
    assert calls == [1]
    assert result.resolve('plain') == {'kept': 1, 'a': 7}


def test_preloaded_buffers_are_used_without_resolution(monkeypatch):
    spec, held = make_spec()
    preloaded = {buffer.get_checksum(): buffer for buffer in held}
    async def forbidden(*args, **kwargs):
        raise AssertionError('preloaded inputs must not be resolved')
    monkeypatch.setattr(Checksum, 'resolution', forbidden)
    result = asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec, buffers=preloaded))
    assert result.resolve('plain')['a'] == 7


def test_expression_fingertip_ignores_celljoin_cache_keys():
    source = Buffer({'a': 'hello'}, 'plain')
    source.tempref()
    expression = Expression(source.get_checksum(), 'a', input_celltype='plain', celltype='str')
    result = expression.compute()
    expression_mod.get_expression_cache()[('celljoin', '11' * 32, 'plain')] = result
    drop_buffer(result)
    assert asyncio.run(result.fingertip('str')) == 'hello'


@pytest.mark.parametrize('divergent', [False, True], ids=['same-result', 'different-result'])
def test_recomputed_buffers_survive_and_are_scratch_even_when_mapping_exists(monkeypatch, divergent):
    from seamless.checksum.hash_type import get_hash_type, get_hash_type_cache

    spec, held = make_spec()
    recorded = Buffer({'recorded-stage2': 77}, 'plain').get_checksum()
    drop_buffer(recorded)
    expression_mod.get_expression_cache()[key(spec)] = recorded
    value = {'recorded-stage2': 78 if divergent else 77}
    produced_checksum = Buffer(value, 'plain').get_checksum()
    drop_buffer(produced_checksum)
    get_hash_type_cache().pop(produced_checksum, None)
    cache = get_buffer_cache()
    # Remove prior test state so this recompute must establish producer status.
    with cache.lock:
        cache._scratch_refs.discard(produced_checksum)
    assert not cache.is_scratch_ref(produced_checksum)

    def recompute(*args, **kwargs):
        buffer = Buffer(value, 'plain')
        # Construction must not mask the evaluator's responsibility to register
        # the produced buffer's HashType and retain it after this frame exits.
        get_hash_type_cache().pop(buffer.get_checksum(), None)
        return buffer

    monkeypatch.setattr(celljoin_mod, 'evaluate_celljoin', recompute)
    result = asyncio.run(celljoin_mod.evaluate_celljoin_local_async(spec, materialize=True))
    assert result == produced_checksum
    assert expression_mod.get_expression_cache()[key(spec)] == recorded
    gc.collect()
    assert result.resolve('plain') == value
    assert get_hash_type(result) is not None
    assert cache.is_scratch_ref(result)
