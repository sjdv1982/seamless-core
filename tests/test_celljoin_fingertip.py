"""Recovery walks celljoin recipes, while ordinary evaluation only resolves."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionError
from seamless import Buffer, CacheMissError, Checksum, Expression, FingertipCategory as Category
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
from seamless.checksum import celljoin as joins, expression
from seamless.error_envelope import ExecutionCanceledError
from seamless_remote import buffer_remote, database_remote


def drop(checksum):
    checksum = Checksum(checksum)
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression._expression_result_buffers.pop(checksum, None)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    expression.get_expression_cache().clear()
    monkeypatch.setattr(database_remote, 'has_read_database', lambda: False)
    monkeypatch.setattr(database_remote, 'has_write_server', lambda: False)
    monkeypatch.setattr(buffer_remote, 'get_buffer', AsyncMock(return_value=None))
    yield
    expression.get_expression_cache().clear()


def recipe(member=None, celltype='plain', root=None):
    held = []
    if member is None:
        member = Buffer(73129, 'plain')
        member.tempref(); held.append(member)
        member = member.get_checksum()
    spec = joins.parse_celljoin(joins.build_celljoin(root, {'a': member}), celltype)
    spec.buffer.tempref()
    held.append(spec.buffer)
    return spec, held


def seed(spec, result):
    expression.get_expression_cache()[joins.celljoin_cache_key(spec.checksum, spec.celltype)] = result


def category(result):
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(result.fingertip())
    assert caught.value.checksum == result
    return caught.value.fingertip_category


def test_missing_result_recovers_locally_without_publication(monkeypatch):
    spec, held = recipe()
    result = asyncio.run(joins.evaluate_celljoin_local_async(spec))
    drop(result)
    writes = []
    monkeypatch.setattr(Buffer, 'transfer_write', lambda *args: writes.append(args))
    monkeypatch.setattr(buffer_writer, 'register_celljoin_result', lambda *args: writes.append(args))
    monkeypatch.setattr(joins, 'evaluate_celljoin_placed', AsyncMock(side_effect=AssertionError('dispatch forbidden')))
    assert asyncio.run(result.fingertip('plain')) == {'a': 73129}
    assert get_buffer_cache().is_scratch_ref(result)
    assert writes == []


def test_database_and_process_candidates_deduplicate_exact_identity(monkeypatch):
    spec, held = recipe()
    wanted = Checksum('17' * 32)
    seed(spec, wanted)
    monkeypatch.setattr(database_remote, 'has_read_database', lambda: True)
    monkeypatch.setattr(database_remote, 'get_rev_transformations', AsyncMock(return_value=[]))
    monkeypatch.setattr(database_remote, 'get_rev_expressions', AsyncMock(return_value=[]))
    monkeypatch.setattr(database_remote, 'get_rev_celljoins', AsyncMock(return_value=[
        {'checksum': spec.checksum.hex(), 'celltype': 'plain'},
        {'checksum': spec.checksum.hex(), 'celltype': 'plain'},
        {'checksum': spec.checksum.hex(), 'celltype': 'mixed'},
    ]))
    evaluate = AsyncMock(side_effect=ValueError('cannot reproduce'))
    monkeypatch.setattr(joins, 'evaluate_celljoin_local_async', evaluate)
    assert category(wanted) == Category.IRREPRODUCIBLE_EXPRESSION
    assert [call.args[0].celltype for call in evaluate.await_args_list] == ['plain', 'mixed']
    for celltype in ('plain', 'mixed'):
        assert expression.get_expression_cache()[joins.celljoin_cache_key(spec.checksum, celltype)] == wanted


def test_expression_input_recovers_recursively_and_result_feeds_expression():
    source = Buffer({'nested': 84231}, 'plain'); source.tempref()
    member = Expression(source.get_checksum(), 'nested', input_celltype='plain', celltype='plain').compute()
    spec, held = recipe(member)
    result = asyncio.run(joins.evaluate_celljoin_local_async(spec))
    output = Expression(result, 'a', input_celltype='plain', celltype='str').compute()
    for checksum in (member, result, output):
        drop(checksum)
    assert asyncio.run(output.fingertip('str')) == '84231'
    assert result.resolve('plain') == {'a': 84231}


@pytest.mark.parametrize('first', [Category.MATERIALIZATION, Category.IRREPRODUCIBLE_TRANSFORMATION])
def test_all_inputs_attempted_and_nested_maximum_retained(monkeypatch, first):
    one, two = Checksum('21' * 32), Checksum('22' * 32)
    spec = joins.parse_celljoin({'a': one.hex(), 'b': two.hex()}, 'plain'); spec.buffer.tempref()
    wanted = Checksum('23' * 32); seed(spec, wanted)
    original = Checksum.fingertip
    calls = []
    async def recover(checksum, *args, **kwargs):
        if checksum in (one, two):
            calls.append(checksum)
            await asyncio.sleep(0)
            raise CacheMissError(checksum, fingertip_category=first if checksum == one else Category.FAILED_TRANSFORMATION)
        return await original(checksum, *args, **kwargs)
    monkeypatch.setattr(Checksum, 'fingertip', recover)
    assert category(wanted) == max(first, Category.FAILED_TRANSFORMATION)
    assert set(calls) == {one, two}


@pytest.mark.parametrize('rooted', [False, True])
def test_deep_walk_only_recovers_root_index(monkeypatch, rooted):
    root = Buffer({'kept': '31' * 32}, 'deepcell'); root.tempref()
    member = Checksum('32' * 32)
    spec, held = recipe(member, 'deepcell', root.get_checksum() if rooted else None)
    result = asyncio.run(joins.evaluate_celljoin_local_async(spec)); drop(result)
    original = Checksum.fingertip
    seen = []
    async def recover(checksum, *args, **kwargs):
        seen.append(checksum)
        assert checksum != member
        return await original(checksum, *args, **kwargs)
    monkeypatch.setattr(Checksum, 'fingertip', recover)
    expected = {'a': member.hex()}
    if rooted:
        expected['kept'] = '31' * 32
    if rooted:
        assert asyncio.run(result.fingertip('deepcell')) == expected
        assert root.get_checksum() in seen
    else:
        # A rootless deep recipe and its result serialize identically. Evicting
        # the result therefore evicts the definition, which cannot be rebuilt.
        assert spec.checksum == result
        assert category(result) == Category.MATERIALIZATION
        assert seen == [result]
        # Restoring the definition restores the wanted result directly.
        spec.buffer.tempref()
        assert asyncio.run(result.fingertip('deepcell')) == expected
        assert seen == [result, result]


@pytest.mark.parametrize('malformed', [False, True])
def test_unavailable_or_invalid_definition_is_skipped_without_fingertipping_it(monkeypatch, malformed):
    definition = Buffer(b'{invalid-json') if malformed else Buffer({'a': '41' * 32}, 'plain')
    definition.tempref()
    checksum = definition.get_checksum()
    wanted = Checksum('42' * 32)
    expression.get_expression_cache()[joins.celljoin_cache_key(checksum, 'plain')] = wanted
    if not malformed:
        drop(checksum)
    original = Checksum.fingertip
    async def recover(target, *args, **kwargs):
        assert target != checksum, 'definitions have no producer and must only resolve'
        return await original(target, *args, **kwargs)
    monkeypatch.setattr(Checksum, 'fingertip', recover)
    monkeypatch.setattr(joins, 'evaluate_celljoin_local_async', AsyncMock(side_effect=AssertionError('must skip definition')))
    assert category(wanted) == Category.MATERIALIZATION


@pytest.mark.parametrize('failure', ['different', 'exception', 'miss'])
def test_recompute_failure_preserves_original_mapping_and_records_nothing(monkeypatch, failure):
    spec, held = recipe()
    wanted = Checksum('51' * 32); seed(spec, wanted)
    writes = []
    monkeypatch.setattr(buffer_writer, 'register_celljoin_result', lambda *args: writes.append(args))
    monkeypatch.setattr(database_remote, 'set_celljoin_result', AsyncMock(side_effect=AssertionError('no DB write')))
    if failure == 'different':
        # Use the real evaluator, including its insert-only recording path.
        expected = Category.IRREPRODUCIBLE_EXPRESSION
    else:
        error = ValueError('broken recipe') if failure == 'exception' else CacheMissError(spec.root or spec.members[0][1], fingertip_category=Category.FAILED_TRANSFORMATION)
        monkeypatch.setattr(joins, 'evaluate_celljoin_local_async', AsyncMock(side_effect=error))
        expected = Category.IRREPRODUCIBLE_EXPRESSION if failure == 'exception' else Category.FAILED_TRANSFORMATION
    assert category(wanted) == expected
    assert expression.get_expression_cache()[joins.celljoin_cache_key(spec.checksum, spec.celltype)] == wanted
    assert writes == []
    database_remote.set_celljoin_result.assert_not_awaited()


@pytest.mark.parametrize('where', ['input', 'evaluate'])
@pytest.mark.parametrize('error', [asyncio.CancelledError(), ExecutionCanceledError('canceled'), ClientConnectionError('offline'), TimeoutError('timeout')])
def test_cancellation_and_infrastructure_errors_propagate(monkeypatch, where, error):
    spec, held = recipe(); wanted = Checksum('61' * 32); seed(spec, wanted)
    if where == 'evaluate':
        monkeypatch.setattr(joins, 'evaluate_celljoin_local_async', AsyncMock(side_effect=error))
    else:
        original = Checksum.fingertip
        async def recover(checksum, *args, **kwargs):
            if checksum == spec.members[0][1]:
                raise error
            return await original(checksum, *args, **kwargs)
        monkeypatch.setattr(Checksum, 'fingertip', recover)
    with pytest.raises(type(error)):
        asyncio.run(wanted.fingertip())


def test_ordinary_evaluation_missing_input_does_not_fingertip(monkeypatch):
    missing = Checksum('71' * 32); spec, held = recipe(missing)
    monkeypatch.setattr(Checksum, 'fingertip', AsyncMock(side_effect=AssertionError('ordinary miss is final')))
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(joins.evaluate_celljoin_placed(spec, execution='local'))
    assert caught.value.checksum == missing
    assert caught.value.fingertip_category is None


def test_database_candidate_without_definition_cannot_recover_json(monkeypatch):
    wanted, definition = Checksum('81' * 32), Checksum('82' * 32)
    monkeypatch.setattr(database_remote, 'has_read_database', lambda: True)
    monkeypatch.setattr(database_remote, 'get_rev_transformations', AsyncMock(return_value=[]))
    monkeypatch.setattr(database_remote, 'get_rev_expressions', AsyncMock(return_value=[]))
    monkeypatch.setattr(database_remote, 'get_rev_celljoins', AsyncMock(return_value=[{'checksum': definition.hex(), 'celltype': 'plain'}]))
    monkeypatch.setattr(joins, 'evaluate_celljoin_local_async', AsyncMock(side_effect=AssertionError('DB stores identity only')))
    assert category(wanted) == Category.MATERIALIZATION
    assert expression.get_expression_cache()[joins.celljoin_cache_key(definition, 'plain')] == wanted


def test_definition_retrieved_from_hashserver_then_local_inputs_recover(monkeypatch):
    spec, held = recipe()
    result = asyncio.run(joins.evaluate_celljoin_local_async(spec))
    payload = spec.buffer.content
    drop(spec.checksum); drop(result)
    async def fetch(checksum, *args, **kwargs):
        return Buffer(payload) if Checksum(checksum) == spec.checksum else None
    remote = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(buffer_remote, 'get_buffer', remote)
    assert asyncio.run(result.fingertip('plain')) == {'a': 73129}
    assert any(Checksum(call.args[0]) == spec.checksum for call in remote.await_args_list)


def test_real_transformation_input_and_downstream_allow_input_fingertip():
    import seamless.config
    from seamless.transformer import delayed
    seamless.config.init()
    try:
        @delayed
        def producer() -> dict:
            return {'message': 'celljoin-transformation-chain'}
        producer.local = True; producer.scratch = True
        producer.celltypes.result = 'plain'
        upstream = producer()
        member = upstream.compute()
        assert isinstance(member, Checksum), upstream.exception
        spec, held = recipe(member)
        join_result = asyncio.run(joins.evaluate_celljoin_local_async(spec))
        @delayed
        def consumer(value) -> str:
            return value['a']['message'] + ':done'
        consumer.local = True; consumer.scratch = True
        consumer.celltypes.value = 'plain'
        consumer.celltypes.result = 'str'
        consumer.meta = {'allow_input_fingertip': True}
        downstream = consumer(join_result)
        output = downstream.compute()
        assert isinstance(output, Checksum), downstream.exception
        for checksum in (member, join_result, output):
            drop(checksum)
        assert asyncio.run(output.fingertip('str')) == 'celljoin-transformation-chain:done'
        assert join_result.resolve('plain') == {'a': {'message': 'celljoin-transformation-chain'}}
    finally:
        seamless.close()
