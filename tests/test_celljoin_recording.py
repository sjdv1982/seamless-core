"""Publication, database reuse and immutable celljoin result recording."""
import asyncio
import gc
from unittest.mock import AsyncMock
import pytest
from seamless import Buffer, Checksum
from seamless.celljoin_class import CellJoin
from seamless.checksum import celljoin as joins
from seamless.checksum import expression
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless_remote import buffer_remote, database_remote

@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    expression.get_expression_cache().clear()
    monkeypatch.setattr(database_remote, 'has_write_server', lambda: False)
    yield
    expression.get_expression_cache().clear()

@pytest.mark.parametrize('hash_write,db_write', [(False, False), (True, False), (False, True), (True, True)])
def test_from_inputs_publishes_definition_only_with_both_write_services(monkeypatch, hash_write, db_write):
    calls = []
    monkeypatch.setattr(buffer_remote, 'has_write_server', lambda: hash_write)
    monkeypatch.setattr(database_remote, 'has_write_server', lambda: db_write)
    monkeypatch.setattr(Buffer, 'transfer_write', lambda self: calls.append(self))
    join = CellJoin.from_inputs('deepfolder', None, {'file': Checksum('d' * 64)})
    assert len(calls) == int(hash_write and db_write)
    if calls:
        assert calls[0].get_checksum() == join.celljoin_checksum
        assert calls[0].get_value('plain') == {'file': 'd' * 64}

def spec(celltype='plain'):
    return joins.parse_celljoin({'member': 'e' * 64}, celltype)

def test_database_hit_requires_no_input_buffers_and_caches_by_celltype(monkeypatch):
    result = Checksum('f' * 64)
    lookup = AsyncMock(return_value=result)
    monkeypatch.setattr(database_remote, 'get_celljoin_result', lookup)
    async def unexpected(*args, **kwargs):
        raise AssertionError('recorded result must bypass input resolution')
    monkeypatch.setattr(joins, 'evaluate_celljoin_local_async', unexpected)
    for celltype in ('plain', 'mixed'):
        item = spec(celltype)
        assert asyncio.run(joins.evaluate_celljoin_placed(item)) == result
        assert asyncio.run(joins.evaluate_celljoin_placed(item)) == result
        assert expression.get_expression_cache()[joins.celljoin_cache_key(item.checksum, celltype)] == result
    assert lookup.await_count == 2
    assert [call.args[1] for call in lookup.await_args_list] == ['plain', 'mixed']

def test_database_miss_evaluates_and_records_local_result(monkeypatch):
    item = joins.parse_celljoin({}, 'deepcell')
    lookup = AsyncMock(return_value=None)
    writes = []
    monkeypatch.setattr(database_remote, 'get_celljoin_result', lookup)
    monkeypatch.setattr(buffer_writer, 'register_celljoin_result', lambda *args: writes.append(args))
    result = asyncio.run(joins.evaluate_celljoin_placed(item))
    assert result.resolve('deepcell') == {}
    assert writes == [(item.checksum.hex(), 'deepcell', result)]
    assert asyncio.run(joins.evaluate_celljoin_placed(item)) == result
    assert lookup.await_count == 1
    assert len(writes) == 1

@pytest.mark.parametrize('divergent', [False, True])
def test_duplicate_record_does_not_rewrite_mapping_but_retains_produced_buffer(monkeypatch, divergent):
    item = spec()
    writes = []
    monkeypatch.setattr(buffer_writer, 'register_celljoin_result', lambda *args: writes.append(args))
    first = Buffer({'recorded': 1}, 'plain')
    recorded = first.get_checksum()
    assert joins._record_celljoin_result(item, recorded, first) is True
    produced = Buffer({'recorded': 2 if divergent else 1}, 'plain')
    checksum = produced.get_checksum()
    assert joins._record_celljoin_result(item, checksum, produced) is False
    del produced
    gc.collect()
    assert checksum.resolve('plain') == {'recorded': 2 if divergent else 1}
    assert get_buffer_cache().is_scratch_ref(checksum)
    assert expression.get_expression_cache()[joins.celljoin_cache_key(item.checksum, 'plain')] == recorded
    assert writes == [(item.checksum.hex(), 'plain', recorded)]

def test_writer_metadata_preserves_celltype_identity_and_is_nonfatal_on_refusal(monkeypatch, caplog):
    buffer_writer.flush()
    calls = []
    monkeypatch.setattr(database_remote, 'has_write_server', lambda: True)
    async def store(checksum, celltype, result):
        calls.append((Checksum(checksum), celltype, Checksum(result)))
        return False if celltype == 'plain' else True
    monkeypatch.setattr(database_remote, 'set_celljoin_result', store)
    definition, result = Checksum('a' * 64), Checksum('b' * 64)
    buffer_writer.register_celljoin_result(definition.hex(), 'plain', result)
    buffer_writer.register_celljoin_result(definition.hex(), 'mixed', result)
    buffer_writer.flush(timeout=5)
    assert set(calls) == {(definition, 'plain', result), (definition, 'mixed', result)}
    assert 'celljoin' in caplog.text
