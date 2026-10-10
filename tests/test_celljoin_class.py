"""CellJoin identity and its boundary as an internal Expression kind."""
import asyncio
import copy

import pytest
from seamless import Buffer, Cell, Checksum, Expression
from seamless.celljoin_class import CellJoin
from seamless.checksum import expression as expression_mod
from seamless.error_envelope import RunningLoopRefusal


@pytest.fixture(autouse=True)
def clear_cache():
    expression_mod.get_expression_cache().clear()
    yield
    expression_mod.get_expression_cache().clear()


def make_join(celltype='plain'):
    member = Buffer(7, celltype)
    member.tempref()
    return CellJoin.from_inputs(celltype, None, {'a': member.get_checksum()})


def test_identity_database_key_and_expression_subclass():
    join = make_join()
    assert isinstance(join, Expression)
    assert join.identity_key == ('celljoin', join.celljoin_checksum.hex(), 'plain')
    assert join.database_key == (join.celljoin_checksum.hex(), 'plain')
    assert join.input_checksum == join.celljoin_checksum
    assert join.definition.get_checksum() == join.celljoin_checksum
    assert join.path == ''
    assert join.input_celltype is None
    assert join.validator is None


def test_equality_hashing_and_celltype_identity():
    member = Checksum('11' * 32)
    plain = CellJoin.from_inputs('plain', None, {'a': member})
    same = CellJoin.from_inputs('plain', None, {'a': member})
    mixed = CellJoin.from_inputs('mixed', None, {'a': member})
    expression = Expression(plain.celljoin_checksum, '', input_celltype='plain', celltype='plain')
    assert plain == same
    assert hash(plain) == hash(same)
    assert plain != mixed
    assert plain != expression
    assert expression != plain
    assert len({plain, same, mixed, expression}) == 3
    assert mixed.celljoin_checksum == plain.celljoin_checksum


@pytest.mark.parametrize('operation', [
    lambda join: join['a'], lambda join: join[0:2], lambda join: join.item('a'),
    lambda join: join.slice(0, 2), lambda join: join.as_celltype('mixed'),
])
def test_projection_and_conversion_are_refused(operation):
    with pytest.raises(TypeError):
        operation(make_join())


def test_unknown_attribute_is_not_a_projection():
    join = make_join()
    with pytest.raises(AttributeError):
        getattr(join, 'misspelled_attribute')


@pytest.mark.parametrize('wrapper', [
    lambda join: Expression(join, '', input_celltype='plain', celltype='plain'),
    lambda join: Expression(join, 'a', input_celltype='plain', celltype='mixed'),
    lambda join: Cell('plain', source=join),
])
def test_join_is_not_an_input_reference(wrapper):
    with pytest.raises(TypeError):
        wrapper(make_join())


def test_copy_and_with_result_preserve_spec_and_identity():
    join = make_join()
    result = Buffer({'a': 7}, 'plain').get_checksum()
    clone = copy.copy(join)
    annotated = join.with_result(result)
    assert clone is not join
    assert clone == join == annotated
    assert clone.definition.content == join.definition.content
    assert annotated._result_checksum_internal() == result
    assert join._result_checksum_internal() is None
    assert annotated.celljoin_checksum == join.celljoin_checksum
    assert clone.input_checksums == join.input_checksums


def test_deep_input_checksums_include_members_and_root():
    root = Buffer({'kept': '22' * 32}, 'deepcell')
    member = Checksum('11' * 32)
    join = CellJoin.from_inputs('deepcell', root.get_checksum(), {'a': member})
    assert set(join.input_checksums) == {root.get_checksum(), member}


def test_compute_and_compute_async_hold_the_result():
    join = make_join()
    result = join.compute()
    assert result.resolve('plain') == {'a': 7}
    assert join.checksum == result
    assert asyncio.run(make_join().compute_async()) == result
    assert join.softcancel() is False


def test_synchronous_compute_refuses_uncached_work_inside_running_loop():
    from seamless.checksum.celljoin import celljoin_cache_key
    join = make_join()
    async def main():
        with pytest.raises(RunningLoopRefusal):
            join.compute()
        result = await join.compute_async()
        assert expression_mod.get_expression_cache()[celljoin_cache_key(join.celljoin_checksum, 'plain')] == result
        assert join.compute() == result
    asyncio.run(main())
