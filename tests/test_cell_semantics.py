import gc

import pytest
from seamless import Buffer, CacheMissError, Cell, Checksum, Expression
from seamless.cell_errors import AuthorityError
from seamless.checksum.null import NULL_CHECKSUM
from seamless.checksum.celltypes import celltypes
from seamless.checksum.hash_type_validation import validate_deserializable_as


def test_retype_and_live_source_type():
    source = Cell('str'); source.set('42')
    target = Cell(source=source)
    assert target.celltype == target.input_celltype == 'str'
    source.celltype = 'int'
    assert target.celltype == 'str'
    assert target.input_celltype == 'int'
    assert target.value == '42'
    target.celltype = 'text'
    assert target.value == '42'
    assert target.checksum == Buffer('42', 'text').get_checksum()
    assert target.build().run() == target.value
    expression = Expression(source, celltype='str')
    source.celltype = 'float'
    assert expression.input_celltype == 'int'
    assert expression.run() == '42'


def test_declared_checksum_and_readonly_input_type():
    source = Buffer(5, 'int'); source.tempref()
    cell = Cell('str', checksum=source.get_checksum(), input_celltype='int')
    assert cell.value == '5'
    assert cell.source is None
    with pytest.raises(AttributeError): cell.input_celltype = 'text'
    cell.set_checksum(source.get_checksum(), input_celltype='int')
    assert cell.value == '5'
    other = Cell('int'); other.set(5)
    with pytest.raises(ValueError): Cell(source=other, input_celltype='str')
    with pytest.raises(ValueError): Expression(other, input_celltype='str')
    with pytest.raises(TypeError): Cell(source=other, checksum=source.get_checksum())


@pytest.mark.parametrize('form', ['value', 'buffer', 'checksum'])
def test_write_matrix_and_authority(form):
    source = Cell('int'); source.set(1)
    cell = Cell('int', source=source)
    buffer = Buffer(7, 'int'); buffer.tempref()
    value = {'value': 7, 'buffer': buffer, 'checksum': buffer.get_checksum()}[form]
    setter = {'value': cell.set, 'buffer': cell.set_buffer, 'checksum': cell.set_checksum}[form]
    with pytest.raises(AuthorityError): setter(value)
    setattr(cell, form, value)
    assert cell.source is None
    assert cell.checksum == buffer.get_checksum()
    setter(value)
    assert cell.value == 7


@pytest.mark.parametrize('form', ['buffer', 'set_buffer'])
def test_buffer_writes_deposit_the_buffer(form):
    # Without a tempref, a dropped buffer that nothing deposited can't be resolved.
    control = Buffer(f'undeposited standalone {form}', 'text'); checksum = control.get_checksum()
    del control; gc.collect()
    with pytest.raises(CacheMissError): checksum.resolve('text')
    cell = Cell('text')
    buffer = Buffer(f'deposited standalone {form}', 'text')
    if form == 'buffer': cell.buffer = buffer
    else: cell.set_buffer(buffer)
    del buffer; gc.collect()
    assert cell.value == f'deposited standalone {form}'


@pytest.mark.parametrize('make', [lambda: Cell('int'), lambda: Cell('plain')['x']], ids=['cell', 'subcell'])
def test_set_and_value_take_values_only(make):
    source = Cell('int'); source.set(1)
    references = [
        (Buffer(2, 'int').get_checksum(), r'use \.set_checksum\(\)'),
        (source, r'Cell\(source=\.\.\.\)'),
        (source.build(), r'Cell\(source=\.\.\.\)'),
    ]
    for reference, message in references:
        cell = make()
        with pytest.raises(TypeError, match=message): cell.set(reference)
        with pytest.raises(TypeError, match=message): cell.value = reference
        assert cell.source is None and cell.state == 'unwired'


@pytest.mark.parametrize('celltype', celltypes + ['deepcell', 'deepfolder', 'folder', 'module'])
def test_null_is_celltype_independent(celltype):
    cell = Cell(celltype); cell.set(None)
    checksum = Checksum(NULL_CHECKSUM)
    assert cell.checksum == checksum
    assert Buffer(None, celltype).get_checksum() == checksum
    assert cell.value == (b'' if celltype == 'bytes' else None)
    validate_deserializable_as(checksum, Buffer._map_celltype(celltype))
    # Cross-celltype retyping is paired with the bound case in
    # test_cells_contract_alignment.py::test_null_retype_keeps_checksum.
    assert checksum.resolve(celltype) == (b'' if celltype == 'bytes' else None)


def test_null_resolves_without_cache_and_empty_bytes_are_null():
    from seamless.caching.buffer_cache import get_buffer_cache
    checksum = Checksum(NULL_CHECKSUM)
    # Trivial-buffer lookup must work even if the cache cannot return a buffer.
    from unittest.mock import patch
    with patch.object(type(get_buffer_cache()), 'get', return_value=None):
        assert checksum.resolve('int') is None
        assert checksum.resolve('bytes') == b''
    assert Buffer(b'', 'bytes').get_checksum() == checksum
    assert Expression(Buffer(b'').get_checksum(), input_celltype='bytes', celltype='bytes').compute() == checksum


def test_null_vs_clear_and_failed_conversion():
    cell = Cell('int'); cell.value = None
    assert cell.state == 'complete' and cell.checksum == Checksum(NULL_CHECKSUM)
    cell.buffer = None
    assert cell.state == 'unwired' and cell.input_celltype is None
    cell.value = 4; cell.checksum = None
    assert cell.state == 'unwired'
    cell.celltype = 'str'; cell.set('hello'); cell.celltype = 'int'
    assert cell.state == 'failed' and cell.exception is not None and cell.checksum is None


def test_reinterpretation_validates_and_keeps_checksum():
    cell = Cell('text'); cell.set('x = 1')
    checksum = cell.checksum
    cell.celltype = 'python'
    assert cell.checksum == checksum


def test_bytes_value_and_run_agree():
    cell = Cell('bytes'); cell.set(b'payload')
    assert cell.value == cell.run() == cell.build().run() == b'payload'


@pytest.mark.parametrize('celltype', ['int', 'binary', 'bytes', 'plain', 'str'])
def test_empty_bytes_input_converts_as_null(celltype):
    from seamless import Expression
    expression = Expression(Buffer(b'').get_checksum(), input_celltype='bytes', celltype=celltype)
    assert expression.compute() == Buffer(None, 'plain').get_checksum()
    assert expression.run() == (b'' if celltype == 'bytes' else None)
