"""Canonical celljoin definitions, independently of execution and storage."""
from dataclasses import FrozenInstanceError

import pytest
from seamless import Buffer, Checksum
from seamless.checksum.celljoin import (
    build_celljoin, celljoin_buffer, parse_celljoin, required_buffers,
    celljoin_cache_key,
)

A = Checksum('11' * 32)
B = Checksum('22' * 32)
R = Checksum('33' * 32)
TYPES = ('mixed', 'plain', 'deepcell', 'deepfolder')


def test_exact_plain_definition_bytes_and_checksum():
    definition = build_celljoin(R, {2: A, 10: B})
    expected = ('{\n  "10": "' + B.hex() + '",\n  "2": "' + A.hex()
                + '",\n  "<numeric>": null,\n  "<root>": "' + R.hex() + '"\n}\n').encode()
    buffer = celljoin_buffer(definition)
    assert buffer.content == expected
    assert buffer.get_checksum() == Buffer(definition, 'plain').get_checksum()
    assert definition == {'<root>': R.hex(), '<numeric>': None, '2': A.hex(), '10': B.hex()}


@pytest.mark.parametrize('celltype', TYPES)
def test_order_independent_definition_and_celltype_independent_checksum(celltype):
    one = build_celljoin(R, {'é': A, 'z': B})
    two = build_celljoin(R, {'z': B, 'é': A})
    spec = parse_celljoin(one, celltype)
    assert spec.buffer.content == celljoin_buffer(two).content
    assert spec.checksum == celljoin_buffer(one).get_checksum()
    assert spec.root == R
    assert dict(spec.members) == {'é': A, 'z': B}
    assert spec.celltype == celltype
    assert spec.numeric is False
    assert celljoin_cache_key(spec.checksum, celltype) == ('celljoin', spec.checksum.hex(), celltype)
    with pytest.raises(FrozenInstanceError):
        spec.numeric = True


@pytest.mark.parametrize('celltype', TYPES)
def test_empty_and_rootless_forms(celltype):
    definition = build_celljoin(None, {})
    assert definition == {}
    spec = parse_celljoin(definition, celltype)
    assert spec.root is None
    assert not spec.numeric
    assert spec.members == ()
    assert required_buffers(spec) == ()
    assert '<root>' not in build_celljoin(None, {'a': A})
    assert '<numeric>' not in build_celljoin(None, {'0': A})
    assert build_celljoin(None, {0: A}) == {'0': A.hex(), '<numeric>': None}


@pytest.mark.parametrize('definition_kind', ['buffer', 'bytes', 'str', 'mapping'])
def test_parse_supported_definition_representations(definition_kind):
    definition = build_celljoin(R, {'a': A})
    buffer = celljoin_buffer(definition)
    representations = {'buffer': buffer, 'bytes': buffer.content,
                       'str': buffer.content.decode(), 'mapping': definition}
    spec = parse_celljoin(representations[definition_kind], 'plain')
    assert spec.checksum == buffer.get_checksum()
    assert spec.root == R
    assert spec.members == (('a', A),)


def test_explicit_null_is_a_root_and_not_rootless():
    null = Buffer(None, 'plain').get_checksum()
    definition = build_celljoin(null, {'a': A})
    assert definition['<root>'] == null.hex()
    assert parse_celljoin(definition, 'plain').root == null


@pytest.mark.parametrize('members,exception', [
    ({0: A, 'a': B}, TypeError), ({1.5: A}, TypeError), ({None: A}, TypeError),
    ({True: A}, TypeError), ({False: A}, TypeError), ({-1: A}, ValueError),
    ({'<root>': A}, ValueError), ({'<numeric>': A}, ValueError),
])
def test_build_refuses_invalid_member_keys(members, exception):
    with pytest.raises(exception):
        build_celljoin(None, members)


def test_build_refuses_noninteger_objects_even_with_index_protocol():
    class IndexOnly:
        def __index__(self):
            return 3

    with pytest.raises(TypeError):
        build_celljoin(None, {IndexOnly(): A})


@pytest.mark.parametrize('definition,celltype', [
    ({'a': 'not-a-checksum'}, 'plain'), ({'a': 'AB' * 32}, 'mixed'),
    ({'a': None}, 'plain'), ({'<root>': 17}, 'plain'),
    ({'<numeric>': True, '0': A.hex()}, 'plain'),
    ({'<numeric>': None, '-1': A.hex()}, 'plain'),
    ({'<numeric>': None, '01': A.hex()}, 'mixed'),
    ({'<numeric>': None, 'a': A.hex()}, 'plain'),
    ({'<numeric>': None, '0': A.hex()}, 'deepcell'),
    ({'<numeric>': None, '0': A.hex()}, 'deepfolder'),
    ({0: A.hex()}, 'plain'), ([], 'plain'), ({}, 'folder'), ({}, 'unknown'),
])
def test_parse_refuses_invalid_definitions(definition, celltype):
    with pytest.raises((TypeError, ValueError)):
        parse_celljoin(definition, celltype)


@pytest.mark.parametrize('celltype', TYPES)
def test_required_buffers_deduplicates_and_ignores_deep_members(celltype):
    spec = parse_celljoin(build_celljoin(A, {'a': A, 'b': B, 'c': B}), celltype)
    expected = (A,) if celltype.startswith('deep') else (A, B)
    assert required_buffers(spec) == expected
