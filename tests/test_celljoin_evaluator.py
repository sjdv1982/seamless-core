"""Pure celljoin evaluator contracts and portable Stage 0 golden equivalence."""
import builtins
import json
from pathlib import Path

import numpy as np
import pytest
from seamless import Buffer, Checksum
from seamless.checksum.celljoin import build_celljoin, parse_celljoin, evaluate_celljoin

INTEGER_ERROR = 'Integer Cell connection targets require an existing sequence'
DEEP = {'deepcell', 'deepfolder'}
GOLDEN = json.loads((Path(__file__).parent / 'data/celljoin_sidework_golden.json').read_text())


def prepare(celltype, root, members, *, root_present=True):
    buffers = {}
    def store(value):
        buffer = Buffer(value, celltype)
        buffers[buffer.get_checksum()] = buffer
        return buffer.get_checksum()
    root_checksum = store(root) if root_present else None
    member_checksums = {key: Checksum(value) if celltype in DEEP else store(value)
                        for key, value in members}
    spec = parse_celljoin(build_celljoin(root_checksum, member_checksums), celltype)
    return spec, buffers


@pytest.mark.parametrize('celltype', ['mixed', 'plain'])
@pytest.mark.parametrize('root,root_present', [({}, True), ({'kept': 1}, True), (None, False)])
def test_numeric_mapping_and_rootless_fail_before_reading_members(celltype, root, root_present):
    spec, buffers = prepare(celltype, root, [(3, 7)], root_present=root_present)
    requests = []
    def get_buffer(checksum):
        requests.append(checksum)
        assert checksum == spec.root, 'numeric guard must precede member reads'
        return buffers[checksum]
    with pytest.raises(TypeError, match='^' + INTEGER_ERROR + '$'):
        evaluate_celljoin(spec, get_buffer)
    assert requests == ([spec.root] if root_present else [])


@pytest.mark.parametrize('celltype', ['mixed', 'plain'])
def test_numeric_list_assignment_and_out_of_range(celltype):
    spec, buffers = prepare(celltype, [1, 2, 3], [(2, 7), (0, 8)])
    assert evaluate_celljoin(spec, buffers.__getitem__).content == Buffer([8, 2, 7], celltype).content
    spec, buffers = prepare(celltype, [1, 2], [(3, 7)])
    with pytest.raises(IndexError, match='^list assignment index out of range$'):
        evaluate_celljoin(spec, buffers.__getitem__)


@pytest.mark.parametrize('celltype', ['mixed', 'plain'])
@pytest.mark.parametrize('root', [[1, 2], 12, None])
def test_string_keys_over_nonmapping_roots_fail(celltype, root):
    spec, buffers = prepare(celltype, root, [('a', 7)])
    with pytest.raises(TypeError):
        evaluate_celljoin(spec, buffers.__getitem__)


def test_mixed_member_key_kinds_fail_before_spec_formation():
    with pytest.raises(TypeError):
        build_celljoin(None, {0: Checksum('11' * 32), 'a': Checksum('22' * 32)})


def test_result_serialization_depends_on_celltype_for_non_ascii_and_small_float():
    results = {}
    for celltype in ('mixed', 'plain'):
        spec, buffers = prepare(celltype, None, [('é', 1e-7)], root_present=False)
        result = evaluate_celljoin(spec, buffers.__getitem__)
        assert result.content == Buffer({'é': 1e-7}, celltype).content
        results[celltype] = result.content
    assert results['mixed'] != results['plain']


@pytest.mark.parametrize('root_present', [False, True])
def test_deep_members_never_request_buffers_and_root_is_preserved(root_present):
    results = []
    for celltype in ('deepcell', 'deepfolder'):
        spec, buffers = prepare(celltype, {'kept': '22' * 32, 'added': '33' * 32},
                                [('added', '11' * 32)], root_present=root_present)
        requests = []
        def get_buffer(checksum):
            requests.append(checksum)
            assert checksum == spec.root, 'deep member buffers must never be requested'
            return buffers[checksum]
        result = evaluate_celljoin(spec, get_buffer)
        expected = {'added': '11' * 32}
        if root_present:
            expected['kept'] = '22' * 32
        assert result.content == Buffer(expected, celltype).content
        assert requests == ([spec.root] if root_present else [])
        results.append(result.content)
    assert results[0] == results[1]


def compatible(case):
    keys = [key for key, _ in case['members']]
    return (all(isinstance(key, str) and key not in {'<root>', '<numeric>'} for key in keys)
            or all(type(key) is int and key >= 0 for key in keys))


@pytest.mark.parametrize('case', [case for case in GOLDEN if compatible(case)],
                         ids=lambda case: case['name'])
def test_stage0_supported_cases_have_identical_bytes_or_errors(case):
    celltype = 'deepfolder' if case['target'] == 'folder' else case['target']
    root = case['root']
    if case.get('root_kind') == 'structured':
        root = np.array((1, 2), dtype=np.dtype([('a', 'i8'), ('b', 'i8')], align=True))[()]
    elif case.get('root_kind') == 'ndarray':
        root = np.array([1, 2, 3])
    spec, buffers = prepare(celltype, root, case['members'], root_present=case['root_present'])
    if 'error_type' in case:
        with pytest.raises(getattr(builtins, case['error_type'])) as exc_info:
            evaluate_celljoin(spec, buffers.__getitem__)
        assert str(exc_info.value) == case['error_message']
    else:
        result = evaluate_celljoin(spec, buffers.__getitem__)
        assert result.content == bytes.fromhex(case['result_hex'])
