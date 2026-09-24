"""Contract tests: canonical file bytes for mounts.

Oracle: seamless/docs/agent/contracts/mounts.md, *Mountable celltypes*,
*Canonical bytes* and *Null files*. ``canon_T`` lives in seamless-core and is
shared with the CLI file loader, so its per-celltype rules are pinned here.
"""
import pytest

from seamless import Buffer
from seamless.checksum.canonical import canon_T, FILE_CELLTYPES, DIRECTORY_CELLTYPES
from seamless.checksum.null import NULL_BUFFER


def test_mountable_celltype_sets():
    # mounts.md *Mountable celltypes*: the file and directory celltype lists.
    assert FILE_CELLTYPES == {'text', 'python', 'ipython', 'yaml', 'plain', 'str',
                              'int', 'float', 'bool', 'bytes', 'binary', 'mixed'}
    assert DIRECTORY_CELLTYPES == {'folder', 'deepfolder'}
    for celltype in ('checksum', 'deepcell', 'module', 'folder', 'deepfolder'):
        with pytest.raises(TypeError):
            canon_T(b'x', celltype)


@pytest.mark.parametrize('celltype', ['text', 'python', 'ipython', 'yaml'])
def test_text_trailing_newline_normalized_to_exactly_one(celltype):
    # "strict UTF-8; the trailing newline is normalized to exactly one"
    assert canon_T(b'abc', celltype) == b'abc\n'
    assert canon_T(b'abc\n', celltype) == b'abc\n'
    assert canon_T(b'abc\n\n\n', celltype) == b'abc\n'


@pytest.mark.parametrize('celltype', ['text', 'python'])
def test_text_crlf_is_content(celltype):
    # "no CRLF conversion — CRLF is content"
    assert canon_T(b'a\r\nb\r\n', celltype).startswith(b'a\r\nb')
    assert canon_T(b'a\r\nb', celltype) != canon_T(b'a\nb', celltype)


@pytest.mark.parametrize('celltype', ['text', 'python', 'ipython', 'yaml'])
def test_text_is_strict_utf8(celltype):
    with pytest.raises((UnicodeDecodeError, ValueError)):
        canon_T(b'\xff\xfe bad', celltype)


@pytest.mark.parametrize('celltype,content', [('python', b'def (:\n'), ('yaml', b'a: [broken\n'),
                                              ('ipython', b'%%magic !!\n')])
def test_canon_T_does_not_syntax_check_code_text(celltype, content):
    # canon_T performs no mount-local parsing of code text (the celltype's own
    # validation, applied equally to assignments, is a separate step).
    assert canon_T(content, celltype) == Buffer(content.decode(), celltype).content


def test_plain_reformatting_maps_to_value_checksum():
    # "a user-formatted file maps to the checksum of its value"
    a = canon_T(b'{ "b" : 2,\n  "a":1 }', 'plain')
    b = canon_T(b'{"a": 1, "b": 2}', 'plain')
    assert a == b == Buffer({'a': 1, 'b': 2}, 'plain').content


def test_str_file_holds_a_quoted_json_string():
    # "the file holds a quoted JSON string; use text for raw text"
    assert canon_T(b'"hi"', 'str') == Buffer('hi', 'str').content
    with pytest.raises((ValueError, TypeError)):
        canon_T(b'hi there', 'str')


@pytest.mark.parametrize('celltype,content,value', [('int', b' 12 ', 12), ('float', b'1.5', 1.5),
                                                    ('bool', b'true', True)])
def test_scalar_json_parse_and_reserialize(celltype, content, value):
    assert canon_T(content, celltype) == Buffer(value, celltype).content


def test_bytes_identity_and_empty_is_null():
    # "bytes: identity for non-empty content; empty -> null"
    assert canon_T(b'\x00\xff raw', 'bytes') == b'\x00\xff raw'
    assert canon_T(b'', 'bytes') == NULL_BUFFER


@pytest.mark.parametrize('celltype', sorted({'text', 'python', 'ipython', 'yaml', 'plain', 'str',
                                             'int', 'float', 'bool', 'bytes', 'binary', 'mixed'}))
def test_zero_byte_and_explicit_null_both_read_as_null(celltype):
    # *Null files*: "A zero-byte file and b"null\n" both read as null, before
    # celltype-specific parsing."
    assert canon_T(b'', celltype) == NULL_BUFFER
    assert canon_T(b'null\n', celltype) == NULL_BUFFER


def test_newline_only_file_is_not_special():
    # "A newline-only file is not special and undergoes normal parsing."
    assert canon_T(b'\n', 'text') != NULL_BUFFER
    assert canon_T(b'\n', 'text') == Buffer('\n', 'text').content
    with pytest.raises((ValueError, TypeError)):
        canon_T(b'\n', 'int')


def test_binary_must_be_pure_binary():
    import numpy as np
    content = Buffer(np.arange(3.), 'binary').content
    assert canon_T(content, 'binary') == content
    with pytest.raises((ValueError, TypeError)):
        canon_T(b'{"a": 1}', 'binary')
