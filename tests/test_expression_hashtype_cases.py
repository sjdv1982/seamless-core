from __future__ import annotations

import pytest

from seamless import Expression
from seamless.checksum.hash_type import HashType
from seamless.expression_class import normalize_path

from helpers.expression_hashtype_cases import (
    database_path_roundtrip,
    expression_case_id,
    iter_expression_cases,
    iter_hashtype_witnesses,
)


WITNESSES = tuple(iter_hashtype_witnesses())
EXPRESSION_CASES = tuple(iter_expression_cases())


def _witness_id(witness):
    return (
        f"{witness.name}|hash_type={witness.expected_hash_type}|"
        f"mic={witness.mic}"
    )


@pytest.mark.parametrize("witness", WITNESSES, ids=_witness_id)
def test_hashtype_witness_words_are_well_formed(witness):
    assert HashType.is_valid_word(witness.expected_hash_type)
    assert witness.decoded_hash_type.word == witness.expected_hash_type
    assert witness.source_checksum
    assert isinstance(witness.raw_buffer, bytes)
    assert witness.valid_expressions
    assert len(witness.invalid_expressions) >= 2


def test_hashtype_witness_names_are_unique():
    names = {witness.name for witness in WITNESSES}

    assert len(names) == len(WITNESSES)


@pytest.mark.parametrize(
    ("witness", "case"),
    EXPRESSION_CASES,
    ids=[expression_case_id(witness, case) for witness, case in EXPRESSION_CASES],
)
def test_helper_expressions_construct_and_normalize(witness, case):
    expression = case.build(witness.source_checksum)

    assert isinstance(expression, Expression)
    assert expression.input_checksum == witness.source_checksum
    assert expression.path == normalize_path(case.path)
    assert expression.input_celltype == case.celltype
    assert expression.celltype == case.target_celltype
    assert expression.database_key == (
        witness.source_checksum.hex(),
        case.path,
        case.celltype,
        case.target_celltype,
    ), expression_case_id(witness, case)


@pytest.mark.parametrize(
    ("witness", "case"),
    EXPRESSION_CASES,
    ids=[expression_case_id(witness, case) for witness, case in EXPRESSION_CASES],
)
def test_helper_paths_roundtrip_through_database_payload(witness, case):
    assert database_path_roundtrip(case.path) == case.path


@pytest.mark.parametrize("witness", WITNESSES, ids=_witness_id)
def test_outside_hashtype_variants_are_marked_only_on_json_containers(witness):
    if witness.outside_hashtype_celltypes:
        assert witness.mic == "plain"
        assert witness.decoded_hash_type.kind.name in {"JSON_OBJECT", "JSON_ARRAY"}
