from __future__ import annotations

import pytest

from seamless import Buffer, Cell, Expression
from seamless.checksum.expression import parse_path

from helpers.expression_hashtype_cases import (
    expression_case_id,
    iter_expression_cases,
)


EXPRESSION_CASES = tuple(iter_expression_cases())
VALID_EXPRESSION_CASES = tuple(
    (witness, case) for witness, case in EXPRESSION_CASES if case.valid
)
INVALID_EXPRESSION_CASES = tuple(
    (witness, case) for witness, case in EXPRESSION_CASES if not case.valid
)


def _cell_from_case(source_checksum, case):
    cell = Cell(input_ref=source_checksum, celltype=case.celltype)
    for kind, payload in parse_path(case.path):
        if kind == "item":
            cell = cell[payload]
        elif kind == "slice":
            cell = cell[payload]
        else:
            raise AssertionError(kind)
    return cell.as_celltype(case.target_celltype)


@pytest.mark.parametrize(
    ("witness", "case"),
    EXPRESSION_CASES,
    ids=[expression_case_id(witness, case) for witness, case in EXPRESSION_CASES],
)
def test_cell_built_expression_matches_direct_expression(witness, case):
    direct = case.build(witness.source_checksum)
    built = _cell_from_case(witness.source_checksum, case).build()

    assert built == direct
    assert built.identity_key == direct.identity_key
    assert built.database_key == direct.database_key


@pytest.mark.parametrize(
    ("witness", "case"),
    VALID_EXPRESSION_CASES,
    ids=[expression_case_id(witness, case) for witness, case in VALID_EXPRESSION_CASES],
)
def test_cell_built_valid_expressions_match_direct_results(witness, case):
    input_buffer = Buffer(witness.raw_buffer, checksum=witness.source_checksum)
    direct = case.build(witness.source_checksum)
    built = _cell_from_case(witness.source_checksum, case).build()

    assert input_buffer.checksum == witness.source_checksum
    assert built.compute() == direct.compute()
    assert _values_equal(built.run(), direct.run())


@pytest.mark.parametrize(
    ("witness", "case"),
    INVALID_EXPRESSION_CASES,
    ids=[
        expression_case_id(witness, case)
        for witness, case in INVALID_EXPRESSION_CASES
    ],
)
def test_cell_built_invalid_expressions_match_direct_failures(witness, case):
    input_buffer = Buffer(witness.raw_buffer, checksum=witness.source_checksum)
    direct = case.build(witness.source_checksum)
    built = _cell_from_case(witness.source_checksum, case).build()

    assert input_buffer.checksum == witness.source_checksum
    with pytest.raises(Exception) as direct_exc:
        direct.compute()
    with pytest.raises(Exception) as built_exc:
        built.compute()
    assert type(built_exc.value) is type(direct_exc.value)


def test_mutating_cell_after_build_does_not_affect_expression():
    first_checksum = Buffer({"a": 1}, "plain").get_checksum()
    second_checksum = Buffer({"a": 2}, "plain").get_checksum()
    cell = Cell(input_ref=first_checksum, celltype="plain").a

    expression = cell.build()
    cell.input_ref = second_checksum
    cell.path = ".b"
    cell.target_celltype = "mixed"

    assert expression.input_checksum == first_checksum
    assert expression.path == "a"
    assert expression.target_celltype == "plain"


def test_single_cell_builds_independent_expressions_after_reassignment():
    first_checksum = Buffer({"a": 1}, "plain").get_checksum()
    second_checksum = Buffer({"a": 2}, "plain").get_checksum()
    cell = Cell(input_ref=first_checksum, celltype="plain").a

    first = cell.build()
    cell.input_ref = second_checksum
    second = cell.build()

    assert first.input_checksum == first_checksum
    assert second.input_checksum == second_checksum
    assert first.path == second.path == "a"


def test_derived_cell_navigation_does_not_mutate_parent_cell():
    parent = Cell(celltype="text")
    child = parent[0].as_celltype("str")

    assert parent.path == ""
    assert parent.target_celltype == "text"
    assert child.path == "[0]"
    assert child.target_celltype == "str"


def test_cell_call_remains_expression_builder():
    checksum = Buffer({"a": 1}, "plain").get_checksum()
    expression = Cell(input_ref=checksum, celltype="plain").a()

    assert isinstance(expression, Expression)
    assert expression.path == "a"


def _values_equal(first, second) -> bool:
    try:
        import numpy as np

        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            return bool(np.array_equal(first, second))
    except ImportError:
        pass
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            return False
        return all(_values_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return False
        return all(_values_equal(item1, item2) for item1, item2 in zip(first, second))
    return first == second
