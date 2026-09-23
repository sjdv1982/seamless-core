from __future__ import annotations

import pytest

from seamless import Buffer, Expression
from seamless.checksum.expression import (
    ExpressionEvaluationError,
    get_expression_cache,
)

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


@pytest.mark.parametrize(
    ("witness", "case"),
    VALID_EXPRESSION_CASES,
    ids=[expression_case_id(witness, case) for witness, case in VALID_EXPRESSION_CASES],
)
def test_valid_helper_expressions_evaluate_locally(witness, case):
    input_buffer = Buffer(witness.raw_buffer, checksum=witness.source_checksum)
    assert input_buffer.checksum == witness.source_checksum

    expression = case.build(witness.source_checksum)
    result_checksum = expression.compute()
    result = expression.run()

    assert result_checksum
    assert result is not None or case.target_celltype == "plain"


@pytest.mark.parametrize(
    ("witness", "case"),
    INVALID_EXPRESSION_CASES,
    ids=[
        expression_case_id(witness, case)
        for witness, case in INVALID_EXPRESSION_CASES
    ],
)
def test_invalid_helper_expressions_fail_locally(witness, case):
    Buffer(witness.raw_buffer, checksum=witness.source_checksum)
    with pytest.raises((ExpressionEvaluationError, TypeError, ValueError, KeyError)):
        case.build(witness.source_checksum).compute()


def test_empty_path_conversion_uses_expression_cache():
    source = Buffer("hello", "text")
    source_checksum = source.get_checksum()
    expression = Expression(
        source_checksum,
        path="",
        input_celltype="text",
        celltype="str",
    )

    cache = get_expression_cache()
    cache.clear()
    result_checksum = expression.compute()

    assert expression.compute() == result_checksum
    assert expression.database_key in cache
    assert expression.run() == "hello"


def test_expression_identity_ignores_validator_fields():
    source_checksum = Buffer({"a": 1}, "plain").get_checksum()

    first = Expression(source_checksum, path=".a", input_celltype="plain")
    second = Expression(
        source_checksum,
        path=".a",
        input_celltype="plain",
        validator=bytes.fromhex("00" * 32),
        validator_language="python",
    )

    assert first.identity_key == second.identity_key
    assert first.database_key == second.database_key
