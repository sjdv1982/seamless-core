from __future__ import annotations

import asyncio

import pytest

from seamless import Buffer, Expression
from seamless.checksum.hash_type_validation import HashTypeValidationError


def test_buffer_deserialization_rejects_impossible_celltype_before_parse():
    buffer = Buffer(b"\xff\xfe\x00")

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        buffer.get_value("text")


def test_async_buffer_deserialization_rejects_impossible_celltype_before_parse():
    async def main():
        buffer = Buffer(b'{"a": 1}')
        await buffer.get_checksum_async()
        with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
            await buffer.get_value_async("binary")

    asyncio.run(main())


def test_expression_path_capability_rejects_before_materialization():
    checksum = Buffer([1, 2, 3], "plain").get_checksum()
    expression = Expression(checksum, "missing", celltype="plain", target_celltype="plain")

    with pytest.raises(HashTypeValidationError, match="requires MAP capability"):
        expression.compute()


def test_identity_expression_keeps_validity_gate_for_overlong_numbers():
    checksum = Buffer(b"1" * 1001).get_checksum()
    expression = Expression(checksum, "", celltype="float", target_celltype="float")

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        expression.compute()
