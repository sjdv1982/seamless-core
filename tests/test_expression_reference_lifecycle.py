import asyncio

from seamless import Buffer
from seamless.expression_class import Expression
from seamless.caching.buffer_cache import get_buffer_cache


def test_dormant_expression_holds_input_until_release():
    buffer = Buffer(b"dormant expression")
    checksum = buffer.get_checksum()
    expression = Expression(checksum, "", input_celltype="text", celltype="text")
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 1
    expression._release_refholds()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_public_expression_result_is_held_once():
    buffer = Buffer(b"public expression")
    checksum = buffer.get_checksum()
    expression = Expression(checksum, "", input_celltype="text", celltype="text")
    result = expression.compute()
    cache = get_buffer_cache()
    assert cache.reference_snapshot()[result][0] == 2  # input and public result
    assert expression.compute() == result
    assert cache.reference_snapshot()[result][0] == 2
    assert expression.result == result
    expression._release_refholds()
    assert cache.reference_snapshot().get(result, (0, 0, False))[0] == 0


def test_public_hold_after_internal_publication_is_acquired_once():
    buffer = Buffer(b"internal publication")
    checksum = buffer.get_checksum()
    expression = Expression(checksum, "", input_celltype="text", celltype="text")
    result = __import__("seamless.checksum.expression", fromlist=["evaluate_expression"]).evaluate_expression(
        checksum, "", "text", "text"
    )
    expression._publish_result(result)
    assert get_buffer_cache().reference_snapshot().get(result, (0, 0, False))[0] == 1
    expression._enable_result_holding()
    assert get_buffer_cache().reference_snapshot()[result][0] == 2
    expression._enable_result_holding()
    assert get_buffer_cache().reference_snapshot()[result][0] == 2
    expression._release_refholds()


def test_equal_expressions_are_registered_and_released_independently():
    buffer = Buffer(b"equal expressions")
    checksum = buffer.get_checksum()
    first = Expression(checksum, "", input_celltype="text", celltype="text")
    second = Expression(checksum, "", input_celltype="text", celltype="text")
    assert first == second
    result = first.compute()
    second._publish_result(result)
    second._enable_result_holding()
    assert get_buffer_cache().reference_snapshot()[result][0] == 4
    first._release_refholds()
    assert get_buffer_cache().reference_snapshot()[result][0] == 2
    second._release_refholds()
    assert get_buffer_cache().reference_snapshot().get(result, (0, 0, False))[0] == 0
