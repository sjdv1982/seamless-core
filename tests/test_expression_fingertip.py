import asyncio

from seamless import Buffer, Checksum, Expression
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
import seamless.checksum.expression as expression_mod


def _drop_buffer(checksum):
    checksum = Checksum(checksum)
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)


def test_fingertip_recovers_missing_expression_result_from_reverse_cache():
    expression_mod.get_expression_cache().clear()
    source_checksum = Buffer({"a": "hello"}, "plain").get_checksum()
    expression = Expression(source_checksum, "a", input_celltype="plain", target_celltype="str")
    result_checksum = expression.compute()
    _drop_buffer(result_checksum)

    value = asyncio.run(result_checksum.fingertip("str"))

    assert value == "hello"
    assert result_checksum.resolve("str") == "hello"


def test_fingertip_recovers_chained_expression_results_recursively():
    expression_mod.get_expression_cache().clear()
    source_checksum = Buffer({"a": {"b": "leaf"}}, "plain").get_checksum()
    first = Expression(source_checksum, "a", input_celltype="plain", target_celltype="plain")
    first_checksum = first.compute()
    second = Expression(first_checksum, "b", input_celltype="plain", target_celltype="str")
    second_checksum = second.compute()
    _drop_buffer(first_checksum)
    _drop_buffer(second_checksum)

    value = asyncio.run(second_checksum.fingertip("str"))

    assert value == "leaf"
    assert first_checksum.resolve("plain") == {"b": "leaf"}
    assert second_checksum.resolve("str") == "leaf"
