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
    expression = Expression(source_checksum, "a", input_celltype="plain", celltype="str")
    result_checksum = expression.compute()
    _drop_buffer(result_checksum)

    value = asyncio.run(result_checksum.fingertip("str"))

    assert value == "hello"
    assert result_checksum.resolve("str") == "hello"


def test_fingertip_recovers_chained_expression_results_recursively():
    expression_mod.get_expression_cache().clear()
    source_checksum = Buffer({"a": {"b": "leaf"}}, "plain").get_checksum()
    first = Expression(source_checksum, "a", input_celltype="plain", celltype="plain")
    first_checksum = first.compute()
    second = Expression(first_checksum, "b", input_celltype="plain", celltype="str")
    second_checksum = second.compute()
    _drop_buffer(first_checksum)
    _drop_buffer(second_checksum)

    value = asyncio.run(second_checksum.fingertip("str"))

    assert value == "leaf"
    assert first_checksum.resolve("plain") == {"b": "leaf"}
    assert second_checksum.resolve("str") == "leaf"


def test_expression_mismatch_reports_category_without_recording(monkeypatch, caplog):
    import pytest
    from seamless import CacheMissError, FingertipCategory
    from seamless.caching import buffer_writer
    expression_mod.get_expression_cache().clear()
    source = Buffer({"a": "actual"}, "plain")
    source.tempref()
    key = (source.get_checksum().hex(), "a", "plain", "str")
    wanted = Checksum("e" * 64)
    expression_mod.get_expression_cache()[key] = wanted
    writes = []
    monkeypatch.setattr(buffer_writer, "register_expression_result", lambda *args: writes.append(args))
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(wanted.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.IRREPRODUCIBLE_EXPRESSION
    assert expression_mod.get_expression_cache()[key] == wanted
    assert writes == []
    assert "already recorded" in caplog.text


def test_recorded_expression_that_raises_is_irreproducible():
    import pytest
    from seamless import CacheMissError, FingertipCategory
    expression_mod.get_expression_cache().clear()
    source = Buffer({"a": "actual"}, "plain")
    source.tempref()
    key = (source.get_checksum().hex(), "missing", "plain", "str")
    wanted = Checksum("f" * 64)
    expression_mod.get_expression_cache()[key] = wanted
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(wanted.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.IRREPRODUCIBLE_EXPRESSION
    assert expression_mod.get_expression_cache()[key] == wanted


def test_expression_input_retains_nested_failure_category(monkeypatch):
    import pytest
    from seamless import CacheMissError, FingertipCategory
    expression_mod.get_expression_cache().clear()
    source, wanted = Checksum("c" * 64), Checksum("b" * 64)
    key = (source.hex(), "a", "plain", "str")
    expression_mod.get_expression_cache()[key] = wanted
    original = Checksum.fingertip
    async def fingertip(checksum, *args, **kwargs):
        if checksum == source:
            raise CacheMissError(source, fingertip_category=FingertipCategory.IRREPRODUCIBLE_TRANSFORMATION)
        return await original(checksum, *args, **kwargs)
    monkeypatch.setattr(Checksum, "fingertip", fingertip)
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(wanted.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.IRREPRODUCIBLE_TRANSFORMATION
