from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from seamless import Buffer, Cell, Checksum, Expression
from seamless.checksum import expression as expression_module
from seamless.checksum.expression import (
    ExpressionEvaluationError,
    evaluate_expression_remote,
    get_expression_cache,
)


def test_expression_projects_before_converting_and_composition_spells_the_reverse():
    source = Buffer({"k": "value"}, "plain")
    source_checksum = source.get_checksum()

    project_then_convert = Expression(
        source_checksum,
        path="k",
        input_celltype="plain",
        celltype="text",
    )
    converted_parent = Expression(
        source_checksum,
        path="",
        input_celltype="plain",
        celltype="text",
    )
    convert_then_project = Expression(
        converted_parent,
        path="[3]",
        input_celltype="text",
        celltype="text",
    )

    assert project_then_convert.run() == "value"
    assert convert_then_project.run() == " "
    assert convert_then_project.identity_key[0][0] == "expression"


def test_expression_builders_are_immutable_and_spell_paths_canonically():
    checksum = Checksum(bytes.fromhex("11" * 32))
    base = Expression(checksum, input_celltype="plain")

    assert base.name.path == "name"
    assert base["path/to.file"].path == "['path/to.file']"
    assert base[1:5:2].path == "[1:5:2]"
    assert base.path == ""
    with pytest.raises(FrozenInstanceError):
        base.path = ".changed"


def test_expression_rejects_an_input_celltype_that_disagrees_with_typed_source():
    with pytest.raises(ValueError, match="disagrees with the typed source"):
        Expression(Cell("plain"), input_celltype="text")


def test_expression_defaults_and_attribute_guards_are_part_of_the_definition():
    checksum = Checksum(bytes.fromhex("12" * 32))

    defaulted = Expression(checksum)
    target_only = Expression(checksum, celltype="text")

    assert (defaulted.input_celltype, defaulted.celltype) == ("mixed", "mixed")
    assert (target_only.input_celltype, target_only.celltype) == ("text", "text")
    with pytest.raises(AttributeError, match="has been retired"):
        _ = defaulted.target_celltype
    with pytest.raises(AttributeError):
        _ = defaulted._projected


def test_expression_identity_keeps_an_unbounded_path_verbatim():
    checksum = Checksum(bytes.fromhex("22" * 32))
    path = "." + "x" * 10_000
    expression = Expression(
        checksum,
        path=path,
        input_celltype="plain",
        celltype="plain",
    )

    assert expression.identity_key == (
        ("checksum", checksum.hex()),
        path,
        "plain",
        "plain",
    )
    assert expression.database_key == (
        checksum.hex(),
        path,
        "plain",
        "plain",
    )


@pytest.mark.parametrize(
    "validator_kwargs",
    [
        {"validator": Checksum(bytes.fromhex("33" * 32))},
        {"validator_language": "python"},
    ],
    ids=["validator", "validator-language"],
)
def test_expression_validators_are_deferred_at_all_evaluation_entrypoints(
    validator_kwargs,
):
    source = Buffer({"value": 1}, "plain")
    source_checksum = source.get_checksum()
    expression = Expression(
        source_checksum,
        path="value",
        input_celltype="plain",
        celltype="plain",
        **validator_kwargs,
    )

    with pytest.raises(NotImplementedError, match="validators are not implemented"):
        expression.compute(execution="local")
    with pytest.raises(NotImplementedError, match="validators are not implemented"):
        asyncio.run(expression.compute_async(execution="local"))
    with pytest.raises(NotImplementedError, match="validators are not implemented"):
        asyncio.run(
            evaluate_expression_remote(
                source_checksum,
                "value",
                "plain",
                "plain",
                execution="auto",
                **validator_kwargs,
            )
        )


def test_expression_failures_are_not_cached_and_are_retried(monkeypatch):
    source = Buffer({"present": 1}, "plain")
    expression = Expression(
        source.get_checksum(),
        path="missing",
        input_celltype="plain",
        celltype="plain",
    )
    cache = get_expression_cache()
    cache.clear()
    calls = 0
    original_apply_step = expression_module._apply_step

    def count_apply_step(value, step):
        nonlocal calls
        calls += 1
        return original_apply_step(value, step)

    monkeypatch.setattr(expression_module, "_apply_step", count_apply_step)
    for _ in range(2):
        with pytest.raises(ExpressionEvaluationError, match="missing"):
            expression.compute(execution="local")
        assert expression.database_key not in cache

    assert calls == 2


def test_expression_evaluation_does_not_publish_without_a_refholder(monkeypatch):
    from seamless.caching.buffer_cache import get_buffer_cache

    source = Buffer({"value": "ephemeral"}, "plain")
    source_checksum = source.get_checksum()
    cache = get_buffer_cache()
    original_tempref = type(cache).tempref
    observed = []

    def observe_tempref(self, checksum, *args, **kwargs):
        observed.append((Checksum(checksum), kwargs.get("scratch", False)))
        return original_tempref(self, checksum, *args, **kwargs)

    monkeypatch.setattr(type(cache), "tempref", observe_tempref)
    expression = Expression(
        source_checksum,
        path="value",
        input_celltype="plain",
        celltype="str",
    )
    result = expression.compute(execution="local")

    assert (result, True) in observed
    assert (result, False) not in observed


def test_dummy_expression_preserves_its_checksum_without_fetching(monkeypatch):
    source = Buffer(True, "bool")
    source_checksum = source.get_checksum()
    expression = Expression(
        source_checksum,
        path="",
        input_celltype="bool",
        celltype="bool",
    )

    def fail_if_fetched(*args, **kwargs):
        pytest.fail("a dummy Expression must not fetch its input buffer")

    monkeypatch.setattr(expression_module, "_get_local_buffer", fail_if_fetched)
    assert expression.compute(execution="local") == source_checksum


@pytest.mark.parametrize(
    "input_celltype,path",
    [
        ("text", ".name"),
        ("str", "[0].name"),
        ("python", ".name"),
        ("yaml", "['name']"),
        ("bytes", ".name"),
        ("bytes", "[0][0]"),
        ("int", "[0]"),
        ("float", "[:]"),
        ("bool", ".name"),
        ("checksum", "[0]"),
    ],
)
def test_statically_illegal_ordinary_paths_are_rejected_at_construction(
    input_celltype, path
):
    with pytest.raises(ValueError):
        Expression(
            Checksum(bytes.fromhex("44" * 32)),
            path=path,
            input_celltype=input_celltype,
            celltype=input_celltype,
        )


def test_malformed_path_syntax_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unclosed|path"):
        Expression(
            Checksum(bytes.fromhex("4a" * 32)),
            path="value[",
            input_celltype="plain",
            celltype="plain",
        )


def test_illegal_pathless_conversion_is_rejected_at_construction():
    with pytest.raises(ValueError):
        Expression(
            Checksum(bytes.fromhex("55" * 32)),
            input_celltype="python",
            celltype="bool",
        )


@pytest.mark.parametrize(
    "source,target",
    [
        ("deepfolder", "deepcell"),
        ("folder", "plain"),
        ("plain", "deepcell"),
        ("plain", "deepfolder"),
        ("plain", "folder"),
        ("deepcell", "mixed"),
    ],
)
def test_illegal_deep_conversions_are_rejected_at_construction(source, target):
    with pytest.raises(ValueError):
        Expression(
            Checksum(bytes.fromhex("5a" * 32)),
            input_celltype=source,
            celltype=target,
        )


def test_adjacent_paths_fuse_into_one_expression_identity():
    checksum = Checksum(bytes.fromhex("66" * 32))
    inner = Expression(
        checksum,
        path="[1]",
        input_celltype="plain",
        celltype="plain",
    )
    outer = Expression(
        inner,
        path="[0]",
        input_celltype="plain",
        celltype="int",
    )

    assert outer.input_checksum == checksum
    assert outer.path == "[1][0]"
    assert outer.identity_key == (
        ("checksum", checksum.hex()),
        "[1][0]",
        "plain",
        "int",
    )


def test_checksum_preserving_conversion_then_path_fuses():
    checksum = Checksum(bytes.fromhex("77" * 32))
    converted = Expression(
        checksum,
        path="",
        input_celltype="python",
        celltype="text",
    )
    projected = Expression(
        converted,
        path="[1]",
        input_celltype="text",
        celltype="text",
    )

    assert projected.input_checksum == checksum
    assert projected.path == "[1]"
    assert projected.input_celltype == "text"
    assert projected.identity_key[0] == ("checksum", checksum.hex())


def test_consecutive_conversions_remain_two_expression_identities():
    checksum = Checksum(bytes.fromhex("88" * 32))
    first = Expression(
        checksum,
        path="",
        input_celltype="plain",
        celltype="text",
    )
    second = Expression(
        first,
        path="",
        input_celltype="text",
        celltype="str",
    )

    assert second.identity_key[0] == ("expression", first.identity_key)


@pytest.mark.parametrize(
    "source_celltype,member_celltype,target",
    [
        ("deepcell", "mixed", "mixed"),
        ("deepcell", "mixed", "checksum"),
        ("deepfolder", "bytes", "bytes"),
        ("deepfolder", "bytes", "checksum"),
        ("folder", "bytes", "bytes"),
        ("folder", "bytes", "checksum"),
    ],
)
def test_deep_one_step_selects_a_child_without_fetching_it(
    source_celltype, member_celltype, target
):
    child_checksum = Checksum(bytes.fromhex("99" * 32))
    index = Buffer({"path/to/file": child_checksum.hex()}, source_celltype)
    expression = Expression(
        index.get_checksum(),
        path="['path/to/file']",
        input_celltype=source_celltype,
        celltype=target,
    )

    if target == member_celltype:
        expected = child_checksum
    else:
        expected = Buffer(child_checksum, "checksum").get_checksum()
    assert expression.compute(execution="local") == expected


@pytest.mark.parametrize("path", ["[0]", "[:1]", "['a']['b']"])
def test_illegal_deep_path_shapes_are_rejected_at_construction(path):
    with pytest.raises(ValueError):
        Expression(
            Checksum(bytes.fromhex("aa" * 32)),
            path=path,
            input_celltype="deepcell",
            celltype="mixed",
        )


@pytest.mark.parametrize(
    "source_celltype,illegal_target",
    [
        ("deepcell", "bytes"),
        ("deepcell", "plain"),
        ("deepfolder", "mixed"),
        ("deepfolder", "text"),
        ("folder", "mixed"),
        ("folder", "text"),
    ],
)
def test_illegal_deep_member_targets_are_rejected_at_construction(
    source_celltype, illegal_target
):
    with pytest.raises(ValueError):
        Expression(
            Checksum(bytes.fromhex("ab" * 32)),
            path="member",
            input_celltype=source_celltype,
            celltype=illegal_target,
        )


def test_deep_value_is_a_flat_index_of_checksum_objects():
    child_checksum = Checksum(bytes.fromhex("bb" * 32))
    value = Buffer({"member": child_checksum.hex()}, "deepcell").get_value("deepcell")

    assert value == {"member": child_checksum}
    assert isinstance(value["member"], Checksum)


def test_reading_a_nested_deep_value_rejects_the_false_deep_claim():
    nested = Buffer({"nested": {"member": "cc" * 32}}, "plain")

    with pytest.raises(ValueError, match="nested"):
        nested.get_value("deepcell")


def test_deep_path_wraps_a_flatness_failure_as_expression_evaluation_error():
    nested = Buffer({"nested": {"member": "cc" * 32}}, "deepcell")
    expression = Expression(
        nested.get_checksum(),
        path="nested",
        input_celltype="deepcell",
        celltype="mixed",
    )

    with pytest.raises(ExpressionEvaluationError, match="nested"):
        expression.compute(execution="local")


def test_folder_to_mixed_uses_one_dimensional_s1_arrays_for_every_child():
    empty = Buffer(b"")
    nul_bytes = Buffer(b"ab\x00\x00")
    index = Buffer(
        {"empty": empty.get_checksum().hex(), "nul": nul_bytes.get_checksum().hex()},
        "folder",
    )
    expression = Expression(
        index.get_checksum(),
        input_celltype="folder",
        celltype="mixed",
    )

    value = expression.run()

    assert value["empty"].dtype == np.dtype("S1")
    assert value["empty"].shape == (0,)
    assert value["nul"].dtype == np.dtype("S1")
    assert value["nul"].shape == (4,)
    assert value["nul"].tobytes() == b"ab\x00\x00"


def test_expression_cancel_is_a_softcancel_alias():
    expression = Expression(
        Checksum(bytes.fromhex("cd" * 32)), input_celltype="plain"
    )

    assert expression.softcancel() is False
    assert expression.cancel() is False
    assert not hasattr(expression_module, "cancel_expression")
