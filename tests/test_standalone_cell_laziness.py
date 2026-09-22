"""Standalone Cell state and configuration inspection never pull evaluation."""

import pytest

from seamless import Cell, Checksum, Expression


@pytest.fixture
def forbid_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("standalone Cell inspection started evaluation")

    monkeypatch.setattr(Expression, "_evaluate_internal", forbidden)
    monkeypatch.setattr(Expression, "_evaluate_internal_async", forbidden)


def test_wired_configuration_and_builders_remain_lazy(forbid_evaluation):
    root = Cell("text")
    root.set('{"items": [1, 2, 3]}')
    converted = root.as_celltype("plain")
    child = converted["items"][1:]

    assert converted.source is root
    assert child.celltype == child.input_celltype == "plain"
    assert child.path == child.path_python == "[1:]"
    repr(child)
    snapshots = [child.build(), child.expression(), child()]
    assert all(isinstance(snapshot, Expression) for snapshot in snapshots)
    child.with_input(converted)
    child.with_validator(Checksum("ab" * 32), language="python")
    root.set('{"items": [4, 5, 6]}')
    root.celltype = "plain"
    child.build()


@pytest.mark.parametrize("operation", ["projection", "conversion", "validator"])
@pytest.mark.parametrize("attribute", ["state", "exception"])
def test_inspection_does_not_evaluate_pending_work(forbid_evaluation, operation, attribute):
    root = Cell("plain")
    root.set({"a": 1})
    if operation == "projection":
        cell = root["a"]
    elif operation == "conversion":
        cell = root.as_celltype("text")
    else:
        cell = root.with_validator(Checksum("ab" * 32), language="python")

    assert root.state == "complete"  # A literal needs no evaluation.
    expected = "waiting" if attribute == "state" else None
    assert getattr(cell, attribute) == expected
    assert getattr(cell, attribute) == expected


def test_state_after_parent_change_waits_for_an_explicit_read():
    root = Cell("text")
    root.set('{"a": 7}')
    cell = root.as_celltype("plain")["a"]

    assert cell.state == "waiting"
    assert cell.value == 7
    assert cell.state == "complete"
    root.set('{"a": 11}')
    assert cell.state == "waiting"
    assert cell.value == 11
    assert cell.state == "complete"


def test_inspection_does_not_discover_or_retry_evaluation_failures():
    root = Cell("str")
    root.set("not an integer")
    cell = root.as_celltype("int")

    assert cell.exception is None
    assert cell.state == "waiting"
    assert cell.checksum is None
    assert cell.state == "failed"
    error = cell.exception
    assert isinstance(error, str) and error
    cell.clear_exception()
    assert cell.exception is None
    assert cell.state == "waiting"
    assert cell.compute() is None
    assert cell.exception == error
    assert cell.state == "failed"
    root.set("13")
    assert cell.exception is None
    assert cell.state == "waiting"
    assert cell.run() == 13
    assert cell.state == "complete"


def test_state_never_resolves_a_literal_checksum(forbid_evaluation, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("state materialized a literal checksum")

    monkeypatch.setattr(Checksum, "resolve", forbidden)
    cell = Cell("plain", checksum=Checksum("de" * 32))
    assert cell.state == "complete"
    cell.checksum = None
    assert cell.state == "unwired"


def test_state_reports_miswiring_without_evaluating(forbid_evaluation):
    root = Cell("text")
    root.set("[1, 2]")
    child = root[0]
    assert child.state == "waiting"
    root.celltype = "plain"
    assert child.state == "miswired"
    root.celltype = "text"
    assert child.state == "waiting"
