"""The six projection writes: standalone refusal versus bound transaction.

Paired cases in seamless-core and seamless-workflow. The documented write bugs
are xfailed individually; authority failures never count as successful writes.
"""
import pytest
from seamless import Buffer, Cell
from seamless.cell_errors import AuthorityError


FORMS = ["value", "buffer", "checksum"]


def write(cell, form, method, value):
    if method:
        getattr(cell, {"value": "set", "buffer": "set_buffer", "checksum": "set_checksum"}[form])(value)
    else:
        setattr(cell, form, value)


@pytest.mark.xfail(strict=False, reason="feature 5 bug 5: standalone projection setters silently mutate a throwaway handle")
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("method", [False, True])
def test_projection_write_matrix(form, method):
    root = Cell("plain")
    root.set({"a": 1, "other": 2})
    projected = root["a"]
    buffer = Buffer(3, "plain")
    hold = buffer.tempref()
    try:
        value = {"value": 3, "buffer": buffer, "checksum": buffer.get_checksum()}[form]
        with pytest.raises((TypeError, AttributeError)):
            write(projected, form, method, value)
        assert root.value == {"a": 1, "other": 2}
        assert projected.value == 1
    finally:
        hold.clear()


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("method", [pytest.param(False, marks=pytest.mark.xfail(strict=False, reason="feature 5 bug 5: standalone projection properties silently detach")), True])
def test_projection_write_under_source_is_refused(form, method):
    source = Cell("plain")
    source.set({"a": 1})
    root = Cell("plain", source=source)
    buffer = Buffer(3, "plain")
    hold = buffer.tempref()
    try:
        value = {"value": 3, "buffer": buffer, "checksum": buffer.get_checksum()}[form]
        with pytest.raises((TypeError, AttributeError, AuthorityError)):
            write(root["a"], form, method, value)
        assert root.value == source.value == {"a": 1}
    finally:
        hold.clear()


def test_subpath_assignment_and_augmented_assignment_are_bound_only():
    root = Cell("plain")
    root.set({"a": 1})
    with pytest.raises(TypeError):
        root["a"] = 2
    with pytest.raises(AttributeError):
        root.a = 2
    with pytest.raises(TypeError):
        root += 2
    assert root.value == {"a": 1}
