"""Mutable structural Cell builder."""

from __future__ import annotations

from typing import Any

from .cell_errors import ProjectionError
from .expression_class import (
    Expression,
    append_item_path,
    append_slice_path,
    normalize_path,
)

_UNSET = object()


class Cell:
    """Mutable structural expression builder.

    Navigation creates derived Cell builders. ``build()`` snapshots the current
    builder state into an immutable ``Expression`` container.

    The positional argument is ``celltype`` (default ``"mixed"``).
    ``input_ref`` is keyword-only and is a reference, never a value: ``None``,
    a ``Checksum``, an ``Expression``, another Cell, or a workflow source. A Cell gets a value
    through ``set()``, which serializes it to a checksum.
    """

    __slots__ = (
        "_workflow_backend",
        "_input_ref",
        "_path",
        "_celltype",
        "_target_celltype",
        "_validator",
        "_validator_language",
        "_refholds_released",
        "__weakref__",
    )

    def __init__(
        self,
        celltype: str = "mixed",
        *,
        input_ref: Any = None,
        path: str | None = None,
        target_celltype: str | None = None,
        validator: Any = None,
        validator_language: str | None = None,
    ) -> None:
        _check_input_ref(input_ref)
        self._workflow_backend = None
        self._input_ref = input_ref
        self._path = normalize_path(path)
        self._celltype = celltype
        self._target_celltype = celltype if target_celltype is None else target_celltype
        self._validator = validator
        self._validator_language = validator_language
        self._refholds_released = False
        from .checksum_class import Checksum
        from .reference_lifecycle import register_refholder

        register_refholder(self)
        if isinstance(input_ref, Checksum):
            input_ref.incref_refholder()

    @classmethod
    def _from_backend(cls, backend) -> "Cell":
        """Create a canonical Cell whose state is entirely backend-owned."""

        self = cls.__new__(cls)
        object.__setattr__(self, "_workflow_backend", backend)
        object.__setattr__(self, "_input_ref", None)
        object.__setattr__(self, "_path", "")
        object.__setattr__(self, "_celltype", "mixed")
        object.__setattr__(self, "_target_celltype", "mixed")
        object.__setattr__(self, "_validator", None)
        object.__setattr__(self, "_validator_language", None)
        object.__setattr__(self, "_refholds_released", True)
        return self

    @property
    def input_ref(self) -> Any:
        if self._workflow_backend is not None:
            return self._workflow_backend.input_ref
        return self._input_ref

    @input_ref.setter
    def input_ref(self, input_ref: Any) -> None:
        if self._workflow_backend is not None:
            raise _bound_state_error("input_ref")
        self._replace_input_ref(input_ref)

    def _replace_input_ref(self, input_ref: Any) -> None:
        from .checksum_class import Checksum

        _check_input_ref(input_ref)
        old = self._input_ref
        if isinstance(input_ref, Checksum):
            input_ref.incref_refholder()
        self._input_ref = input_ref
        if isinstance(old, Checksum):
            old.decref_refholder()

    @property
    def path(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.path
        return self._path

    @path.setter
    def path(self, path: str | None) -> None:
        if self._workflow_backend is not None:
            raise _bound_state_error("path")
        self._path = normalize_path(path)

    @property
    def path_python(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.path_python
        return self._path

    @property
    def celltype(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.celltype
        return self._celltype

    @celltype.setter
    def celltype(self, celltype: str) -> None:
        if self._workflow_backend is not None:
            self._workflow_backend.celltype = celltype
            return
        self._celltype = celltype

    @property
    def target_celltype(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.target_celltype
        return self._target_celltype

    @target_celltype.setter
    def target_celltype(self, target_celltype: str | None) -> None:
        if self._workflow_backend is not None:
            self._workflow_backend.target_celltype = target_celltype
            return
        self._target_celltype = (
            self._celltype if target_celltype is None else target_celltype
        )

    @property
    def validator(self) -> Any:
        if self._workflow_backend is not None:
            return self._workflow_backend.validator
        return self._validator

    @validator.setter
    def validator(self, validator: Any) -> None:
        if self._workflow_backend is not None:
            self._workflow_backend.validator = validator
            return
        self._validator = validator

    @property
    def validator_language(self) -> str | None:
        if self._workflow_backend is not None:
            return self._workflow_backend.validator_language
        return self._validator_language

    @validator_language.setter
    def validator_language(self, validator_language: str | None) -> None:
        if self._workflow_backend is not None:
            self._workflow_backend.validator_language = validator_language
            return
        self._validator_language = validator_language

    @property
    def checksum(self):
        if self._workflow_backend is None:
            raise AttributeError("checksum is only available for bound workflow cells")
        return self._workflow_backend.checksum

    @property
    def buffer(self):
        if self._workflow_backend is None:
            raise AttributeError("buffer is only available for bound workflow cells")
        return self._workflow_backend.buffer

    @property
    def value(self):
        if self._workflow_backend is None:
            raise AttributeError("value is only available for bound workflow cells")
        return self._workflow_backend.value

    @property
    def state(self) -> str:
        """Return the node state of a bound workflow cell.

        The six-state vocabulary of the node itself: ``unwired``, ``blocked``,
        ``waiting``, ``computing``, ``complete``, ``failed``.  With
        :attr:`block_reason` this is the whole of a node's lifecycle report;
        the display string ``status`` used to return is gone, because it
        collapsed ``waiting`` and ``computing`` into one word.  For a summary,
        ``repr`` of the handle carries the state.

        This name is API, so a cell whose value happens to hold a ``"state"``
        key reaches that key as ``cell["state"]`` rather than by attribute.

        On a sub-path projection this reports the state of the node the
        projection is taken from; a projection has no state of its own.
        """

        if self._workflow_backend is None:
            raise AttributeError("state is only available for bound workflow cells")
        return self._workflow_backend.state

    @property
    def block_reason(self) -> str | None:
        """Why a ``blocked`` cell is blocked: ``blocked-by-unwired``, ``blocked-by-error``, or None."""

        if self._workflow_backend is None:
            raise AttributeError("block_reason is only available for bound workflow cells")
        return self._workflow_backend.block_reason

    @property
    def mount(self):
        """Attach a whole Context cell to a file or directory."""
        if self._workflow_backend is None:
            raise AttributeError("mount is only available for bound workflow cells")
        from seamless_workflow.attachments.api import MountHandle
        return MountHandle(self._workflow_backend)

    @mount.deleter
    def mount(self):
        self.mount.unmount()

    @property
    def exception(self):
        """Return the exception associated with a failed workflow cell."""

        if self._workflow_backend is None:
            raise AttributeError("exception is only available for bound workflow cells")
        return self._workflow_backend.exception

    def set(self, value: Any) -> None:
        """Set the input to ``value``.

        A reference (see ``input_ref``) becomes the input as it is.  Any other
        value is serialized now, with the current celltype, and its checksum
        becomes the input.
        """
        if self._workflow_backend is not None:
            self._workflow_backend.set(value)
            return None
        value = _capture_workflow_source(value)
        if not _is_input_ref(value):
            value = _serialize_value(value, self._celltype)
        self.input_ref = value
        return None

    def set_checksum(self, checksum) -> None:
        if self._workflow_backend is not None:
            self._workflow_backend.set_checksum(checksum)
            return None
        from .checksum_class import Checksum

        self.input_ref = Checksum(checksum)
        return None

    def _refheld_checksums(self):
        from .checksum_class import Checksum

        if getattr(self, "_refholds_released", False):
            return ()
        if isinstance(self._input_ref, Checksum):
            return ((self._input_ref, "input"),)
        return ()

    def _release_refholds(self) -> None:
        if getattr(self, "_refholds_released", False):
            return
        object.__setattr__(self, "_refholds_released", True)
        from .checksum_class import Checksum

        input_ref = getattr(self, "_input_ref", None)
        if isinstance(input_ref, Checksum):
            input_ref.decref_refholder()

    def __del__(self):
        try:
            from .reference_lifecycle import safe_release_refholder

            safe_release_refholder(self)
        except Exception:
            pass

    def _derive(self, **updates: Any) -> "Cell":
        cls = updates.pop("_cls", None) or type(self)
        if self._workflow_backend is not None:
            result = self._workflow_backend.derive(**updates)
            return result if isinstance(result, Cell) else cls._from_backend(result)
        clone = cls(
            input_ref=self._input_ref,
            path=self._path,
            celltype=self._celltype,
            target_celltype=self._target_celltype,
            validator=self._validator,
            validator_language=self._validator_language,
        )
        for name, value in updates.items():
            setattr(clone, name, value)
        return clone

    def item(self, key: Any) -> "Cell":
        cls = SubCell if type(self) is Cell else type(self)
        if self._workflow_backend is not None:
            return cls._from_backend(self._workflow_backend.derive_item(key))
        return self._derive(path=append_item_path(self.path_python, key), _cls=cls)

    def slice(self, start: Any = None, stop: Any = None, step: Any = None) -> "Cell":
        cls = SubCell if type(self) is Cell else type(self)
        if self._workflow_backend is not None:
            return cls._from_backend(
                self._workflow_backend.derive_slice(start, stop, step)
            )
        return self._derive(
            path=append_slice_path(self.path_python, start, stop, step), _cls=cls
        )

    def as_celltype(self, target_celltype: str) -> "Cell":
        return self._derive(target_celltype=target_celltype)

    def with_input(self, input_ref: Any) -> "Cell":
        return self._derive(input_ref=input_ref)

    def with_validator(
        self, validator: Any, *, language: str | None = None
    ) -> "Cell":
        return self._derive(validator=validator, validator_language=language)

    def build(self, input_ref: Any = _UNSET) -> Expression:
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return self._workflow_backend.build(input_ref)
        if input_ref is _UNSET:
            input_ref = _capture_workflow_source(self._input_ref)
        return Expression(
            input_ref,
            path=self._path,
            celltype=self._celltype,
            target_celltype=self._target_celltype,
            validator=self._validator,
            validator_language=self._validator_language,
    )

    expression = build

    def __call__(self, input_ref: Any = _UNSET) -> Expression:
        return self.build(input_ref)

    def compute(self, input_ref: Any = _UNSET, *, timeout=None):
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return self._workflow_backend.compute(input_ref, timeout=timeout)
        return self.build(input_ref).compute()

    def run(self, input_ref: Any = _UNSET):
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return self._workflow_backend.run(input_ref)
        return self.build(input_ref).run()

    async def compute_async(self, input_ref: Any = _UNSET, *, timeout=None):
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return await self._workflow_backend.compute_async(input_ref, timeout=timeout)
        return await self.build(input_ref).compute_async()

    async def computation(self, timeout=None):
        return await self.compute_async(timeout=timeout)

    def prune(self):
        if self._workflow_backend is None:
            raise AttributeError("prune is only available for bound workflow cells")
        return self._workflow_backend.prune()

    def clear_exception(self):
        if self._workflow_backend is None:
            raise AttributeError(
                "clear_exception is only available for bound workflow cells"
            )
        return self._workflow_backend.clear_exception()

    def _workflow_endpoint(self):
        backend = self._workflow_backend
        return backend._workflow_endpoint() if backend is not None else None

    def _workflow_capture_source(self):
        backend = self._workflow_backend
        if backend is None:
            return self
        return backend.capture_source()

    def __getitem__(self, item: Any) -> "Cell":
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        return self.item(item)

    def __getattr__(self, name: str) -> "Cell":
        if name.startswith("_"):
            raise AttributeError(name)
        # A class-defined API member is authoritative even when its getter raises
        # a deliberate bound-only AttributeError.  Only genuinely unknown names
        # participate in structural projection.
        if _class_attribute(type(self), name) is not None:
            raise AttributeError(name)
        return self.item(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or _class_attribute(type(self), name) is not None:
            object.__setattr__(self, name, value)
            return
        if self._workflow_backend is None:
            raise AttributeError(name)
        self._workflow_backend.assign(self.path_python, name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_") or _class_attribute(type(self), name) is not None:
            object.__delattr__(self, name)
            return
        if self._workflow_backend is None:
            raise AttributeError(name)
        self._workflow_backend.delete(self.path_python, name)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._workflow_backend is None:
            raise TypeError("Standalone Cell item assignment is not supported")
        self._workflow_backend.assign_item(self.path_python, key, value)

    def __delitem__(self, key: Any) -> None:
        if self._workflow_backend is None:
            raise TypeError("Standalone Cell item deletion is not supported")
        self._workflow_backend.delete_item(self.path_python, key)

    def __iadd__(self, value: Any) -> "Cell":
        return self._augmented(value, "add")

    def __isub__(self, value: Any) -> "Cell":
        return self._augmented(value, "sub")

    def __imul__(self, value: Any) -> "Cell":
        return self._augmented(value, "mul")

    def __itruediv__(self, value: Any) -> "Cell":
        return self._augmented(value, "truediv")

    def _augmented(self, value: Any, operation: str) -> "Cell":
        if self._workflow_backend is None:
            raise TypeError("Augmented Cell updates require a bound Cell")
        self._workflow_backend.augmented(self.path_python, operation, value)
        return self

    def _repr_state(self) -> str:
        """``state=...`` for a bound cell, empty otherwise.  Never raises.

        A repr is what a REPL shows, and it is now the only place a bound cell
        summarises itself: the ``status`` string used to do that and is gone.  A stale
        or standalone handle simply omits the field rather than failing to print.
        """

        try:
            if self._workflow_backend is None:
                return ""
            return f", state={self._workflow_backend.state!r}"
        except Exception:
            return ""

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(input_ref={self.input_ref!r}, path={self.path_python!r}, "
            f"celltype={self.celltype!r}, target_celltype={self.target_celltype!r}"
            f"{self._repr_state()})"
        )


class SubCell(Cell):
    """A sub-path projection of a Cell: a handle, and never a value.

    Attribute access on a Cell is *structural projection* — ``ctx.a.x`` is the
    canonical way to source a sub-path — so a name that is not Cell API resolves
    to a projection rather than raising.  That makes two mistakes silent, and
    they are the same mistake: a typo (``ctx.a.vlaue``), and a name that used to
    be API and no longer is (``ctx.a.status``).  Either way the result is an
    object that compares unequal to everything and is truthy, so ``==`` fails
    confusingly while ``!=``, ``is not None`` and a bare ``assert`` all pass.

    A projection is loud instead.  Every operation that treats it as a *value*
    raises :class:`~seamless.cell_errors.ProjectionError`; the operations that
    treat it as a handle — further projection, assignment, ``.value``,
    ``.checksum``, ``.state`` — are untouched.  A bound ``Transformer`` needs
    none of this: it has no projection fallback, so a bad name there is already
    an ``AttributeError`` (with Python's own "Did you mean" suggestion).

    One gap has no fix.  ``x is None`` compiles to the ``IS_OP`` bytecode and
    compares pointers, with no protocol to intercept, so
    ``assert ctx.a.vlaue is not None`` still passes silently.  ``is None`` in the
    other direction is safe: it evaluates false, and the assertion fails.
    """

    #: Defining ``__eq__`` would otherwise set this to ``None`` and make every
    #: projection unhashable, breaking any set or dict of handles.
    __hash__ = Cell.__hash__

    def _not_a_value(self, operation: str):
        path = self.path_python or "<root>"
        raise ProjectionError(
            f"cannot {operation} the sub-path projection {path!r}: it is a handle, "
            f"not a value.  Read it with .value, or check whether {path!r} is a "
            f"misspelling of a Cell attribute."
        )

    def __eq__(self, other):
        self._not_a_value("compare")

    def __lt__(self, other):
        self._not_a_value("order")

    def __le__(self, other):
        self._not_a_value("order")

    def __gt__(self, other):
        self._not_a_value("order")

    def __ge__(self, other):
        self._not_a_value("order")

    def __bool__(self):
        self._not_a_value("test the truth of")

    def __len__(self):
        self._not_a_value("take the length of")

    def __iter__(self):
        self._not_a_value("iterate")

    def __repr__(self) -> str:
        path = self.path_python or "<root>"
        return f"<SubCell {path!r}: a handle, not a value \u2014 read it with .value>"


def _class_attribute(cls, name: str):
    """Return a statically defined member without invoking descriptors."""

    for parent in cls.__mro__:
        if name in parent.__dict__:
            return parent.__dict__[name]
    return None


def _bound_state_error(name: str):
    from .cell_errors import BoundStateError

    return BoundStateError(f"{name} is standalone-only for bound Cells")


def _is_input_ref(value: Any) -> bool:
    """Whether ``value`` is a reference that a Cell may hold as its input.

    Workflow sources (``_workflow_endpoint``) and transformation dependencies
    (``_compute_dependency``) are duck-typed, like ``_capture_workflow_source``.
    """

    from .checksum_class import Checksum

    if value is None or isinstance(value, (Checksum, Expression, Cell)):
        return True
    return any(
        callable(getattr(value, hook, None))
        for hook in ("_workflow_endpoint", "_compute_dependency")
    )


def _check_input_ref(value: Any) -> Any:
    if _is_input_ref(value):
        return value
    hint = "use .set() to give a Cell a value"
    if isinstance(value, (str, bytes)):
        hint += ", or wrap a checksum in Checksum(...)"
    raise TypeError(
        "Cell input_ref must be None, a Checksum, an Expression, a Cell or a "
        f"workflow source, not {type(value).__name__}; {hint}"
    )


def _input_override(input_ref: Any) -> Any:
    """Capture and type-check an input passed to build/compute/run."""

    if input_ref is _UNSET:
        return input_ref
    return _check_input_ref(_capture_workflow_source(input_ref))


def _serialize_value(value: Any, celltype: str):
    from .buffer_class import Buffer

    buffer = Buffer(value, celltype)
    # The tempref keeps the buffer resolvable until the Cell's refhold adopts it.
    buffer.tempref()
    return buffer.get_checksum()


def _capture_workflow_source(value: Any) -> Any:
    # This is intentionally a duck-typed protocol.  Core must remain importable
    # without the workflow package and must not know workflow view classes.
    capture = getattr(value, "_workflow_capture_source", None)
    if callable(capture):
        return capture()
    return value


__all__ = ["Cell"]
