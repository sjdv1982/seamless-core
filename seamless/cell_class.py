"""Mutable structural Cell builder."""

from __future__ import annotations

from typing import Any

from .cell_errors import ProjectionError
from .retired_names import check_retired_name
from .expression_class import (
    Expression,
    append_item_path,
    append_slice_path,
)

_UNSET = object()


class CellBase:
    """Shared value, type, evaluation, and ownership API for Cells and Pins.

    Structural navigation and workflow source protocols belong to Cell only.
    Pin backends supply Transformer-owned state through the same value API.
    """

    __slots__ = ("_workflow_backend", "_standalone_input_ref", "_input_celltype",
                 "_celltype", "_standalone_exception", "_standalone_result_checksum",
                 "_refholds_released", "_standalone_recipe_key", "_standalone_expression", "__weakref__")

    __hash__ = object.__hash__

    def _not_a_value(self, operation):
        path = self.path if isinstance(self, Cell) else ""
        detail = (f"sub-path projection {path!r}; check for a misspelled Cell attribute"
                  if path else type(self).__name__)
        raise ProjectionError(f"cannot {operation} {detail}: it is a handle, not a value; read .value")

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

    def build(self):
        return self._workflow_backend.build(_UNSET)

    def __getattr__(self, name):
        check_retired_name(name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        check_retired_name(name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        check_retired_name(name)
        object.__delattr__(self, name)

    def __repr__(self):
        return (f"{type(self).__name__}(input_celltype={self.input_celltype!r}, "
                f"celltype={self.celltype!r}{self._repr_state()})")

    @property
    def _input_ref(self):
        if self._workflow_backend is not None:
            return self._workflow_backend._input_ref
        return self._standalone_input_ref

    @_input_ref.setter
    def _input_ref(self, value):
        self._standalone_input_ref = value

    @property
    def source(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.source
        from .checksum_class import Checksum
        return None if isinstance(self._input_ref, Checksum) else self._input_ref

    def _replace_input_ref(self, input_ref: Any, *, input_celltype=None) -> None:
        from .checksum_class import Checksum
        _check_input_ref(input_ref)
        typed = _typed_input_celltype(input_ref)
        if typed is not None and input_celltype is not None and typed != input_celltype:
            raise ValueError("input_celltype disagrees with the typed source's celltype")
        old = self._input_ref
        if isinstance(input_ref, Checksum):
            input_ref.incref_refholder()
        self._input_ref = input_ref
        self._input_celltype = None if input_ref is None else typed or input_celltype or self.celltype
        self._standalone_exception = None
        self._set_result_checksum(None)
        if isinstance(old, Checksum):
            old.decref_refholder()

    @property
    def input_celltype(self) -> str | None:
        if self._workflow_backend is not None:
            return self._workflow_backend.input_celltype
        return _typed_input_celltype(self._input_ref) or self._input_celltype

    @property
    def celltype(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.celltype
        return self._celltype

    @celltype.setter
    def celltype(self, celltype: str | None) -> None:
        if self._workflow_backend is not None:
            self._workflow_backend.celltype = celltype
            return
        from .buffer_class import Buffer
        if celltype is None:
            raise TypeError("celltype must name a supported type")
        Buffer._map_celltype(celltype)
        _check_projected_source(self.source, celltype)
        if self._path and self.input_celltype not in ("deepcell", "deepfolder", "folder") and celltype != self.input_celltype:
            raise TypeError("Cannot convert behind a projection; use as_celltype()")
        self._celltype = celltype
        self._standalone_exception = None
        self._set_result_checksum(None)

    @property
    def checksum(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.checksum
        self._sync_recipe()
        if self._input_ref is None or self._miswired():
            return None
        if self._standalone_exception is not None:
            return None
        if self._standalone_result_checksum is not None:
            return self._standalone_result_checksum
        from .checksum_class import Checksum
        if (isinstance(self._input_ref, Checksum) and not self._path
                and self.input_celltype == self.celltype and self._validator is None):
            from .checksum.null import canonicalize_checksum
            return canonicalize_checksum(self._input_ref, self.celltype)
        from .error_envelope import RunningLoopRefusal
        try:
            expression = self.build()
            self._standalone_expression = expression
            result = _available_input_checksum(expression)
        except RunningLoopRefusal:
            return None
        except Exception as exc:
            from .error_envelope import execution_error
            self._standalone_exception = execution_error(exc)
            return None
        self._set_result_checksum(result)
        return result

    @checksum.setter
    def checksum(self, value):
        self._write_checksum(value, detach=True)

    @property
    def buffer(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.buffer
        checksum = self.checksum
        if checksum is None:
            if self._standalone_exception is not None:
                raise self._standalone_exception
            return None
        try:
            from .checksum.hash_type_validation import validate_deserializable_as
            from .buffer_class import Buffer
            celltype = Buffer._map_celltype(self.celltype)
            validate_deserializable_as(checksum, celltype)
            buffer = checksum.resolve()
            validate_deserializable_as(checksum, celltype, buffer=buffer)
            return buffer
        except Exception as exc:
            return self._handle_materialization_error(exc)

    @buffer.setter
    def buffer(self, value):
        self._write_buffer(value, detach=True)

    @property
    def value(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.value
        checksum = self.checksum
        if checksum is None:
            if self._standalone_exception is not None:
                raise self._standalone_exception
            return None
        try:
            value = checksum.resolve(self.celltype)
        except Exception as exc:
            return self._handle_materialization_error(exc)
        return value.content if self.celltype == "bytes" and hasattr(value, "content") else value

    @value.setter
    def value(self, value):
        self._write_value(value, detach=True)

    @property
    def state(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.state
        self._sync_recipe()
        if self._miswired():
            return "miswired"
        if self._input_ref is None:
            return "unwired"
        if self._standalone_exception is not None:
            return "failed"
        if self._standalone_result_checksum is not None:
            return "complete"
        # State inspection must not pull the recipe. A literal dummy result
        # is already available; all other recipes wait for a demanding read
        # or an explicit compute, even if their inputs are locally cached.
        from .checksum_class import Checksum
        if (isinstance(self._input_ref, Checksum) and not self._path
                and self.input_celltype == self.celltype and self._validator is None):
            return "complete"
        return "waiting"

    @property
    def exception(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.exception
        self._sync_recipe()
        return str(self._standalone_exception) if self._standalone_exception is not None else None

    def clear_exception(self):
        if self._workflow_backend is None:
            self._standalone_exception = None
            return
        return self._workflow_backend.clear_exception()

    def _handle_materialization_error(self, exc):
        from . import CacheMissError
        if isinstance(exc, CacheMissError):
            raise exc
        from .error_envelope import execution_error
        self._standalone_exception = execution_error(exc)
        raise self._standalone_exception

    def _check_write_authority(self, detach):
        if self._path:
            raise TypeError("Standalone projection writes are not supported")
        if not detach and self.source is not None:
            from .cell_errors import AuthorityError
            raise AuthorityError("The input is controlled by a source; assign .value, .buffer or .checksum to replace it")

    def _write_value(self, value, *, detach=False):
        # .set() and .value take values; a reference is connected by assignment.
        # A Checksum is a value exactly when the celltype is checksum.
        from .checksum_class import Checksum
        checksum_value = isinstance(value, Checksum) and self.celltype == "checksum"
        if value is not None and not checksum_value and _is_input_ref(value):
            if isinstance(value, Checksum):
                raise TypeError(
                    "A Checksum is not a value: use .set_checksum() "
                    "(a Checksum is a value only for celltype 'checksum')"
                )
            raise TypeError(
                f"A {type(value).__name__} is not a value: connect it by assignment, "
                "or with Cell(source=...)"
            )
        if self._workflow_backend is not None:
            return self._workflow_backend.write_value(value, detach=detach)
        self._check_write_authority(detach)
        self._replace_input_ref(_serialize_value(value, self.celltype), input_celltype=self.celltype)

    def _write_checksum(self, checksum, *, input_celltype=None, detach=False):
        if self._workflow_backend is not None:
            return self._workflow_backend.write_checksum(checksum, input_celltype=input_celltype, detach=detach)
        from .checksum_class import Checksum
        self._check_write_authority(detach)
        self._replace_input_ref(None if checksum is None else Checksum(checksum), input_celltype=input_celltype)

    def _write_buffer(self, buffer, *, detach=False):
        if self._workflow_backend is not None:
            return self._workflow_backend.write_buffer(buffer, detach=detach)
        self._check_write_authority(detach)
        checksum = _checksum_for_buffer(buffer, self.celltype)
        self._write_checksum(checksum, detach=detach)

    def set(self, value: Any) -> None:
        self._write_value(value)

    def set_checksum(self, checksum, *, input_celltype=None) -> None:
        self._write_checksum(checksum, input_celltype=input_celltype)

    def set_buffer(self, buffer) -> None:
        self._write_buffer(buffer)

    def _set_result_checksum(self, result):
        old = getattr(self, "_standalone_result_checksum", None)
        if result is not None:
            result.incref_refholder()
        self._standalone_result_checksum = result
        if result is None:
            self._standalone_expression = None
        if old is not None:
            old.decref_refholder()

    def _refheld_checksums(self):
        from .checksum_class import Checksum

        if getattr(self, "_refholds_released", False):
            return ()
        claims = []
        if isinstance(self._input_ref, Checksum):
            claims.append((self._input_ref, "input"))
        result = getattr(self, "_standalone_result_checksum", None)
        if result is not None:
            claims.append((result, "result"))
        return tuple(claims)

    def _release_refholds(self) -> None:
        if getattr(self, "_refholds_released", False):
            return
        object.__setattr__(self, "_refholds_released", True)
        from .checksum_class import Checksum

        input_ref = getattr(self, "_input_ref", None)
        if isinstance(input_ref, Checksum):
            input_ref.decref_refholder()
        self._set_result_checksum(None)
        self._standalone_expression = None

    def __del__(self):
        try:
            from .reference_lifecycle import safe_release_refholder

            safe_release_refholder(self)
        except Exception:
            pass

    def compute(self, input_ref: Any = _UNSET, *, timeout=None):
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return self._workflow_backend.compute(input_ref, timeout=timeout)
        if input_ref is not _UNSET:
            return self.with_input(input_ref).compute()
        self._sync_recipe()
        if self._miswired():
            return None
        try:
            self._standalone_expression = self.build(input_ref)
            result = self._standalone_expression.compute()
        except Exception as exc:
            from .error_envelope import execution_error
            self._standalone_exception = execution_error(exc)
            self._set_result_checksum(None)
            return None
        self._standalone_exception = None
        self._set_result_checksum(result)
        return result

    def run(self, input_ref: Any = _UNSET):
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return self._workflow_backend.run(input_ref)
        if input_ref is not _UNSET:
            return self.with_input(input_ref).run()
        result = self.compute(input_ref)
        if result is None:
            if self._standalone_exception is not None:
                raise self._standalone_exception
            return None
        return self.value

    async def compute_async(self, input_ref: Any = _UNSET, *, timeout=None):
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return await self._workflow_backend.compute_async(input_ref, timeout=timeout)
        if input_ref is not _UNSET:
            return await self.with_input(input_ref).compute_async()
        self._sync_recipe()
        if self._miswired():
            return None
        try:
            self._standalone_expression = self.build(input_ref)
            result = await self._standalone_expression.compute_async()
        except Exception as exc:
            from .error_envelope import execution_error
            self._standalone_exception = execution_error(exc)
            self._set_result_checksum(None)
            return None
        self._standalone_exception = None
        self._set_result_checksum(result)
        return result

    async def computation(self, timeout=None):
        return await self.compute_async(timeout=timeout)

    def _sync_recipe(self):
        key = _cell_recipe_key(self)
        if key != self._standalone_recipe_key:
            self._standalone_recipe_key = key
            self._standalone_exception = None
            self._set_result_checksum(None)
            self._standalone_expression = None

    def _miswired(self):
        source = self.source
        if isinstance(source, Cell) and source._workflow_backend is None and source._miswired():
            return True
        return bool(self._path and self.input_celltype not in ("deepcell", "deepfolder", "folder")
                    and self.input_celltype != self.celltype)

    def fingertip(self):
        if self._workflow_backend is not None:
            checksum = self._workflow_backend.checksum
        else:
            self._sync_recipe()
            if self._standalone_exception is not None or self._miswired():
                return None
            checksum = self._standalone_result_checksum
            from .checksum_class import Checksum
            if (checksum is None and isinstance(self._input_ref, Checksum)
                    and not self._path and self.input_celltype == self.celltype
                    and self._validator is None):
                from .checksum.null import canonicalize_checksum
                checksum = canonicalize_checksum(self._input_ref, self.celltype)
        if checksum is None:
            return None
        return checksum.fingertip_sync()

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



class Cell(CellBase):
    """Mutable structural expression builder.

    Navigation creates derived Cell builders. ``build()`` snapshots the current
    builder state into an immutable ``Expression`` container.

    The positional argument is the produced ``celltype``. Give a typed input
    with ``source=`` or a checksum with ``checksum=``; use ``set()`` for values.
    The input type is read-only and follows the source or the stored checksum's
    declaration. Retyping changes the output conversion, not the stored input.
    """

    __slots__ = ("_path", "_validator", "_validator_language")

    def __init__(
        self, celltype: str | None = None, *, checksum: Any = _UNSET,
        source: Any = _UNSET, input_celltype: str | None = None,
        validator: Any = None,
        validator_language: str | None = None,
    ) -> None:
        from .checksum_class import Checksum
        from .buffer_class import Buffer
        from .reference_lifecycle import register_refholder

        if checksum is not _UNSET and source is not _UNSET:
            raise TypeError("checksum and source are mutually exclusive inputs")
        if source is not _UNSET:
            if source is not None and (isinstance(source, Checksum) or not _is_input_ref(source)):
                raise TypeError("source must be a typed reference; use checksum= for a checksum")
            ref = source
        else:
            ref = None if checksum is _UNSET or checksum is None else Checksum(checksum)
        declared = _typed_input_celltype(ref)
        if declared is not None and input_celltype is not None and input_celltype != declared:
            raise ValueError("input_celltype disagrees with the typed source's celltype")
        celltype = celltype if celltype is not None else declared or "mixed"
        Buffer._map_celltype(celltype)
        self._workflow_backend = None
        self._input_ref = ref
        self._path = ""
        _check_projected_source(ref, celltype)
        self._celltype = celltype
        self._input_celltype = None if ref is None else declared or input_celltype or celltype
        self._validator = validator
        self._validator_language = validator_language
        self._standalone_exception = None
        self._set_result_checksum(None)
        self._standalone_recipe_key = None
        self._standalone_expression = None
        self._refholds_released = False
        register_refholder(self)
        if isinstance(ref, Checksum):
            ref.incref_refholder()

    @classmethod
    def _from_backend(cls, backend) -> "Cell":
        """Create a canonical Cell whose state is entirely backend-owned."""

        self = cls.__new__(cls)
        object.__setattr__(self, "_workflow_backend", backend)
        object.__setattr__(self, "_input_ref", None)
        object.__setattr__(self, "_path", "")
        object.__setattr__(self, "_input_celltype", "mixed")
        object.__setattr__(self, "_celltype", "mixed")
        object.__setattr__(self, "_validator", None)
        object.__setattr__(self, "_validator_language", None)
        object.__setattr__(self, "_standalone_exception", None)
        object.__setattr__(self, "_standalone_result_checksum", None)
        object.__setattr__(self, "_refholds_released", True)
        return self





    @property
    def path(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.path
        return self._path

    @property
    def path_python(self) -> str:
        if self._workflow_backend is not None:
            return self._workflow_backend.path_python
        return self._path




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
        self._standalone_exception = None
        self._set_result_checksum(None)

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
        self._standalone_exception = None
        self._set_result_checksum(None)








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












    def _derive(self, **updates: Any) -> "Cell":
        if self._workflow_backend is not None:
            result = self._workflow_backend.derive(**updates)
            return result if isinstance(result, Cell) else Cell._from_backend(result)
        ref = updates.pop("_input_ref", self._input_ref)
        from .checksum_class import Checksum
        recipe = {"checksum": ref} if ref is None or isinstance(ref, Checksum) else {"source": ref}
        clone = Cell(**recipe, celltype=self.celltype,
                     input_celltype=self.input_celltype if ref is self._input_ref else None,
                     validator=self.validator, validator_language=self.validator_language)
        clone._path = self._path
        for name, value in updates.items():
            setattr(clone, name, value)
        return clone

    def item(self, key: Any) -> "Cell":
        if self._workflow_backend is not None:
            return Cell._from_backend(self._workflow_backend.derive_item(key))
        child = Cell(source=self)
        child._path = append_item_path("", key)
        if self.celltype in ("deepcell", "deepfolder", "folder"):
            child._celltype = "mixed" if self.celltype == "deepcell" else "bytes"
        return child

    def slice(self, start: Any = None, stop: Any = None, step: Any = None) -> "Cell":
        if self._workflow_backend is not None:
            return Cell._from_backend(self._workflow_backend.derive_slice(start, stop, step))
        child = Cell(source=self)
        child._path = append_slice_path("", start, stop, step)
        return child

    def as_celltype(self, celltype: str) -> "Cell":
        if self._workflow_backend is not None:
            return self._derive(celltype=celltype)
        # An explicit conversion is a separate, pathless link.
        child = Cell(source=self)
        from .buffer_class import Buffer
        Buffer._map_celltype(celltype)
        child._celltype = celltype
        return child

    def with_input(self, input_ref: Any) -> "Cell":
        return self._derive(_input_ref=input_ref)

    def with_validator(self, validator: Any, *, language: str | None = None) -> "Cell":
        return self._derive(validator=validator, validator_language=language)

    def build(self, input_ref: Any = _UNSET) -> Expression:
        overridden = input_ref is not _UNSET
        input_ref = _input_override(input_ref)
        if self._workflow_backend is not None:
            return self._workflow_backend.build(input_ref)
        if input_ref is _UNSET:
            input_ref = _capture_workflow_source(self._input_ref)
        return Expression(
            input_ref, path=self._path,
            input_celltype=(_typed_input_celltype(input_ref) or self.celltype) if overridden else self.input_celltype,
            celltype=self._celltype, validator=self._validator,
            validator_language=self._validator_language,
        )

    expression = build

    def __call__(self, input_ref: Any = _UNSET) -> Expression:
        return self.build(input_ref)





    def prune(self):
        if self._workflow_backend is None:
            raise AttributeError("prune is only available for bound workflow cells")
        return self._workflow_backend.prune()

    def clear_exception(self):
        if self._workflow_backend is None:
            self._standalone_exception = None
            return
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
        check_retired_name(name)
        if name.startswith("_"):
            raise AttributeError(name)
        # A class-defined API member is authoritative even when its getter raises
        # a deliberate bound-only AttributeError.  Only genuinely unknown names
        # participate in structural projection.
        if _class_attribute(type(self), name) is not None:
            raise AttributeError(name)
        return self.item(name)

    def __setattr__(self, name: str, value: Any) -> None:
        check_retired_name(name)
        if name.startswith("_") or _class_attribute(type(self), name) is not None:
            object.__setattr__(self, name, value)
            return
        if self._workflow_backend is None:
            raise AttributeError(name)
        self._workflow_backend.assign(self.path_python, name, value)

    def __delattr__(self, name: str) -> None:
        check_retired_name(name)
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


    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(source={self.source!r}, path={self.path_python!r}, "
            f"input_celltype={self.input_celltype!r}, celltype={self.celltype!r}"
            f"{self._repr_state()})"
        )



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

    if isinstance(value, CellBase) and not isinstance(value, Cell):
        raise TypeError("a Pin can't be a source; connect pin.source instead")
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
        "Cell input must be None, a Checksum, an Expression, a Cell or a "
        f"workflow source, not {type(value).__name__}; {hint}"
    )


def _input_override(input_ref: Any) -> Any:
    """Capture and type-check an input passed to build/compute/run."""

    if input_ref is _UNSET:
        return input_ref
    return _check_input_ref(_capture_workflow_source(input_ref))


def _serialize_value(value: Any, input_celltype: str):
    from .buffer_class import Buffer

    buffer = Buffer(value, input_celltype)
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


__all__ = ["Cell", "CellBase"]


def _available_input_checksum(value):
    """Read an input result without starting transformation work."""
    from .checksum_class import Checksum

    if isinstance(value, Checksum):
        return value
    if isinstance(value, Expression):
        if isinstance(value._input_ref, Checksum):
            return value.compute()
        checksum = _available_input_checksum(value._input_ref)
        if checksum is None:
            return None
        # Freeze the available input so compute cannot start (or wait on) a
        # Transformation, even when its recorded result is already available.
        concrete = Expression(
            checksum, path=value.path, input_celltype=value.input_celltype,
            celltype=value.celltype, validator=value.validator,
            validator_language=value.validator_language,
        )
        result = concrete.compute()
        value._enable_result_holding()
        return value._publish_result(result) if result is not None else None
    if isinstance(value, Cell):
        return value.checksum
    result_getter = getattr(value, "_result_checksum_internal", None)
    if callable(result_getter):
        return result_getter()
    return None


def _typed_input_celltype(value):
    if value is None:
        return None
    from .checksum_class import Checksum
    if isinstance(value, Checksum):
        return None
    if isinstance(value, (Cell, Expression)) or any(callable(getattr(value, hook, None)) for hook in ("_workflow_endpoint", "_compute_dependency")):
        return value.celltype
    return None


def _checksum_for_buffer(value, celltype):
    if value is None:
        return None
    from .buffer_class import Buffer
    from .checksum.hash_type_validation import validate_deserializable_as
    from .checksum.null import NULL_BUFFER
    buffer = value if isinstance(value, Buffer) else Buffer(value)
    if celltype == "bytes" and buffer.content == b"":
        buffer = Buffer(NULL_BUFFER)
    checksum = buffer.get_checksum()
    validate_deserializable_as(checksum, Buffer._map_celltype(celltype), buffer=buffer)
    buffer.get_value(celltype)
    buffer.tempref()
    return checksum


def _check_projected_source(source, celltype):
    if isinstance(source, (Cell, Expression)) and source.path and source.celltype != celltype:
        raise TypeError("Cannot implicitly convert behind a projection; use as_celltype() before or after projecting")


def _cell_recipe_key(cell):
    """Configuration-only revision key; never compare handles or pull work."""
    ref = cell._input_ref
    if isinstance(ref, Cell) and ref._workflow_backend is None:
        source = _cell_recipe_key(ref)
    else:
        from .checksum_class import Checksum
        source = ("checksum", ref.hex()) if isinstance(ref, Checksum) else ("object", id(ref))
    return (source, cell.celltype, cell.input_celltype, cell._path,
            id(cell._validator), cell._validator_language)
