"""Container class for Seamless structural expressions."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any

from .checksum_class import Checksum
from .retired_names import check_retired_name


def normalize_path(path: str | None) -> str:
    """Normalize the expression path container field.

    Path parsing, validation, and application are intentionally deferred.
    """

    if path is None:
        return ""
    if not isinstance(path, str):
        raise TypeError("Expression path must be a string")
    return path


def append_item_path(path: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        prefix = "." if path else ""
        return f"{path}{prefix}{key}"
    return f"{path}[{key!r}]"


def append_slice_path(
    path: str,
    start: Any = None,
    stop: Any = None,
    step: Any = None,
) -> str:
    parts = [
        "" if start is None else repr(start),
        "" if stop is None else repr(stop),
    ]
    if step is not None:
        parts.append(repr(step))
    return path + "[" + ":".join(parts) + "]"


def _input_ref_key(input_ref: Any) -> tuple[str, Any]:
    if isinstance(input_ref, Checksum):
        return ("checksum", input_ref.hex())
    if isinstance(input_ref, Expression):
        return ("expression", input_ref.identity_key)
    try:
        checksum = Checksum(input_ref)
    except (TypeError, ValueError):
        return ("object", id(input_ref))
    else:
        return ("checksum", checksum.hex())


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class Expression:
    """Immutable structural expression definition.

    This class intentionally carries only the definition shape. Resolution,
    caching, reverse lookup, cancellation, path validation, and value
    materialization are later mechanics.
    """

    _input_ref: Any
    path: str | None = ""
    _: KW_ONLY
    input_celltype: str | None = None
    celltype: str | None = None
    validator: Checksum | str | bytes | None = None
    validator_language: str | None = None
    _result_checksum: Checksum | None = field(
        init=False, default=None, compare=False, repr=False
    )
    _refhold_result: bool = field(
        init=False, default=False, compare=False, repr=False
    )
    _result_refheld: bool = field(
        init=False, default=False, compare=False, repr=False
    )
    _refholds_released: bool = field(
        init=False, default=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        from .cell_class import Cell, _typed_input_celltype, _check_input_ref
        ref = self._input_ref
        _check_input_ref(ref)
        typed = _typed_input_celltype(ref)
        if typed is not None and self.input_celltype is not None and self.input_celltype != typed:
            raise ValueError("input_celltype disagrees with the typed source's celltype")
        if isinstance(ref, Cell):
            ref = ref.build()
            object.__setattr__(self, "_input_ref", ref)
        if self.input_celltype is None:
            object.__setattr__(self, "input_celltype", typed or self.celltype or "mixed")
        path = normalize_path(self.path)
        celltype = (
            self.input_celltype if self.celltype is None else self.celltype
        )
        validator = None if self.validator is None else Checksum(self.validator)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "celltype", celltype)
        object.__setattr__(self, "validator", validator)
        from .reference_lifecycle import register_refholder

        if self.input_checksum is not None:
            self.input_checksum.incref_refholder()
        register_refholder(self)

    @property
    def source(self):
        return None if isinstance(self._input_ref, Checksum) else self._input_ref

    @property
    def checksum(self):
        return self.result

    @property
    def result(self) -> Checksum | None:
        """Return the published result and express user result interest."""

        self._enable_result_holding()
        return self._result_checksum

    def _result_checksum_internal(self) -> Checksum | None:
        """Read the result without changing lifecycle ownership."""

        return self._result_checksum

    def _evaluate_internal(self, *, execution: str = "local") -> Checksum | None:
        """Evaluate and publish without expressing user result interest.

        Dependency schedulers use this entry point.  Publication is still
        centralized here, so a downstream holder can adopt the concrete
        checksum even when the Expression itself remains neutral.
        """

        from .checksum.expression import evaluate_expression, evaluate_expression_remote

        input_ref = self._input_ref
        if isinstance(input_ref, Expression):
            input_checksum = input_ref._evaluate_internal(execution=execution)
        elif hasattr(input_ref, "_compute_dependency"):
            input_ref._compute_dependency()
            input_checksum = input_ref._result_checksum_internal()
        else:
            input_checksum = self.input_checksum
        if input_checksum is None:
            raise ValueError("Expression input is not a concrete checksum yet")
        if execution != "local":
            # The async remote evaluator is the only non-blocking evaluator.
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._evaluate_internal_async(execution=execution))
            raise RuntimeError(
                "Cannot block on remote expression evaluation in a running loop"
            )
        result = evaluate_expression(
            input_checksum,
            self.path,
            self.input_celltype,
            self.celltype,
            validator=self.validator,
            validator_language=self.validator_language,
        )
        return self._publish_result(result)

    async def _evaluate_internal_async(
        self, *, execution: str = "local"
    ) -> Checksum | None:
        """Async counterpart of :meth:`_evaluate_internal`."""

        from .checksum.expression import evaluate_expression_async, evaluate_expression_remote

        input_ref = self._input_ref
        if isinstance(input_ref, Expression):
            input_checksum = await input_ref._evaluate_internal_async(execution=execution)
        elif hasattr(input_ref, "_compute_dependency_async"):
            await input_ref._compute_dependency_async(require_value=False)
            input_checksum = input_ref._result_checksum_internal()
        elif hasattr(input_ref, "_compute_dependency"):
            input_ref._compute_dependency()
            input_checksum = input_ref._result_checksum_internal()
        else:
            input_checksum = self.input_checksum
        if input_checksum is None:
            raise ValueError("Expression input is not a concrete checksum yet")
        if execution == "local":
            result = await evaluate_expression_async(
                input_checksum,
                self.path,
                self.input_celltype,
                self.celltype,
                validator=self.validator,
                validator_language=self.validator_language,
            )
        else:
            result = await evaluate_expression_remote(
                input_checksum,
                self.path,
                self.input_celltype,
                self.celltype,
                validator=self.validator,
                validator_language=self.validator_language,
                execution=execution,
                member_id=id(self),
            )
        return self._publish_result(result)

    def _enable_result_holding(self) -> None:
        if self._refholds_released:
            return
        if not self._refhold_result:
            object.__setattr__(self, "_refhold_result", True)
        if self._result_checksum is not None and not self._result_refheld:
            self._result_checksum.incref_refholder()
            object.__setattr__(self, "_result_refheld", True)

    def _publish_result(self, result: Checksum | str | bytes | None) -> Checksum | None:
        """Publish a result, keeping a tempref and optional user hold."""

        if result is None:
            return None
        checksum = Checksum(result)
        checksum.tempref()
        old = self._result_checksum
        if old is not None and old == checksum:
            if self._refhold_result and not self._result_refheld:
                checksum.incref_refholder()
                object.__setattr__(self, "_result_refheld", True)
            return checksum

        old_refheld = self._result_refheld
        new_refheld = self._refhold_result and not self._refholds_released
        if new_refheld:
            checksum.incref_refholder()
        object.__setattr__(self, "_result_checksum", checksum)
        object.__setattr__(self, "_result_refheld", new_refheld)
        if old is not None and old_refheld:
            old.decref_refholder()
        return checksum

    def _refheld_checksums(self):
        if self._refholds_released:
            return ()
        claims = []
        if self.input_checksum is not None:
            claims.append((self.input_checksum, "input"))
        if self._refhold_result and self._result_checksum is not None:
            claims.append((self._result_checksum, "result"))
        return tuple(claims)

    def _release_refholds(self) -> None:
        if self._refholds_released:
            return
        object.__setattr__(self, "_refholds_released", True)
        if self.input_checksum is not None:
            self.input_checksum.decref_refholder()
        if self._result_refheld and self._result_checksum is not None:
            self._result_checksum.decref_refholder()
            object.__setattr__(self, "_result_refheld", False)

    def __del__(self):
        try:
            from .reference_lifecycle import safe_release_refholder

            safe_release_refholder(self)
        except Exception:
            pass

    @property
    def input_checksum(self) -> Checksum | None:
        try:
            return Checksum(self._input_ref)
        except (TypeError, ValueError):
            return self._input_ref if isinstance(self._input_ref, Checksum) else None

    @property
    def identity_key(self) -> tuple[Any, str, str, str]:
        return (
            _input_ref_key(self._input_ref),
            self.path,
            self.input_celltype,
            self.celltype,
        )

    @property
    def path_python(self) -> str:
        return self.path

    @property
    def database_key(self) -> tuple[str, str, str, str]:
        input_checksum = self.input_checksum
        if input_checksum is None:
            raise ValueError("Expression input is not a concrete checksum yet")
        return (
            input_checksum.hex(),
            self.path,
            self.input_celltype,
            self.celltype,
        )

    def with_result(self, result: Checksum | str | bytes | None) -> "Expression":
        clone = replace(self)
        if result is not None:
            object.__setattr__(clone, "_result_checksum", Checksum(result))
        return clone

    def item(self, key: Any) -> "Expression":
        return replace(self, path=append_item_path(self.path, key))

    def slice(
        self,
        start: Any = None,
        stop: Any = None,
        step: Any = None,
    ) -> "Expression":
        return replace(self, path=append_slice_path(self.path, start, stop, step))

    def as_celltype(self, celltype: str) -> "Expression":
        return replace(self, celltype=celltype)

    def __getitem__(self, item: Any) -> "Expression":
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        return self.item(item)

    def __getattr__(self, name: str) -> "Expression":
        check_retired_name(name)
        if name.startswith("_"):
            raise AttributeError(name)
        return self.item(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            return False
        return self.identity_key == other.identity_key

    def __hash__(self) -> int:
        return hash(self.identity_key)

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}({self._input_ref!r}, path={self.path!r}, "
            f"input_celltype={self.input_celltype!r}, celltype={self.celltype!r})"
        )

    async def compute_async(self, *, execution: str = "local") -> Checksum | None:
        self._enable_result_holding()
        return await self._evaluate_internal_async(execution=execution)

    def compute(self, *, execution: str = "local") -> Checksum | None:
        self._enable_result_holding()
        return self._evaluate_internal(execution=execution)

    def run(self) -> Any:
        from .checksum.expression import resolve_expression_value

        result = self.compute()
        if result is None:
            return None
        return resolve_expression_value(result, self.celltype)

    __call__ = run

    def cancel(self) -> bool:
        """Soft-cancel this expression's active remote evaluation, if any."""

        input_checksum = self.input_checksum
        if input_checksum is None:
            return False
        from .checksum.expression import cancel_expression

        return cancel_expression(
            input_checksum,
            self.path,
            self.input_celltype,
            self.celltype,
            member_id=id(self),
        )

__all__ = [
    "Expression",
    "append_item_path",
    "append_slice_path",
    "normalize_path",
]
