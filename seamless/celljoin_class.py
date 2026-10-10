"""Expression-like container for a celljoin definition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

from .checksum.celljoin import (
    CellJoinSpec,
    build_celljoin,
    celljoin_buffer,
    celljoin_cache_key,
    parse_celljoin,
    publish_celljoin_definition,
    required_buffers,
)
from .checksum_class import Checksum
from .expression_class import Expression


@dataclass(frozen=True, slots=True, eq=False)
class CellJoin(Expression):
    """An immutable, multi-input expression for a cell-level join."""

    _spec: CellJoinSpec | None = field(
        default=None, kw_only=True, compare=False, repr=False
    )

    @classmethod
    def from_inputs(cls, celltype, root, members) -> "CellJoin":
        definition = build_celljoin(root, members)
        buffer = celljoin_buffer(definition)
        spec = parse_celljoin(buffer, celltype)
        publish_celljoin_definition(spec)
        return cls(
            spec.checksum,
            path="",
            input_celltype=None,
            celltype=spec.celltype,
            _spec=spec,
        )

    def __post_init__(self) -> None:
        spec = self._spec
        if not isinstance(spec, CellJoinSpec):
            raise TypeError("CellJoin must be constructed with CellJoin.from_inputs")
        if self.validator is not None or self.validator_language is not None:
            raise NotImplementedError("CellJoin validators are not implemented")

        object.__setattr__(self, "_input_ref", spec.checksum)
        object.__setattr__(self, "path", "")
        object.__setattr__(self, "input_celltype", None)
        object.__setattr__(self, "celltype", spec.celltype)
        object.__setattr__(self, "validator", None)
        object.__setattr__(self, "validator_language", None)
        spec.buffer.tempref()

        from .reference_lifecycle import register_refholder

        register_refholder(self)

    def __copy__(self):
        duplicate = replace(self)
        result = self._result_checksum
        if result is not None:
            result.tempref()
            object.__setattr__(duplicate, "_result_checksum", result)
        object.__setattr__(duplicate, "_refhold_result", self._refhold_result)
        if result is not None and self._refhold_result and not self._refholds_released:
            result.incref_refholder(scratch=None)
            object.__setattr__(duplicate, "_result_refheld", True)
        if self._refholds_released:
            duplicate._release_refholds()
        return duplicate

    def _available_result(self, *, wait: bool = True) -> Checksum | None:
        result = self._result_checksum_internal()
        if result is not None or not wait:
            return result
        from .checksum.expression import wait_for_active_key

        result = wait_for_active_key(
            celljoin_cache_key(self.celljoin_checksum, self.celltype)
        )
        return self._hold_result(result) if result is not None else None

    def _evaluate_internal(
        self,
        *,
        execution: str = "auto",
        scratch: bool = True,
        materialize: bool | None = None,
        run_source: bool = True,
    ) -> Checksum | None:
        if materialize is None:
            materialize = not scratch
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._evaluate_internal_async(
                    execution=execution,
                    scratch=scratch,
                    materialize=materialize,
                    run_source=run_source,
                )
            )

        from .checksum.expression import _has_local_buffer, get_expression_cache
        from .error_envelope import RunningLoopRefusal

        result = self._result_checksum_internal()
        if result is None:
            result = get_expression_cache().get(
                celljoin_cache_key(self.celljoin_checksum, self.celltype)
            )
        if result is not None and (not materialize or _has_local_buffer(result)):
            return self._hold_result(result)
        raise RunningLoopRefusal(
            "Cannot block on celljoin evaluation in a running loop"
        )

    async def _evaluate_internal_async(
        self,
        *,
        execution: str = "auto",
        scratch: bool = True,
        materialize: bool | None = None,
        run_source: bool = True,
    ) -> Checksum | None:
        from .checksum.celljoin import evaluate_celljoin_placed

        result = await evaluate_celljoin_placed(
            self._celljoin_spec,
            execution=execution,
            member_id=id(self),
            scratch=scratch,
        )
        return self._hold_result(result)

    @property
    def _celljoin_spec(self) -> CellJoinSpec:
        spec = self._spec
        if spec is None:
            raise TypeError("CellJoin has no definition")
        return spec

    @property
    def celljoin_checksum(self) -> Checksum:
        return self._celljoin_spec.checksum

    @property
    def definition(self):
        return self._celljoin_spec.buffer

    @property
    def input_checksums(self) -> tuple[Checksum, ...]:
        spec = self._celljoin_spec
        checksums = list(required_buffers(spec))
        if spec.celltype in ("deepcell", "deepfolder"):
            for _, checksum in spec.members:
                if checksum not in checksums:
                    checksums.append(checksum)
        return tuple(checksums)

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return celljoin_cache_key(self.celljoin_checksum, self.celltype)

    @property
    def database_key(self) -> tuple[str, str]:
        return self.celljoin_checksum.hex(), self.celltype

    def softcancel(self) -> bool:
        from .checksum.expression import softcancel_expression

        return softcancel_expression(self.identity_key, id(self))

    def item(self, key):
        raise TypeError("CellJoin cannot be projected")

    def slice(self, start=None, stop=None, step=None):
        raise TypeError("CellJoin cannot be projected")

    def as_celltype(self, celltype):
        raise TypeError("CellJoin cannot be converted")

    def __getitem__(self, key):
        raise TypeError("CellJoin cannot be projected")

    def __getattr__(self, name: str):
        raise AttributeError(name)

    def __repr__(self) -> str:
        return (
            f"CellJoin(checksum={self.celljoin_checksum.hex()!r}, "
            f"celltype={self.celltype!r})"
        )
