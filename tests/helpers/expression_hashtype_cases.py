"""Witness corpus for expression and HashType tests."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from typing import Any, Iterable

from seamless import Buffer, Checksum, Expression
from seamless.checksum.hash_type import (
    DType,
    Flag,
    HashType,
    Kind,
    Length,
    Rank,
    pack,
)


@dataclass(frozen=True, slots=True)
class ExpressionCase:
    name: str
    path: str
    celltype: str
    target_celltype: str
    valid: bool
    reason: str

    def build(self, input_checksum: Checksum) -> Expression:
        return Expression(
            input_checksum,
            path=self.path,
            input_celltype=self.celltype,
            celltype=self.target_celltype,
        )


@dataclass(frozen=True, slots=True)
class HashTypeWitness:
    name: str
    raw_buffer: bytes
    value: Any
    source_checksum: Checksum
    expected_hash_type: int
    mic: str
    expressions: tuple[ExpressionCase, ...]
    deep_celltypes: tuple[str, ...] = ()

    @property
    def decoded_hash_type(self) -> HashType:
        return HashType.unpack(self.expected_hash_type)

    @property
    def valid_expressions(self) -> tuple[ExpressionCase, ...]:
        return tuple(case for case in self.expressions if case.valid)

    @property
    def invalid_expressions(self) -> tuple[ExpressionCase, ...]:
        return tuple(case for case in self.expressions if not case.valid)


def iter_hashtype_witnesses() -> Iterable[HashTypeWitness]:
    return tuple(_build_witnesses())


def iter_expression_cases() -> Iterable[tuple[HashTypeWitness, ExpressionCase]]:
    for witness in iter_hashtype_witnesses():
        for case in witness.expressions:
            yield witness, case


def database_path_roundtrip(path: str) -> str:
    """Mirror the current database path boundary representation."""

    return json.loads(json.dumps(path))


def expression_case_id(witness: HashTypeWitness, case: ExpressionCase) -> str:
    return (
        f"{witness.name}|hash_type={witness.expected_hash_type}|"
        f"{case.celltype}->{case.target_celltype}|path={case.path!r}|"
        f"{'valid' if case.valid else 'invalid'}"
    )


def _build_witnesses() -> list[HashTypeWitness]:
    witnesses: list[HashTypeWitness] = []
    for semantic in (False, True):
        flags = Flag.SEMANTIC if semantic else Flag(0)
        prefix = "raw_text_semantic" if semantic else "raw_text"
        for length, raw in _raw_text_buffers().items():
            witnesses.append(
                _witness(
                    f"{prefix}_{length.name.lower()}",
                    raw,
                    Kind.RAW_TEXT,
                    length,
                    flags=flags,
                    value=raw.decode(),
                    mic="text",
                    expressions=_flat_text_expressions(),
                )
            )

    for length, raw in _raw_bytes_buffers().items():
        witnesses.append(
            _witness(
                f"raw_bytes_{length.name.lower()}",
                raw,
                Kind.RAW_BYTES,
                length,
                value=raw,
                mic="bytes",
                expressions=_flat_bytes_expressions(),
            )
        )

    for dtype, rank, numpy_bytes, medium, long_ in _numpy_buffers():
        for length, raw in ((Length.MEDIUM, medium), (Length.LONG, long_)):
            flags = Flag.NUMPY_BYTES if numpy_bytes else Flag(0)
            witnesses.append(
                _witness(
                    (
                        "numpy_"
                        f"{dtype.name.lower()}_{rank.name.lower()}_"
                        f"{'bytes_' if numpy_bytes else ''}{length.name.lower()}"
                    ),
                    raw,
                    Kind.NUMPY,
                    length,
                    dtype=dtype,
                    rank=rank,
                    flags=flags,
                    mic="binary",
                    expressions=_numpy_expressions(dtype, rank),
                )
            )

    for kind, value in (
        (Kind.MIXED_OBJECT, {"a": _np_arange(3), "b": 2}),
        (Kind.MIXED_ARRAY, [_np_arange(3), {"x": 1}]),
    ):
        for length, raw in _mixed_buffers(kind, value).items():
            witnesses.append(
                _witness(
                    f"{kind.name.lower()}_{length.name.lower()}",
                    raw,
                    kind,
                    length,
                    value=value,
                    mic="mixed",
                    expressions=_mixed_mapping_expressions()
                    if kind == Kind.MIXED_OBJECT
                    else _sequence_expressions("mixed"),
                )
            )

    json_specs = (
        ("json_object", Kind.JSON_OBJECT, _json_object_buffers(), "plain", _mapping_expressions(), Flag(0)),
        ("json_array", Kind.JSON_ARRAY, _json_array_buffers(), "plain", _sequence_expressions("plain"), Flag(0)),
        ("json_string", Kind.JSON_STRING, _json_string_buffers(False), "str", _json_string_expressions(), Flag(0)),
        (
            "json_numeric_string",
            Kind.JSON_STRING,
            _json_string_buffers(True),
            "str",
            _json_string_expressions(),
            Flag.NUMERIC_SCALAR,
        ),
        (
            "json_number",
            Kind.JSON_NUMBER,
            _json_number_buffers(),
            "float",
            _scalar_expressions("float"),
            Flag.NUMERIC_SCALAR,
        ),
    )
    for base, kind, buffers, mic, expressions, flags in json_specs:
        for length, raw in buffers.items():
            case_expressions = expressions
            if kind == Kind.JSON_NUMBER and length == Length.LONG:
                case_expressions = _long_json_number_expressions()
            deep_celltypes = ()
            if kind in (Kind.JSON_OBJECT, Kind.JSON_ARRAY):
                deep_celltypes = ("deepcell", "deepfolder", "folder", "module")
            witnesses.append(
                _witness(
                    f"{base}_{length.name.lower()}",
                    raw,
                    kind,
                    length,
                    flags=flags,
                    value=None,
                    mic=mic,
                    expressions=case_expressions,
                    deep_celltypes=deep_celltypes,
                )
            )

    for name, raw, mic in (
        ("json_const_true", b"true", "bool"),
        ("json_const_false", b"false", "bool"),
        ("json_const_null", b"null", "plain"),
    ):
        witnesses.append(
            _witness(
                name,
                raw,
                Kind.JSON_STRING,
                Length.SHORT,
                value=None,
                mic=mic,
                expressions=_scalar_expressions(mic),
            )
        )

    return witnesses


def _witness(
    name: str,
    raw: bytes,
    kind: Kind,
    length: Length,
    *,
    dtype: DType = DType.NA,
    rank: Rank = Rank.SCALAR,
    flags: Flag = Flag(0),
    value: Any = None,
    mic: str | None = None,
    expressions: tuple[ExpressionCase, ...],
    deep_celltypes: tuple[str, ...] = (),
) -> HashTypeWitness:
    checksum = Buffer(raw).get_checksum()
    if mic is None:
        mic = HashType(kind, length, dtype, rank, flags).mic
    return HashTypeWitness(
        name=name,
        raw_buffer=raw,
        value=value,
        source_checksum=checksum,
        expected_hash_type=pack(kind, length, dtype, rank, flags),
        mic=mic,
        expressions=expressions,
        deep_celltypes=deep_celltypes,
    )


def _expr(
    name: str,
    path: str,
    celltype: str,
    target_celltype: str | None = None,
    *,
    valid: bool = True,
    reason: str = "",
) -> ExpressionCase:
    return ExpressionCase(
        name=name,
        path=path,
        celltype=celltype,
        target_celltype=celltype if target_celltype is None else target_celltype,
        valid=valid,
        reason=reason,
    )


def _flat_text_expressions() -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", "text"),
        _expr("first_char", "[0]", "text", "str"),
        _expr("slice", "[:4]", "text", "text"),
        _expr("as_bytes", "", "text", "bytes"),
        _expr("map_on_text", "missing", "text", valid=False, reason="text has no MAP capability"),
        _expr("plain_read", "", "plain", valid=False, reason="raw text is not plain JSON"),
    )


def _flat_bytes_expressions() -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", "bytes"),
        _expr("first_byte", "[0]", "bytes", "int"),
        _expr("slice", "[:2]", "bytes", "bytes"),
        _expr("as_binary", "", "bytes", "binary"),
        _expr("map_on_bytes", "missing", "bytes", valid=False, reason="bytes has no MAP capability"),
        _expr("text_read", "", "text", valid=False, reason="raw bytes are not UTF-8"),
    )


def _mapping_expressions() -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", "mixed"),
        _expr("field", "a", "mixed", "mixed"),
        _expr("field_plain", "a", "plain", "plain"),
        _expr("as_bytes", "", "mixed", "bytes"),
        _expr("sequence_on_map", "[0]", "mixed", valid=False, reason="map root has no SEQ capability"),
        _expr("binary_read", "", "binary", valid=False, reason="mapping buffer is not binary"),
    )


def _mixed_mapping_expressions() -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", "mixed"),
        _expr("field", "a", "mixed", "mixed"),
        _expr("as_bytes", "", "mixed", "bytes"),
        _expr("sequence_on_map", "[0]", "mixed", valid=False, reason="map root has no SEQ capability"),
        _expr("plain_read", "a", "plain", valid=False, reason="mixed buffer is not plain JSON"),
        _expr("binary_read", "", "binary", valid=False, reason="mapping buffer is not binary"),
    )


def _sequence_expressions(celltype: str) -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", celltype),
        _expr("item", "[0]", celltype, celltype),
        _expr("slice", "[:1]", celltype, celltype),
        _expr("as_bytes", "", celltype, "bytes"),
        _expr("map_on_sequence", "missing", celltype, valid=False, reason="sequence root has no MAP capability"),
        _expr("binary_read", "", "binary", valid=False, reason="sequence buffer is not binary"),
    )


def _json_string_expressions() -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", "str"),
        _expr("first_char", "[0]", "str", "str"),
        _expr("as_plain", "", "str", "plain"),
        _expr("as_text", "", "str", "text"),
        _expr("map_on_string", "missing", "str", valid=False, reason="string has no MAP capability"),
        _expr("binary_read", "", "binary", valid=False, reason="JSON string is not binary"),
    )


def _long_json_number_expressions() -> tuple[ExpressionCase, ...]:
    return (
        _expr("bytes_identity", "", "bytes"),
        _expr("first_byte", "[0]", "bytes", "int"),
        _expr("text_identity", "", "text"),
        _expr("float_read", "", "float", valid=False, reason="over-long JSON number cannot materialize as float"),
        _expr("map_on_bytes", "missing", "bytes", valid=False, reason="bytes has no MAP capability"),
    )


def _scalar_expressions(celltype: str) -> tuple[ExpressionCase, ...]:
    return (
        _expr("identity", "", celltype),
        _expr("as_plain", "", celltype, "plain"),
        _expr("as_bytes", "", celltype, "bytes"),
        _expr("item_on_scalar", "[0]", celltype, valid=False, reason="scalar has no SEQ capability"),
        _expr("map_on_scalar", "missing", celltype, valid=False, reason="scalar has no MAP capability"),
    )


def _numpy_expressions(dtype: DType, rank: Rank) -> tuple[ExpressionCase, ...]:
    valid = [_expr("identity", "", "binary"), _expr("as_mixed", "", "binary", "mixed")]
    invalid = [
        _expr("plain_read", "", "plain", valid=False, reason="numpy is not plain JSON"),
    ]
    if rank == Rank.SCALAR:
        invalid.append(
            _expr("item_on_scalar", "[0]", "binary", valid=False, reason="scalar numpy has no SEQ capability")
        )
    else:
        valid.extend(
            (
                _expr("item", "[0]", "binary", "binary"),
                _expr("slice", "[:1]", "binary", "binary"),
            )
        )
    if dtype == DType.STRUCTURED:
        valid.append(_expr("field", "a", "binary", "binary"))
        invalid.append(
            _expr(
                "missing_field",
                "missing",
                "binary",
                valid=False,
                reason="structured numpy lacks this field",
            )
        )
    else:
        invalid.append(
            _expr("field_on_unstructured", "a", "binary", valid=False, reason="unstructured numpy has no MAP capability")
        )
    return tuple(valid + invalid[:2])


def _raw_text_buffers() -> dict[Length, bytes]:
    return {
        Length.SHORT: b"hello world",
        Length.EQ64: b"a" * 64,
        Length.MEDIUM: b"x" * 65,
        Length.LONG: b"x" * 1001,
    }


def _raw_bytes_buffers() -> dict[Length, bytes]:
    return {
        Length.SHORT: b"\xff\xfe\x00",
        Length.EQ64: b"\xff" * 64,
        Length.MEDIUM: b"\xff" * 65,
        Length.LONG: b"\xff" * 1001,
    }


def _json_object_buffers() -> dict[Length, bytes]:
    return {
        Length.SHORT: b'{"a":1}',
        Length.EQ64: b'{"a":"' + b"x" * 56 + b'"}',
        Length.MEDIUM: b'{"a":"' + b"x" * 80 + b'"}',
        Length.LONG: b'{"a":"' + b"x" * 1000 + b'"}',
    }


def _json_array_buffers() -> dict[Length, bytes]:
    return {
        Length.SHORT: b"[1,2]",
        Length.EQ64: b'["' + b"x" * 60 + b'"]',
        Length.MEDIUM: b'["' + b"x" * 80 + b'"]',
        Length.LONG: b'["' + b"x" * 1000 + b'"]',
    }


def _json_string_buffers(numeric: bool) -> dict[Length, bytes]:
    fill = b"7" if numeric else b"x"
    return {
        Length.SHORT: b'"42"' if numeric else b'"hello"',
        Length.EQ64: b'"' + fill * 62 + b'"',
        Length.MEDIUM: b'"' + fill * 80 + b'"',
        Length.LONG: b'"' + fill * 1000 + b'"',
    }


def _json_number_buffers() -> dict[Length, bytes]:
    return {
        Length.SHORT: b"42",
        Length.EQ64: b"1" * 64,
        Length.MEDIUM: b"1" * 65,
        Length.LONG: b"1" * 1001,
    }


def _numpy_buffers() -> Iterable[tuple[DType, Rank, bool, bytes, bytes]]:
    import numpy as np

    specs = [
        (DType.NUMERIC, Rank.SCALAR, False, np.array(3.0), np.array(3.0)),
        (DType.NUMERIC, Rank.D1, False, np.arange(3), np.arange(300)),
        (DType.NUMERIC, Rank.D2, False, np.zeros((2, 3)), np.zeros((40, 40))),
        (DType.NUMERIC, Rank.D3PLUS, False, np.zeros((2, 3, 4)), np.zeros((12, 12, 12))),
        (DType.NONNUMERIC, Rank.SCALAR, True, np.array(b"abc"), np.array(b"x" * 950)),
        (DType.NONNUMERIC, Rank.SCALAR, False, np.array("abc"), np.array("x" * 950)),
        (DType.NONNUMERIC, Rank.D1, False, np.array([b"a", b"bb"]), np.array([b"x" * 500] * 3)),
        (DType.NONNUMERIC, Rank.D2, False, np.array([["a"], ["b"]]), np.array([["x" * 100] * 4] * 4)),
        (
            DType.NONNUMERIC,
            Rank.D3PLUS,
            False,
            np.array([b"x"] * 8).reshape(2, 2, 2),
            np.array([b"x" * 100] * 27).reshape(3, 3, 3),
        ),
        (
            DType.STRUCTURED,
            Rank.SCALAR,
            False,
            np.array((1, 2.0), dtype=[("a", "<i4"), ("b", "<f8")]),
            np.array((1, 2.0), dtype=[("a", "<i4"), ("b", "<f8")]),
        ),
        (
            DType.STRUCTURED,
            Rank.D1,
            False,
            np.zeros((2,), dtype=[("a", "<i4"), ("b", "<f8")]),
            np.zeros((100,), dtype=[("a", "<i4"), ("b", "<f8")]),
        ),
        (
            DType.STRUCTURED,
            Rank.D2,
            False,
            np.zeros((2, 2), dtype=[("a", "<i4"), ("b", "<f8")]),
            np.zeros((20, 20), dtype=[("a", "<i4"), ("b", "<f8")]),
        ),
        (
            DType.STRUCTURED,
            Rank.D3PLUS,
            False,
            np.zeros((2, 2, 2), dtype=[("a", "<i4"), ("b", "<f8")]),
            np.zeros((10, 10, 10), dtype=[("a", "<i4"), ("b", "<f8")]),
        ),
    ]
    for dtype, rank, numpy_bytes, medium_value, long_value in specs:
        medium = _npy_bytes(medium_value)
        long_ = _npy_bytes(long_value)
        if len(long_) <= 1000:
            long_ = _pad_npy_to_long(long_)
        yield dtype, rank, numpy_bytes, medium, long_


def _npy_bytes(value: Any) -> bytes:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _pad_npy_to_long(raw: bytes) -> bytes:
    if not raw.startswith(b"\x93NUMPY"):
        raise ValueError("not an .npy buffer")
    major = raw[6]
    if major != 1:
        raise ValueError("only .npy v1 padding is supported")
    header_length = int.from_bytes(raw[8:10], "little")
    header_start = 10
    header_end = header_start + header_length
    header = raw[header_start:header_end]
    if not header.endswith(b"\n"):
        raise ValueError("unexpected .npy header")
    extra = 1001 - len(raw)
    if extra <= 0:
        return raw
    new_header = header[:-1] + (b" " * extra) + b"\n"
    return raw[:8] + len(new_header).to_bytes(2, "little") + new_header + raw[header_end:]


def _mixed_buffers(kind: Kind, value: Any) -> dict[Length, bytes]:
    raw = Buffer(value, "mixed").content
    long_value = {"a": _np_arange(300)} if kind == Kind.MIXED_OBJECT else [_np_arange(300), {"x": 1}]
    long_raw = Buffer(long_value, "mixed").content
    medium = raw
    return {
        Length.MEDIUM: medium,
        Length.LONG: long_raw
        if len(long_raw) > 1000
        else _long_mixed_fallback(kind),
    }


def _np_arange(size: int):
    import numpy as np

    return np.arange(size)


def _long_mixed_fallback(kind: Kind) -> bytes:
    if kind == Kind.MIXED_OBJECT:
        return Buffer({"a": _np_arange(2000)}, "mixed").content
    return Buffer([_np_arange(2000), {"x": 1}], "mixed").content
