"""Dependency-neutral execution failure identities and wire codec."""

import asyncio
import re
import traceback
from uuid import uuid4


class WorkflowExecutionError(RuntimeError):
    def __init__(self, message, *, failure_id=None, kind="execution"):
        super().__init__(message)
        self.failure_id = failure_id or uuid4().hex
        self.kind = kind

    def __deepcopy__(self, memo):
        return type(self)(str(self), failure_id=self.failure_id, kind=self.kind)


class ExecutionCanceledError(RuntimeError):
    """An answered job was canceled."""


def _kinds():
    from seamless import CacheMissError
    from seamless.checksum.conversion import SeamlessConversionError
    from seamless.checksum.expression import ExpressionEvaluationError
    from seamless.checksum.hash_type_validation import HashTypeValidationError

    return {
        "cache_miss": CacheMissError,
        "hash_type_validation": HashTypeValidationError,
        "expression_evaluation": ExpressionEvaluationError,
        "conversion": SeamlessConversionError,
        "canceled": ExecutionCanceledError,
    }


def error_kind(exc):
    if isinstance(exc, asyncio.CancelledError):
        return "canceled"
    for kind, cls in _kinds().items():
        if isinstance(exc, cls):
            return kind
    return getattr(exc, "kind", "execution")


def format_exception(exc):
    formatted = traceback.TracebackException.from_exception(exc)
    if isinstance(exc, AssertionError):
        last = formatted.stack[-1].filename if formatted.stack else ""
        if not (last.startswith("transformer-") and not last.endswith(".py")):
            return "".join(formatted.format()).strip("\n") + "\n"
    if isinstance(exc, ValueError) and "fromhex" in str(exc):
        return "".join(formatted.format()).strip("\n") + "\n"
    pending = [formatted]
    while pending:
        current = pending.pop()
        start = next(
            (
                i
                for i, frame in enumerate(current.stack)
                if frame.filename.startswith("transformer-")
                and not frame.filename.endswith(".py")
            ),
            0,
        )
        current.stack = traceback.StackSummary.from_list(current.stack[start:])
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        pending.extend(getattr(current, "exceptions", None) or ())
    return "".join(formatted.format()).strip("\n") + "\n"


def encode_error(exc):
    from seamless import Checksum

    kind = error_kind(exc)
    message = (
        str(exc)
        if kind in _kinds() or isinstance(exc, WorkflowExecutionError)
        else format_exception(exc)
    )
    error = {"kind": kind, "message": message}
    if exc.args:
        try:
            checksum = Checksum(exc.args[0])
            digest = checksum.hex()
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                error["checksum"] = digest
        except (TypeError, ValueError, AttributeError):
            pass
    return {"error": error}


def decode_error(payload):
    from seamless import Checksum

    if isinstance(payload, dict) and "kind" in payload:
        payload = {"error": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        raise ValueError("Malformed execution error envelope")
    error = payload["error"]
    kind, message = error.get("kind"), error.get("message")
    if not isinstance(kind, str) or not kind or not isinstance(message, str):
        raise ValueError("Malformed execution error envelope")
    checksum = error.get("checksum")
    if "checksum" in error and (
        not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum)
    ):
        raise ValueError("Malformed execution error checksum")
    cls = _kinds().get(kind)
    if cls is None:
        return WorkflowExecutionError(message, kind=kind)
    if checksum is not None:
        return cls(Checksum(checksum))
    if kind == "cache_miss" and not message:
        return cls()
    return cls(message)


def execution_error(exc):
    if isinstance(exc, dict):
        exc = decode_error(exc)
    if error_kind(exc) not in _kinds() and not isinstance(exc, WorkflowExecutionError):
        exc = WorkflowExecutionError(str(exc))
    exc = exc.with_traceback(None)
    exc.__cause__ = exc.__context__ = None
    if getattr(exc, "failure_id", None) is None:
        exc.failure_id = uuid4().hex
    return exc


envelope_to_error = decode_error
error_to_envelope = encode_error


class RunningLoopRefusal(RuntimeError):
    """A synchronous caller cannot await remote work on its running loop."""
