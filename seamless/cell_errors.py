"""Errors used by the canonical Cell builder API."""


class BoundStateError(AttributeError):
    """Raised when standalone-only Cell state is changed on a bound Cell."""


class ProjectionError(TypeError, AttributeError):
    """Raised when a sub-path projection is used as if it were a value.

    Both bases are deliberate.  ``TypeError`` is what it is: the operand of a
    comparison or a truth test was the wrong kind of object, and the attribute
    lookup that produced it *succeeded* — so reporting a missing attribute would
    be a lie about what happened.  ``AttributeError`` is what it is *about*: the
    cause is almost always a misspelled or removed attribute name, and a bound
    ``Transformer`` already raises ``AttributeError`` for exactly that mistake
    (it has no projection fallback).  Inheriting from both means one ``except``
    clause catches the same user error on either handle.
    """


__all__ = ["BoundStateError", "ProjectionError", "WorkflowError", "AuthorityError"]


class WorkflowError(Exception):
    """Base error for workflow ownership and configuration."""


class AuthorityError(WorkflowError):
    """A local write would be overwritten by an upstream source."""
