"""Class for Seamless buffers."""

from .checksum_class import Checksum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seamless.caching.buffer_cache import TempRef


class Buffer:
    """Class for Seamless buffers."""

    __slots__ = ["_checksum", "_content", "__weakref__"]

    def __init__(
        self,
        value_or_buffer,
        celltype: str | None = None,
        *,
        checksum: Checksum | None = None,
    ):
        from .checksum.serialize import serialize_sync as serialize
        from seamless.caching.buffer_cache import get_buffer_cache

        if isinstance(value_or_buffer, Buffer):
            value_or_buffer = value_or_buffer.content
        elif isinstance(value_or_buffer, Checksum) and celltype != "checksum":
            raise TypeError("A Checksum is a value only for celltype 'checksum'")

        if celltype is None:
            if isinstance(value_or_buffer, Buffer):
                value_or_buffer = value_or_buffer.content
            if value_or_buffer is None:
                if checksum is not None:
                    raise TypeError(
                        "Constructing Buffer from None, but checksum is not None"
                    )
            elif not isinstance(value_or_buffer, bytes):
                raise TypeError(
                    "Constructing Buffer from raw buffer, but raw buffer is not a bytes object"
                )
            buf = value_or_buffer
        else:
            celltype = self._map_celltype(celltype)
            buf = serialize(value_or_buffer, celltype)
        self._content = buf
        self._checksum = None
        if checksum:
            self._checksum = Checksum(checksum)
            get_buffer_cache().register(self._checksum, self, size=len(self.content))
            from .checksum.hash_type import register_hash_type_for_buffer

            register_hash_type_for_buffer(self._checksum, self)

    @staticmethod
    def _map_celltype(celltype: str) -> str:
        from .checksum.celltypes import celltypes

        allowed_celltypes = celltypes + [
            "deepcell",
            "deepfolder",
            "folder",
            "module",
        ]
        if celltype is not None and celltype not in allowed_celltypes:
            raise TypeError(celltype, allowed_celltypes)
        if celltype in ("deepcell", "deepfolder", "folder", "module"):
            celltype = "plain"
        return celltype

    @classmethod
    def load(cls, filename) -> "Buffer":
        """Loads the buffer from a file"""
        with open(filename, "rb") as f:
            buf = f.read()
        return cls(buf)

    @classmethod
    async def from_async(
        cls,
        value,
        celltype: str,
        *,
        use_cache: bool = True,
        checksum: Checksum | None = None,
    ) -> "Buffer":
        """Init from value, asynchronously"""
        from .checksum.serialize import serialize

        celltype = cls._map_celltype(celltype)
        buf = await serialize(value, celltype, use_cache=use_cache)
        return cls(buf, checksum=checksum)

    @property
    def checksum(self) -> Checksum:
        """Returns the buffer's Checksum object, which must have been calculated already"""
        if self._checksum is None:
            raise AttributeError(
                "Checksum has not yet been calculated, use .get_checksum()"
            )
        return self._checksum

    def get_checksum(self) -> Checksum:
        """Returns the buffer's Checksum object, calculating it if needed"""
        from seamless.diagnostics import record
        record("hash", nbytes=len(self)) if self._checksum is None else None
        from .checksum.cached_calculate_checksum import (
            cached_calculate_checksum_sync as cached_calculate_checksum,
        )
        from seamless.caching.buffer_cache import get_buffer_cache

        if self._checksum is None:
            checksum = cached_calculate_checksum(self)
            assert isinstance(checksum, Checksum)
            self._checksum = checksum
            get_buffer_cache().register(self._checksum, self, size=len(self.content))
            return checksum
        else:
            return self._checksum

    async def get_checksum_async(self) -> Checksum:
        """Returns the buffer's Checksum object, calculating it asynchronously if needed"""
        from seamless.diagnostics import record
        record("hash", nbytes=len(self)) if self._checksum is None else None
        from .checksum.cached_calculate_checksum import (
            cached_calculate_checksum,
        )
        from seamless.caching.buffer_cache import get_buffer_cache

        if self._checksum is None:
            checksum = await cached_calculate_checksum(self)
            assert isinstance(checksum, Checksum)
            self._checksum = checksum
            get_buffer_cache().register(self._checksum, self, size=len(self.content))
            return checksum
        else:
            return self._checksum

    @property
    def content(self) -> bytes:
        """Return the buffer value"""
        return self._content

    def save(self, filename: str) -> None:
        """Saves the buffer to a file"""
        with open(filename, "wb") as f:
            f.write(self.content)

    def get_value(self, celltype: str):
        """Converts the buffer to a value.
        The checksum must have been computed already."""
        from .checksum.parse_buffer import parse_buffer_sync as parse_buffer

        celltype = self._map_celltype(celltype)
        checksum = self.get_checksum()
        return parse_buffer(self, checksum, celltype, copy=True)

    def decode(self):
        return self.content.decode()

    async def get_value_async(self, celltype: str, *, copy: bool = True):
        """Converts the buffer to a value.
        The checksum must have been computed already.

        If copy=False, the value can be returned from cache.
        It must not be modified.
        """
        from .checksum.parse_buffer import parse_buffer

        celltype = self._map_celltype(celltype)
        return await parse_buffer(self, self.checksum, celltype, copy=copy)

    def incref(self) -> None:
        """Increment normal refcount in the buffer cache."""
        # local import to avoid importing caching at module import time
        from seamless.caching.buffer_cache import get_buffer_cache

        checksum = self.get_checksum()
        get_buffer_cache().incref(checksum, buffer=self)

    def decref(self):
        """Decrement a normal cache ref and return whether one was held."""

        checksum = self.get_checksum()
        return checksum.decref()

    def incref_refholder(self, *, scratch: bool = False) -> None:
        """Acquire an internal lifecycle reference for this buffer's checksum."""

        from seamless.caching.buffer_cache import get_buffer_cache

        checksum = self.get_checksum()
        get_buffer_cache().incref_refholder(
            checksum, buffer=self, scratch=scratch
        )

    def decref_refholder(self) -> bool:
        """Release an internal lifecycle reference for this buffer's checksum."""

        checksum = self.get_checksum()
        return checksum.decref_refholder()

    def tempref(
        self,
        interest: float = 128.0,
        fade_factor: float = 2.0,
        fade_interval: float = 2.0,
        scratch: bool = False,
    ) -> "TempRef":
        """Add or refresh a single tempref. Only one tempref allowed per checksum.

        If scratch is True, keep the tempref scratch-only (no remote registration).
        """
        # local import to avoid importing caching at module import time
        from seamless.caching.buffer_cache import get_buffer_cache

        checksum = self.get_checksum()
        return get_buffer_cache().tempref(
            checksum,
            buffer=self,
            interest=interest,
            fade_factor=fade_factor,
            fade_interval=fade_interval,
            scratch=scratch,
        )

    async def write(self) -> bool:
        """Write the buffer to remote server(s), if any have been configured
        Returns True if the write has succeeded.
        """
        try:
            import seamless_remote.buffer_remote
        except ImportError:
            return False
        from seamless import ensure_open
        try:
            mark_required = seamless_remote.buffer_remote.has_write_server()
        except Exception:
            mark_required = False
        ensure_open("buffer write", mark_required=mark_required)
        try:
            from seamless.caching import buffer_writer
        except ImportError:  # pragma: no cover - defensive, module should exist
            buffer_writer = None
        checksum = await self.get_checksum_async()
        if buffer_writer is not None:
            background_result = await buffer_writer.await_existing_task(checksum)
            if background_result is not None:
                return background_result

        await seamless_remote.buffer_remote.promise(checksum)
        result = await seamless_remote.buffer_remote.write_buffer(checksum, self)
        return result

    def __str__(self):
        return str(self.content)

    def __repr__(self):
        result = "Seamless Buffer"
        if self._checksum is not None:
            result += f", checksum {self.checksum}"
        result += f", length {len(self.content)}"
        return result

    def __eq__(self, other):
        if not isinstance(other, Buffer):
            other = Buffer(other)
        return self.checksum == other.checksum

    def __len__(self):
        return len(self.content)

    def __hash__(self):
        return hash(self.content)
