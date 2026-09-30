"""The ``StreamSource`` contract.

Everything downstream (STT, rules, outputs) consumes a :class:`StreamSource`
and never learns whether the bytes came from a live broadcast, a local file or
a recording on disk. That indirection is what makes replay-driven tests and
benchmarks possible: the same pipeline runs against any source.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from ..models import AudioChunk, CaptureStats, StreamInfo

__all__ = [
    "StreamError",
    "StreamNotLiveError",
    "StreamResolutionError",
    "StreamSource",
]


class StreamError(RuntimeError):
    """Base class for stream failures."""


class StreamResolutionError(StreamError):
    """The source URL could not be resolved to playable media."""


class StreamNotLiveError(StreamResolutionError):
    """The source is listed as live but resolves as a finished broadcast.

    Not a failure to open: a live listing lags a stream end by a probe round or
    two, so this is the stream going offline. Callers treat it as "not live yet"
    and back off instead of counting it as an error.
    """


class StreamSource(abc.ABC):
    """An async source of time-stamped audio chunks.

    :meth:`get_audio` is consumed by a single reader. Implementations must be
    safe to :meth:`close` more than once, and ``async with`` connects on entry
    and closes on exit.
    """

    @abc.abstractmethod
    async def connect(self) -> StreamInfo:
        """Resolve and open the source. Returns what is known about it."""

    @abc.abstractmethod
    def get_audio(self) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks until the source ends or :meth:`close` is called."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release every resource. Idempotent."""

    @property
    @abc.abstractmethod
    def stats(self) -> CaptureStats:
        """Counters accumulated since the source was created."""

    async def __aenter__(self) -> StreamSource:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
