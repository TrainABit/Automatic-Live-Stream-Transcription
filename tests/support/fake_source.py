"""A scripted :class:`StreamSource`, so session tests need no ffmpeg and no network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from livestream_transcriber.models import AudioChunk, CaptureStats, SegmentInfo, StreamInfo
from livestream_transcriber.stream.base import StreamError, StreamSource

from .audio import chunk as make_chunk
from .audio import tone

__all__ = ["FakeSource", "audible_chunks"]


def audible_chunks(count: int, seconds: float = 2.5, *, start: float = 0.0) -> list[AudioChunk]:
    return [make_chunk(start + i * seconds, tone(seconds), index=i) for i in range(count)]


class FakeSource(StreamSource):
    """Yields ``chunks``, then ends, raises ``error`` or waits until closed.

    ``connect_error`` makes :meth:`connect` fail. ``on_chunk(n)`` runs after the n-th
    chunk has been handed over (a test can set a stop event there).
    """

    def __init__(
        self,
        chunks: list[AudioChunk] | None = None,
        *,
        is_live: bool = False,
        error: StreamError | None = None,
        connect_error: Exception | None = None,
        hang: bool = False,
        on_chunk: Callable[[int], None] | None = None,
        title: str = "fake stream",
    ) -> None:
        self.chunks = list(chunks or [])
        self.is_live = is_live
        self.error = error
        self.connect_error = connect_error
        self.hang = hang
        self.on_chunk = on_chunk
        self.title = title
        self.connected = False
        self.close_calls = 0
        self.require_live = False
        self._closed = asyncio.Event()
        self._stats = CaptureStats()
        self.segments = [SegmentInfo(index=0, started_at=0.0, offset=0.0)]

    @property
    def stats(self) -> CaptureStats:
        return self._stats

    async def connect(self) -> StreamInfo:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True
        return StreamInfo(url="fake://source", title=self.title, is_live=self.is_live)

    async def get_audio(self) -> AsyncIterator[AudioChunk]:  # type: ignore[override]
        for n, chunk in enumerate(self.chunks, start=1):
            if self._closed.is_set():
                return
            self._stats.audio_chunks_emitted += 1
            self._stats.audio_seconds += chunk.duration
            yield chunk
            if self.on_chunk is not None:
                self.on_chunk(n)
            await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        if self.hang:
            await self._closed.wait()

    async def close(self) -> None:
        self.close_calls += 1
        self._closed.set()
