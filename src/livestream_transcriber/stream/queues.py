"""Bounded queues with an explicit overflow policy.

A live pipeline must never grow without bound: an unbounded audio queue on a
small server is the fastest way to get OOM-killed. When a consumer falls behind
we would rather lose data *and know it* than buffer indefinitely.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

T = TypeVar("T")

__all__ = ["QUEUE_CLOSED", "DropOldestQueue"]


class _Sentinel:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<QUEUE_CLOSED>"


QUEUE_CLOSED = _Sentinel()


class DropOldestQueue(Generic[T]):
    """A fixed-capacity queue that discards the *oldest* item when full.

    Dropping the oldest is the right policy for a live stream: a stale chunk
    has little value once a newer one exists. Every drop is counted so the
    caller can log and alarm on it.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        # One slot more than ``maxsize`` exists so :meth:`close` can always
        # deliver its sentinel without evicting a queued item; only the
        # dropping :meth:`put` is held to ``maxsize``.
        self._q: asyncio.Queue[T | _Sentinel] = asyncio.Queue(maxsize=maxsize + 1)
        self._maxsize = maxsize
        self.dropped = 0
        self._closed = False
        self._error: Exception | None = None
        self._space = asyncio.Event()
        self._space.set()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return self._q.qsize()

    def put(self, item: T) -> bool:
        """Enqueue without blocking.

        Returns False when the item was not enqueued or an older one was dropped to make
        room: :attr:`dropped` says which, and a closed queue accepts nothing.
        """
        if self._closed:
            return False
        dropped = False
        while self._q.qsize() >= self._maxsize:
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race-free in practice
                break
            self.dropped += 1
            dropped = True
        self._q.put_nowait(item)
        return not dropped

    async def put_wait(self, item: T) -> bool:
        """Enqueue, waiting for room instead of discarding the oldest item.

        Backpressure rather than loss. A live capture must never do this, since
        a stalled consumer would stall ffmpeg, but a replay or a local file must:
        dropping a chunk because the consumer was slow would make the same input
        produce different output on different machines.

        Written for a single producer. Returns False if the queue closes while
        waiting.
        """
        while not self._closed and self._q.qsize() >= self._maxsize:
            self._space.clear()
            await self._space.wait()
        if self._closed:
            return False
        self._q.put_nowait(item)
        return True

    def close(self, error: Exception | None = None) -> None:
        """Signal end-of-stream to consumers. Idempotent.

        The first non-None ``error`` wins: a later ``close()`` cannot clear or
        replace it. That lets a supervisor record a crash while a subsequent
        ``source.close()`` keeps the reason for the consumer.
        """
        if error is not None and self._error is None:
            self._error = error
        if self._closed:
            return
        self._closed = True
        self._space.set()  # release any waiting producer
        try:
            self._q.put_nowait(QUEUE_CLOSED)
        except asyncio.QueueFull:  # pragma: no cover
            self._q.get_nowait()
            self._q.put_nowait(QUEUE_CLOSED)

    async def get(self) -> T | _Sentinel:
        item = await self._q.get()
        if self._q.qsize() < self._maxsize:
            self._space.set()
        if isinstance(item, _Sentinel):
            # Leave the end marker for the next reader: a second get() or iteration
            # after close must see the end too, not wait forever. The slot it just
            # freed guarantees room.
            self._q.put_nowait(item)
            if self._error is not None:
                raise self._error
        return item

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self.get()
            if isinstance(item, _Sentinel):
                return
            yield item
