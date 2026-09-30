"""Put asynchronous speech-to-text results back into media order.

A transcription request is issued when its audio chunk closes, but it comes back
whenever the provider feels like answering: a retry, a slow shard or a cache hit
next to a cache miss are enough to reverse two chunks. Nothing downstream may see
that. :class:`~livestream_transcriber.audio.speech.SpeechStream` stitches each
transcript against *the previous one*, and rules evaluate on stream time; both
are statements about the recording, not about the network.

So this is a reorder buffer on the session clock. Chunks are announced in media
order (:meth:`MediaOrderGate.submit`, called before the request is made, from the
one task that consumes the audio queue), results are handed back in any order
(:meth:`MediaOrderGate.complete`), and the gate releases them strictly in media
order, never releasing a chunk while an earlier one is still outstanding.

The :attr:`MediaOrderGate.frontier` is the honest answer to "how far has the
audio lane actually got?": the end of the longest contiguous run of finished
chunks. With chunks 100-105, 105-110 and 110-115 outstanding and only 105-110
back, the frontier is still where it was, because 100-105 might yet say something.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = ["MediaOrderGate", "Slot"]

T = TypeVar("T")


@dataclass(slots=True)
class Slot(Generic[T]):
    """One announced chunk and, once it is back, its result."""

    ticket: int
    start: float
    end: float
    done: bool = False
    payload: T | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class MediaOrderGate(Generic[T]):
    """Release asynchronous per-chunk results in media order."""

    def __init__(self) -> None:
        self._slots: deque[Slot[T]] = deque()
        self._by_ticket: dict[int, Slot[T]] = {}
        self._next = 0
        self._frontier = -math.inf
        self.reordered = 0
        """How many results arrived out of order. Operational, not semantic: a
        healthy sequential run reports zero, and a run that reports many still
        produces the identical output."""

    def submit(self, start: float, end: float) -> int:
        """Announce a chunk. Call in media order, before the request is made.

        Returns the ticket to hand back to :meth:`complete`.
        """
        ticket = self._next
        self._next += 1
        slot: Slot[T] = Slot(ticket=ticket, start=float(start), end=float(end))
        self._slots.append(slot)
        self._by_ticket[ticket] = slot
        return ticket

    def complete(self, ticket: int, payload: T | None = None, **extra: Any) -> list[Slot[T]]:
        """Record a result and return everything now releasable, in media order.

        Returns an empty list while an earlier chunk is still outstanding; the
        held results come out later, behind the one that was blocking them.
        Completing an unknown (or already released) ticket raises ``KeyError``.
        """
        slot = self._by_ticket.get(ticket)
        if slot is None:
            raise KeyError(f"unknown ticket {ticket}")
        slot.done = True
        slot.payload = payload
        slot.extra = dict(extra)
        if self._slots[0] is not slot:
            self.reordered += 1

        released: list[Slot[T]] = []
        while self._slots and self._slots[0].done:
            head = self._slots.popleft()
            del self._by_ticket[head.ticket]
            self._frontier = max(self._frontier, head.end)
            released.append(head)
        return released

    @property
    def frontier(self) -> float:
        """End of the longest contiguous run of finished chunks."""
        return self._frontier

    @property
    def outstanding(self) -> int:
        return len(self._slots)

    def describe(self) -> dict[str, Any]:
        return {
            "outstanding": self.outstanding,
            "reordered": self.reordered,
            "frontier": None if self._frontier == -math.inf else round(self._frontier, 2),
        }
