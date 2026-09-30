"""The segment type sinks consume, and the sink contract."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeGuard

from ..logging_setup import get_logger

log = get_logger(__name__)

__all__ = ["CompositeSink", "TimedText", "TranscriptSegment", "TranscriptSink"]


@dataclass(frozen=True, slots=True)
class TimedText:
    """A stretch of text with its own start and end, on the session timeline."""

    start: float
    end: float
    text: str


def _is_seconds(value: Any) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _timed_parts(items: Any, *, start: float, end: float) -> tuple[TimedText, ...]:
    """The provider's own segments as timed parts, or ``()`` when they cannot be trusted.

    All-or-nothing: one record without usable times, or times running backwards, means the
    segment is better written as a single cue than as a partly guessed list. Times are
    clamped into the segment's span so cues never overlap the neighbouring chunks.
    """
    if not items:
        return ()
    parts: list[TimedText] = []
    previous_end = start
    for item in items:
        if not isinstance(item, dict):
            return ()
        raw_start, raw_end = item.get("start"), item.get("end")
        text = str(item.get("text") or "").strip()
        if not text or not _is_seconds(raw_start) or not _is_seconds(raw_end):
            return ()
        part_start = max(previous_end, min(float(raw_start), end))
        part_end = max(part_start, min(float(raw_end), end))
        parts.append(TimedText(part_start, part_end, text))
        previous_end = part_end
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One final piece of transcript, placed on the session timeline."""

    start: float
    """Seconds since the session started."""
    end: float
    text: str
    language: str | None = None
    provider: str | None = None
    model: str | None = None
    latency: float | None = None
    """Seconds the STT provider took for this segment."""
    confidence: float | None = None
    wallclock: float = field(default_factory=time.time)
    """Unix time at which the segment became final."""
    parts: tuple[TimedText, ...] = ()
    """The provider's own timed segments, when it gave any: subtitle cues follow these
    instead of covering the whole chunk."""

    @classmethod
    def from_transcript(
        cls,
        transcript: Any,
        *,
        language: str | None = None,
        wallclock: float | None = None,
    ) -> TranscriptSegment:
        """Adapt an STT result: anything with ``start``, ``end`` and ``text``.

        Optional attributes (``provider``, ``model``, ``provider_latency``,
        ``confidence``, ``segments``) are read when present. ``segments`` must already be
        on the session clock and must belong to ``text``.
        """
        start, end = float(transcript.start), float(transcript.end)
        return cls(
            start=start,
            end=end,
            text=str(transcript.text),
            parts=_timed_parts(getattr(transcript, "segments", None), start=start, end=end),
            language=language or getattr(transcript, "language", None),
            provider=getattr(transcript, "provider", None),
            model=getattr(transcript, "model", None),
            latency=getattr(transcript, "provider_latency", None),
            confidence=getattr(transcript, "confidence", None),
            wallclock=time.time() if wallclock is None else wallclock,
        )


class TranscriptSink(Protocol):
    """A destination for final transcript segments."""

    def write(self, segment: TranscriptSegment) -> None: ...

    def close(self) -> None: ...


class CompositeSink:
    """Write to several sinks; one failing sink never starves the others."""

    def __init__(self, sinks: list[TranscriptSink]) -> None:
        self.sinks = list(sinks)

    def write(self, segment: TranscriptSegment) -> None:
        for sink in self.sinks:
            try:
                sink.write(segment)
            except Exception:
                log.exception("sink write failed", extra={"sink": type(sink).__name__})

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                log.exception("sink close failed", extra={"sink": type(sink).__name__})
