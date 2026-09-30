"""Fakes and helpers for pipeline tests: transcribers with scripted timing, recording sinks."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from livestream_transcriber.models import AudioChunk
from livestream_transcriber.notify.base import Event
from livestream_transcriber.outputs.base import TranscriptSegment
from livestream_transcriber.stt.base import Transcript, unavailable

from .audio import RATE, chunk, silence, tone

__all__ = ["ListSink", "RecordingNotifier", "ScriptedTranscriber", "audible", "quiet"]


def audible(ts: float, seconds: float = 2.5, *, index: int | None = None) -> AudioChunk:
    return chunk(ts, tone(seconds), index=index)


def quiet(ts: float, seconds: float = 2.5) -> AudioChunk:
    return chunk(ts, silence(seconds))


class ScriptedTranscriber:
    """A blocking transcriber whose text, delay and failures are decided per call.

    ``text`` maps the chunk start (seconds) to the text to return; a chunk without an
    entry gets ``"chunk <start>"``. ``delay`` maps a start to seconds of blocking sleep.
    ``gate`` (a ``threading.Event``), when given, blocks every call until it is set.
    """

    def __init__(
        self,
        text: dict[float, str] | Callable[[float], str | None] | None = None,
        *,
        delay: dict[float, float] | float = 0.0,
        fail: set[float] | None = None,
        gate: threading.Event | None = None,
        segments: dict[float, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._segments = segments or {}
        self._text = text
        self._delay = delay
        self._fail = fail or set()
        self.gate = gate
        self.calls: list[float] = []
        self._lock = threading.Lock()

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        with self._lock:
            self.calls.append(start)
        if self.gate is not None:
            self.gate.wait(30)
        delay = self._delay.get(start, 0.0) if isinstance(self._delay, dict) else self._delay
        if delay:
            time.sleep(delay)
        if start in self._fail:
            return unavailable(start, end, model="scripted")
        if callable(self._text):
            text = self._text(start)
        elif self._text is not None and start in self._text:
            text = self._text[start]
        else:
            text = f"chunk {start:g}"
        if text is None:
            return None
        return Transcript(
            start=start,
            end=end,
            text=text,
            confidence=0.9,
            provider="scripted",
            segments=self._segments.get(start),
        )


class ListSink:
    """Collects the segments written to it."""

    def __init__(self) -> None:
        self.segments: list[TranscriptSegment] = []
        self.closed = False

    def write(self, segment: TranscriptSegment) -> None:
        self.segments.append(segment)

    def close(self) -> None:
        self.closed = True

    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.segments]


class RecordingNotifier:
    """A notifier that remembers every event it is asked to send."""

    def __init__(self, *, result: bool = True) -> None:
        self.events: list[Event] = []
        self.result = result

    def send(self, event: Event) -> bool:
        self.events.append(event)
        return self.result


__all__ += ["RATE"]
