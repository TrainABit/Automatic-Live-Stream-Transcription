"""One JSON object per final segment, appended and flushed line by line."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import IO

from .base import TranscriptSegment

__all__ = ["JsonlSink"]


class JsonlSink:
    """Newline-delimited JSON, safe to ``tail -f`` and to parse after a crash.

    Each line is flushed as soon as it is written, so a killed process loses at
    most the segment in flight and never leaves a half-written record behind
    an earlier one.
    """

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("a" if append else "w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, segment: TranscriptSegment) -> None:
        record = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text,
            "language": segment.language,
            "provider": segment.provider,
            "model": segment.model,
            "confidence": None if segment.confidence is None else round(segment.confidence, 3),
            "latency": None if segment.latency is None else round(segment.latency, 3),
            "wallclock": round(segment.wallclock, 3),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            if self._fh is None:
                raise ValueError("sink is closed")
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
