"""SubRip (.srt) and WebVTT (.vtt) writers that append cue by cue.

Live transcription never knows the end of the file, so both writers are
incremental: a cue is written and flushed as soon as its segment is final, and
a partially written file is still a valid subtitle file.
"""

from __future__ import annotations

import textwrap
import threading
from pathlib import Path
from typing import IO

from .base import TimedText, TranscriptSegment

__all__ = ["SrtSink", "VttSink", "format_timestamp"]

MIN_CUE_SECONDS = 0.5
"""Cues with no positive duration are stretched to this length."""


def format_timestamp(seconds: float, *, decimal: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SubRip) or, with ``decimal="."``, ``HH:MM:SS.mmm`` (WebVTT).

    Works in integer milliseconds so that 59.9996 s rolls over to ``00:01:00``
    instead of printing ``00:00:60``.
    """
    ms = max(0, int(seconds * 1000 + 0.5))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


class _SubtitleSink:
    decimal = ","
    header = ""

    def __init__(self, path: str | Path, *, append: bool = False, max_line_chars: int = 42) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_line_chars = max(10, max_line_chars)
        self._lock = threading.Lock()
        existing = self.path.read_text(encoding="utf-8") if append and self.path.exists() else ""
        self._cues = existing.count(" --> ")
        self._fh: IO[str] | None = self.path.open("a" if append else "w", encoding="utf-8")
        if self.header and not existing.strip():
            self._fh.write(self.header)
            self._fh.flush()

    def _clean(self, text: str) -> str:
        return " ".join(text.split()).replace("-->", "->")

    def _body(self, text: str) -> str:
        return "\n".join(textwrap.wrap(text, self.max_line_chars, break_long_words=False))

    def _cue_id(self, number: int) -> str:
        return ""

    def write(self, segment: TranscriptSegment) -> None:
        """One cue per provider segment when there are any, else one for the whole segment."""
        cues = segment.parts or (TimedText(segment.start, segment.end, segment.text),)
        with self._lock:
            if self._fh is None:
                raise ValueError("sink is closed")
            for cue in cues:
                text = self._clean(cue.text)
                if not text:
                    continue
                start = max(0.0, cue.start)
                end = cue.end if cue.end > start else start + MIN_CUE_SECONDS
                self._cues += 1
                times = (
                    f"{format_timestamp(start, decimal=self.decimal)} --> "
                    f"{format_timestamp(end, decimal=self.decimal)}"
                )
                self._fh.write(f"{self._cue_id(self._cues)}{times}\n{self._body(text)}\n\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class SrtSink(_SubtitleSink):
    """SubRip: numbered cues, comma before the milliseconds."""

    decimal = ","

    def _cue_id(self, number: int) -> str:
        return f"{number}\n"


class VttSink(_SubtitleSink):
    """WebVTT: ``WEBVTT`` header, dot before the milliseconds, markup escaped."""

    decimal = "."
    header = "WEBVTT\n\n"

    def _clean(self, text: str) -> str:
        cleaned = super()._clean(text)
        return cleaned.replace("&", "&amp;").replace("<", "&lt;")
