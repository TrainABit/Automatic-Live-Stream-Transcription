"""Live transcript lines for a terminal."""

from __future__ import annotations

import sys
import threading
from typing import TextIO

from ..ansi import paint, supports_color
from ..rules.engine import RuleHit
from ..rules.model import Severity
from .base import TranscriptSegment

__all__ = ["ConsoleSink"]

_HIT_STYLE = {
    Severity.INFO: "bold_cyan",
    Severity.WARNING: "bold_yellow",
    Severity.CRITICAL: "bold_red",
}


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class ConsoleSink:
    """``[0:01:23] text`` per segment; rule hits are printed right below it."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
        show_language: bool = False,
    ) -> None:
        self._stream = stream
        self._color = color
        self.show_language = show_language
        self._lock = threading.Lock()

    def _out(self) -> tuple[TextIO, bool]:
        stream = self._stream or sys.stdout
        return stream, supports_color(stream, force=self._color)

    def write(self, segment: TranscriptSegment) -> None:
        stream, color = self._out()
        stamp = paint(f"[{_clock(segment.start)}]", "dim", color)
        lang = (
            paint(f" ({segment.language})", "dim", color)
            if self.show_language and segment.language
            else ""
        )
        with self._lock:
            stream.write(f"{stamp}{lang} {segment.text.strip()}\n")
            stream.flush()

    def write_hit(self, hit: RuleHit) -> None:
        """Highlight a rule hit; the pipeline calls this for each hit of a segment."""
        stream, color = self._out()
        style = _HIT_STYLE[hit.severity]
        label = f">> {hit.severity.value.upper()} {hit.rule_id}: {hit.matched_text.strip()}"
        with self._lock:
            stream.write(f"    {paint(label, style, color)}\n")
            stream.flush()

    def close(self) -> None:
        stream, _ = self._out()
        stream.flush()
