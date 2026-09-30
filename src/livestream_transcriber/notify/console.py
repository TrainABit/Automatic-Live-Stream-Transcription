"""Print alerts to the terminal."""

from __future__ import annotations

import sys
from typing import TextIO

from ..ansi import paint, supports_color
from ..rules.model import Severity
from .base import Event, format_event

__all__ = ["ConsoleNotifier"]

_SEVERITY_STYLE = {
    Severity.INFO: "bold_cyan",
    Severity.WARNING: "bold_yellow",
    Severity.CRITICAL: "bold_red",
}


class ConsoleNotifier:
    """Always available, needs no configuration; the default target of a rule."""

    name = "console"

    def __init__(self, stream: TextIO | None = None, *, color: bool | None = None) -> None:
        self._stream = stream
        self._color = color

    def send(self, event: Event) -> bool:
        # Resolve the stream late so output capture and redirection keep working.
        stream = self._stream or sys.stdout
        color = supports_color(stream, force=self._color)
        head, _, rest = format_event(event).partition("\n")
        style = _SEVERITY_STYLE[event.severity]
        stream.write(f"{paint('ALERT', style, color)} {paint(head, style, color)}\n")
        if rest:
            stream.write("".join(f"  {line}\n" for line in rest.splitlines()))
        stream.flush()
        return True
