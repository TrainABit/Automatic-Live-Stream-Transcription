"""Minimal ANSI styling shared by the console notifier and the console sink."""

from __future__ import annotations

import os
from typing import Any

__all__ = ["paint", "supports_color"]

_STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "bold_red": "\033[1;31m",
    "bold_yellow": "\033[1;33m",
    "bold_cyan": "\033[1;36m",
}
_RESET = "\033[0m"


def supports_color(stream: Any, *, force: bool | None = None) -> bool:
    """Whether to colour output to ``stream``.

    ``force`` wins when given (a ``--no-color`` flag); otherwise the ``NO_COLOR``
    convention is honoured and colour is used only on a terminal.
    """
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def paint(text: str, style: str, enabled: bool) -> str:
    """Wrap ``text`` in a named style when ``enabled``; a no-op otherwise."""
    if not enabled or style not in _STYLES:
        return text
    return f"{_STYLES[style]}{text}{_RESET}"
