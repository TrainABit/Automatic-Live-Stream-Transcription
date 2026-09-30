"""Structured logging.

Two sinks:

* **console**: compact and human readable, for a person watching a run;
* **JSONL file**: one object per line, so a debug run can be diffed and
  post-processed later.

Both carry the same structured fields: anything passed via ``extra=`` that is
not a stdlib :class:`logging.LogRecord` attribute is treated as a field. Every
handler carries a :class:`~livestream_transcriber.redact.RedactingFilter`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any

from .redact import RedactingFilter

__all__ = ["ConsoleFormatter", "JsonlFormatter", "get_logger", "setup_logging"]

_PACKAGE_PREFIX = __name__.rsplit(".", 1)[0] + "."

# Attributes the stdlib puts on every record; everything else came from `extra`.
_STD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;197m",
}
_DIM = "\033[38;5;245m"
_RESET = "\033[0m"

_LOG_FILE_MAX_BYTES = 32 * 1024 * 1024
_LOG_FILE_BACKUPS = 3


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS}


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, separators=(",", "="))
    text = str(value)
    return f'"{text}"' if " " in text else text


class FlushingStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler that flushes after every record (piped stderr, journald)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class ConsoleFormatter(logging.Formatter):
    """``HH:MM:SS.mmm LEVEL logger message  key=value ...``, optionally coloured."""

    def __init__(self, *, color: bool = True) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(record.created))
        clock = f"{clock}.{int(record.msecs):03d}"
        level = record.levelname
        name = record.name.removeprefix(_PACKAGE_PREFIX)

        c = _LEVEL_COLORS.get(level, "") if self.color else ""
        r = _RESET if self.color else ""
        d = _DIM if self.color else ""

        line = f"{d}{clock}{r} {c}{level:<7}{r} {d}{name:<22}{r} {record.getMessage()}"

        extras = _extras(record)
        if extras:
            fields = " ".join(f"{k}={_fmt_value(v)}" for k, v in extras.items())
            line += f"  {d}{fields}{r}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonlFormatter(logging.Formatter):
    """One JSON object per record: ``time``, ``level``, ``logger``, ``message`` + extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    level: str | int = "INFO",
    *,
    json_path: str | os.PathLike[str] | None = None,
    color: bool | None = None,
    quiet_libraries: bool = True,
) -> None:
    """Install console (and optional JSONL) handlers on the root logger.

    Replaces any handlers installed earlier, so it is safe to call repeatedly.
    The JSONL file always records at DEBUG; ``level`` only gates the console.
    """
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    if color is None:
        color = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    # Defence in depth: scrub secrets on the way out, so a careless call site
    # or a third-party logger (yt-dlp, urllib) cannot leak a signed URL.
    redactor = RedactingFilter()

    console = FlushingStreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter(color=color))
    console.setLevel(level)
    console.addFilter(redactor)
    root.addHandler(console)

    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate so a long-running capture cannot fill the disk.
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=_LOG_FILE_MAX_BYTES,
            backupCount=_LOG_FILE_BACKUPS,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonlFormatter())
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)
        root.setLevel(min(level, logging.DEBUG))

    if quiet_libraries:
        for noisy in ("urllib3", "asyncio", "charset_normalizer", "httpx", "httpcore", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
