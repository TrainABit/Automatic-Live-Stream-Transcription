from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from livestream_transcriber.logging_setup import (
    ConsoleFormatter,
    JsonlFormatter,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    yield
    setup_logging("INFO", color=False)


def _record(
    msg: str = "hello world", name: str = "livestream_transcriber.stream.ffmpeg", **extra: object
):
    rec = logging.LogRecord(name, logging.WARNING, __file__, 1, msg, (), None)
    rec.__dict__.update(extra)
    return rec


def test_console_line_shows_short_logger_name_level_and_fields():
    line = ConsoleFormatter(color=False).format(_record(chunk=3, note="two words", ratio=0.12345))
    assert "WARNING" in line
    assert "stream.ffmpeg" in line
    assert "livestream_transcriber" not in line
    assert line.endswith('chunk=3 note="two words" ratio=0.123')


def test_console_color_is_optional():
    assert "\033[" not in ConsoleFormatter(color=False).format(_record())
    assert "\033[" in ConsoleFormatter(color=True).format(_record())


def test_jsonl_formatter_emits_one_valid_object_with_extras():
    payload = json.loads(JsonlFormatter().format(_record("héllo", chunk=3)))
    assert payload["message"] == "héllo"
    assert payload["level"] == "WARNING"
    assert payload["chunk"] == 3
    assert payload["time"].endswith("Z")


def test_exceptions_are_included():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    assert "RuntimeError: boom" in json.loads(JsonlFormatter().format(rec))["exception"]
    assert "RuntimeError: boom" in ConsoleFormatter(color=False).format(rec)


def test_setup_is_idempotent_and_replaces_handlers(tmp_path: Path):
    setup_logging("INFO", color=False)
    setup_logging("INFO", json_path=tmp_path / "logs" / "run.jsonl", color=False)
    setup_logging("INFO", json_path=tmp_path / "logs" / "run.jsonl", color=False)
    assert len(logging.getLogger().handlers) == 2  # console + file, not accumulated


def test_file_captures_debug_while_console_honours_the_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "run.jsonl"
    setup_logging("WARNING", json_path=path, color=False)
    log = get_logger("livestream_transcriber.test")
    log.debug("only in the file")
    log.warning("in both")
    for handler in logging.getLogger().handlers:
        handler.flush()
    messages = [json.loads(line)["message"] for line in path.read_text().splitlines()]
    assert messages == ["only in the file", "in both"]
    err = capsys.readouterr().err
    assert "in both" in err
    assert "only in the file" not in err


def test_numeric_and_unknown_levels():
    setup_logging(logging.ERROR, color=False)
    assert logging.getLogger().level == logging.ERROR
    setup_logging("not-a-level", color=False)
    assert logging.getLogger().level == logging.INFO
