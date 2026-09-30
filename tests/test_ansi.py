"""ANSI styling and the colour decision."""

from __future__ import annotations

import io

import pytest

from livestream_transcriber.ansi import paint, supports_color


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_paint_wraps_text_in_a_style_and_resets_it() -> None:
    assert paint("hi", "red", True) == "\033[31mhi\033[0m"


def test_paint_is_a_no_op_when_disabled_or_the_style_is_unknown() -> None:
    assert paint("hi", "red", False) == "hi"
    assert paint("hi", "no-such-style", True) == "hi"


def test_force_wins_over_the_terminal_and_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(io.StringIO(), force=True) is True
    monkeypatch.delenv("NO_COLOR")
    assert supports_color(_Tty(), force=False) is False


def test_colour_is_used_only_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert supports_color(_Tty()) is True
    assert supports_color(io.StringIO()) is False
    assert supports_color(object()) is False


def test_no_color_disables_colour_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(_Tty()) is False
