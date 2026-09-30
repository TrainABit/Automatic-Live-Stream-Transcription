from __future__ import annotations

import os
import threading

import pytest

from livestream_transcriber import process_priority
from livestream_transcriber.process_priority import apply_thread_nice, run_with_nice


@pytest.fixture
def nice_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace ``os.nice`` with a recorder, so the test never changes real priority."""
    current = {"value": 0}
    calls: list[int] = []

    def fake_nice(increment: int) -> int:
        calls.append(increment)
        current["value"] += increment
        return current["value"]

    monkeypatch.setattr(os, "nice", fake_nice)
    process_priority._applied = threading.local()
    return calls


def test_nice_is_applied_once_per_thread_as_an_absolute_target(nice_calls: list[int]):
    apply_thread_nice(5)
    apply_thread_nice(5)
    apply_thread_nice(5)
    assert nice_calls == [0, 5]  # one query, one adjustment of exactly 5


def test_zero_or_negative_leaves_priority_alone(nice_calls: list[int]):
    apply_thread_nice(0)
    apply_thread_nice(-3)
    assert nice_calls == []


def test_already_nicer_thread_is_not_touched(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    def fake_nice(increment: int) -> int:
        calls.append(increment)
        return 10

    monkeypatch.setattr(os, "nice", fake_nice)
    process_priority._applied = threading.local()
    apply_thread_nice(5)
    assert calls == [0]


def test_lack_of_permission_is_swallowed(monkeypatch: pytest.MonkeyPatch):
    def refuse(increment: int) -> int:
        raise PermissionError("not allowed")

    monkeypatch.setattr(os, "nice", refuse)
    process_priority._applied = threading.local()
    apply_thread_nice(5)  # must not raise


def test_run_with_nice_returns_the_result(nice_calls: list[int]):
    assert run_with_nice(3, lambda a, b=0: a + b, 1, b=2) == 3
    assert nice_calls == [0, 3]
