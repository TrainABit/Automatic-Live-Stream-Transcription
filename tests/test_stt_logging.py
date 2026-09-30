"""STT failures are logged once per state change plus a periodic count.

A dead cloud endpoint used to write several warning lines for every 2.5 s chunk.
"""

from __future__ import annotations

import logging

from livestream_transcriber.netutil import HttpError
from livestream_transcriber.stt.base import (
    CircuitBreaker,
    ConditionLog,
    MockTranscriber,
    Transcript,
)
from livestream_transcriber.stt.providers import _cloud
from livestream_transcriber.stt.providers.openrouter import OpenRouterTranscriber
from livestream_transcriber.stt.wrappers import FallbackSttTranscriber
from tests.support.audio import RATE, Clock, tone

POST = "livestream_transcriber.stt.providers.openrouter.post_json"
PACKAGE = "livestream_transcriber"


def records(caplog, level=logging.WARNING):
    return [r for r in caplog.records if r.name.startswith(PACKAGE) and r.levelno >= level]


def test_condition_log_starts_once_then_summarises_with_counts(caplog):
    clock = Clock(0.0)
    logger = logging.getLogger(f"{PACKAGE}.test.condition")
    cond = ConditionLog(
        "provider failing", logger=logger, summary_seconds=300, quiet_seconds=60, clock=clock
    )
    with caplog.at_level(logging.INFO, logger=PACKAGE):
        for _ in range(264):  # eleven minutes of 2.5 s chunks
            cond.hit(model="m")
            clock.advance(2.5)
        assert not cond.ok()  # 2.5 s after the last hit: not over yet
        clock.advance(60)
        assert cond.ok()
    lines = [r for r in caplog.records if r.name == logger.name]
    messages = [r.getMessage() for r in lines]
    assert messages[0] == "provider failing"
    assert messages[1:3] == ["provider failing (still occurring)"] * 2
    assert messages[3] == "provider failing - recovered"
    assert len(lines) == 4
    assert lines[1].occurrences == 120 and lines[2].occurrences == 120
    assert lines[3].occurrences == 264
    assert [r.levelno for r in lines] == [logging.WARNING] * 3 + [logging.INFO]


def test_a_new_episode_logs_its_start_again(caplog):
    clock = Clock(0.0)
    logger = logging.getLogger(f"{PACKAGE}.test.episodes")
    cond = ConditionLog("provider failing", logger=logger, quiet_seconds=60, clock=clock)
    with caplog.at_level(logging.INFO, logger=PACKAGE):
        cond.hit()
        clock.advance(61)
        cond.ok()
        cond.hit()
    assert [r.getMessage() for r in caplog.records].count("provider failing") == 2
    assert cond.episodes == 2


def test_a_cloud_outage_with_a_fallback_does_not_log_per_chunk(monkeypatch, caplog):
    def down(url, payload, *, headers=None, timeout=30.0):
        raise HttpError(503, "upstream down")

    monkeypatch.setattr(POST, down)
    primary = OpenRouterTranscriber(
        "your-openrouter-key", model="m", breaker=CircuitBreaker("m", failure_threshold=3)
    )
    local = MockTranscriber(
        script=[Transcript(start=0, end=1, text="local words", degraded=True)] * 200
    )
    stt = FallbackSttTranscriber(primary, local)
    with caplog.at_level(logging.INFO, logger=PACKAGE):
        for i in range(200):
            got = stt.transcribe(tone(), RATE, start=float(i), end=float(i) + 1.0)
            assert got is not None and got.degraded
    warnings = records(caplog)
    messages = [r.getMessage() for r in warnings]
    assert len(warnings) <= 5, messages
    assert messages.count("stt circuit open; provider paused") == 1


def test_rate_limit_retries_are_info_not_warning(monkeypatch, caplog):
    calls = {"n": 0}

    def flaky(url, payload, *, headers=None, timeout=30.0):
        calls["n"] += 1
        if calls["n"] % 2:
            raise HttpError(429, "slow down")
        return {"text": "ok", "usage": {"cost": 0.0}}

    monkeypatch.setattr(POST, flaky)
    monkeypatch.setattr(_cloud.time, "sleep", lambda s: None)
    stt = OpenRouterTranscriber("your-openrouter-key", model="m")
    with caplog.at_level(logging.INFO, logger=PACKAGE):
        for i in range(50):
            assert stt.transcribe(tone(), RATE, start=float(i), end=float(i) + 1.0).text == "ok"
    assert records(caplog) == []
