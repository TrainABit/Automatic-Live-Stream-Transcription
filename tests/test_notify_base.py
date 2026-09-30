"""Event identity and serialisation, the rate limiter and its notifier wrapper."""

from __future__ import annotations

import pytest

from livestream_transcriber.notify import (
    Event,
    NotifyError,
    RateLimitedNotifier,
    RateLimiter,
    Verdict,
    format_event,
    make_event_id,
)
from livestream_transcriber.rules import RuleEngine, RuleSet, Severity, TextSegment


def make_event(**kw: object) -> Event:
    base: dict[str, object] = {
        "event_id": "e1",
        "rule_id": "giveaway",
        "text": "we run a giveaway today",
        "matched_text": "giveaway",
        "start": 12.0,
        "end": 14.0,
        "wallclock": 1_700_000_000.0,
        "severity": Severity.WARNING,
        "targets": ("console",),
    }
    base.update(kw)
    return Event(**base)  # type: ignore[arg-type]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestEventId:
    def test_is_deterministic(self) -> None:
        a = make_event_id("r", "key", 12.0, Severity.INFO)
        assert a == make_event_id("r", "key", 12.0, "info")
        assert len(a) == 20 and int(a, 16) >= 0

    def test_a_different_stream_time_is_a_different_event(self) -> None:
        assert make_event_id("r", "k", 10.0, "info") == make_event_id("r", "k", 10.0004, "info")
        assert make_event_id("r", "k", 10.0, "info") != make_event_id("r", "k", 10.5, "info")

    @pytest.mark.parametrize(
        "other",
        [
            ("other", "k", 10.0, "info", ""),
            ("r", "other", 10.0, "info", ""),
            ("r", "k", 10.0, "critical", ""),
            ("r", "k", 10.0, "info", "7"),
        ],
    )
    def test_every_component_matters(self, other: tuple[str, str, float, str, str]) -> None:
        base = make_event_id("r", "k", 10.0, "info")
        rule, key, start, sev, scope = other
        assert make_event_id(rule, key, start, sev, scope=scope) != base

    def test_replaying_a_hit_yields_the_same_event_id(self) -> None:
        rules = RuleSet.from_yaml("rules: [{id: g, type: keyword, keywords: [giveaway]}]")
        ids = set()
        for _ in range(2):
            (hit,) = RuleEngine(rules).evaluate(TextSegment("a giveaway", start=7.0, end=9.0))
            ids.add(Event.from_hit(hit, session_id=3).event_id)
        assert len(ids) == 1


class TestEvent:
    def test_roundtrips_through_a_dict(self) -> None:
        event = make_event(session_id=4, source_url="https://example.com/live", extra={"a": [1]})
        assert Event.from_dict(event.to_dict()) == event

    def test_from_hit_copies_rule_details_and_redacts_the_source(self) -> None:
        rules = RuleSet.from_yaml(
            """
rules:
  - {id: v, type: regex, pattern: 'v(?P<n>\\d+)', description: Version, severity: critical,
     notify: [console, webhook]}
"""
        )
        (hit,) = RuleEngine(rules).evaluate(TextSegment("we ship v3", start=1.0, end=2.0))
        event = Event.from_hit(
            hit,
            source_url="https://example.com/live?token=abc123secret",
            session_id=9,
            wallclock=5.0,
        )
        assert event.rule_id == "v" and event.severity is Severity.CRITICAL
        assert event.targets == ("console", "webhook")
        assert event.extra == {"description": "Version", "groups": {"n": "3"}}
        assert event.session_id == 9 and event.wallclock == 5.0
        assert event.source_url is not None and "abc123secret" not in event.source_url

    def test_format_event_is_readable_plain_text(self) -> None:
        text = format_event(make_event(source_url="https://example.com/live"))
        lines = text.splitlines()
        assert lines[0] == "[WARNING] giveaway"
        assert "Matched: giveaway" in text
        assert "At 0:00:12 in https://example.com/live" in text

    def test_format_event_truncates(self) -> None:
        text = format_event(make_event(text="word " * 500), max_chars=100)
        assert len(text) == 100 and text.endswith("…")


class TestRateLimiter:
    def test_repeat_guard_drops_the_same_text_twice_in_a_row(self) -> None:
        clock = Clock()
        limiter = RateLimiter(repeat_seconds=10, clock=clock)
        assert limiter.acquire("a") is Verdict.ALLOW
        assert limiter.acquire("a") is Verdict.REPEAT
        assert limiter.acquire("b") is Verdict.ALLOW
        assert limiter.acquire("a") is Verdict.ALLOW  # not "in a row" any more
        clock.now += 11
        assert limiter.acquire("a") is Verdict.ALLOW

    def test_token_bucket_allows_a_burst_then_limits_then_refills(self) -> None:
        clock = Clock()
        limiter = RateLimiter(repeat_seconds=0, burst=3, refill_seconds=30, clock=clock)
        assert [limiter.acquire(str(i)) for i in range(4)] == [
            Verdict.ALLOW,
            Verdict.ALLOW,
            Verdict.ALLOW,
            Verdict.LIMITED,
        ]
        clock.now += 10  # 30 s refill for 3 tokens = one per 10 s
        assert limiter.acquire("x") is Verdict.ALLOW
        assert limiter.acquire("y") is Verdict.LIMITED

    def test_held_back_count_is_reported_once(self) -> None:
        limiter = RateLimiter(repeat_seconds=0, burst=1, refill_seconds=100, clock=Clock())
        limiter.acquire("a")
        limiter.acquire("b")
        limiter.acquire("c")
        assert limiter.take_held_back() == 2
        assert limiter.take_held_back() == 0

    def test_refund_undoes_an_allow(self) -> None:
        limiter = RateLimiter(repeat_seconds=60, burst=1, refill_seconds=1000, clock=Clock())
        assert limiter.acquire("a") is Verdict.ALLOW
        limiter.refund("a")
        assert limiter.acquire("a") is Verdict.ALLOW


class Inner:
    name = "inner"

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.sent: list[Event] = []

    def send(self, event: Event) -> bool:
        self.sent.append(event)
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if isinstance(outcome, Exception):
            raise outcome
        return bool(outcome)


class TestRateLimitedNotifier:
    def test_repeats_are_dropped_without_error(self) -> None:
        inner = Inner()
        notifier = RateLimitedNotifier(inner, RateLimiter(repeat_seconds=60, clock=Clock()))
        assert notifier.send(make_event()) is True
        assert notifier.send(make_event(event_id="e2")) is False
        assert len(inner.sent) == 1

    def test_an_empty_bucket_raises_so_the_event_stays_pending(self) -> None:
        limiter = RateLimiter(repeat_seconds=0, burst=1, refill_seconds=100, clock=Clock())
        notifier = RateLimitedNotifier(Inner(), limiter)
        notifier.send(make_event(text="one"))
        with pytest.raises(NotifyError, match="rate limit"):
            notifier.send(make_event(text="two"))

    def test_a_failed_send_does_not_turn_the_retry_into_a_repeat(self) -> None:
        inner = Inner(NotifyError("down"), True)
        notifier = RateLimitedNotifier(inner, RateLimiter(repeat_seconds=60, clock=Clock()))
        with pytest.raises(NotifyError):
            notifier.send(make_event())
        assert notifier.send(make_event()) is True

    def test_the_next_alert_says_how_many_were_held_back(self) -> None:
        clock = Clock()
        limiter = RateLimiter(repeat_seconds=0, burst=1, refill_seconds=10, clock=clock)
        inner = Inner()
        notifier = RateLimitedNotifier(inner, limiter)
        notifier.send(make_event(text="one"))
        for text in ("two", "three"):
            with pytest.raises(NotifyError):
                notifier.send(make_event(text=text))
        clock.now += 10
        notifier.send(make_event(text="four"))
        assert inner.sent[-1].extra["held_back"] == 2
