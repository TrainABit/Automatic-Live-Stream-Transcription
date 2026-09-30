"""The process heartbeat and the source-health alerts behind it."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import pytest

from livestream_transcriber.heartbeat import HEARTBEAT_FIELDS, ProcessHeartbeat, format_fields
from livestream_transcriber.source_health import (
    HealthWatch,
    SourceHealth,
    TrackedStatusMonitor,
)
from livestream_transcriber.stream.auto_resume import ProbeBackoff
from livestream_transcriber.stream.fallback import LiveSourceSelector

from .support.audio import Clock
from .support.scripted_probes import BOT, ERR, LIVE, OFF, AlertRecorder, scripted_selector


async def selection(*statuses: str) -> Any:
    selector, _probe = scripted_selector(statuses)  # type: ignore[arg-type]
    assert isinstance(selector, LiveSourceSelector)
    return await selector.select()


class TestFormatFields:
    def test_none_values_are_left_out(self) -> None:
        assert format_fields({"capture": "live", "dropped": None}, ("capture", "dropped")) == (
            " capture=live"
        )

    def test_lists_are_comma_joined_and_empty_lists_are_a_dash(self) -> None:
        summary = {"health_reasons": ["a", "b"], "stt_paused": False, "queue_depth": []}
        text = format_fields(summary, ("health_reasons", "stt_paused", "queue_depth"))
        assert text == " health_reasons=a,b stt_paused=false queue_depth=-"

    def test_a_value_stays_one_token(self) -> None:
        assert format_fields({"capture": "not live"}, ("capture",)) == " capture=not_live"

    def test_only_the_named_fields_are_used(self) -> None:
        assert format_fields({"capture": "live", "secret": "x"}, ("capture",)) == " capture=live"


class TestBeat:
    def heartbeat(self, **kwargs: Any) -> tuple[ProcessHeartbeat, io.StringIO, Clock]:
        out = io.StringIO()
        clock = Clock(100.0)
        return ProcessHeartbeat(out=out, clock=clock, **kwargs), out, clock

    async def test_a_session_beat_carries_the_session_summary(self) -> None:
        beat, out, clock = self.heartbeat()
        beat.attach(lambda: {"queue_depth": 3, "dropped": 0, "health": "ok"})
        clock.advance(12.5)
        await beat.beat()
        line = out.getvalue().splitlines()[0]
        assert line.startswith("LST_HEARTBEAT tick=1 uptime_s=12.5 version=")
        assert "capture=live" in line
        assert "queue_depth=3" in line
        assert "health=ok" in line

    async def test_every_line_is_distinct(self) -> None:
        beat, out, clock = self.heartbeat()
        beat.attach(lambda: {})
        for _ in range(3):
            clock.advance(1)
            await beat.beat()
        lines = out.getvalue().splitlines()
        assert len(set(lines)) == 3

    async def test_idle_process_says_so_when_asked(self) -> None:
        beat, out, _ = self.heartbeat(idle_beats=True)
        await beat.beat()
        assert "capture=idle" in out.getvalue()

    async def test_idle_process_is_silent_by_default(self) -> None:
        beat, out, _ = self.heartbeat()
        await beat.beat()
        assert out.getvalue() == ""
        assert beat.tick == 0

    async def test_detach_returns_to_idle(self) -> None:
        beat, out, _ = self.heartbeat(idle_beats=True)
        beat.attach(lambda: {"queue_depth": 1})
        await beat.beat()
        beat.detach()
        await beat.beat()
        first, second = out.getvalue().splitlines()
        assert "capture=live" in first
        assert "capture=idle" in second

    async def test_an_idle_bot_check_degrades_health(self) -> None:
        beat, out, _ = self.heartbeat(idle_beats=True)
        beat.source.observe(await selection(BOT, BOT))
        await beat.beat()
        line = out.getvalue()
        assert "health=degraded" in line
        assert "source_bot_check" in line
        assert "selection_state=bot_check" in line

    async def test_a_delivering_capture_clears_the_blocked_clock(self) -> None:
        beat, _, clock = self.heartbeat()
        beat.source.observe(await selection(ERR, ERR))
        assert beat.source.blocked_since is not None
        beat.attach(lambda: {"delivering": True})
        clock.advance(1)
        await beat.beat()
        assert beat.source.blocked_since is None

    async def test_the_line_is_logged_too(self, caplog: pytest.LogCaptureFixture) -> None:
        beat, _, _ = self.heartbeat()
        beat.attach(lambda: {})
        with caplog.at_level(logging.INFO):
            await beat.beat()
        assert any(r.getMessage() == "process heartbeat" for r in caplog.records)

    async def test_memory_over_the_limit_degrades_health(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from livestream_transcriber import heartbeat as hb

        monkeypatch.setattr(hb, "current_rss_mb", lambda: 900.0)
        beat, out, _ = self.heartbeat(memory_limit_mb=1000.0)
        beat.attach(lambda: {"health": "ok"})
        await beat.beat()
        line = out.getvalue()
        assert "rss_mb=900" in line
        assert "health=degraded" in line
        assert "health_reasons=memory" in line

    async def test_a_broken_output_stream_does_not_stop_the_beat(self) -> None:
        beat, out, _ = self.heartbeat()
        out.close()
        beat.attach(lambda: {})
        await beat.beat()
        assert beat.tick == 1

    async def test_every_documented_field_is_named_once(self) -> None:
        assert len(set(HEARTBEAT_FIELDS)) == len(HEARTBEAT_FIELDS)


class TestRunLoop:
    async def test_beats_until_stopped(self) -> None:
        beat = ProcessHeartbeat(out=io.StringIO(), idle_beats=True)
        stop = asyncio.Event()
        task = asyncio.create_task(beat.run(stop, 0.01))
        for _ in range(200):
            if beat.tick >= 3:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 2)
        assert beat.tick >= 3

    async def test_a_beat_that_raises_does_not_end_the_loop(self) -> None:
        beat = ProcessHeartbeat(out=io.StringIO())
        calls = 0

        def broken() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            raise RuntimeError("summary bug")

        beat.attach(broken)
        stop = asyncio.Event()
        task = asyncio.create_task(beat.run(stop, 0.01))
        for _ in range(200):
            if calls >= 2:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 2)
        assert calls >= 2


class TestSourceHealth:
    async def test_it_starts_unknown(self) -> None:
        fields = SourceHealth().fields()
        assert fields["selection_state"] is None
        assert fields["selection_blocked_s"] is None
        assert fields["bot_check"] is None

    async def test_a_conclusive_answer_has_no_blocked_time(self) -> None:
        health = SourceHealth()
        health.observe(await selection(OFF, OFF))
        assert health.blocked_seconds() is None
        assert health.fields()["selection_state"] == "offline"

    async def test_probe_errors_start_the_blocked_clock_and_an_answer_stops_it(self) -> None:
        clock = Clock(0.0)
        health = SourceHealth(clock=clock)
        health.observe(await selection(ERR, ERR))
        clock.advance(30)
        assert health.blocked_seconds() == pytest.approx(30)
        health.observe(await selection(LIVE, OFF))
        assert health.blocked_seconds() is None

    async def test_the_same_round_observed_twice_counts_once(self) -> None:
        clock = Clock(0.0)
        health = SourceHealth(clock=clock)
        round_ = await selection(ERR, ERR)
        health.observe(round_)
        clock.advance(10)
        health.observe(round_)
        assert health.observed_at == 0.0

    async def test_next_probe_and_age_come_from_the_backoff_and_clock(self) -> None:
        clock = Clock(0.0)
        health = SourceHealth(clock=clock)
        health.observe(await selection(OFF, OFF))
        clock.advance(4)
        fields = health.fields(ProbeBackoff(30.0, 60.0))
        assert fields["last_probe_age_s"] == 4.0
        assert fields["next_probe_s"] is not None


class TestTrackedStatusMonitor:
    async def test_it_feeds_the_health_and_alerts_on_a_change(self) -> None:
        health = SourceHealth()
        alerts = AlertRecorder()
        monitor = TrackedStatusMonitor(alerts, health)
        await monitor.observe(await selection(BOT, BOT), in_session=False, immediate=False)
        assert health.fields()["selection_state"] == "bot_check"


class TestHealthWatch:
    async def test_a_long_stt_outage_alerts_once_per_episode(self) -> None:
        alerts = AlertRecorder()
        watch = HealthWatch(alerts, stt_outage_seconds=300)
        source = SourceHealth()
        capture = {"stt_outage_s": 400.0, "stt_paused": True}
        await watch.check(capture=capture, source=source)
        await watch.check(capture=capture, source=source)
        assert alerts.titles == ["Speech-to-text is failing"]
        assert "paused: yes" in alerts.bodies[0]

        await watch.check(capture={"stt_outage_s": None}, source=source)  # the provider is back
        await watch.check(capture=capture, source=source)
        assert len(alerts.sent) == 2

    async def test_a_short_outage_is_quiet(self) -> None:
        alerts = AlertRecorder()
        watch = HealthWatch(alerts, stt_outage_seconds=300)
        await watch.check(capture={"stt_outage_s": 20.0}, source=SourceHealth())
        assert alerts.sent == []

    async def test_a_zero_threshold_turns_the_check_off(self) -> None:
        alerts = AlertRecorder()
        watch = HealthWatch(alerts, stt_outage_seconds=0, source_blocked_seconds=0)
        source = SourceHealth()
        source.observe(await selection(ERR, ERR))
        await watch.check(capture={"stt_outage_s": 10_000.0}, source=source)
        assert alerts.sent == []

    async def test_a_blocked_source_alerts_after_the_threshold(self) -> None:
        clock = Clock(0.0)
        source = SourceHealth(clock=clock)
        alerts = AlertRecorder()
        watch = HealthWatch(alerts, source_blocked_seconds=600)
        source.observe(await selection(BOT, BOT))
        clock.advance(300)
        await watch.check(capture=None, source=source)
        assert alerts.sent == []
        clock.advance(400)
        await watch.check(capture=None, source=source, next_probe_s=120)
        await watch.check(capture=None, source=source)
        assert alerts.titles == ["Source blocked"]
        assert "next probe in 120 s" in alerts.bodies[0]

    async def test_a_conclusive_answer_re_arms_the_source_alert(self) -> None:
        clock = Clock(0.0)
        source = SourceHealth(clock=clock)
        alerts = AlertRecorder()
        watch = HealthWatch(alerts, source_blocked_seconds=60)
        source.observe(await selection(ERR, ERR))
        clock.advance(100)
        await watch.check(capture=None, source=source)
        source.observe(await selection(OFF, OFF))
        await watch.check(capture=None, source=source)
        assert watch.blocked_alerted is False
        source.observe(await selection(ERR, ERR))
        clock.advance(100)
        await watch.check(capture=None, source=source)
        assert len(alerts.sent) == 2

    async def test_a_failing_alert_hook_does_not_raise(self) -> None:
        async def broken(_title: str, _body: str) -> None:
            raise RuntimeError("delivery down")

        watch = HealthWatch(broken, stt_outage_seconds=1)
        await watch.check(capture={"stt_outage_s": 5.0}, source=SourceHealth())
        assert watch.stt_alerted is True

    async def test_without_an_alert_hook_it_only_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        watch = HealthWatch(None, stt_outage_seconds=1)
        with caplog.at_level(logging.WARNING):
            await watch.check(capture={"stt_outage_s": 5.0}, source=SourceHealth())
        assert any(r.getMessage() == "health alert" for r in caplog.records)
