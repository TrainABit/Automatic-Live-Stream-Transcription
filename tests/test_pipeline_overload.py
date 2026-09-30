"""The overload guard: pause on sustained lag or drops, probe for recovery, back off on relapse."""

from __future__ import annotations

import threading

import pytest

from livestream_transcriber.config import Settings
from livestream_transcriber.pipeline.engine import PipelineOptions, TranscriptionPipeline
from livestream_transcriber.pipeline.overload import (
    REASON_DROPS,
    REASON_LAG,
    OverloadConfig,
    OverloadGuard,
)

from .support.audio import Clock
from .support.pipeline import ListSink, ScriptedTranscriber, audible, quiet
from .support.waits import wait_until

CONFIG = OverloadConfig(
    max_lag_seconds=5.0,
    max_drop_ratio=0.5,
    window_seconds=10.0,
    probe_interval_seconds=20.0,
    min_samples=4,
    stable_seconds=100.0,
    max_probe_seconds=80.0,
    max_fallback_hold_seconds=500.0,
)


def guard(clock: Clock, config: OverloadConfig = CONFIG) -> OverloadGuard:
    return OverloadGuard(config, clock=clock)


class TestLag:
    def test_a_single_slow_moment_is_not_an_overload(self) -> None:
        clock = Clock()
        g = guard(clock)
        assert g.evaluate(30.0) is None
        clock.advance(9)
        assert g.evaluate(30.0) is None

    def test_lag_sustained_for_a_whole_window_pauses(self) -> None:
        clock = Clock()
        g = guard(clock)
        assert g.evaluate(30.0) is None
        clock.advance(10)
        verdict = g.evaluate(31.0)
        assert verdict is not None
        assert verdict.reason == REASON_LAG
        assert verdict.lag_seconds == 31.0

    def test_catching_up_restarts_the_clock(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.evaluate(30.0)
        clock.advance(8)
        assert g.evaluate(1.0) is None  # caught up: the sustained period is over
        clock.advance(8)
        assert g.evaluate(30.0) is None  # a new period starts here
        clock.advance(9)
        assert g.evaluate(30.0) is None

    def test_unknown_lag_never_pauses(self) -> None:
        clock = Clock()
        g = guard(clock)
        clock.advance(1000)
        assert g.evaluate(None) is None

    def test_zero_limit_disables_the_lag_signal(self) -> None:
        clock = Clock()
        g = guard(clock, OverloadConfig(max_lag_seconds=0))
        g.evaluate(1e6)
        clock.advance(1e6)
        assert g.evaluate(1e6) is None


class TestDrops:
    def test_needs_enough_samples(self) -> None:
        g = guard(Clock())
        for _ in range(3):
            g.record_outcome(True)
        assert g.evaluate(0.0) is None

    def test_a_high_drop_ratio_pauses(self) -> None:
        g = guard(Clock())
        for dropped in (True, True, True, False):
            g.record_outcome(dropped)
        verdict = g.evaluate(0.0)
        assert verdict is not None
        assert verdict.reason == REASON_DROPS
        assert verdict.drop_ratio == pytest.approx(0.75)

    def test_old_outcomes_leave_the_window(self) -> None:
        clock = Clock()
        g = guard(clock)
        for _ in range(4):
            g.record_outcome(True)
        clock.advance(11)
        assert g.drop_ratio() == (None, 0)
        assert g.evaluate(0.0) is None

    def test_ratio_at_the_limit_is_tolerated(self) -> None:
        g = guard(Clock())
        for dropped in (True, True, False, False):
            g.record_outcome(dropped)
        assert g.evaluate(0.0) is None


class TestPauseAndProbe:
    def test_probe_waits_for_the_interval_and_an_audible_chunk(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.pause(REASON_LAG)
        assert not g.take_probe(audible=True)  # too early
        clock.advance(20)
        assert not g.take_probe(audible=False)  # silence proves nothing
        assert g.take_probe(audible=True)
        assert not g.take_probe(audible=True)  # one probe at a time

    def test_a_probe_that_reached_no_provider_is_repeated_at_once(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.pause(REASON_LAG)
        clock.advance(20)
        assert g.take_probe(audible=True)
        g.probe_ended(no_verdict=True)
        assert g.take_probe(audible=True)

    def test_a_probe_with_a_verdict_waits_out_the_interval(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.pause(REASON_LAG)
        clock.advance(20)
        assert g.take_probe(audible=True)
        g.probe_ended(no_verdict=False)
        assert not g.take_probe(audible=True)
        clock.advance(20)
        assert g.take_probe(audible=True)

    def test_no_probes_while_running(self) -> None:
        assert not guard(Clock()).take_probe(audible=True)

    def test_outcomes_are_not_recorded_while_paused(self) -> None:
        g = guard(Clock())
        g.pause(REASON_LAG)
        g.record_outcome(True)
        assert g.drop_ratio() == (None, 0)

    def test_resume_clears_the_paused_state(self) -> None:
        g = guard(Clock())
        g.pause(REASON_LAG)
        assert g.resume() is None
        assert not g.paused
        assert g.pause_reason is None
        assert g.resumes == 1
        assert g.evaluate(0.0) is None


class TestRelapse:
    def test_relapse_doubles_the_probe_interval_up_to_the_cap(self) -> None:
        clock = Clock()
        g = guard(clock)
        assert g.pause(REASON_LAG) == 20.0
        g.resume()
        clock.advance(10)  # well inside stable_seconds
        assert g.pause(REASON_LAG) == 40.0
        g.resume()
        clock.advance(10)
        assert g.pause(REASON_LAG) == 80.0
        g.resume()
        clock.advance(10)
        assert g.pause(REASON_LAG) == 80.0  # capped by max_probe_seconds

    def test_a_pause_after_a_stable_stretch_starts_over(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.pause(REASON_LAG)
        g.resume()
        clock.advance(10)
        assert g.pause(REASON_LAG) == 40.0
        g.resume()
        clock.advance(101)
        assert g.pause(REASON_LAG) == 20.0
        assert g.relapses == 0


class TestFallbackHold:
    def test_an_overloaded_fallback_stays_off_after_recovery(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.pause(REASON_LAG, fallback_active=True)
        hold = g.resume()
        assert hold == 100.0
        assert not g.release_fallback_hold()
        clock.advance(99)
        assert not g.release_fallback_hold()
        clock.advance(2)
        assert g.release_fallback_hold()
        assert not g.release_fallback_hold()  # released only once

    def test_an_outage_does_not_blame_the_fallback(self) -> None:
        g = guard(Clock())
        g.pause("outage", fallback_active=True)
        assert g.resume() is None

    def test_hold_grows_with_relapses_and_is_capped(self) -> None:
        clock = Clock()
        g = guard(clock)
        g.pause(REASON_LAG, fallback_active=True)
        g.resume()
        clock.advance(10)
        g.pause(REASON_LAG, fallback_active=True)
        assert g.resume() == 200.0
        clock.advance(10)
        g.pause(REASON_LAG, fallback_active=True)
        g.resume()
        clock.advance(10)
        g.pause(REASON_LAG, fallback_active=True)
        assert g.resume() == 500.0  # 400 would be next, 800 after; capped at 500 eventually


def test_config_from_settings() -> None:
    cfg = OverloadConfig.from_settings(
        Settings(stt_max_lag_seconds=12, stt_overload_window_seconds=30, stt_max_drop_ratio=0.1)
    )
    assert (cfg.max_lag_seconds, cfg.window_seconds, cfg.max_drop_ratio) == (12.0, 30.0, 0.1)


def test_describe_is_json_friendly() -> None:
    import json

    g = guard(Clock())
    g.pause(REASON_LAG)
    json.dumps(g.describe())


class TestInThePipeline:
    """The guard wired to a real pipeline, on a fake clock."""

    async def test_sustained_lag_pauses_probes_and_resumes(self) -> None:
        clock = Clock()
        gate = threading.Event()
        stt = ScriptedTranscriber(gate=gate)
        sink = ListSink()
        alerts: list[str] = []

        async def alert(title: str, body: str) -> None:
            alerts.append(title)

        pipeline = TranscriptionPipeline(
            stt,
            options=PipelineOptions(
                queue_size=8,
                spill_chunks=0,
                overload=OverloadConfig(
                    max_lag_seconds=5.0,
                    max_drop_ratio=0.0,
                    window_seconds=10.0,
                    probe_interval_seconds=20.0,
                ),
            ),
            sinks=[sink],
            alert=alert,
            clock=clock,
        )
        guard_ = pipeline.guard
        assert guard_ is not None

        await pipeline.on_audio(audible(0.0))
        await wait_until(lambda: pipeline.stats.started == 1, what="the first call to start")
        for i in range(1, 5):
            await pipeline.on_audio(audible(i * 2.5))
        assert not guard_.paused  # lag is high, but not yet for a whole window

        clock.advance(11)
        await pipeline.on_audio(audible(12.5))
        assert guard_.paused
        assert guard_.pause_reason == REASON_LAG
        # Everything that was waiting was dropped, counted, and the lane caught up.
        assert pipeline.stats.dropped >= 4
        assert alerts == ["Speech-to-text paused"]

        gate.set()
        await wait_until(lambda: pipeline.stats.completed == 1, what="the in-flight call")

        # While paused audio is skipped, not queued.
        before = len(stt.calls)
        clock.advance(5)
        await pipeline.on_audio(audible(15.0))
        assert len(stt.calls) == before
        assert pipeline.summary()["audio"]["skipped_while_paused"] == 1

        # Silence never probes, an audible chunk after the interval does.
        clock.advance(20)
        await pipeline.on_audio(quiet(17.5))
        assert guard_.paused
        segments = await pipeline.on_audio(audible(20.0), wait=True)
        assert not guard_.paused
        assert [s.text for s in segments] == ["chunk 20"]
        await wait_until(lambda: "Speech-to-text resumed" in alerts, what="the resume alert")
        await pipeline.drain()
        assert guard_.resumes == 1

    async def test_queue_drops_alone_can_pause(self) -> None:
        gate = threading.Event()
        pipeline = TranscriptionPipeline(
            ScriptedTranscriber(gate=gate),
            options=PipelineOptions(
                queue_size=1,
                spill_chunks=0,
                overload=OverloadConfig(
                    max_lag_seconds=0.0, max_drop_ratio=0.5, window_seconds=60.0, min_samples=4
                ),
            ),
        )
        await pipeline.on_audio(audible(0.0))
        await wait_until(lambda: pipeline.stats.started == 1, what="the first call to start")
        for i in range(1, 12):
            await pipeline.on_audio(audible(i * 2.5))
        assert pipeline.guard is not None
        assert pipeline.guard.paused
        assert pipeline.guard.pause_reason == REASON_DROPS
        gate.set()
        await pipeline.drain()

    async def test_a_sustained_outage_pauses_speech(self) -> None:
        clock = Clock()
        alerts: list[str] = []

        async def alert(title: str, body: str) -> None:
            alerts.append(title)

        pipeline = TranscriptionPipeline(
            ScriptedTranscriber(fail={0.0, 2.5}),
            options=PipelineOptions(
                spill_chunks=0,
                outage_seconds=60.0,
                overload=OverloadConfig(max_lag_seconds=0.0, max_drop_ratio=0.0),
            ),
            alert=alert,
            clock=clock,
        )
        await pipeline.on_audio(audible(0.0), wait=True)
        clock.advance(61)
        await pipeline.on_audio(audible(2.5), wait=True)
        assert pipeline.guard is not None
        await wait_until(lambda: pipeline.guard is not None and pipeline.guard.paused, what="pause")
        assert pipeline.guard.pause_reason == "outage"
        await pipeline.drain()
        assert "Speech-to-text paused" in alerts

    async def test_lossless_runs_have_no_guard(self) -> None:
        pipeline = TranscriptionPipeline(
            ScriptedTranscriber(), options=PipelineOptions(lossless=True, spill_chunks=0)
        )
        assert pipeline.guard is None
        assert pipeline.stt_drop_ratio() == (None, 0)
        await pipeline.drain()
