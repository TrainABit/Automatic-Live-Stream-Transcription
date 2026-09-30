"""The transcription pipeline: ordering, backpressure, shutdown, rules and alerts."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from livestream_transcriber.notify.dispatcher import NotificationDispatcher
from livestream_transcriber.pipeline.engine import (
    PipelineOptions,
    TranscriptionPipeline,
    utterance_delta,
)
from livestream_transcriber.rules import RuleEngine, RuleSet

from .support.pipeline import ListSink, RecordingNotifier, ScriptedTranscriber, audible, quiet
from .support.waits import wait_until

KEYWORD_RULES = """
version: 1
rules:
  - id: release
    type: keyword
    keywords: ["new release"]
    severity: warning
    notify: [console]
    cooldown_seconds: 0
"""


def make_pipeline(
    transcriber: ScriptedTranscriber | None, sink: ListSink | None = None, **options: object
) -> tuple[TranscriptionPipeline, ListSink]:
    sink = sink or ListSink()
    pipeline = TranscriptionPipeline(
        transcriber,
        options=PipelineOptions(spill_chunks=0, lossless=True, **options),  # type: ignore[arg-type]
        sinks=[sink],
    )
    return pipeline, sink


class TestUtteranceDelta:
    def test_prefix_extension_returns_only_the_new_words(self) -> None:
        assert utterance_delta("we are going live in", "We are going live in five minutes") == (
            "five minutes"
        )

    def test_punctuation_and_case_do_not_break_the_prefix(self) -> None:
        assert utterance_delta("Hello, world", "hello world again") == "again"

    def test_rewording_keeps_only_unknown_words(self) -> None:
        assert utterance_delta("the big red car", "a big red car drives") == "a drives"

    def test_no_previous_utterance_is_the_whole_text(self) -> None:
        assert utterance_delta(None, "  fresh start ") == "fresh start"

    def test_nothing_new_is_empty(self) -> None:
        assert utterance_delta("one two three", "one two three") == ""


class TestSubtitleCues:
    async def test_provider_segments_reach_the_sink_as_session_time_parts(self) -> None:
        stt = ScriptedTranscriber(
            {100.0: "hello there friend"},
            segments={
                100.0: [
                    {"start": 0.2, "end": 1.0, "text": "hello there"},
                    {"start": 1.2, "end": 2.0, "text": "friend"},
                ]
            },
        )
        pipeline, sink = make_pipeline(stt)
        await pipeline.on_audio(audible(100.0))
        await pipeline.drain()
        (segment,) = sink.segments
        assert [(p.start, p.end, p.text) for p in segment.parts] == [
            (100.2, 101.0, "hello there"),
            (101.2, 102.0, "friend"),
        ]


class TestOrdering:
    async def test_out_of_order_results_are_written_in_media_order(self) -> None:
        # The first chunk answers last; three workers make the reordering real.
        stt = ScriptedTranscriber(delay={0.0: 0.4, 2.5: 0.2})
        pipeline, sink = make_pipeline(stt, workers=3)
        for i in range(6):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()

        assert sink.texts == [f"chunk {i * 2.5:g}" for i in range(6)]
        assert pipeline.order.reordered > 0
        assert [s.start for s in sink.segments] == sorted(s.start for s in sink.segments)

    async def test_wait_returns_the_utterances_of_that_chunk(self) -> None:
        pipeline, _ = make_pipeline(ScriptedTranscriber({0.0: "hello there"}))
        segments = await pipeline.on_audio(audible(0.0), wait=True)
        await pipeline.drain()
        assert [s.text for s in segments] == ["hello there"]
        assert (segments[0].start, segments[0].end) == (0.0, 2.5)

    async def test_failed_chunk_leaves_a_gap_but_does_not_block_later_ones(self) -> None:
        pipeline, sink = make_pipeline(ScriptedTranscriber(fail={2.5}))
        for i in range(4):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()
        assert sink.texts == ["chunk 0", "chunk 5", "chunk 7.5"]
        assert pipeline.summary()["speech"]["gaps"] == 1
        assert pipeline.stats.failed == 1


class TestSilenceAndDuplicates:
    async def test_silent_chunks_never_reach_the_provider(self) -> None:
        stt = ScriptedTranscriber()
        pipeline, sink = make_pipeline(stt)
        await pipeline.on_audio(quiet(0.0))
        await pipeline.on_audio(audible(2.5))
        await pipeline.on_audio(quiet(5.0))
        await pipeline.drain()
        assert stt.calls == [2.5]
        assert sink.texts == ["chunk 2.5"]
        assert pipeline.stats.empty == 2

    async def test_repeated_text_is_written_once(self) -> None:
        pipeline, sink = make_pipeline(
            ScriptedTranscriber(lambda start: "same sentence again"), workers=1
        )
        for i in range(3):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()
        assert sink.texts == ["same sentence again"]
        assert pipeline.summary()["speech"]["duplicates_suppressed"] == 2

    async def test_overlapping_chunks_are_stitched_without_repeating_words(self) -> None:
        stt = ScriptedTranscriber({0.0: "we are going live in", 2.5: "live in five minutes"})
        pipeline, sink = make_pipeline(stt)
        for i in range(2):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()
        assert sink.texts == ["we are going live in", "five minutes"]

    async def test_no_speech_answer_produces_nothing(self) -> None:
        pipeline, sink = make_pipeline(ScriptedTranscriber(lambda start: None))
        await pipeline.on_audio(audible(0.0))
        await pipeline.drain()
        assert sink.segments == []
        assert pipeline.stats.empty == 1

    async def test_speech_off_counts_and_drops_chunks(self) -> None:
        pipeline, sink = make_pipeline(None)
        await pipeline.on_audio(audible(0.0))
        await pipeline.drain()
        assert sink.segments == []
        assert pipeline.stats.produced == 1


class TestBackpressure:
    async def test_slow_stt_does_not_stall_capture(self) -> None:
        stt = ScriptedTranscriber(delay=0.25)
        pipeline = TranscriptionPipeline(
            stt, options=PipelineOptions(queue_size=2, spill_chunks=0), sinks=[ListSink()]
        )
        started = time.monotonic()
        for i in range(30):
            await pipeline.on_audio(audible(i * 2.5))
        elapsed = time.monotonic() - started
        # Thirty chunks at 0.25 s each would take 7.5 s if capture waited for STT.
        assert elapsed < 2.0
        await pipeline.drain(wait=False)

    async def test_live_queue_drops_oldest_and_counts_it(self) -> None:
        gate = threading.Event()
        stt = ScriptedTranscriber(gate=gate)
        sink = ListSink()
        pipeline = TranscriptionPipeline(
            stt, options=PipelineOptions(queue_size=2, spill_chunks=0), sinks=[sink]
        )
        await pipeline.on_audio(audible(0.0))
        await wait_until(lambda: pipeline.stats.started == 1, what="the first call to start")
        for i in range(1, 8):
            await pipeline.on_audio(audible(i * 2.5))
        # One chunk is in flight and two wait; the other five were evicted, oldest first.
        await wait_until(lambda: pipeline.stats.dropped >= 5, what="the queue to overflow")
        gate.set()
        await pipeline.drain()

        assert pipeline.stats.dropped == 5
        audio = pipeline.summary()["audio"]
        accounted = audio["completed"] + audio["dropped"] + audio["failed"] + audio["empty"]
        assert accounted == audio["produced"] == 8
        # The survivors are the first (already in flight) and the two newest.
        assert sink.texts == ["chunk 0", "chunk 15", "chunk 17.5"]
        assert pipeline.summary()["speech"]["gaps"] == 5

    async def test_lossless_mode_waits_instead_of_dropping(self) -> None:
        stt = ScriptedTranscriber(delay=0.02)
        sink = ListSink()
        pipeline = TranscriptionPipeline(
            stt,
            options=PipelineOptions(queue_size=1, spill_chunks=0, lossless=True),
            sinks=[sink],
        )
        for i in range(10):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()
        assert pipeline.stats.dropped == 0
        assert len(sink.segments) == 10

    async def test_overflow_spills_to_disk_before_dropping(self, tmp_path: object) -> None:
        gate = threading.Event()
        pipeline = TranscriptionPipeline(
            ScriptedTranscriber(gate=gate),
            options=PipelineOptions(queue_size=1, spill_chunks=20),
            sinks=[ListSink()],
        )
        for i in range(10):
            await pipeline.on_audio(audible(i * 2.5))
        assert pipeline.summary()["queue"]["spilled"] > 0
        assert pipeline.stats.dropped == 0
        gate.set()
        await pipeline.drain()
        assert pipeline.stats.completed == 10


class TestShutdown:
    async def test_drain_is_idempotent(self) -> None:
        pipeline, sink = make_pipeline(ScriptedTranscriber())
        await pipeline.on_audio(audible(0.0))
        await pipeline.drain()
        await pipeline.drain()
        await pipeline.drain(wait=False)
        assert sink.texts == ["chunk 0"]

    async def test_audio_after_drain_is_ignored(self) -> None:
        stt = ScriptedTranscriber()
        pipeline, _ = make_pipeline(stt)
        await pipeline.drain()
        assert await pipeline.on_audio(audible(0.0), wait=True) == []
        assert stt.calls == []

    async def test_drain_without_any_audio(self) -> None:
        pipeline, _ = make_pipeline(ScriptedTranscriber())
        await pipeline.drain()
        assert pipeline.health()["health"] == "ok"

    async def test_abandoning_does_not_wait_for_a_wedged_provider(self) -> None:
        gate = threading.Event()
        pipeline, _ = make_pipeline(ScriptedTranscriber(gate=gate))
        await pipeline.on_audio(audible(0.0))
        await wait_until(lambda: pipeline.stats.started == 1, what="the call to start")
        started = time.monotonic()
        await pipeline.drain(wait=False)
        assert time.monotonic() - started < 2.0
        gate.set()

    async def test_drain_timeout_abandons_stuck_work(self) -> None:
        gate = threading.Event()
        pipeline, _ = make_pipeline(ScriptedTranscriber(gate=gate))
        await pipeline.on_audio(audible(0.0))
        await wait_until(lambda: pipeline.stats.started == 1, what="the call to start")
        started = time.monotonic()
        await pipeline.drain(timeout=0.2)
        assert time.monotonic() - started < 2.0
        gate.set()

    async def test_spill_directory_is_removed(self) -> None:
        pipeline = TranscriptionPipeline(
            ScriptedTranscriber(), options=PipelineOptions(spill_chunks=4), sinks=[ListSink()]
        )
        spill = pipeline._queue.spill_dir
        assert spill is not None and spill.is_dir()
        await pipeline.drain()
        assert not spill.exists()


class TestRulesAndNotifications:
    def _pipeline(
        self, stt: ScriptedTranscriber, notifier: RecordingNotifier
    ) -> tuple[TranscriptionPipeline, ListSink]:
        sink = ListSink()
        engine = RuleEngine(RuleSet.from_yaml(KEYWORD_RULES))
        pipeline = TranscriptionPipeline(
            stt,
            options=PipelineOptions(spill_chunks=0, lossless=True),
            sinks=[sink],
            rules=engine,
            dispatcher=NotificationDispatcher({"console": notifier}, default_targets=("console",)),
            source_url="https://example.com/stream?token=secret-value",
            session_id=7,
        )
        return pipeline, sink

    async def test_a_hit_becomes_a_delivered_event(self) -> None:
        notifier = RecordingNotifier()
        pipeline, _ = self._pipeline(
            ScriptedTranscriber({0.0: "today there is a new release for everyone"}), notifier
        )
        await pipeline.on_audio(audible(0.0))
        await pipeline.drain()

        assert len(notifier.events) == 1
        event = notifier.events[0]
        assert event.rule_id == "release"
        assert event.matched_text.lower() == "new release"
        assert event.session_id == 7
        assert "secret-value" not in (event.source_url or "")
        assert pipeline.summary()["speech"]["rule_hits"] == 1
        assert pipeline.summary()["speech"]["events"] == 1

    async def test_a_phrase_split_by_a_chunk_boundary_still_matches_once(self) -> None:
        notifier = RecordingNotifier()
        pipeline, _ = self._pipeline(
            ScriptedTranscriber({0.0: "we have a new", 2.5: "release today"}), notifier
        )
        for i in range(2):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()
        assert [e.matched_text.lower() for e in notifier.events] == ["new release"]

    async def test_words_seen_in_the_previous_chunk_do_not_fire_again(self) -> None:
        notifier = RecordingNotifier()
        pipeline, _ = self._pipeline(
            ScriptedTranscriber({0.0: "a new release", 2.5: "is coming soon"}), notifier
        )
        for i in range(2):
            await pipeline.on_audio(audible(i * 2.5))
        await pipeline.drain()
        assert len(notifier.events) == 1

    async def test_no_carry_across_a_long_silence(self) -> None:
        notifier = RecordingNotifier()
        pipeline, _ = self._pipeline(
            ScriptedTranscriber({0.0: "we have a new", 60.0: "release of the week"}), notifier
        )
        await pipeline.on_audio(audible(0.0))
        await pipeline.on_audio(audible(60.0))
        await pipeline.drain()
        assert notifier.events == []

    async def test_a_failing_notifier_does_not_disturb_transcription(self) -> None:
        class Broken:
            def send(self, event: object) -> bool:
                raise RuntimeError("boom")

        sink = ListSink()
        pipeline = TranscriptionPipeline(
            ScriptedTranscriber({0.0: "a new release"}),
            options=PipelineOptions(spill_chunks=0, lossless=True),
            sinks=[sink],
            rules=RuleEngine(RuleSet.from_yaml(KEYWORD_RULES)),
            dispatcher=NotificationDispatcher({"console": Broken()}, default_targets=("console",)),  # type: ignore[dict-item]
        )
        await pipeline.on_audio(audible(0.0))
        await pipeline.drain()
        assert sink.texts == ["a new release"]
        assert pipeline.summary()["speech"]["notify_failures"] == 1

    async def test_a_hung_notifier_with_a_full_queue_cannot_hold_shutdown(self) -> None:
        gate = threading.Event()

        class Hung:
            def send(self, event: object) -> bool:
                gate.wait(10)
                return True

        pipeline = TranscriptionPipeline(
            ScriptedTranscriber(
                {0.0: "a new release", 60.0: "a new release", 120.0: "a new release"}
            ),
            options=PipelineOptions(
                spill_chunks=0, lossless=True, notify_queue_size=1, flush_seconds=0.3
            ),
            sinks=[ListSink()],
            rules=RuleEngine(RuleSet.from_yaml(KEYWORD_RULES)),
            dispatcher=NotificationDispatcher({"console": Hung()}, default_targets=("console",)),  # type: ignore[dict-item]
        )
        for start in (0.0, 60.0, 120.0):
            await pipeline.on_audio(audible(start))
        started = time.monotonic()
        try:
            await asyncio.wait_for(pipeline.drain(), 5)
            assert time.monotonic() - started < 3.0
        finally:
            gate.set()

    async def test_on_hit_callback_sees_the_hit(self) -> None:
        seen: list[str] = []
        pipeline = TranscriptionPipeline(
            ScriptedTranscriber({0.0: "a new release"}),
            options=PipelineOptions(spill_chunks=0, lossless=True),
            rules=RuleEngine(RuleSet.from_yaml(KEYWORD_RULES)),
            on_hit=lambda hit: seen.append(hit.rule_id),
        )
        await pipeline.on_audio(audible(0.0))
        await pipeline.drain()
        assert seen == ["release"]


class TestHealth:
    async def test_lag_is_zero_when_nothing_is_outstanding(self) -> None:
        pipeline, _ = make_pipeline(ScriptedTranscriber())
        assert pipeline.stt_lag_seconds() is None
        await pipeline.on_audio(audible(0.0), wait=True)
        assert pipeline.stt_lag_seconds() == 0.0
        await pipeline.drain()

    async def test_lag_is_measured_against_the_audio_head(self) -> None:
        gate = threading.Event()
        pipeline, _ = make_pipeline(ScriptedTranscriber(gate=gate))
        for i in range(4):
            await pipeline.on_audio(audible(i * 2.5))
        # The oldest chunk starts at 0 and the head is at 10.
        assert pipeline.stt_lag_seconds() == pytest.approx(10.0)
        assert pipeline.health()["stt_lag_s"] == 10.0
        gate.set()
        await pipeline.drain()

    async def test_health_uses_the_names_the_heartbeat_prints(self) -> None:
        from livestream_transcriber.heartbeat import HEARTBEAT_FIELDS

        pipeline, _ = make_pipeline(ScriptedTranscriber())
        health = pipeline.health()
        for name in ("health", "health_reasons", "stt_lag_s", "stt_paused", "queue_depth"):
            assert name in HEARTBEAT_FIELDS
            assert name in health
        await pipeline.drain()

    async def test_health_reports_degradation_reasons(self) -> None:
        gate = threading.Event()
        pipeline, _ = make_pipeline(ScriptedTranscriber(gate=gate), health_lag_seconds=5.0)
        for i in range(4):
            await pipeline.on_audio(audible(i * 2.5))
        health = pipeline.health()
        assert health["health"] == "degraded"
        assert "stt_lag" in health["health_reasons"]
        gate.set()
        await pipeline.drain()

    async def test_summary_has_the_stage_counters(self) -> None:
        pipeline, _ = make_pipeline(ScriptedTranscriber())
        await pipeline.on_audio(audible(0.0), wait=True)
        await pipeline.drain()
        summary = pipeline.summary()
        assert set(summary) >= {"audio", "queue", "order", "speech", "cost_usd"}
        assert summary["audio"]["completed"] == 1
        assert summary["order"]["outstanding"] == 0


class TestOptions:
    def test_invalid_values_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            PipelineOptions(workers=0)
        with pytest.raises(ValueError):
            PipelineOptions(queue_size=0)
        with pytest.raises(ValueError):
            PipelineOptions(spill_chunks=-1)

    def test_from_settings_separates_live_and_finite_runs(self) -> None:
        from livestream_transcriber.config import Settings

        settings = Settings()
        live = PipelineOptions.from_settings(settings, live=True)
        finite = PipelineOptions.from_settings(settings, live=False)
        assert live.overload is not None and not live.lossless
        assert finite.overload is None and finite.lossless
