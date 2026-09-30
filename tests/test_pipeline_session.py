"""The session loop: lifecycle, exit codes, resuming, recording and replay, signals."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from livestream_transcriber.config import Settings
from livestream_transcriber.pipeline.session import (
    ExitCode,
    SessionPlan,
    SessionRunner,
    free_transcript_stem,
    is_remote_url,
)
from livestream_transcriber.rules import RuleSet
from livestream_transcriber.store import Database
from livestream_transcriber.stream import (
    LiveSourceSelector,
    ProbeBackoff,
    SourceCandidate,
    StreamError,
    StreamNotLiveError,
    StreamResolutionError,
)
from livestream_transcriber.stream.fallback import STATUS_LIVE, STATUS_OFFLINE, ProbeResult
from livestream_transcriber.stt.base import MockTranscriber
from livestream_transcriber.stt.wrappers import CachedTranscriber

from .support.fake_source import FakeSource, audible_chunks
from .support.loopback import serve
from .support.pipeline import ScriptedTranscriber
from .support.waits import wait_until

URL = "https://example.com/live"


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "out_dir": tmp_path / "out",
        "stt_provider": "mock",
        "heartbeat_seconds": 0,
        "resume_online_stable_seconds": 0,
        "resume_end_settle_seconds": 5,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def runner_for(
    tmp_path: Path,
    plan: SessionPlan,
    sources: list[FakeSource],
    *,
    transcriber: object | None = None,
    **kwargs: object,
) -> SessionRunner:
    queue = list(sources)
    made: list[str] = []

    def factory(url: str) -> FakeSource:
        made.append(url)
        return queue.pop(0)

    settings = kwargs.pop("settings", None) or settings_for(tmp_path)  # type: ignore[assignment]
    stt = transcriber if transcriber is not None else ScriptedTranscriber()
    runner = SessionRunner(
        settings,  # type: ignore[arg-type]
        plan,
        source_factory=factory,
        backoff=ProbeBackoff(0.05, 0.2),
        transcriber_factory=lambda _settings: stt,  # type: ignore[arg-type,return-value]
        install_signals=False,
        color=False,
        **kwargs,  # type: ignore[arg-type]
    )
    runner.made = made  # type: ignore[attr-defined]
    return runner


def out_files(tmp_path: Path) -> set[str]:
    return {p.name for p in (tmp_path / "out").iterdir()}


class TestFiniteSession:
    async def test_transcribes_everything_and_writes_every_output(self, tmp_path: Path) -> None:
        source = FakeSource(audible_chunks(4))
        # A local path is checked for existence up front; the fake source stands in for it.
        (tmp_path / "f").write_text("x")
        plan = SessionPlan(url=str(tmp_path / "f"), out_dir=tmp_path / "out")
        runner = runner_for(tmp_path, plan, [source])

        assert await runner.run() == ExitCode.OK

        assert {"transcript.jsonl", "transcript.srt", "transcript.vtt", "lst.db"} <= out_files(
            tmp_path
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "transcript.jsonl").read_text().splitlines()
        ]
        assert [r["text"] for r in rows] == [f"chunk {i * 2.5:g}" for i in range(4)]
        assert (tmp_path / "out" / "transcript.vtt").read_text().startswith("WEBVTT")
        assert source.close_calls >= 1

        (report,) = runner.reports
        assert (report.chunks, report.utterances) == (4, 4)
        assert report.seconds == pytest.approx(10.0)

        with Database(tmp_path / "out" / "lst.db") as db:
            (row,) = db.list_sessions()
            assert row["ended_at"] is not None
            assert row["mode"] == "file"
            assert len(db.recent_transcripts()) == 4

    async def test_a_finite_source_is_lossless(self, tmp_path: Path) -> None:
        # More chunks than the queue holds, with a slow provider: nothing may be lost.
        (tmp_path / "f").write_text("x")
        source = FakeSource(audible_chunks(30))
        plan = SessionPlan(url=str(tmp_path / "f"), out_dir=tmp_path / "out")
        settings = settings_for(tmp_path, stt_queue_size=2, stt_spill_chunks=0)
        runner = runner_for(
            tmp_path,
            plan,
            [source],
            settings=settings,
            transcriber=ScriptedTranscriber(delay=0.005),
        )
        assert await runner.run() == ExitCode.OK
        assert runner.reports[0].utterances == 30
        assert runner.pipeline is not None and runner.pipeline.stats.dropped == 0

    async def test_no_audio_is_exit_4(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_text("x")
        runner = runner_for(tmp_path, SessionPlan(url=str(tmp_path / "f")), [FakeSource([])])
        assert await runner.run() == ExitCode.NO_DATA

    async def test_a_capture_error_is_stream_lost(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_text("x")
        source = FakeSource(audible_chunks(2), error=StreamError("could not read the input"))
        runner = runner_for(tmp_path, SessionPlan(url=str(tmp_path / "f")), [source])
        assert await runner.run() == ExitCode.STREAM_LOST
        # Whatever arrived before the failure is still finished and stored.
        assert runner.reports[0].stream_lost
        assert runner.reports[0].utterances == 2
        assert (tmp_path / "out" / "transcript.jsonl").read_text().count("\n") == 2

    async def test_speech_off_records_without_transcribing(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_text("x")
        stt = ScriptedTranscriber()
        plan = SessionPlan(url=str(tmp_path / "f"), transcribe=False, console=False)
        runner = runner_for(tmp_path, plan, [FakeSource(audible_chunks(3))], transcriber=stt)
        assert await runner.run() == ExitCode.OK
        assert stt.calls == []
        assert runner.reports[0].chunks == 3


class TestOpeningFailures:
    async def test_not_live_without_resume_is_exit_3(self, tmp_path: Path) -> None:
        source = FakeSource(connect_error=StreamNotLiveError("the broadcast is over"))
        plan = SessionPlan(url=URL, resume=False)
        runner = runner_for(tmp_path, plan, [source])
        assert await runner.run() == ExitCode.NOT_LIVE

    async def test_an_unresolvable_source_is_exit_3(self, tmp_path: Path) -> None:
        source = FakeSource(connect_error=StreamResolutionError("unsupported url"))
        runner = runner_for(tmp_path, SessionPlan(url=URL, resume=False), [source])
        assert await runner.run() == ExitCode.NOT_LIVE

    async def test_a_missing_local_file_is_a_config_error(self, tmp_path: Path) -> None:
        plan = SessionPlan(url=str(tmp_path / "missing.mp4"))
        runner = runner_for(tmp_path, plan, [FakeSource([])])
        assert await runner.run() == ExitCode.CONFIG

    async def test_a_missing_recording_is_a_config_error(self, tmp_path: Path) -> None:
        plan = SessionPlan(url=str(tmp_path / "nope"), mode="replay")
        runner = runner_for(tmp_path, plan, [FakeSource([])])
        assert await runner.run() == ExitCode.CONFIG

    async def test_a_missing_api_key_is_caught_before_capture(self, tmp_path: Path) -> None:
        settings = settings_for(tmp_path, stt_provider="openai")
        runner = SessionRunner(
            settings,
            SessionPlan(url=URL, resume=False),
            source_factory=lambda url: pytest.fail("no source may be opened"),
            install_signals=False,
        )
        assert await runner.run() == ExitCode.CONFIG

    async def test_a_provider_that_cannot_build_is_a_config_error(self, tmp_path: Path) -> None:
        from livestream_transcriber.config import ConfigError

        def broken(_settings: Settings) -> MockTranscriber:
            raise ConfigError("model files are missing")

        (tmp_path / "f").write_text("x")
        runner = SessionRunner(
            settings_for(tmp_path),
            SessionPlan(url=str(tmp_path / "f")),
            source_factory=lambda url: FakeSource(audible_chunks(1)),
            transcriber_factory=broken,
            install_signals=False,
        )
        assert await runner.run() == ExitCode.CONFIG

    async def test_a_non_empty_recording_directory_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "rec"
        target.mkdir()
        (target / "keep.txt").write_text("precious")
        plan = SessionPlan(url=URL, mode="record", record_dir=target, transcribe=False)
        runner = runner_for(tmp_path, plan, [FakeSource(audible_chunks(1))])
        assert await runner.run() == ExitCode.CONFIG
        assert (target / "keep.txt").read_text() == "precious"


class TestStopping:
    async def test_stop_finishes_the_outputs_and_exits_130(self, tmp_path: Path) -> None:
        stop = asyncio.Event()
        source = FakeSource(
            audible_chunks(3),
            is_live=True,
            hang=True,
            on_chunk=lambda n: stop.set() if n == 3 else None,
        )
        runner = runner_for(tmp_path, SessionPlan(url=URL, resume=False), [source], stop=stop)
        assert await runner.run() == ExitCode.INTERRUPTED
        assert source.close_calls >= 1
        assert (tmp_path / "out" / "transcript.srt").exists()
        with Database(tmp_path / "out" / "lst.db") as db:
            assert db.list_sessions()[0]["ended_at"] is not None

    async def test_stop_while_a_provider_is_wedged_does_not_hang(self, tmp_path: Path) -> None:
        import threading

        gate = threading.Event()
        stop = asyncio.Event()
        source = FakeSource(
            audible_chunks(2),
            is_live=True,
            hang=True,
            on_chunk=lambda n: stop.set() if n == 2 else None,
        )
        runner = runner_for(
            tmp_path,
            SessionPlan(url=URL, resume=False),
            [source],
            stop=stop,
            transcriber=ScriptedTranscriber(gate=gate),
        )
        started = time.monotonic()
        try:
            assert await runner.run() == ExitCode.INTERRUPTED
            assert time.monotonic() - started < 10
        finally:
            gate.set()

    async def test_first_signal_stops_second_exits_now(self, tmp_path: Path) -> None:
        runner = runner_for(tmp_path, SessionPlan(url=URL), [])
        runner._request_stop("SIGINT")
        assert runner.stop.is_set()
        with pytest.raises(SystemExit) as excinfo:
            runner._request_stop("SIGINT")
        assert excinfo.value.code == 130

    async def test_signal_handlers_are_installed_and_removed(self, tmp_path: Path) -> None:
        import signal

        runner = runner_for(tmp_path, SessionPlan(url=URL), [])
        runner.install_signals = True
        loop = asyncio.get_running_loop()
        with runner._signals():
            assert loop.remove_signal_handler(signal.SIGTERM) is True  # it was installed
            loop.add_signal_handler(signal.SIGTERM, lambda: None)
        assert loop.remove_signal_handler(signal.SIGTERM) is False  # and removed on exit


def fake_probe(states: list[str]):
    """A probe that answers LIVE or OFFLINE from ``states``, one per round (last repeats)."""
    calls: list[str] = []

    async def probe(candidate: SourceCandidate, fmt: str, **_kw: object) -> ProbeResult:
        state = states[min(len(calls), len(states) - 1)]
        calls.append(state)
        live = state == STATUS_LIVE
        return ProbeResult(
            name=candidate.name,
            probe_start=0.0,
            probe_end=0.0,
            status=state,
            is_live=live,
            stream_id="s" if live else None,
            url=candidate.url if live else None,
        )

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


class TestResuming:
    async def test_a_ended_live_stream_is_resumed_in_a_new_session(self, tmp_path: Path) -> None:
        stop = asyncio.Event()
        first = FakeSource(audible_chunks(2), is_live=True)
        second = FakeSource(
            audible_chunks(2),
            is_live=True,
            hang=True,
            on_chunk=lambda n: stop.set() if n == 2 else None,
        )
        probe = fake_probe([STATUS_LIVE])
        selector = LiveSourceSelector([SourceCandidate("primary", URL)], probe=probe)
        runner = runner_for(
            tmp_path, SessionPlan(url=URL), [first, second], stop=stop, selector=selector
        )
        assert await runner.run() == ExitCode.INTERRUPTED

        assert [r.chunks for r in runner.reports] == [2, 2]
        # Each session has its own transcript files, so the second does not overwrite the first.
        assert {"transcript.srt", "transcript-2.srt", "transcript-2.jsonl"} <= out_files(tmp_path)
        assert probe.calls  # type: ignore[attr-defined]
        with Database(tmp_path / "out" / "lst.db") as db:
            assert len(db.list_sessions()) == 2

    async def test_a_stream_that_is_not_live_yet_is_waited_for(self, tmp_path: Path) -> None:
        stop = asyncio.Event()
        never_opened = FakeSource(connect_error=StreamNotLiveError("not live"))
        live = FakeSource(
            audible_chunks(2),
            is_live=True,
            hang=True,
            on_chunk=lambda n: stop.set() if n == 2 else None,
        )
        probe = fake_probe([STATUS_LIVE])
        selector = LiveSourceSelector([SourceCandidate("primary", URL)], probe=probe)
        runner = runner_for(
            tmp_path, SessionPlan(url=URL), [never_opened, live], stop=stop, selector=selector
        )
        assert await runner.run() == ExitCode.INTERRUPTED
        assert runner.reports[-1].chunks == 2
        assert runner.made == [URL, URL]  # type: ignore[attr-defined]

    async def test_no_resume_flag_exits_when_the_stream_ends(self, tmp_path: Path) -> None:
        source = FakeSource(audible_chunks(2), is_live=True)
        runner = runner_for(tmp_path, SessionPlan(url=URL, resume=False), [source])
        assert await runner.run() == ExitCode.OK

    async def test_a_duration_limited_run_never_resumes(self, tmp_path: Path) -> None:
        source = FakeSource(audible_chunks(2), is_live=True)
        runner = runner_for(tmp_path, SessionPlan(url=URL, duration=5.0), [source])
        assert await runner.run() == ExitCode.OK
        assert len(runner.reports) == 1

    async def test_candidates_are_probed_and_the_live_one_is_captured(self, tmp_path: Path) -> None:
        backup = "https://example.com/backup"
        stop = asyncio.Event()
        source = FakeSource(
            audible_chunks(2),
            is_live=True,
            hang=True,
            on_chunk=lambda n: stop.set() if n == 2 else None,
        )

        async def probe(candidate: SourceCandidate, fmt: str, **_kw: object) -> ProbeResult:
            live = candidate.url == backup
            return ProbeResult(
                candidate.name, 0.0, 0.0, STATUS_LIVE if live else STATUS_OFFLINE, live,
                "id" if live else None, candidate.url if live else None,
            )  # fmt: skip

        selector = LiveSourceSelector(
            [SourceCandidate("primary", URL), SourceCandidate("fallback-1", backup)], probe=probe
        )
        plan = SessionPlan(url=URL, fallbacks=(backup,))
        runner = runner_for(tmp_path, plan, [source], stop=stop, selector=selector)
        assert await runner.run() == ExitCode.INTERRUPTED
        assert runner.made == [backup]  # type: ignore[attr-defined]

    async def test_no_live_candidate_without_resume_is_exit_3(self, tmp_path: Path) -> None:
        probe = fake_probe([STATUS_OFFLINE])
        selector = LiveSourceSelector([SourceCandidate("primary", URL)], probe=probe)
        plan = SessionPlan(url=URL, fallbacks=("https://example.com/b",), resume=False)
        runner = runner_for(tmp_path, plan, [], selector=selector)
        assert await runner.run() == ExitCode.NOT_LIVE

    async def test_stop_while_waiting_for_a_stream_exits_130(self, tmp_path: Path) -> None:
        stop = asyncio.Event()
        never_opened = FakeSource(connect_error=StreamNotLiveError("not live"))
        probe = fake_probe([STATUS_OFFLINE])
        selector = LiveSourceSelector([SourceCandidate("primary", URL)], probe=probe)
        runner = runner_for(
            tmp_path, SessionPlan(url=URL), [never_opened], stop=stop, selector=selector
        )
        task = asyncio.create_task(runner.run())
        await wait_until(lambda: probe.calls, what="the first probe")  # type: ignore[attr-defined]
        stop.set()
        assert await asyncio.wait_for(task, 10) == ExitCode.INTERRUPTED


class TestTranscriptCache:
    def test_the_cache_directory_setting_reaches_the_transcriber(self, tmp_path: Path) -> None:
        settings = settings_for(
            tmp_path,
            stt_provider="openai",
            openai_api_key="your-openai-key",
            stt_cache_dir=tmp_path / "cache",
        )
        runner = SessionRunner(settings, SessionPlan(url=URL), install_signals=False)
        transcriber = runner._new_transcriber(URL, None)
        assert isinstance(transcriber, CachedTranscriber)


class TestRerunIntoTheSameDirectory:
    def test_existing_transcripts_are_never_overwritten(self, tmp_path: Path) -> None:
        assert free_transcript_stem(tmp_path) == "transcript"
        (tmp_path / "transcript.srt").write_text("kept")
        assert free_transcript_stem(tmp_path) == "transcript-2"
        (tmp_path / "transcript-2.jsonl").write_text("kept")
        assert free_transcript_stem(tmp_path) == "transcript-3"
        assert free_transcript_stem(tmp_path, 5) == "transcript-5"

    async def test_a_second_run_keeps_the_first_runs_transcript(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_text("x")
        plan = SessionPlan(url=str(tmp_path / "f"), out_dir=tmp_path / "out")
        for _ in range(2):
            runner = runner_for(tmp_path, plan, [FakeSource(audible_chunks(3))])
            assert await runner.run() == ExitCode.OK
        assert {"transcript.jsonl", "transcript-2.jsonl"} <= out_files(tmp_path)


class TestRecordingAlongsideALiveRun:
    def test_run_with_record_keeps_fallbacks_and_auto_resume(self, tmp_path: Path) -> None:
        plan = SessionPlan(
            url=URL,
            fallbacks=("https://example.com/b",),
            record_dir=tmp_path / "rec",
        )
        runner = runner_for(tmp_path, plan, [])
        assert runner._resume_enabled() is True
        assert runner._build_selector(needed=True) is not None


class TestRecordAndReplay:
    async def test_a_recording_replays_to_the_same_transcript(self, tmp_path: Path) -> None:
        recording = tmp_path / "recordings" / "demo"
        plan = SessionPlan(
            url=URL,
            mode="record",
            record_dir=recording,
            out_dir=tmp_path / "rec-out",
            transcribe=False,
            console=False,
        )
        runner = runner_for(tmp_path, plan, [FakeSource(audible_chunks(4))])
        assert await runner.run() == ExitCode.OK
        assert (recording / "manifest.json").is_file()
        assert (recording / "audio.jsonl").read_text().count("\n") == 4

        replay_plan = SessionPlan(
            url=str(recording), mode="replay", out_dir=tmp_path / "replay-out", console=False
        )
        replay_runner = SessionRunner(
            settings_for(tmp_path, out_dir=tmp_path / "replay-out"),
            replay_plan,
            transcriber_factory=lambda _s: ScriptedTranscriber(),  # type: ignore[arg-type,return-value]
            install_signals=False,
        )
        assert await replay_runner.run() == ExitCode.OK
        rows = [
            json.loads(line)
            for line in (tmp_path / "replay-out" / "transcript.jsonl").read_text().splitlines()
        ]
        assert [r["text"] for r in rows] == [f"chunk {i * 2.5:g}" for i in range(4)]

    async def test_replay_is_deterministic(self, tmp_path: Path) -> None:
        recording = tmp_path / "rec"
        record = runner_for(
            tmp_path,
            SessionPlan(url=URL, mode="record", record_dir=recording, transcribe=False),
            [FakeSource(audible_chunks(5))],
        )
        assert await record.run() == ExitCode.OK
        assert not list((tmp_path / "out").glob("transcript*")), "record-only writes no transcripts"

        outputs = []
        for n in range(2):
            out = tmp_path / f"out{n}"
            delay = 0.003 * (1 + n)
            runner = SessionRunner(
                settings_for(tmp_path, out_dir=out),
                SessionPlan(url=str(recording), mode="replay", out_dir=out, console=False),
                transcriber_factory=lambda _s, d=delay: ScriptedTranscriber(delay=d),  # type: ignore[arg-type,return-value]
                install_signals=False,
            )
            assert await runner.run() == ExitCode.OK
            outputs.append((out / "transcript.srt").read_text())
        assert outputs[0] == outputs[1]


class TestRulesAndAlerts:
    async def test_a_rule_hit_reaches_the_webhook(self, tmp_path: Path) -> None:
        rules = RuleSet.from_yaml(
            """
version: 1
rules:
  - id: giveaway
    type: keyword
    keywords: [giveaway]
    severity: warning
    notify: [webhook]
"""
        )
        (tmp_path / "f").write_text("x")
        with serve() as server:
            settings = settings_for(tmp_path, notify_webhook_url=f"{server.url}/hook")
            runner = runner_for(
                tmp_path,
                SessionPlan(url=str(tmp_path / "f")),
                [FakeSource(audible_chunks(2))],
                settings=settings,
                transcriber=ScriptedTranscriber({0.0: "we run a giveaway today"}),
                ruleset=rules,
            )
            assert await runner.run() == ExitCode.OK
            bodies = server.json_bodies()
        assert len(bodies) == 1
        assert "giveaway" in json.dumps(bodies[0])
        assert runner.reports[0].hits == 1
        with Database(tmp_path / "out" / "lst.db") as db:
            (event,) = db.list_events()
            assert event["rule_id"] == "giveaway"

    async def test_stream_alerts_go_to_the_webhook(self, tmp_path: Path) -> None:
        source = FakeSource(audible_chunks(1), error=StreamError("connection lost"))
        with serve() as server:
            settings = settings_for(tmp_path, notify_webhook_url=f"{server.url}/hook")
            (tmp_path / "f").write_text("x")
            runner = runner_for(
                tmp_path, SessionPlan(url=str(tmp_path / "f")), [source], settings=settings
            )
            assert await runner.run() == ExitCode.STREAM_LOST
            assert any("Stream lost" in json.dumps(b) for b in server.json_bodies())


def test_is_remote_url() -> None:
    assert is_remote_url("https://example.com/x.m3u8")
    assert is_remote_url("RTMP://host/app")
    assert not is_remote_url("/tmp/file.mp4")
    assert not is_remote_url("relative/file.wav")


def test_exit_codes_are_the_documented_ones() -> None:
    assert {c.name: int(c) for c in ExitCode} == {
        "OK": 0,
        "CONFIG": 2,
        "NOT_LIVE": 3,
        "NO_DATA": 4,
        "STREAM_LOST": 5,
        "INTERRUPTED": 130,
    }
