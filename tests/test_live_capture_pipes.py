"""Capture-pipe isolation: stderr drain, queues, one ffmpeg, watchdog and supervisor."""

from __future__ import annotations

import asyncio
import time
from itertools import pairwise
from pathlib import Path

import pytest

from livestream_transcriber.models import AudioChunk, SegmentInfo, StreamInfo
from livestream_transcriber.stream import source as source_mod
from livestream_transcriber.stream.base import StreamError
from livestream_transcriber.stream.ffmpeg import FFmpegAudioPipe, FFmpegSpec, build_ffmpeg_command
from livestream_transcriber.stream.queues import DropOldestQueue
from livestream_transcriber.stream.source import (
    CaptureOptions,
    LiveStreamSource,
    await_capture_tasks,
    should_attempt_reselect,
)
from tests.support.ffmpeg_stubs import stub_fake_ffmpeg, stub_stalled
from tests.support.waits import wait_until

LIVE_URL = "https://example.test/live"


def _live_info(url: str = LIVE_URL) -> StreamInfo:
    return StreamInfo(url=url, is_live=True, media_url=f"{url}/audio.mp3")


def _fast(**kw: object) -> CaptureOptions:
    base: dict[str, object] = {
        "reconnect_initial_delay": 0.01,
        "reconnect_max_delay": 0.01,
        "max_reconnect_attempts": 3,
    }
    base.update(kw)
    return CaptureOptions(**base)


@pytest.fixture
def fake_spec(tmp_path: Path) -> FFmpegSpec:
    return FFmpegSpec(
        url="https://example.test/a.mp3",
        sample_rate=16000,
        binary=stub_fake_ffmpeg(tmp_path),
        hls_live_start_index=None,
        duration=2.0,
    )


# --------------------------------------------------------------------- pump


async def _pump(source: LiveStreamSource, spec: FFmpegSpec, segment: SegmentInfo) -> None:
    pipe = FFmpegAudioPipe(spec)
    await pipe.start()
    try:
        await source._pump_audio(pipe, segment, is_live=True)
    finally:
        await pipe.stop()


async def test_pump_audio_media_ts_increments_within_segment(fake_spec: FFmpegSpec) -> None:
    """Live chunks must not all stay at media_ts=0."""
    options = CaptureOptions(live_chunk_seconds=0.5)
    source = LiveStreamSource(LIVE_URL, options)
    chunks: list[AudioChunk] = []

    async def collect() -> None:
        async for chunk in source.get_audio():
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    try:
        await _pump(source, fake_spec, SegmentInfo(index=0, started_at=time.time(), offset=0.0))
    finally:
        source._audio_q.close()
        await task

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert chunk.media_ts == pytest.approx(i * 0.5, rel=1e-6)
    for prev, nxt in pairwise(chunks):
        assert nxt.media_ts == pytest.approx(prev.media_ts_end, abs=1e-6)
        assert nxt.ts >= prev.ts


async def test_pump_flushes_a_partial_tail_chunk(fake_spec: FFmpegSpec) -> None:
    """The stub writes 96000 bytes (3 s): with 2 s chunks the last 1 s is a tail."""
    source = LiveStreamSource(LIVE_URL, CaptureOptions(live_chunk_seconds=2.0))
    await _pump(source, fake_spec, SegmentInfo(index=0, started_at=time.time(), offset=0.0))
    source._audio_q.close()
    chunks = [c async for c in source.get_audio()]
    assert [round(c.duration, 3) for c in chunks] == [2.0, 1.0]
    assert chunks[1].media_ts == pytest.approx(2.0)
    assert source.stats.audio_seconds == pytest.approx(3.0)


async def test_media_ts_resets_per_segment_but_session_samples_continue(
    fake_spec: FFmpegSpec,
) -> None:
    source = LiveStreamSource(LIVE_URL, CaptureOptions(live_chunk_seconds=0.5))
    seg0 = SegmentInfo(index=0, started_at=time.time(), offset=0.0, audio_sample_base=0)
    await _pump(source, fake_spec, seg0)
    seg1 = SegmentInfo(
        index=1,
        started_at=time.time(),
        offset=5.0,
        audio_sample_base=source._session_audio_samples,
    )
    await _pump(source, fake_spec, seg1)
    source._audio_q.close()
    chunks = [c async for c in source.get_audio()]

    seg0_chunks = [c for c in chunks if c.segment == 0]
    seg1_chunks = [c for c in chunks if c.segment == 1]
    assert len(seg0_chunks) >= 2 and seg1_chunks
    # New ffmpeg segment: media_ts restarts, but the sample counter and chunk index do not.
    assert seg1_chunks[0].media_ts == pytest.approx(0.0)
    assert seg1_chunks[0].ts == pytest.approx(5.0)
    assert seg1_chunks[1].media_ts == pytest.approx(0.5)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert source._session_audio_samples > seg1.audio_sample_base


# --------------------------------------------------------------------- pipe


async def test_stderr_flood_does_not_deadlock(fake_spec: FFmpegSpec) -> None:
    pipe = FFmpegAudioPipe(fake_spec)
    await pipe.start()
    reads = 0

    async def audio() -> None:
        nonlocal reads
        async for _ in pipe.read_audio():
            reads += 1

    try:
        await asyncio.wait_for(audio(), timeout=8)
    finally:
        await pipe.stop()
    assert reads >= 1
    assert int(pipe.diagnostics.get("stderr_lines") or 0) >= 100
    assert pipe.diagnostics.get("first_audio_byte_s") is not None
    assert pipe.diagnostics["audio_bytes"] == 12 * 8000


async def test_a_slow_reader_does_not_lose_bytes(fake_spec: FFmpegSpec) -> None:
    pipe = FFmpegAudioPipe(fake_spec)
    await pipe.start()
    total = 0
    try:
        async for data in pipe.read_audio():
            total += len(data)
            await asyncio.sleep(0.02)
    finally:
        await pipe.stop()
    assert total == 12 * 8000


async def test_graceful_shutdown_closes_the_pipe(fake_spec: FFmpegSpec) -> None:
    pipe = FFmpegAudioPipe(fake_spec)
    await pipe.start()
    assert pipe.returncode is None
    await pipe.stop()
    assert pipe.returncode is not None
    await pipe.stop()  # idempotent


def test_queues_remain_bounded() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(4)
    for i in range(50):
        q.put(i)
    assert q.qsize() == 4
    assert q.dropped == 46


def test_build_command_never_spawns_a_second_binary() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.m3u8"))
    assert [tok for tok in cmd if tok == "ffmpeg"] == ["ffmpeg"]


# --------------------------------------------------------------------- real ffmpeg (slow)


@pytest.mark.slow
async def test_one_ffmpeg_child_for_a_local_clip(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip))
    await source.connect()
    try:
        await source.wait_for_first_data(timeout=30)
        pipe = source._pipe
        if pipe is not None:  # the 6 s clip may already be fully read
            cmd = pipe.command
            assert cmd.count("-i") == 1
            assert sum(1 for tok in cmd if tok == "ffmpeg" or tok.endswith("/ffmpeg")) <= 1
    finally:
        await source.close()


@pytest.mark.slow
async def test_local_file_gets_no_http_options_and_is_lossless(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip))
    info = await source.connect()
    try:
        assert info.is_live is False and source.lossless
        await source.wait_for_first_data(timeout=30)
        pipe = source._pipe
        if pipe is not None:
            assert "-protocol_whitelist" not in pipe.command
            assert "-reconnect" not in pipe.command
    finally:
        await source.close()


@pytest.mark.slow
async def test_synthetic_timestamps_stay_monotonic(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), CaptureOptions(file_chunk_seconds=1.0))
    await source.connect()
    chunks: list[AudioChunk] = []

    async def audio() -> None:
        async for chunk in source.get_audio():
            chunks.append(chunk)

    try:
        await asyncio.wait_for(audio(), timeout=60)
    finally:
        await source.close()
    # One ffmpeg run from start to end. A restart would capture the clip again from
    # media_ts 0; say so here rather than as a confusing ordering failure below.
    assert [s.reason for s in source.segments] == ["stream_ended"]
    assert chunks
    for prev, nxt in pairwise(chunks):
        assert nxt.media_ts >= prev.media_ts
        assert nxt.ts >= prev.ts
    assert chunks[-1].media_ts_end == pytest.approx(6.0, abs=0.2)


@pytest.mark.slow
async def test_audio_gap_is_filled_so_timestamps_stay_true(ffmpeg_bin: str, tmp_path: Path) -> None:
    """Nothing fills a gap in an audio input by itself: the PCM would simply lack those
    seconds, the sample counter would fall behind, and every later chunk would be
    stamped early by the gap. With ``aresample=async=1`` the gap becomes silence.

    The input is an HLS playlist whose listing skips two 1 s segments at t=4..6 (what
    the HLS demuxer does when a live window slid past unread segments). The tone
    switches from 440 Hz to 1 kHz at t=6, right after the gap."""
    import subprocess

    import numpy as np

    out = tmp_path / "hls"
    out.mkdir()
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*if(lt(t\\,6)\\,440\\,1000)*t)*0.5:s=48000:d=10",
            "-c:a", "aac", "-f", "hls", "-hls_time", "1", "-hls_list_size", "0",
            "-hls_segment_filename", str(out / "a%02d.ts"), str(out / "full.m3u8"),
        ],
        check=True,
    )  # fmt: skip
    lines = (out / "full.m3u8").read_text().splitlines()
    kept: list[str] = []
    for i, line in enumerate(lines):
        skipped = ("a04.ts", "a05.ts")
        if line in skipped or (line.startswith("#EXTINF") and lines[i + 1] in skipped):
            continue
        kept.append(line)
    (out / "gap.m3u8").write_text("\n".join(kept) + "\n")

    spec = FFmpegSpec(url=str(out / "gap.m3u8"), sample_rate=16000, binary=ffmpeg_bin)
    pipe = FFmpegAudioPipe(spec)
    await pipe.start()
    pcm = bytearray()
    try:
        async for data in pipe.read_audio():
            pcm.extend(data)
        # The output is at EOF: ffmpeg is exiting by itself. Let it, so a clean end
        # keeps its exit code instead of racing our SIGTERM.
        await pipe.stop(grace=5.0)
    finally:
        await pipe.stop()
    assert pipe.returncode == 0, pipe.stderr_tail

    samples = np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float32)
    window = spec.sample_rate // 20
    high_tone_at = next(
        k * window / spec.sample_rate
        for k in range(len(samples) // window)
        if np.count_nonzero(np.diff(np.signbit(samples[k * window : (k + 1) * window]))) > 70
    )  # more than 700 Hz: the 1 kHz tone after the gap
    assert abs(high_tone_at - 6.0) <= 0.5, high_tone_at
    assert len(samples) / spec.sample_rate == pytest.approx(10.0, abs=0.5)


# --------------------------------------------------------------------- supervisor


async def test_supervisor_crash_closes_the_queue_with_a_stream_error() -> None:
    source = LiveStreamSource(LIVE_URL, _fast(max_reconnect_attempts=1))
    notified: list[Exception] = []
    source.on_fatal = notified.append

    async def boom(_info: StreamInfo, _segment: SegmentInfo, _remaining: float | None) -> str:
        raise RuntimeError("ffmpeg exploded")

    source._run_segment = boom
    await source._supervise(_live_info())

    assert len(notified) == 1
    assert isinstance(notified[0], StreamError)
    assert "crashed" in str(notified[0])
    with pytest.raises(StreamError, match="crashed"):
        await source._audio_q.get()


async def test_give_up_closes_the_queue_with_a_stream_error_and_notifies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve(*_a: object, **_k: object) -> StreamInfo:
        return _live_info()

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    source = LiveStreamSource(LIVE_URL, _fast(max_reconnect_attempts=1))
    notified: list[Exception] = []
    source.on_fatal = notified.append

    async def dead(_info: StreamInfo, _segment: SegmentInfo, _remaining: float | None) -> str:
        return "pipe_died"

    source._run_segment = dead
    await source._supervise(_live_info())

    assert len(notified) == 1
    assert isinstance(notified[0], StreamError)
    assert "gave up" in str(notified[0])
    source._close_queue(StreamError("second close"))
    assert len(notified) == 1
    with pytest.raises(StreamError, match="gave up"):
        await source._audio_q.get()


async def test_a_failing_fatal_callback_does_not_break_the_close() -> None:
    source = LiveStreamSource(LIVE_URL)

    def broken(_error: Exception) -> None:
        raise RuntimeError("callback bug")

    source.on_fatal = broken
    source._close_queue(StreamError("lost"))
    with pytest.raises(StreamError, match="lost"):
        await source._audio_q.get()


def test_should_attempt_reselect_on_live_loss_not_transient_drop() -> None:
    assert should_attempt_reselect(reason="stream_ended", is_live=True) is True
    assert should_attempt_reselect(reason="stream_ended", is_live=False) is False
    assert should_attempt_reselect(reason="ffmpeg_exit_1", is_live=True, empty_streak=0) is False
    assert should_attempt_reselect(reason="pipe_died", is_live=True, empty_streak=2) is True
    assert should_attempt_reselect(reason="stale_playlist", is_live=True) is True


async def test_audio_stall_watchdog_triggers_a_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(*_a: object, **_k: object) -> StreamInfo:
        return _live_info()

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    source = LiveStreamSource(LIVE_URL, _fast(audio_stall_seconds=0.25))
    calls: list[str] = []

    async def fake_segment(
        _info: StreamInfo, _segment: SegmentInfo, _remaining: float | None
    ) -> str:
        calls.append("segment")
        if len(calls) == 1:
            return "audio_stalled"
        source._closing.set()
        return "duration_reached"

    source._run_segment = fake_segment
    await source._supervise(_live_info())

    assert calls == ["segment", "segment"]
    assert source.stats.reconnects >= 1


# --------------------------------------------------------------------- watchdog


async def _stall_verdict(source: LiveStreamSource, segment: SegmentInfo) -> tuple[str | None, bool]:
    """Run the segment watchdog the way ``_run_segment`` does: under
    ``await_capture_tasks``, against a pump that runs until the pipe is stopped.
    Returns ``(verdict, pipe_was_stopped)``."""
    stopped = asyncio.Event()

    async def pump_until_stopped() -> None:
        await stopped.wait()

    async def stop_pipe() -> None:
        stopped.set()

    pump = asyncio.ensure_future(pump_until_stopped())
    watchdog = asyncio.create_task(source._watch_segment_audio(segment, is_live=True))
    try:
        reason = await asyncio.wait_for(
            await_capture_tasks(pump, watchdog, stop_pipe=stop_pipe, unblock_timeout=1.0),
            timeout=2.0,
        )
    finally:
        watchdog.cancel()
        pump.cancel()
        await asyncio.gather(watchdog, pump, return_exceptions=True)
    return reason, stopped.is_set()


async def test_audio_stall_watchdog_stops_the_pipe() -> None:
    source = LiveStreamSource(LIVE_URL, CaptureOptions(audio_stall_seconds=0.25))
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    reason, stopped = await _stall_verdict(source, segment)
    assert reason == "audio_stalled"
    assert stopped


async def test_a_stalled_live_segment_is_cut_at_the_requested_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stall threshold is far longer than the duration: the wall clock must win."""
    monkeypatch.setattr("livestream_transcriber.stream.source._DURATION_GRACE_S", 0.1)
    source = LiveStreamSource(LIVE_URL, CaptureOptions(audio_stall_seconds=30))
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    started = time.monotonic()
    verdict = await asyncio.wait_for(
        source._watch_segment_audio(segment, is_live=True, remaining=0.3), timeout=3
    )
    assert verdict == "duration_reached"
    assert time.monotonic() - started < 2


async def test_the_watchdog_leaves_a_segment_with_flowing_audio_alone() -> None:
    source = LiveStreamSource(LIVE_URL, CaptureOptions(audio_stall_seconds=0.6))
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)

    async def feed() -> None:
        while True:
            segment.audio_chunks += 1
            await asyncio.sleep(0.05)

    feeder = asyncio.create_task(feed())
    watchdog = asyncio.create_task(source._watch_segment_audio(segment, is_live=True))
    try:
        done, _ = await asyncio.wait({watchdog}, timeout=1.5)
        assert not done, "audio is flowing, so there is nothing to report"
    finally:
        feeder.cancel()
        watchdog.cancel()
        await asyncio.gather(feeder, watchdog, return_exceptions=True)


@pytest.mark.parametrize(
    ("options", "is_live", "lossless"),
    [
        (CaptureOptions(audio_stall_seconds=0), True, None),
        (CaptureOptions(audio_stall_seconds=0.2), False, None),
        (CaptureOptions(audio_stall_seconds=0.2), True, True),
    ],
)
async def test_the_watchdog_is_off_where_it_would_be_wrong(
    options: CaptureOptions, is_live: bool, lossless: bool | None
) -> None:
    """Off when disabled, for finite sources, and for lossless capture, where a slow
    consumer legitimately stops chunks from being emitted."""
    source = LiveStreamSource(LIVE_URL, options, lossless=lossless)
    source._lossless = bool(lossless)
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    assert await source._watch_segment_audio(segment, is_live=is_live) is None


async def test_watchdog_verdict_stops_the_pipe_and_wins_over_the_pump() -> None:
    """Stopping ffmpeg ends the pump (EOF). When the watchdog stopped it itself, the
    pump could finish first; ``await_capture_tasks`` then saw no verdict and the stall
    was filed by ffmpeg's exit code instead."""
    pump_done = asyncio.Event()

    async def pump_until_eof() -> None:
        await pump_done.wait()

    pump = asyncio.create_task(pump_until_eof())

    async def watchdog() -> str:
        return "audio_stalled"

    wd = asyncio.create_task(watchdog())
    stops: list[str] = []

    async def stop() -> None:
        stops.append("stop")
        pump_done.set()  # what closing ffmpeg's pipe does to the reader
        await asyncio.sleep(0)

    reason = await await_capture_tasks(pump, wd, stop_pipe=stop, unblock_timeout=1.0)
    assert reason == "audio_stalled"
    assert stops == ["stop"]


async def test_finished_pump_does_not_wait_on_the_watchdog() -> None:
    async def pump_done() -> None:
        return None

    pump = asyncio.create_task(pump_done())

    async def watchdog() -> str:
        await asyncio.Event().wait()
        return "audio_stalled"

    wd = asyncio.create_task(watchdog())
    stopped = asyncio.Event()

    async def stop() -> None:
        stopped.set()

    try:
        reason = await asyncio.wait_for(
            await_capture_tasks(pump, wd, stop_pipe=stop, unblock_timeout=0.2), timeout=1.0
        )
    finally:
        wd.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wd

    assert reason is None
    assert not stopped.is_set()


async def test_a_pump_that_stays_stuck_is_reported_and_cancelled() -> None:
    async def hung() -> None:
        await asyncio.Event().wait()

    pump = asyncio.create_task(hung())

    async def watchdog() -> str:
        return "audio_stalled"

    wd = asyncio.create_task(watchdog())
    stuck: list[str] = []

    async def stop() -> None:
        return None

    reason = await await_capture_tasks(
        pump, wd, stop_pipe=stop, unblock_timeout=0.05, on_stuck=stuck.append
    )
    assert reason == "audio_stalled"
    assert stuck == ["audio_stalled"]
    assert pump.cancelled(), "the supervisor must be able to go on without the hung pump"


async def test_a_pump_that_stays_stuck_needs_no_callback() -> None:
    async def hung() -> None:
        await asyncio.Event().wait()

    pump = asyncio.create_task(hung())

    async def watchdog() -> str:
        return "audio_stalled"

    async def stop() -> None:
        return None

    reason = await await_capture_tasks(
        pump, asyncio.create_task(watchdog()), stop_pipe=stop, unblock_timeout=0.05
    )
    assert reason == "audio_stalled" and pump.cancelled()


async def test_a_pump_error_is_raised_to_the_caller() -> None:
    async def failing() -> None:
        raise RuntimeError("pump bug")

    pump = asyncio.create_task(failing())
    wd: asyncio.Task[str | None] = asyncio.create_task(asyncio.Event().wait())

    async def stop() -> None:
        return None

    try:
        with pytest.raises(RuntimeError, match="pump bug"):
            await await_capture_tasks(pump, wd, stop_pipe=stop)
    finally:
        wd.cancel()
        await asyncio.gather(wd, return_exceptions=True)


async def test_a_stall_is_filed_as_a_stall_with_a_real_pipe(tmp_path: Path) -> None:
    """The watchdog must not stop ffmpeg itself and only then return its verdict.

    ffmpeg closes its output as soon as SIGTERM arrives but exits a moment later, so
    the pump reaches EOF while the watchdog would still be inside stop(): the segment
    would look like ffmpeg ended by itself and be filed as ``ffmpeg_exit_None``."""
    options = CaptureOptions(
        ffmpeg_binary=stub_stalled(tmp_path, exit_delay=0.4),
        audio_stall_seconds=0.3,
        live_chunk_seconds=0.1,  # the stub writes 0.1 s blocks: one chunk each
    )
    source = LiveStreamSource(LIVE_URL, options)
    info = StreamInfo(url=LIVE_URL, is_live=True, media_url="https://example.test/a.mp3")
    watch = source._watch_segment_audio

    async def watch_once_the_audio_is_in(
        segment: SegmentInfo, *, is_live: bool, remaining: float | None = None
    ) -> str | None:
        # The stall clock starts with the segment, and on a loaded machine the stub's
        # interpreter can take longer than the threshold just to start. Start the real
        # watchdog once the stub's three blocks are in, so the stall it judges is the
        # stub's sleep and not its start-up.
        await wait_until(lambda: segment.audio_chunks >= 1, what="the stub's first chunk")
        return await watch(segment, is_live=is_live, remaining=remaining)

    source._watch_segment_audio = watch_once_the_audio_is_in
    for index in range(3):
        segment = SegmentInfo(index=index, started_at=time.time(), offset=0.0)
        reason = await asyncio.wait_for(source._run_segment(info, segment, None), 60)
        assert reason == "audio_stalled"
        assert segment.audio_chunks >= 1
    source._close_queue()


# --------------------------------------------------------------------- reselect


async def test_reselect_switches_url_after_live_stream_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, backup = "https://streams.example.test/a/live", "https://streams.example.test/b/live"
    resolved: list[str] = []

    async def fake_resolve(url: str, **_k: object) -> StreamInfo:
        resolved.append(url)
        return _live_info(url)

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    asked: list[tuple[str, str, int]] = []

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        asked.append((current, reason, attempt))
        return backup

    source = LiveStreamSource(primary, _fast(), reselect=reselect)
    infos: list[str] = []

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        infos.append(info.url)
        segment.audio_chunks = 4
        return "stream_ended" if len(infos) == 1 else "duration_reached"

    source._run_segment = run_segment
    await source._supervise(_live_info(primary))
    assert infos == [primary, backup]
    assert source.url == backup
    assert asked and asked[0] == (primary, "stream_ended", 1)
    assert backup in resolved


async def test_reselect_after_the_current_source_fails_to_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livestream_transcriber.stream.base import StreamResolutionError

    primary, backup = "https://streams.example.test/a/live", "https://streams.example.test/b/live"

    async def fake_resolve(url: str, **_k: object) -> StreamInfo:
        if url == primary:
            raise StreamResolutionError("The channel is not currently live")
        return _live_info(url)

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        return backup

    source = LiveStreamSource(primary, _fast(), reselect=reselect)
    infos: list[str] = []

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        infos.append(info.url)
        segment.audio_chunks = 2
        return "ffmpeg_exit_1" if len(infos) == 1 else "duration_reached"

    source._run_segment = run_segment
    await source._supervise(_live_info(primary))
    assert infos == [primary, backup]


async def test_a_transient_drop_with_data_does_not_reselect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []

    async def fake_resolve(url: str, **_k: object) -> StreamInfo:
        return _live_info()

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        calls.append((current, reason, attempt))
        return None

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    source = LiveStreamSource(LIVE_URL, _fast(), reselect=reselect)
    seen = {"n": 0}

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        seen["n"] += 1
        segment.audio_chunks = 3
        return "ffmpeg_exit_1" if seen["n"] == 1 else "duration_reached"

    source._run_segment = run_segment
    await source._supervise(_live_info())
    assert calls == []


async def test_a_switched_but_unresolved_source_does_not_replay_the_stale_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livestream_transcriber.stream.base import StreamResolutionError

    primary, backup = "https://streams.example.test/a/live", "https://streams.example.test/b/live"

    async def fake_resolve(url: str, **_k: object) -> StreamInfo:
        raise StreamResolutionError("offline")

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        return backup

    source = LiveStreamSource(primary, _fast(max_reconnect_attempts=2), reselect=reselect)
    infos: list[str] = []

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        infos.append(info.url)
        segment.audio_chunks = 1
        return "stream_ended"

    source._run_segment = run_segment
    await source._supervise(_live_info(primary))
    assert infos == [primary]
    assert source.url == backup


async def test_a_reselect_that_raises_is_treated_as_no_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve(url: str, **_k: object) -> StreamInfo:
        return _live_info()

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        raise RuntimeError("selector bug")

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    source = LiveStreamSource(LIVE_URL, _fast(), reselect=reselect)
    seen = {"n": 0}

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        seen["n"] += 1
        segment.audio_chunks = 1
        return "stream_ended" if seen["n"] == 1 else "duration_reached"

    source._run_segment = run_segment
    await source._supervise(_live_info())
    assert seen["n"] == 2 and source.url == LIVE_URL
