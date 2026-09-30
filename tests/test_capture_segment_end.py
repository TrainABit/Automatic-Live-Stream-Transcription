"""How a capture segment ends: exit code, classification, diagnostics.

The stub writes a few PCM blocks, then exits with a chosen code. Offline, no network,
no real decoder.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest

from livestream_transcriber.models import SegmentInfo, StreamInfo
from livestream_transcriber.stream import source as source_mod
from livestream_transcriber.stream.ffmpeg import FFmpegAudioPipe, FFmpegSpec
from livestream_transcriber.stream.source import CaptureOptions, LiveStreamSource
from tests.support.ffmpeg_stubs import stub_pcm_then_exit, stub_stubborn
from tests.support.waits import wait_until

URL = "https://example.test/watch/1"


def _source(binary: str, **kw: object) -> LiveStreamSource:
    # 0.1 s chunks: the stub's 1600-sample blocks are exactly one chunk each.
    options = CaptureOptions(
        ffmpeg_binary=binary,
        audio_stall_seconds=0,
        live_chunk_seconds=0.1,
        file_chunk_seconds=0.1,
        **kw,
    )
    return LiveStreamSource(URL, options)


def _info(*, is_live: bool = False) -> StreamInfo:
    # Not an m3u8 URL, so no HLS window is opened even when live; the watchdog is off,
    # so the segment ends exactly when the stub exits.
    return StreamInfo(url=URL, is_live=is_live, media_url="https://example.test/a.mp3")


def _diagnostics(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "ffmpeg capture diagnostics"]


@pytest.mark.parametrize("is_live", [False, True])
@pytest.mark.parametrize("attempt", range(5))
async def test_clean_ffmpeg_exit_is_classified_stream_ended(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, attempt: int, is_live: bool
) -> None:
    """The exit code must be read after stop() reaped ffmpeg, or every segment end would
    say ``ffmpeg_rc=None`` and be filed as ``ffmpeg_exit_None`` instead of
    ``stream_ended``. Repeated because the failure mode is a race."""
    source = _source(stub_pcm_then_exit(tmp_path, rc=0))
    source._lossless = not is_live
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    with caplog.at_level(logging.INFO, logger="livestream_transcriber.stream.source"):
        reason = await source._run_segment(_info(is_live=is_live), segment, None)
    source._close_queue()
    assert segment.audio_chunks == 4
    assert reason == "stream_ended"
    diag = _diagnostics(caplog)
    assert len(diag) == 1
    assert diag[0].ffmpeg_rc == 0


async def test_finished_clip_is_captured_once_when_ffmpeg_exits_after_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output reaches EOF a moment before ffmpeg is reaped; on a loaded machine the
    gap is wide. Reading the exit code in that gap would file a clip that played to the
    end as ``ffmpeg_exit_None``, and the supervisor would reconnect and capture the
    whole clip again (media_ts back at 0, every chunk twice). Here the stub lives 0.5 s
    past closing its output, which forces that ordering every time."""
    resolves: list[str] = []

    async def fake_resolve(url: str, **_k: object) -> StreamInfo:
        # A reconnect: record it and end the session so the test ends too.
        resolves.append(url)
        source._closing.set()
        return _info(is_live=True)

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    source = _source(
        stub_pcm_then_exit(tmp_path, rc=0, exit_delay=0.5),
        reconnect_initial_delay=0.01,
        reconnect_max_delay=0.01,
    )
    await asyncio.wait_for(source._supervise(_info(is_live=False)), 60)
    chunks = [chunk async for chunk in source.get_audio()]

    assert [s.reason for s in source.segments] == ["stream_ended"]
    assert resolves == []
    assert [round(c.media_ts, 3) for c in chunks] == [0.0, 0.1, 0.2, 0.3]


async def test_failing_ffmpeg_exit_code_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(stub_pcm_then_exit(tmp_path, rc=3))
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    with caplog.at_level(logging.INFO, logger="livestream_transcriber.stream.source"):
        reason = await source._run_segment(_info(is_live=True), segment, None)
    source._close_queue()
    assert reason == "ffmpeg_exit_3"
    assert _diagnostics(caplog)[0].ffmpeg_rc == 3


async def test_a_binary_that_cannot_start_ends_the_segment_cleanly() -> None:
    source = _source("definitely-not-a-binary-name")
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    assert await source._run_segment(_info(is_live=True), segment, None) == "ffmpeg_start_failed"
    assert source._pipe is None
    source._close_queue()


async def test_a_requested_duration_is_reported_as_reached(tmp_path: Path) -> None:
    source = _source(stub_pcm_then_exit(tmp_path, rc=0, chunks=10))  # 1 s of audio
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    reason = await source._run_segment(_info(is_live=True), segment, 1.0)
    source._close_queue()
    assert reason == "duration_reached"


async def test_the_ffmpeg_command_follows_the_resolved_stream(tmp_path: Path) -> None:
    """The user agent and headers the resolver reported reach ffmpeg's command."""
    source = _source(stub_pcm_then_exit(tmp_path, rc=0))
    info = StreamInfo(
        url=URL,
        is_live=True,
        media_url="https://example.test/a.mp3",
        headers={"User-Agent": "ResolverUA/9", "Referer": "https://example.test/"},
    )
    spec = source._ffmpeg_spec(info, source_mod.HlsInput(info.media_url or ""), None)
    assert spec.user_agent == "ResolverUA/9"
    assert spec.headers["Referer"] == "https://example.test/"

    plain = StreamInfo(url=URL, is_live=True, media_url="https://example.test/a.mp3")
    assert source._ffmpeg_spec(
        plain, source_mod.HlsInput("https://example.test/a.mp3"), None
    ).user_agent
    local = StreamInfo(url="/tmp/a.wav", is_live=False, media_url="/tmp/a.wav")
    assert source._ffmpeg_spec(local, source_mod.HlsInput("/tmp/a.wav"), None).user_agent is None


async def test_pipe_keeps_its_exit_code_after_stop(tmp_path: Path) -> None:
    spec = FFmpegSpec(
        url="https://example.test/a.mp3",
        binary=stub_pcm_then_exit(tmp_path, rc=0),
        hls_live_start_index=None,
    )
    pipe = FFmpegAudioPipe(spec)
    await pipe.start()
    async for _ in pipe.read_audio():
        pass
    await pipe.stop(grace=5.0)
    # stop() drops the process handle; the exit code must survive it.
    assert pipe.returncode == 0
    assert pipe.diagnostics["ffmpeg_rc"] == 0


async def test_stop_does_not_hang_when_nobody_reads_the_output(tmp_path: Path) -> None:
    """A cancelled pump leaves stdout unread: the pipe fills, ffmpeg blocks in write()
    and shrugs off SIGTERM, and asyncio's wait() waits for a pipe EOF that a paused
    transport never delivers. stop() must not hang there (kill and all), or close() and
    the capture supervisor would hang with it."""
    spec = FFmpegSpec(
        url="https://example.test/a.mp3",
        binary=stub_stubborn(tmp_path),
        hls_live_start_index=None,
    )
    pipe = FFmpegAudioPipe(spec)
    await pipe.start()
    assert pipe._proc is not None
    stdout = pipe._proc.stdout
    # Nothing reads stdout meanwhile. The scenario needs the stub's SIGTERM handler in
    # place (it prints its banner right after installing it) and stdout backed up until
    # asyncio paused the transport, which leaves the stub blocked in write().
    await wait_until(lambda: "stubborn stub up" in pipe.stderr_tail, what="the stub's banner")
    await wait_until(
        lambda: stdout._paused,
        what="asyncio to pause the full stdout pipe",
    )
    started = time.monotonic()
    await asyncio.wait_for(pipe.stop(timeout=1.0), timeout=15.0)
    assert time.monotonic() - started < 5.0
    assert pipe.returncode is not None
    assert pipe.diagnostics["ffmpeg_rc"] == pipe.returncode
    assert "stubborn stub up" in pipe.stderr_tail


async def test_a_clean_live_end_offers_reselect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real exit code a clean live end is ``stream_ended``, the one reason that
    asks the source selector for another candidate."""
    asked: list[str] = []

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        asked.append(reason)
        # What a session does when every candidate is offline: end.
        source._closing.set()
        return None

    async def fake_resolve(*_a: object, **_k: object) -> StreamInfo:
        return _info(is_live=True)

    monkeypatch.setattr(source_mod, "resolve_stream", fake_resolve)
    source = _source(
        stub_pcm_then_exit(tmp_path, rc=0),
        max_reconnect_attempts=1,
        reconnect_initial_delay=0.01,
        reconnect_max_delay=0.01,
    )
    source.reselect = reselect
    await source._supervise(_info(is_live=True))
    assert [s.reason for s in source.segments][:1] == ["stream_ended"]
    assert asked and asked[0] == "stream_ended"


async def test_the_stub_interpreter_is_this_python() -> None:
    """Guards the helper the tests above rely on."""
    assert Path(sys.executable).exists()
