"""End-to-end capture against a real ffmpeg run.

These use a locally generated clip, so they are deterministic and offline, but they
exercise the actual subprocess, the stdout pipe and the pump.
"""

from __future__ import annotations

import asyncio
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from livestream_transcriber.models import AudioChunk
from livestream_transcriber.stream.base import StreamResolutionError
from livestream_transcriber.stream.source import CaptureOptions, LiveStreamSource

pytestmark = pytest.mark.slow

OPTIONS = CaptureOptions(file_chunk_seconds=1.0)


async def _collect(source: LiveStreamSource, timeout: float = 90.0) -> list[AudioChunk]:
    chunks: list[AudioChunk] = []

    async def audio() -> None:
        async for chunk in source.get_audio():
            chunks.append(chunk)

    await asyncio.wait_for(audio(), timeout=timeout)
    return chunks


async def test_captures_the_whole_clip(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), OPTIONS)
    info = await source.connect()
    assert info.is_live is False
    try:
        chunks = await _collect(source)
    finally:
        await source.close()

    assert sum(c.duration for c in chunks) == pytest.approx(6.0, abs=0.1)
    assert len(chunks) == 6
    assert source.stats.audio_chunks_emitted == 6
    assert source.stats.audio_chunks_dropped == 0
    assert [s.reason for s in source.segments] == ["stream_ended"]


async def test_an_mp4_container_is_demuxed(synthetic_clip_mp4: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip_mp4), OPTIONS)
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()
    # AAC adds an encoder delay and padding; the duration is only approximately 6 s.
    assert sum(c.duration for c in chunks) == pytest.approx(6.0, abs=0.3)
    assert all(c.sample_rate == 16000 for c in chunks)


async def test_chunks_tile_the_timeline_with_exact_timestamps(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), OPTIONS)
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()

    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert chunk.media_ts == pytest.approx(i * 1.0), "the clock is derived from the counter"
    for prev, nxt in pairwise(chunks):
        assert nxt.media_ts == pytest.approx(prev.media_ts_end, abs=1e-6)


async def test_audio_carries_actual_signal(synthetic_clip: Path) -> None:
    """A silent capture looks healthy but is useless; assert on the level."""
    source = LiveStreamSource(str(synthetic_clip), OPTIONS)
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()

    # The clip is a 440 Hz tone at roughly -18 dBFS.
    assert chunks[0].peak_dbfs() > -30.0
    assert float(np.abs(chunks[0].as_float32()).max()) <= 1.0


async def test_resampling_to_another_rate(synthetic_clip: Path) -> None:
    options = CaptureOptions(sample_rate=8000, file_chunk_seconds=2.0)
    source = LiveStreamSource(str(synthetic_clip), options)
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()
    assert {c.sample_rate for c in chunks} == {8000}
    assert sum(c.n_samples for c in chunks) == pytest.approx(6 * 8000, abs=800)


async def test_duration_limits_the_capture(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), OPTIONS, duration=3.0)
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()
    assert sum(c.duration for c in chunks) == pytest.approx(3.0, abs=0.2)
    assert [s.reason for s in source.segments] == ["duration_reached"]


async def test_seek_starts_later_in_the_clip(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), OPTIONS, input_seek=3.0)
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()
    assert sum(c.duration for c in chunks) == pytest.approx(3.0, abs=0.2)


async def test_realtime_paces_a_file_like_a_live_feed(synthetic_clip: Path) -> None:
    """``-re`` reads at the native rate: a 6 s clip takes about 6 s, not milliseconds."""
    source = LiveStreamSource(str(synthetic_clip), OPTIONS, duration=2.0, realtime=True)
    started = asyncio.get_running_loop().time()
    await source.connect()
    try:
        chunks = await _collect(source)
    finally:
        await source.close()
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed > 1.2, f"expected about 2 s of pacing, got {elapsed:.2f}"
    assert sum(c.duration for c in chunks) == pytest.approx(2.0, abs=0.2)


async def test_a_slow_consumer_loses_nothing_from_a_file(synthetic_clip: Path) -> None:
    """A file decodes far faster than real time. With a drop-oldest queue a slow
    consumer would silently lose audio; the lossless path waits instead."""
    options = CaptureOptions(file_chunk_seconds=0.25, file_queue_size=2)
    source = LiveStreamSource(str(synthetic_clip), options)
    await source.connect()
    total = 0.0
    try:
        async for chunk in source.get_audio():
            total += chunk.duration
            await asyncio.sleep(0.02)
    finally:
        await source.close()
    assert total == pytest.approx(6.0, abs=0.1)
    assert source.stats.audio_chunks_dropped == 0


async def test_forcing_the_live_policy_on_a_file_can_drop(synthetic_clip: Path) -> None:
    options = CaptureOptions(file_chunk_seconds=0.25, queue_size=2)
    source = LiveStreamSource(str(synthetic_clip), options, lossless=False)
    await source.connect()
    assert source.lossless is False
    try:
        await source._supervisor
        chunks = await _collect(source, timeout=10)
    finally:
        await source.close()
    assert source.stats.audio_chunks_dropped > 0
    assert len(chunks) <= 3


async def test_close_is_idempotent(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), OPTIONS)
    await source.connect()
    await source.close()
    await source.close()


async def test_close_terminates_ffmpeg(synthetic_clip: Path) -> None:
    """A leaked ffmpeg keeps pulling bandwidth and RAM forever."""
    options = CaptureOptions(file_chunk_seconds=1.0, file_queue_size=1)
    source = LiveStreamSource(str(synthetic_clip), options, realtime=True)
    await source.connect()
    await source.wait_for_first_data(timeout=30)
    pipe = source._pipe
    assert pipe is not None and pipe.returncode is None
    await source.close()
    assert source._pipe is None
    assert pipe.returncode is not None


async def test_connecting_twice_is_refused(synthetic_clip: Path) -> None:
    source = LiveStreamSource(str(synthetic_clip), OPTIONS)
    await source.connect()
    try:
        with pytest.raises(RuntimeError, match="already connected"):
            await source.connect()
    finally:
        await source.close()


async def test_missing_source_fails_cleanly(tmp_path: Path) -> None:
    source = LiveStreamSource(str(tmp_path / "absent.wav"), OPTIONS)
    with pytest.raises(StreamResolutionError):
        await source.connect()
    await source.close()


async def test_an_unreadable_file_ends_with_an_error_instead_of_retrying(tmp_path: Path) -> None:
    """A finite source that fails is not retried: a retry would replay it from the start
    and duplicate whatever was already emitted, and an unreadable file never heals."""
    from livestream_transcriber.stream.base import StreamError

    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio at all" * 10)
    source = LiveStreamSource(str(bad), OPTIONS)
    await source.connect()
    try:
        with pytest.raises(StreamError, match="could not read"):
            await _collect(source, timeout=30)
    finally:
        await source.close()
    assert len(source.segments) == 1
    assert source.stats.reconnects == 0


async def test_async_context_manager(synthetic_clip: Path) -> None:
    async with LiveStreamSource(str(synthetic_clip), OPTIONS) as source:
        chunks = await _collect(source)
    assert chunks
