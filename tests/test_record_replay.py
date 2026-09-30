"""Record/replay round trip.

Benchmarks and rule tests lean on this: whatever a replay yields must match what the
live capture yielded, or tuning against recordings tunes the wrong thing.
"""

from __future__ import annotations

import asyncio
import json
import wave
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from livestream_transcriber.models import AudioChunk, SegmentInfo, StreamInfo
from livestream_transcriber.stream.queues import DropOldestQueue
from livestream_transcriber.stream.recorder import MANIFEST_VERSION, Recorder
from livestream_transcriber.stream.replay import (
    RecordingError,
    ReplayStreamSource,
    iter_audio_chunks,
    load_audio_chunks,
    read_manifest,
)
from livestream_transcriber.stream.source import CaptureOptions, LiveStreamSource

RATE = 16000


def _chunk(
    index: int, *, ts: float | None = None, seconds: float = 1.0, segment: int = 0
) -> AudioChunk:
    """A synthetic chunk whose samples all equal ``index + 1``: easy to tell apart."""
    n = int(RATE * seconds)
    start = float(index) if ts is None else ts
    return AudioChunk(
        index=index,
        segment=segment,
        media_ts=start,
        ts=start,
        wallclock=0.0,
        sample_rate=RATE,
        pcm=np.full(n, index + 1, dtype="<i2").tobytes(),
    )


async def _record(directory: Path, chunks: list[AudioChunk], **kw: Any) -> Path:
    recorder = Recorder(directory, **kw)
    recorder.open()
    for chunk in chunks:
        await recorder.add_audio(chunk)
    recorder.close()
    return directory


async def _drain(source: ReplayStreamSource, timeout: float = 30.0) -> list[AudioChunk]:
    chunks: list[AudioChunk] = []

    async def audio() -> None:
        async for chunk in source.get_audio():
            chunks.append(chunk)

    await asyncio.wait_for(audio(), timeout=timeout)
    return chunks


# --------------------------------------------------------------------- layout and manifest


async def test_recording_layout_and_manifest(tmp_path: Path) -> None:
    info = StreamInfo(
        url="https://example.test/watch?v=demoVideo01&token=your-secret-token",
        title="A stream",
        channel="A channel",
        is_live=True,
        media_url="https://cdn.example.test/videoplayback?sig=abc&ip=203.0.113.9",
        headers={"Cookie": "session=your-cookie-value", "User-Agent": "UA"},
    )
    rec = Recorder(
        tmp_path / "rec", sample_rate=RATE, chunk_seconds=1.0, stream_info=info, note="a note"
    )
    rec.open()
    for i in range(3):
        await rec.add_audio(_chunk(i))
    rec.close([SegmentInfo(index=0, started_at=1.0, offset=0.0, audio_chunks=3, reason="eof")])

    root = tmp_path / "rec"
    assert sorted(p.name for p in root.iterdir()) == ["audio", "audio.jsonl", "manifest.json"]
    assert (root / "audio" / "seg000.wav").is_file()
    manifest = read_manifest(root)
    assert manifest["version"] == MANIFEST_VERSION == 1
    assert manifest["note"] == "a note"
    assert manifest["capture"] == {"sample_rate": RATE, "chunk_seconds": 1.0}
    assert manifest["counts"] == {"audio_chunks": 3, "audio_seconds": 3.0}
    assert manifest["segments"][0]["reason"] == "eof"
    assert manifest["created_at_iso"].endswith("Z")


async def test_manifest_never_stores_signed_urls_or_credentials(tmp_path: Path) -> None:
    """Recordings get shared; signed URLs carry our address and credentials, and expire."""
    info = StreamInfo(
        url="https://example.test/watch?v=demoVideo01&token=your-secret-token",
        media_url="https://cdn.example.test/videoplayback?sig=abc&ip=203.0.113.9",
        headers={"Cookie": "session=your-cookie-value"},
    )
    await _record(tmp_path / "rec", [_chunk(0)], stream_info=info)
    text = (tmp_path / "rec" / "manifest.json").read_text()
    manifest = json.loads(text)
    assert manifest["stream"]["media_url"] is None
    assert manifest["stream"]["headers"] == {}
    for secret in ("your-secret-token", "your-cookie-value", "203.0.113.9", "sig=abc"):
        assert secret not in text
    assert "v=demoVideo01" in text, "the harmless part of the URL survives"


async def test_audio_rows_carry_offsets_into_the_wav(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0), _chunk(1, seconds=0.5), _chunk(2)])
    rows = [json.loads(line) for line in (root / "audio.jsonl").read_text().splitlines()]
    assert [r["sample_offset"] for r in rows] == [0, RATE, RATE + RATE // 2]
    assert {r["wav"] for r in rows} == {"audio/seg000.wav"}
    assert [r["n_samples"] for r in rows] == [RATE, RATE // 2, RATE]


async def test_the_wav_is_a_valid_mono_16_bit_file(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0), _chunk(1)])
    with wave.open(str(root / "audio" / "seg000.wav"), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, RATE)
        assert wav.getnframes() == 2 * RATE


async def test_each_segment_gets_its_own_wav(tmp_path: Path) -> None:
    chunks = [_chunk(0), _chunk(1, segment=1, ts=5.0), _chunk(2, segment=1, ts=6.0)]
    root = await _record(tmp_path / "rec", chunks)
    assert sorted(p.name for p in (root / "audio").iterdir()) == ["seg000.wav", "seg001.wav"]
    assert [c.pcm for c in load_audio_chunks(root)] == [c.pcm for c in chunks]


async def test_recorder_refuses_to_overwrite_an_existing_recording(tmp_path: Path) -> None:
    rec = Recorder(tmp_path / "rec")
    rec.open()
    rec.close()
    before = (rec.dir / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        Recorder(rec.dir).open()
    # Cleanup after a refused open must not rewrite the existing recording.
    refused = Recorder(rec.dir)
    with pytest.raises(FileExistsError):
        refused.open()
    refused.close()
    assert (rec.dir / "manifest.json").read_bytes() == before


async def test_add_after_close_or_before_open_is_ignored(tmp_path: Path) -> None:
    rec = Recorder(tmp_path / "rec")
    await rec.add_audio(_chunk(0))  # never opened
    assert not (tmp_path / "rec").exists()
    rec.open()
    rec.close()
    rec.close()  # idempotent
    await rec.add_audio(_chunk(1))
    assert rec.audio_chunks_written == 0


async def test_context_manager_opens_and_closes(tmp_path: Path) -> None:
    with Recorder(tmp_path / "rec") as rec:
        await rec.add_audio(_chunk(0))
    assert rec.audio_seconds_written == pytest.approx(1.0)
    assert read_manifest(tmp_path / "rec")["counts"]["audio_chunks"] == 1


# --------------------------------------------------------------------- crash safety


async def test_a_crashed_recording_replays_up_to_the_last_flush(tmp_path: Path) -> None:
    """The process dies without close(): WAV headers are patched and files flushed
    periodically, so the recording is still a valid, replayable prefix."""
    rec = Recorder(tmp_path / "rec", flush_interval_s=0.0)  # flush on every chunk
    rec.open()
    for i in range(3):
        await rec.add_audio(_chunk(i))
    # No close(): read the directory as a crashed process would have left it.
    chunks = load_audio_chunks(tmp_path / "rec")
    assert [c.pcm for c in chunks] == [_chunk(i).pcm for i in range(3)]
    with wave.open(str(tmp_path / "rec" / "audio" / "seg000.wav"), "rb") as wav:
        assert wav.getnframes() == 3 * RATE
    assert read_manifest(tmp_path / "rec")["counts"]["audio_chunks"] == 3
    rec.close()


async def test_unflushed_chunks_are_lost_but_flushed_ones_survive(tmp_path: Path) -> None:
    rec = Recorder(tmp_path / "rec", flush_interval_s=3600.0)
    rec.open()
    await rec.add_audio(_chunk(0))
    rec.flush()
    await rec.add_audio(_chunk(1))  # buffered, not flushed
    # The wav header on disk only knows about the flushed chunk.
    with wave.open(str(tmp_path / "rec" / "audio" / "seg000.wav"), "rb") as wav:
        assert wav.getnframes() == RATE
    rec.close()


async def test_a_torn_last_index_line_is_ignored(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0), _chunk(1)])
    with (root / "audio.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"index": 2, "segm')  # killed mid-write
    assert [c.index for c in load_audio_chunks(root)] == [0, 1]


async def test_a_corrupt_line_in_the_middle_is_an_error(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0), _chunk(1)])
    lines = (root / "audio.jsonl").read_text().splitlines()
    (root / "audio.jsonl").write_text("\n".join([lines[0], "{not json", lines[1]]) + "\n")
    with pytest.raises(RecordingError, match=r"audio\.jsonl:2"):
        load_audio_chunks(root)


async def test_an_index_row_past_the_end_of_its_wav_stops_the_replay(tmp_path: Path) -> None:
    """A crash between the two writes can leave an index row without its samples."""
    root = await _record(tmp_path / "rec", [_chunk(0), _chunk(1)])
    row = json.loads((root / "audio.jsonl").read_text().splitlines()[-1])
    row.update(index=2, sample_offset=2 * RATE, ts=2.0, media_ts=2.0)
    with (root / "audio.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    assert [c.index for c in load_audio_chunks(root)] == [0, 1]


# --------------------------------------------------------------------- replay


async def test_replay_reproduces_counts_timestamps_and_audio(tmp_path: Path) -> None:
    original = [_chunk(i) for i in range(4)]
    root = await _record(tmp_path / "rec", original)
    source = ReplayStreamSource(root, speed=0)
    info = await source.connect()
    try:
        chunks = await _drain(source)
    finally:
        await source.close()

    assert info.is_live is False and info.format_id == "replay" and info.title == "rec"
    assert [c.pcm for c in chunks] == [c.pcm for c in original], "PCM round trip is lossless"
    assert [(c.index, c.segment, c.media_ts, c.ts) for c in chunks] == [
        (c.index, c.segment, c.media_ts, c.ts) for c in original
    ]
    assert source.stats.audio_chunks_emitted == 4
    assert source.stats.audio_seconds == pytest.approx(4.0)
    for prev, nxt in pairwise(chunks):
        assert nxt.media_ts == pytest.approx(prev.media_ts_end, abs=1e-6)


async def test_replayed_chunks_get_a_fresh_wallclock(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    source = ReplayStreamSource(root, speed=0)
    await source.connect()
    try:
        (chunk,) = await _drain(source)
    finally:
        await source.close()
    assert chunk.wallclock > 1e9, "the recorded wallclock is ignored, replay stamps its own"


async def test_replay_range_selects_a_window(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(i) for i in range(6)])
    source = ReplayStreamSource(root, speed=0, start_ts=2.0, end_ts=4.0)
    await source.connect()
    try:
        chunks = await _drain(source)
    finally:
        await source.close()
    assert [c.index for c in chunks] == [2, 3, 4]
    assert [c.pcm for c in chunks] == [_chunk(i).pcm for i in (2, 3, 4)]


async def test_an_empty_range_is_an_error(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    with pytest.raises(RecordingError, match="no audio in the requested range"):
        await ReplayStreamSource(root, start_ts=100.0).connect()


async def test_replay_honours_wall_clock_pacing(tmp_path: Path) -> None:
    """speed=1 reproduces the original timeline, not a race through it."""
    root = await _record(tmp_path / "rec", [_chunk(i, seconds=0.25) for i in range(1)] + [
        _chunk(1, ts=1.0, seconds=0.25),
        _chunk(2, ts=2.0, seconds=0.25),
    ])  # fmt: skip
    source = ReplayStreamSource(root, speed=2.0)  # 2 s span at double rate: about 1 s
    await source.connect()
    start = asyncio.get_running_loop().time()
    try:
        chunks = await _drain(source)
    finally:
        await source.close()
    elapsed = asyncio.get_running_loop().time() - start
    assert len(chunks) == 3
    assert 0.8 < elapsed < 3.0, f"expected about 1 s of pacing, got {elapsed:.2f}"


async def test_speed_zero_is_unthrottled(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(i, ts=i * 100.0) for i in range(3)])
    source = ReplayStreamSource(root, speed=0)
    await source.connect()
    start = asyncio.get_running_loop().time()
    try:
        chunks = await _drain(source)
    finally:
        await source.close()
    assert len(chunks) == 3
    assert asyncio.get_running_loop().time() - start < 2.0


async def test_loop_repeats_with_monotonic_session_time(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0), _chunk(1)])
    source = ReplayStreamSource(root, speed=0, loop=True)
    await source.connect()
    got: list[AudioChunk] = []
    try:
        async for chunk in source.get_audio():
            got.append(chunk)
            if len(got) == 6:
                break
    finally:
        await source.close()
    # ts and index keep counting up across iterations; media_ts restarts, as after a
    # live reconnect.
    assert [c.index for c in got] == [0, 1, 2, 3, 4, 5]
    assert [c.ts for c in got] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert [c.media_ts for c in got] == pytest.approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert [c.pcm for c in got[:2]] == [c.pcm for c in got[2:4]]


async def test_speed_must_not_be_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="speed"):
        ReplayStreamSource(tmp_path, speed=-1)


async def test_connecting_twice_is_refused(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    source = ReplayStreamSource(root, speed=0)
    await source.connect()
    try:
        with pytest.raises(RuntimeError, match="already connected"):
            await source.connect()
    finally:
        await source.close()


# --------------------------------------------------------------------- errors


async def test_missing_recording_raises(tmp_path: Path) -> None:
    with pytest.raises(RecordingError):
        await ReplayStreamSource(tmp_path / "nope").connect()


async def test_directory_without_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(RecordingError, match="manifest"):
        await ReplayStreamSource(tmp_path / "empty").connect()


async def test_corrupt_manifest_raises(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / "manifest.json").write_text("{not json")
    with pytest.raises(RecordingError, match="corrupt"):
        await ReplayStreamSource(d).connect()
    (d / "manifest.json").write_text("[1, 2]")
    with pytest.raises(RecordingError, match="not an object"):
        read_manifest(d)


async def test_an_unknown_manifest_version_is_refused(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["version"] = 99
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecordingError, match="unsupported manifest version 99"):
        await ReplayStreamSource(root).connect()


@pytest.mark.parametrize("name", ["../outside.wav", "/etc/hosts"])
async def test_an_audio_index_cannot_point_outside_the_recording(tmp_path: Path, name: str) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    rows = [json.loads(line) for line in (root / "audio.jsonl").read_text().splitlines()]
    rows[0]["wav"] = name
    (root / "audio.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RecordingError, match="outside the recording"):
        load_audio_chunks(root)


async def test_a_recording_without_audio_raises(tmp_path: Path) -> None:
    rec = Recorder(tmp_path / "rec")
    rec.open()
    rec.close()
    with pytest.raises(RecordingError, match="no audio"):
        await ReplayStreamSource(rec.dir).connect()
    with pytest.raises(RecordingError, match="no audio chunks"):
        load_audio_chunks(rec.dir)


async def test_missing_recorded_media_reaches_the_consumer(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    (root / "audio" / "seg000.wav").unlink()
    source = ReplayStreamSource(root, speed=0)
    await source.connect()
    try:
        with pytest.raises(RecordingError, match="missing audio file"):
            await _drain(source, timeout=5)
    finally:
        await source.close()
    await source.close()


async def test_an_unreadable_wav_is_a_recording_error(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    (root / "audio" / "seg000.wav").write_bytes(b"not a wav file")
    with pytest.raises(RecordingError, match="unreadable audio file"):
        list(iter_audio_chunks(root, [json.loads((root / "audio.jsonl").read_text())]))


async def test_audio_round_trip_after_dropped_chunks(tmp_path: Path) -> None:
    """Chunk indices need not be contiguous (a live capture drops on purpose)."""
    chunks = [_chunk(i) for i in (2, 4, 5)]
    root = await _record(tmp_path / "rec", chunks)
    assert [c.pcm for c in load_audio_chunks(root)] == [c.pcm for c in chunks]
    source = ReplayStreamSource(root, speed=0, start_ts=4)
    await source.connect()
    try:
        replayed = await _drain(source, timeout=5)
    finally:
        await source.close()
    assert [c.pcm for c in replayed] == [c.pcm for c in chunks[1:]]


# --------------------------------------------------------------------- lossless replay


async def test_replay_is_lossless_by_default(tmp_path: Path) -> None:
    """More chunks than the queue holds, and a slow consumer: nothing may be dropped."""
    root = await _record(tmp_path / "rec", [_chunk(i, seconds=0.1, ts=i * 0.1) for i in range(400)])
    source = ReplayStreamSource(root, speed=0.0)
    assert source.deterministic is True
    await source.connect()
    got = 0
    try:
        async for _ in source.get_audio():
            got += 1
            if got % 100 == 0:
                await asyncio.sleep(0.01)
    finally:
        await source.close()
    assert got == 400
    assert source.stats.audio_chunks_dropped == 0


async def test_non_deterministic_replay_may_drop_like_a_live_capture(tmp_path: Path) -> None:
    root = await _record(tmp_path / "rec", [_chunk(i, seconds=0.1, ts=i * 0.1) for i in range(400)])
    source = ReplayStreamSource(root, speed=0.0, deterministic=False)
    await source.connect()
    assert source._task is not None
    await source._task  # nobody is reading: the queue overflows
    try:
        got = await _drain(source, timeout=5)
    finally:
        await source.close()
    assert source.stats.audio_chunks_dropped > 0
    assert len(got) + source.stats.audio_chunks_dropped == 400


async def test_a_full_queue_waits_instead_of_dropping() -> None:
    """A slow consumer must cost time, never evidence. The live queue drops the oldest
    item on purpose, but in a replay that would make the surviving chunks a function of
    how long speech-to-text happened to take."""
    queue: DropOldestQueue[int] = DropOldestQueue(2)
    for value in (1, 2, 3):
        queue.put(value)
    assert queue.dropped == 1  # the live policy

    patient: DropOldestQueue[int] = DropOldestQueue(2)
    await patient.put_wait(1)
    await patient.put_wait(2)
    waiting = asyncio.create_task(patient.put_wait(3))
    await asyncio.sleep(0)
    assert not waiting.done() and patient.dropped == 0
    assert await patient.get() == 1  # room appears
    await waiting
    assert patient.dropped == 0
    # The spare slot stays free for the end-of-stream sentinel, so closing a full queue
    # cannot evict the item that was waiting in it.
    patient.close()
    assert [await patient.get(), await patient.get()] == [2, 3]


async def test_replay_producer_failure_reaches_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = await _record(tmp_path / "rec", [_chunk(0)])
    source = ReplayStreamSource(root, speed=0)

    def fail(*_a: object, **_k: object) -> Any:
        raise RecordingError("unreadable audio")

    monkeypatch.setattr("livestream_transcriber.stream.replay.iter_audio_chunks", fail)
    await source.connect()
    try:
        with pytest.raises(RecordingError, match="unreadable audio"):
            await _drain(source, timeout=5)
    finally:
        await source.close()


# --------------------------------------------------------------------- capture -> record -> replay


@pytest.mark.slow
async def test_capture_record_replay_round_trip(synthetic_clip: Path, tmp_path: Path) -> None:
    """Record a real ffmpeg capture, then replay it: the PCM must be bit-identical."""
    options = CaptureOptions(file_chunk_seconds=1.0)
    source = LiveStreamSource(str(synthetic_clip), options)
    info = await source.connect()
    rec = Recorder(
        tmp_path / "rec", sample_rate=options.sample_rate, stream_info=info, note="round-trip"
    )
    rec.open()
    captured: list[AudioChunk] = []
    try:
        async for chunk in source.get_audio():
            captured.append(chunk)
            await rec.add_audio(chunk)
    finally:
        await source.close()
        rec.close(source.segments)
    # The clip is one ffmpeg run. A second segment would mean the capture restarted and
    # recorded the clip again; fail here, where the cause shows.
    assert [s.reason for s in source.segments] == ["stream_ended"]
    assert read_manifest(tmp_path / "rec")["segments"], "segment history survives"

    replay = ReplayStreamSource(tmp_path / "rec", speed=0)
    await replay.connect()
    try:
        replayed = await _drain(replay, timeout=60)
    finally:
        await replay.close()
    assert [c.pcm for c in replayed] == [c.pcm for c in captured]
    with wave.open(str(tmp_path / "rec" / "audio" / "seg000.wav"), "rb") as wav:
        assert b"".join(c.pcm for c in replayed) == wav.readframes(wav.getnframes())
