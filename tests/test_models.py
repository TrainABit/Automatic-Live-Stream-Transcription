from __future__ import annotations

import math

import numpy as np

from livestream_transcriber.models import AudioChunk, CaptureStats, SegmentInfo, StreamInfo


def _chunk(pcm: bytes, **kw: object) -> AudioChunk:
    base: dict[str, object] = {
        "index": 0,
        "segment": 0,
        "media_ts": 0.0,
        "ts": 0.0,
        "wallclock": 0.0,
        "sample_rate": 16000,
    }
    return AudioChunk(**{**base, "pcm": pcm, **kw})  # type: ignore[arg-type]


def test_audio_duration_and_end():
    chunk = _chunk(b"\x00\x00" * 8000, media_ts=4.0)  # 0.5 s at 16 kHz
    assert chunk.n_samples == 8000
    assert chunk.duration == 0.5
    assert chunk.media_ts_end == 4.5


def test_float32_conversion_is_normalised():
    pcm = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    out = _chunk(pcm).as_float32()
    assert out.dtype == np.float32
    assert np.allclose(out, [0.0, 0.5, -0.5, 0.99997], atol=1e-4)
    assert np.abs(out).max() <= 1.0


def test_peak_dbfs_reports_silence_and_full_scale():
    assert _chunk(b"\x00\x00" * 100).peak_dbfs() == -math.inf
    assert _chunk(b"").peak_dbfs() == -math.inf
    full = np.array([32767, -32768], dtype="<i2").tobytes()
    assert _chunk(full).peak_dbfs() == 0.0
    half = np.array([16384], dtype="<i2").tobytes()
    assert round(_chunk(half).peak_dbfs(), 1) == -6.0


def test_chunk_description_is_json_friendly():
    described = _chunk(b"\x00\x40" * 16000, index=3).describe()
    assert described["index"] == 3
    assert described["duration"] == 1.0
    assert described["peak_dbfs"] == -6.0
    assert _chunk(b"").describe()["peak_dbfs"] is None


def test_stream_info_describe_never_exposes_the_media_url_or_headers():
    info = StreamInfo(
        url="https://example.com/live",
        title="Show",
        is_live=True,
        media_url="https://cdn.example.com/a.m3u8?sig=secret",
        headers={"Authorization": "Bearer token-value"},
    )
    described = info.describe()
    assert described["title"] == "Show"
    assert described["is_live"] is True
    assert "secret" not in str(described)
    assert "token-value" not in repr(info)


def test_segment_info_defaults():
    seg = SegmentInfo(index=1, started_at=10.0, offset=30.0)
    assert seg.ended_at is None
    assert seg.audio_chunks == 0
    assert seg.audio_sample_base == 0


def test_stats_description_is_json_friendly():
    described = CaptureStats(audio_chunks_emitted=3, audio_seconds=2.54).describe()
    assert described["audio_chunks"] == 3
    assert described["audio_seconds"] == 2.5
    assert set(described) == {
        "audio_chunks",
        "audio_dropped",
        "audio_seconds",
        "segments",
        "reconnects",
    }
