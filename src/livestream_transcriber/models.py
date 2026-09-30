"""Core data types shared across the pipeline.

Timing model
------------
Every audio chunk carries three clocks, and they mean different things:

``media_ts``
    Seconds since the start of the current *capture segment*, derived by
    counting samples (sample index / sample rate). ffmpeg emits constant-rate
    PCM from a single demuxed input, so this is exact within a segment and
    immune to scheduling jitter. Use it to place text on the media timeline.

``ts``
    Seconds since the capture *session* started, continuous across reconnects:
    ``ts = segment_offset + media_ts``. It is monotonic, so it is safe to order
    and window on, but it includes the (logged) gaps where the stream dropped.

``wallclock``
    Unix time at which the chunk was read from ffmpeg. Useful for reporting and
    for measuring end-to-end latency; it is *not* the time the audio was
    broadcast (live streams run several seconds behind).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["AudioChunk", "CaptureStats", "SegmentInfo", "StreamInfo"]


@dataclass(slots=True)
class AudioChunk:
    """A fixed-length slice of mono PCM (signed 16-bit little-endian)."""

    index: int
    """Chunk counter within the session (0-based)."""
    segment: int
    """Index of the capture segment (one uninterrupted ffmpeg run)."""
    media_ts: float
    """Start of the chunk, seconds since segment start."""
    ts: float
    """Start of the chunk, seconds since session start."""
    wallclock: float
    """Unix time at which the chunk was read."""
    sample_rate: int
    pcm: bytes

    @property
    def n_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    @property
    def media_ts_end(self) -> float:
        return self.media_ts + self.duration

    def as_float32(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Samples normalised to [-1, 1], the shape Whisper-style models expect."""
        samples = np.frombuffer(self.pcm, dtype="<i2").astype(np.float32)
        return samples / np.float32(32768.0)

    def peak_dbfs(self) -> float:
        """Peak level in dBFS; ``-inf`` for digital silence. A cheap health check."""
        if not self.pcm:
            return float("-inf")
        peak = int(np.abs(np.frombuffer(self.pcm, dtype="<i2").astype(np.int32)).max())
        if peak == 0:
            return float("-inf")
        return float(20.0 * np.log10(peak / 32768.0))

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "segment": self.segment,
            "media_ts": round(self.media_ts, 3),
            "ts": round(self.ts, 3),
            "duration": round(self.duration, 3),
            "peak_dbfs": round(self.peak_dbfs(), 1) if self.pcm else None,
        }


@dataclass(slots=True)
class StreamInfo:
    """What the resolver learned about a source."""

    url: str
    """The URL the user asked for."""
    title: str | None = None
    channel: str | None = None
    is_live: bool = False
    media_url: str | None = None
    """Direct media URL for ffmpeg; ``None`` when ``url`` can be opened as is."""
    format_id: str | None = None
    stream_id: str | None = None
    """Platform id of the broadcast a channel-level URL resolved to, if any."""
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    """HTTP headers ffmpeg must send with ``media_url``. Never logged."""
    resolved_at: float = field(default_factory=time.time)

    def describe(self) -> dict[str, Any]:
        """Log-safe summary: no media URL, no headers."""
        return {
            "title": self.title,
            "channel": self.channel,
            "is_live": self.is_live,
            "format_id": self.format_id,
            "stream_id": self.stream_id,
        }


@dataclass(slots=True)
class SegmentInfo:
    """One uninterrupted ffmpeg run. A reconnect starts a new segment."""

    index: int
    started_at: float
    offset: float
    """``ts`` value corresponding to ``media_ts == 0`` for this segment."""
    audio_sample_base: int = 0
    """Mono PCM samples already emitted when this segment started."""
    ended_at: float | None = None
    audio_chunks: int = 0
    reason: str | None = None
    """Why the segment ended (``"eof"``, ``"stall"``, ``"error"``, ...)."""


@dataclass(slots=True)
class CaptureStats:
    """Counters a capture source keeps for its whole lifetime."""

    audio_chunks_emitted: int = 0
    audio_chunks_dropped: int = 0
    audio_seconds: float = 0.0
    segments: int = 0
    reconnects: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "audio_chunks": self.audio_chunks_emitted,
            "audio_dropped": self.audio_chunks_dropped,
            "audio_seconds": round(self.audio_seconds, 1),
            "segments": self.segments,
            "reconnects": self.reconnects,
        }
