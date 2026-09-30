"""Tiny synthetic PCM helpers shared by the speech-to-text tests."""

from __future__ import annotations

from livestream_transcriber.models import AudioChunk

RATE = 16000


def tone(seconds: float = 1.0, sample_rate: int = RATE) -> bytes:
    """A constant, audible signal (amplitude 4096 of 32768, about -18 dBFS)."""
    return b"\x00\x10" * int(sample_rate * seconds)


def silence(seconds: float = 1.0, sample_rate: int = RATE) -> bytes:
    return b"\x00\x00" * int(sample_rate * seconds)


def chunk(
    ts: float, pcm: bytes, *, index: int | None = None, sample_rate: int = RATE
) -> AudioChunk:
    return AudioChunk(
        index=int(ts) if index is None else index,
        segment=0,
        media_ts=ts,
        ts=ts,
        wallclock=0.0,
        sample_rate=sample_rate,
        pcm=pcm,
    )


class Clock:
    """A manually advanced monotonic clock for breaker and log tests."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
