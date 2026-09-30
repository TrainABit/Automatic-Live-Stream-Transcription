"""Write a capture to disk in a form :mod:`.replay` can reproduce exactly.

A recording makes everything downstream testable: a segment containing a known
phrase can be replayed a hundred times while tuning rules or comparing STT models,
without waiting for a broadcaster to say it again.

Layout (manifest ``version`` 1)::

    recordings/<name>/
        manifest.json        capture settings, stream metadata, counts, segments
        audio.jsonl          one row per chunk: index, segment, media_ts, ts, ...
        audio/seg000.wav     continuous 16-bit mono PCM per capture segment

The recorder is written to survive a crash. WAV headers are patched and both files
are flushed every ``flush_interval_s`` seconds, and the manifest is rewritten
atomically at the same moments, so a killed process leaves a recording that
replays up to the last flush instead of a directory of unreadable files.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from ..logging_setup import get_logger
from ..models import AudioChunk, SegmentInfo, StreamInfo
from ..redact import redact_url

log = get_logger(__name__)

__all__ = ["MANIFEST_VERSION", "Recorder"]

MANIFEST_VERSION = 1
_WAV_HEADER_BYTES = 44


def _safe_stream_info(info: StreamInfo | None) -> dict[str, Any] | None:
    """Manifest-safe view of the stream metadata.

    A resolved media URL is usually signed and carries the capturing machine's
    public address, and request headers can carry credentials. Recordings get
    copied around and attached to bug reports, so none of that travels with them
    (and signed URLs expire within hours, which makes them worthless to keep). The
    URL the user asked for is kept, with any secret query parameter removed.
    """
    if info is None:
        return None
    safe = replace(info, url=redact_url(info.url), media_url=None, headers={})
    return asdict(safe)


class _WavWriter:
    """Append-only mono 16-bit WAV file whose header is always consistent.

    ``wave.Wave_write`` only finalises its header on ``close()``. This writer can
    patch the RIFF and data sizes on every :meth:`flush`, which is what lets a
    crashed recording still open.
    """

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.samples = 0
        self._fh: BinaryIO | None = path.open("wb")
        self._fh.write(self._header(0))

    def _header(self, data_bytes: int) -> bytes:
        rate = self.sample_rate
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_bytes,
            b"WAVE",
            b"fmt ",
            16,  # PCM fmt chunk size
            1,  # PCM
            1,  # mono
            rate,
            rate * 2,  # byte rate
            2,  # block align
            16,  # bits per sample
            b"data",
            data_bytes,
        )

    def write(self, pcm: bytes) -> None:
        assert self._fh is not None
        self._fh.write(pcm)
        self.samples += len(pcm) // 2

    def flush(self) -> None:
        """Make what was written so far a valid, self-describing WAV file."""
        if self._fh is None:
            return
        end = self._fh.tell()
        self._fh.seek(0)
        self._fh.write(self._header(self.samples * 2))
        self._fh.seek(end)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is None:
            return
        self.flush()
        self._fh.close()
        self._fh = None


class Recorder:
    """Persists audio chunks. Safe to use from the capture loop."""

    def __init__(
        self,
        directory: str | Path,
        *,
        sample_rate: int = 16000,
        chunk_seconds: float | None = None,
        stream_info: StreamInfo | None = None,
        note: str | None = None,
        flush_interval_s: float = 2.0,
    ) -> None:
        self.dir = Path(directory)
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.stream_info = stream_info
        self.note = note
        self.flush_interval_s = flush_interval_s

        self._audio_fp: TextIO | None = None
        self._wavs: dict[int, _WavWriter] = {}
        self._started_at = 0.0
        self._chunks = 0
        self._audio_samples = 0
        self._last_flush = 0.0
        self._closed = False
        self._opened = False
        # add_audio hands the write to a worker thread; close() takes this lock too,
        # so it never finalises a file a write is still appending to.
        self._lock = threading.Lock()

    @property
    def audio_chunks_written(self) -> int:
        return self._chunks

    @property
    def audio_seconds_written(self) -> float:
        return self._audio_samples / max(1, self.sample_rate)

    def open(self) -> None:
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(f"recording dir not empty: {self.dir}")
        (self.dir / "audio").mkdir(parents=True, exist_ok=True)
        self._started_at = time.time()
        self._last_flush = time.monotonic()
        self._audio_fp = (self.dir / "audio.jsonl").open("w", encoding="utf-8")
        self._opened = True
        self._write_manifest()
        log.info("recording started", extra={"dir": str(self.dir)})

    async def add_audio(self, chunk: AudioChunk) -> None:
        """Append one chunk. The disk write runs in a worker thread."""
        if self._closed or not self._opened:
            return
        await asyncio.to_thread(self._append, chunk)

    def _append(self, chunk: AudioChunk) -> None:
        with self._lock:
            if self._closed:
                return
            wav = self._wav_for(chunk.segment, chunk.sample_rate)
            row = {
                "index": chunk.index,
                "segment": chunk.segment,
                "media_ts": round(chunk.media_ts, 6),
                "ts": round(chunk.ts, 6),
                "wallclock": round(chunk.wallclock, 6),
                "sample_rate": chunk.sample_rate,
                "n_samples": chunk.n_samples,
                "sample_offset": wav.samples,
                "wav": f"audio/seg{chunk.segment:03d}.wav",
            }
            wav.write(chunk.pcm)
            assert self._audio_fp is not None
            self._audio_fp.write(json.dumps(row) + "\n")
            self._chunks += 1
            self._audio_samples += chunk.n_samples
            now = time.monotonic()
            if now - self._last_flush >= self.flush_interval_s:
                self._flush_locked()
                self._last_flush = now

    def _wav_for(self, segment: int, sample_rate: int) -> _WavWriter:
        wav = self._wavs.get(segment)
        if wav is None:
            wav = _WavWriter(self.dir / "audio" / f"seg{segment:03d}.wav", sample_rate)
            self._wavs[segment] = wav
        return wav

    def flush(self) -> None:
        """Push everything written so far to disk in a replayable state."""
        with self._lock:
            if not self._closed and self._opened:
                self._flush_locked()

    def _flush_locked(self) -> None:
        # WAV data first: an index row must never point past the end of its file.
        for wav in self._wavs.values():
            wav.flush()
        if self._audio_fp is not None:
            self._audio_fp.flush()
        self._write_manifest()

    def _write_manifest(self, segments: list[SegmentInfo] | None = None) -> None:
        manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "created_at": self._started_at,
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._started_at)),
            "note": self.note,
            "capture": {
                "sample_rate": self.sample_rate,
                "chunk_seconds": self.chunk_seconds,
            },
            "stream": _safe_stream_info(self.stream_info),
            "counts": {
                "audio_chunks": self._chunks,
                "audio_seconds": round(self.audio_seconds_written, 3),
            },
            "duration_seconds": round(max(0.0, time.time() - self._started_at), 3),
            "segments": [asdict(s) for s in (segments or [])],
        }
        path = self.dir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)  # readers never see a half-written manifest

    def close(self, segments: list[SegmentInfo] | None = None) -> None:
        """Finalise WAV headers, flush the index, rewrite the manifest. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for wav in self._wavs.values():
                try:
                    wav.close()
                except OSError:  # pragma: no cover
                    log.warning("failed to close wav", exc_info=True)
            self._wavs.clear()
            if self._audio_fp is not None:
                self._audio_fp.flush()
                self._audio_fp.close()
                self._audio_fp = None
            # Never invent a manifest in a directory ``open()`` refused (an existing
            # recording): the cleanup path must not rewrite someone else's files.
            if self._opened:
                self._write_manifest(segments)
        log.info(
            "recording finished",
            extra={
                "dir": str(self.dir),
                "audio_chunks": self._chunks,
                "audio_seconds": round(self.audio_seconds_written, 1),
            },
        )

    def __enter__(self) -> Recorder:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
