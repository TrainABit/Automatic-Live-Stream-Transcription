"""Replay a recording through the :class:`StreamSource` interface.

Downstream code cannot tell a replay from a live capture, which is the point: rules,
STT providers and outputs get exercised against a known recording as often as needed,
deterministically and for free.

``speed`` controls pacing: 1.0 reproduces the original timeline, 2.0 runs at double
rate, and 0 means "as fast as the consumer can take it" (tests and benchmarks).

``deterministic`` (the default) makes the source *lossless*: when a consumer falls
behind, the replay waits instead of discarding the oldest queued chunk. A live capture
drops on purpose, since a stale chunk is worth little and ffmpeg must not be stalled,
but in a replay that policy is a bug generator: how many chunks survive would depend on
how long speech-to-text happened to take, so the same recording would give different
transcripts on different machines. Set it to False only to reproduce the live queue
behaviour on purpose.

With ``loop=True`` the recording repeats. Session time (``ts``) and the chunk index keep
counting up across iterations, so the ordering machinery downstream never sees time run
backwards; ``media_ts`` restarts, exactly as it does after a live reconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import wave
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from ..models import AudioChunk, CaptureStats, StreamInfo
from .base import StreamError, StreamSource
from .queues import DropOldestQueue
from .recorder import MANIFEST_VERSION

log = get_logger(__name__)

__all__ = [
    "RecordingError",
    "ReplayStreamSource",
    "iter_audio_chunks",
    "load_audio_chunks",
    "read_manifest",
]

_QUEUE_SIZE = 256


class RecordingError(StreamError):
    """The recording directory is missing or malformed."""


def read_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise RecordingError(f"no manifest.json in {directory}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordingError(f"corrupt manifest in {directory}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RecordingError(f"corrupt manifest in {directory}: not an object")
    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise RecordingError(
            f"unsupported manifest version {version!r} in {directory} "
            f"(this build reads version {MANIFEST_VERSION})"
        )
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file.

    A torn *last* line is what a crash mid-write leaves behind; it is dropped with a
    warning so the recording still replays up to that point. A bad line anywhere else
    is real corruption and raises.
    """
    if not path.is_file():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    rows: list[dict[str, Any]] = []
    last = max((i for i, ln in enumerate(lines) if ln), default=-1)
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == last:
                log.warning("ignoring a torn last line", extra={"file": path.name})
                continue
            raise RecordingError(f"{path}:{i + 1}: {exc}") from exc
    return rows


def _audio_offsets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows with ``sample_offset`` filled in from the running total when absent."""
    offsets: dict[str, int] = {}
    result = []
    for row in rows:
        name = str(row["wav"])
        offset = int(row.get("sample_offset", offsets.get(name, 0)))
        result.append({**row, "sample_offset": offset})
        offsets[name] = offset + int(row["n_samples"])
    return result


def _inside(directory: Path, name: str) -> Path:
    """``directory / name``, refusing anything that leaves the recording directory.

    The audio index is data from disk; a ``../`` or absolute name in it must not make
    replay open some other file.
    """
    root = directory.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise RecordingError(f"audio file outside the recording: {name!r}")
    return path


def iter_audio_chunks(
    directory: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    wallclock: float | None = None,
) -> Iterator[AudioChunk]:
    """Yield PCM chunks from a recording's WAV files and ``audio.jsonl`` rows.

    This is the same slicing replay uses. ``wallclock`` is a constant when given;
    otherwise each chunk is stamped with ``time.time()``. If a WAV holds fewer samples
    than its index row promises (a crash between the two writes), iteration stops
    there instead of yielding a short chunk.
    """
    directory = Path(directory)
    readers: dict[str, wave.Wave_read] = {}
    try:
        for row in _audio_offsets(rows):
            name = str(row["wav"])
            reader = readers.get(name)
            if reader is None:
                path = _inside(directory, name)
                if not path.is_file():
                    raise RecordingError(f"missing audio file: {path}")
                try:
                    reader = wave.open(str(path), "rb")  # noqa: SIM115 - closed in the finally below
                except (wave.Error, EOFError) as exc:
                    raise RecordingError(f"unreadable audio file: {path}: {exc}") from exc
                readers[name] = reader
            wanted = int(row["n_samples"])
            reader.setpos(min(int(row["sample_offset"]), reader.getnframes()))
            pcm = reader.readframes(wanted)
            if len(pcm) < wanted * 2:
                log.warning("recording ends early", extra={"file": name})
                return
            yield AudioChunk(
                index=int(row["index"]),
                segment=int(row.get("segment", 0)),
                media_ts=float(row["media_ts"]),
                ts=float(row["ts"]),
                wallclock=time.time() if wallclock is None else wallclock,
                sample_rate=int(row["sample_rate"]),
                pcm=pcm,
            )
    finally:
        for reader in readers.values():
            with contextlib.suppress(Exception):
                reader.close()


def load_audio_chunks(directory: str | Path, *, wallclock: float = 0.0) -> list[AudioChunk]:
    """Load every recorded audio chunk with its original timestamps. No asyncio."""
    directory = Path(directory)
    read_manifest(directory)
    rows = _read_jsonl(directory / "audio.jsonl")
    if not rows:
        raise RecordingError(f"{directory} contains no audio chunks")
    return list(iter_audio_chunks(directory, rows, wallclock=wallclock))


class ReplayStreamSource(StreamSource):
    """Feeds a recorded session back through the live pipeline's interface."""

    def __init__(
        self,
        directory: str | Path,
        *,
        speed: float = 1.0,
        loop: bool = False,
        start_ts: float = 0.0,
        end_ts: float | None = None,
        deterministic: bool = True,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0 (0 = unthrottled)")
        self.dir = Path(directory)
        self.speed = speed
        self.deterministic = deterministic
        self.loop = loop
        self.start_ts = start_ts
        self.end_ts = end_ts

        self._manifest: dict[str, Any] = {}
        self._rows: list[dict[str, Any]] = []
        self._audio_q: DropOldestQueue[AudioChunk] = DropOldestQueue(_QUEUE_SIZE)
        self._stats = CaptureStats()
        self._task: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()

    @property
    def stats(self) -> CaptureStats:
        self._stats.audio_chunks_dropped = self._audio_q.dropped
        return self._stats

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    async def connect(self) -> StreamInfo:
        if self._task is not None:
            raise RuntimeError("already connected")
        if not self.dir.is_dir():
            raise RecordingError(f"recording directory not found: {self.dir}")
        self._manifest = read_manifest(self.dir)
        self._rows = [
            r for r in _audio_offsets(_read_jsonl(self.dir / "audio.jsonl")) if self._in_range(r)
        ]
        if not self._rows:
            raise RecordingError(f"{self.dir} contains no audio in the requested range")

        stream = self._manifest.get("stream") or {}
        info = StreamInfo(
            url=str(self.dir),
            title=stream.get("title") or self.dir.name,
            channel=stream.get("channel"),
            is_live=False,
            format_id="replay",
        )
        log.info(
            "replaying recording",
            extra={
                "dir": str(self.dir),
                "audio_chunks": len(self._rows),
                "speed": self.speed or "unthrottled",
                "deterministic": self.deterministic,
                "recorded_at": self._manifest.get("created_at_iso"),
            },
        )
        self._task = asyncio.create_task(self._run(), name="replay")
        return info

    def _in_range(self, row: Mapping[str, Any]) -> bool:
        ts = float(row.get("ts", 0.0))
        if ts < self.start_ts:
            return False
        return self.end_ts is None or ts <= self.end_ts

    def get_audio(self) -> AsyncIterator[AudioChunk]:
        return self._audio_q.__aiter__()

    async def close(self) -> None:
        self._closing.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._audio_q.close()

    async def _run(self) -> None:
        error: Exception | None = None
        try:
            iteration = 0
            while True:
                await self._replay_once(iteration)
                if not self.loop or self._closing.is_set():
                    break
                iteration += 1
                log.info("replay looping", extra={"iteration": iteration})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
            log.exception("replay failed")
        finally:
            self._audio_q.close(error)

    async def _replay_once(self, iteration: int) -> None:
        """Emit every recorded chunk once, in ``ts`` order, paced by ``speed``."""
        rows = sorted(self._rows, key=lambda r: float(r["ts"]))
        origin = float(rows[0]["ts"])
        span = max(float(r["ts"]) + int(r["n_samples"]) / int(r["sample_rate"]) for r in rows)
        span -= origin
        # A looped replay continues the session clock instead of restarting it.
        shift = iteration * span
        index_shift = iteration * len(rows)
        started = time.monotonic()
        chunks = iter_audio_chunks(self.dir, rows)
        for row in rows:
            if self._closing.is_set():
                return
            if self.speed > 0:
                delay = (float(row["ts"]) - origin) / self.speed - (time.monotonic() - started)
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._closing.wait(), delay)
                        return
                    except TimeoutError:
                        pass
            chunk = next(chunks, None)
            if chunk is None:
                return
            if iteration:
                chunk.ts += shift
                chunk.index += index_shift
            self._stats.audio_chunks_emitted += 1
            self._stats.audio_seconds += chunk.duration
            if self.deterministic:
                await self._audio_q.put_wait(chunk)
            else:
                self._audio_q.put(chunk)
