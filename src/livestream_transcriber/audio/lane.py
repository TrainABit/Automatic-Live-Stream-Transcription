"""Bounded STT work queue: capture produces, workers transcribe.

The capture path must not wait on a network transcription; this queue is the
handoff. RAM use is small on purpose (a few chunks of PCM). Further jobs spill
their PCM to disk, so a slow provider does not silently discard speech and a
small machine is not asked to hold an hour of audio in memory.

Live capture drops the oldest *waiting* job when both memory and spill are full,
which is counted and reported by the caller, never silent. Replay waits instead:
the same recording must not lose a chunk because the machine was busy.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import AudioChunk

__all__ = [
    "DEFAULT_STT_QUEUE_SIZE",
    "DEFAULT_STT_SPILL_CHUNKS",
    "DEFAULT_STT_WORKERS",
    "SPILL_DIR_PREFIX",
    "AudioJob",
    "AudioJobQueue",
    "AudioLaneStats",
]

DEFAULT_STT_WORKERS = 1
DEFAULT_STT_QUEUE_SIZE = 8
DEFAULT_STT_SPILL_CHUNKS = 240
#: Prefix of the temporary spill directory the queue creates for itself.
SPILL_DIR_PREFIX = "lst-stt-"
# A spill file still on disk this long after it was written belongs to a crashed
# or killed process: a live queue takes or evicts its jobs long before that.
SPILL_ORPHAN_MAX_AGE_SECONDS = 3600.0
_SPILL_NAME = re.compile(r"^\d+_\d+_\d+\.pcm$")


def _prune_spill_orphans(
    spill_dir: Path, max_age_seconds: float = SPILL_ORPHAN_MAX_AGE_SECONDS
) -> int:
    """Delete the queue's own ``<ticket>_<index>_<segment>.pcm`` files older than the limit.

    Only files matching the queue's naming scheme are touched, so pointing
    ``spill_dir`` at a directory with other content is safe. Never raises.
    """
    cutoff = time.time() - max_age_seconds
    removed = 0
    try:
        candidates = [p for p in spill_dir.iterdir() if _SPILL_NAME.match(p.name)]
    except OSError:
        return 0
    for path in candidates:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def _release_pcm(job: AudioJob) -> None:
    """Let go of a queued job's PCM without emptying anyone else's chunk.

    The capture consumer may hand the *same* :class:`AudioChunk` to the health
    reporter and the recorder. Blanking its ``pcm`` in place (on a spill or an
    eviction) would make every one of them see an empty chunk while the audio is
    fine. The job keeps a PCM-less copy instead; the others keep theirs.
    """
    if job.chunk.pcm:
        job.chunk = dataclasses.replace(job.chunk, pcm=b"")


@dataclass(slots=True)
class AudioLaneStats:
    """Operational counters of the audio lane. Never part of a semantic comparison."""

    produced: int = 0
    queued: int = 0
    started: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    empty: int = 0
    max_queue_depth: int = 0
    transcription_seconds: float = 0.0
    e2e_seconds: float = 0.0
    last_media_ts: float | None = None
    last_progress_mono: float | None = None
    last_progress_wallclock: float | None = None
    latest_completed_media_ts: float | None = None

    def describe(self) -> dict[str, Any]:
        # ``started`` includes silence skips, so latency is averaged over real
        # calls only.
        real = self.completed + self.failed
        finished = real + self.empty

        def opt(value: float | None) -> float | None:
            return None if value is None else round(value, 3)

        return {
            "produced": self.produced,
            "queued": self.queued,
            "started": self.started,
            "completed": self.completed,
            "failed": self.failed,
            "dropped": self.dropped,
            "empty": self.empty,
            "max_queue_depth": self.max_queue_depth,
            "mean_transcription_s": round(self.transcription_seconds / real, 3) if real else None,
            "mean_e2e_s": round(self.e2e_seconds / finished, 3) if finished else None,
            "last_media_ts": opt(self.last_media_ts),
            "latest_completed_media_ts": opt(self.latest_completed_media_ts),
            "stt_calls": real,
        }


@dataclass
class AudioJob:
    """One captured chunk waiting for, or in, transcription."""

    ticket: int
    chunk: AudioChunk
    silent: bool
    enqueued_mono: float
    spill_path: Path | None = None
    transcribed: asyncio.Future[None] | None = None
    ingested: asyncio.Future[list[Any]] | None = None
    spill_lost: bool = False
    """The spilled PCM could not be read back (pruned or deleted file). The job
    has no audio left and must be finished as a drop, not transcribed."""
    probe: bool = False
    """Sent while speech is paused, only to learn whether STT is healthy."""
    media_start: float = field(init=False)
    media_end: float = field(init=False)
    """Session-clock span, fixed at construction: a spill empties the PCM and
    with it ``chunk.duration``."""

    def __post_init__(self) -> None:
        self.media_start = self.chunk.ts
        self.media_end = self.chunk.ts + self.chunk.duration

    def ensure_futures(self, loop: asyncio.AbstractEventLoop) -> None:
        if self.transcribed is None or self.transcribed.get_loop() is not loop:
            self.transcribed = loop.create_future()
        if self.ingested is None or self.ingested.get_loop() is not loop:
            self.ingested = loop.create_future()

    def mark_transcribed(self) -> None:
        if self.transcribed is not None and not self.transcribed.done():
            self.transcribed.set_result(None)

    def mark_ingested(self, events: list[Any]) -> None:
        self.mark_transcribed()
        if self.ingested is not None and not self.ingested.done():
            self.ingested.set_result(events)


class AudioJobQueue:
    """Bounded waiting room for :class:`AudioJob`.

    ``memory_size`` jobs may keep PCM in RAM. Further jobs, up to ``spill_size``,
    write PCM to ``spill_dir`` (a fresh ``lst-stt-*`` temp directory when not
    given) and keep only metadata. Workers get the bytes back just before
    transcription. Capacity is the waiting room, not in-flight work: a worker
    that has already taken a job does not occupy a slot.
    """

    def __init__(
        self,
        memory_size: int = DEFAULT_STT_QUEUE_SIZE,
        spill_size: int = DEFAULT_STT_SPILL_CHUNKS,
        spill_dir: str | Path | None = None,
    ) -> None:
        if memory_size < 1:
            raise ValueError("memory_size must be >= 1")
        if spill_size < 0:
            raise ValueError("spill_size must be >= 0")
        self.memory_size = memory_size
        self.spill_size = spill_size
        self._waiting: list[AudioJob] = []
        self._not_empty = asyncio.Event()
        self._space = asyncio.Event()
        self._space.set()
        self._closed = False
        # Producers take turns: a spill is written off the event loop, and two puts
        # interleaving around it would reorder the waiting room.
        self._put_lock = asyncio.Lock()
        self.dropped = 0
        self.stale_evicted = 0
        self.max_depth = 0
        self._owned_spill = spill_dir is None and spill_size > 0
        self.spill_dir: Path | None
        if spill_size > 0:
            self.spill_dir = (
                Path(spill_dir)
                if spill_dir is not None
                else Path(tempfile.mkdtemp(prefix=SPILL_DIR_PREFIX))
            )
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            self.orphans_pruned = _prune_spill_orphans(self.spill_dir)
        else:
            self.spill_dir = None
            self.orphans_pruned = 0

    @property
    def capacity(self) -> int:
        return self.memory_size + self.spill_size

    def qsize(self) -> int:
        return len(self._waiting)

    def empty(self) -> bool:
        return not self._waiting

    def oldest_media_ts(self) -> float | None:
        return self._waiting[0].chunk.media_ts if self._waiting else None

    def _in_ram(self) -> int:
        return sum(1 for job in self._waiting if job.spill_path is None)

    def _spilled(self) -> int:
        return sum(1 for job in self._waiting if job.spill_path is not None)

    def _room(self) -> bool:
        return len(self._waiting) < self.capacity

    async def _accept(self, job: AudioJob) -> None:
        if self._in_ram() >= self.memory_size:
            await self._spill(job)
        self._waiting.append(job)
        self.max_depth = max(self.max_depth, len(self._waiting))
        self._not_empty.set()
        if not self._room():
            self._space.clear()

    async def _spill(self, job: AudioJob) -> None:
        """Move the job's PCM to disk. The write runs in a thread: a slow disk must not
        stall the event loop, which is also reading the live stream."""
        if self.spill_dir is None:
            return
        path = self.spill_dir / f"{job.ticket}_{job.chunk.index}_{job.chunk.segment}.pcm"
        await asyncio.to_thread(path.write_bytes, job.chunk.pcm)
        _release_pcm(job)
        job.spill_path = path

    async def _restore(self, job: AudioJob) -> None:
        if job.spill_path is None:
            return
        try:
            job.chunk.pcm = await asyncio.to_thread(job.spill_path.read_bytes)
        except asyncio.CancelledError:
            self._discard_pcm(job)  # a cancelled worker must not leave the file behind
            raise
        except OSError:
            # A vanished spill file must not raise out of ``get()`` and kill the
            # worker with the job's media-order ticket still open.
            job.chunk.pcm = b""
            job.spill_lost = True
        job.spill_path.unlink(missing_ok=True)
        job.spill_path = None

    def _discard_pcm(self, job: AudioJob) -> None:
        if job.spill_path is not None:
            job.spill_path.unlink(missing_ok=True)
            job.spill_path = None
        _release_pcm(job)

    def _evict_oldest(self) -> AudioJob:
        evicted = self._waiting.pop(0)
        self._discard_pcm(evicted)
        self.dropped += 1
        self._space.set()
        return evicted

    def evict_older_than(self, cutoff: float) -> list[AudioJob]:
        """Take every waiting job whose media span ended strictly before ``cutoff``.

        Their PCM is discarded without being read back from the spill. The
        caller finishes them (counted, never silent). ``math.inf`` empties the
        waiting room.
        """
        gone = [job for job in self._waiting if job.media_end < cutoff]
        if not gone:
            return []
        self._waiting = [job for job in self._waiting if job.media_end >= cutoff]
        for job in gone:
            self._discard_pcm(job)
        self.stale_evicted += len(gone)
        self._space.set()
        if not self._waiting:
            self._not_empty.clear()
        return gone

    async def put(self, job: AudioJob, *, wait: bool) -> AudioJob | None:
        """Enqueue ``job``. Returns an evicted job (or ``job`` itself if closed), else None.

        ``wait=True`` (replay) blocks until there is room and never drops.
        ``wait=False`` (live) evicts the oldest waiter when full.
        """
        async with self._put_lock:
            while not self._closed and not self._room():
                if wait:
                    self._space.clear()
                    await self._space.wait()
                    continue
                evicted = self._evict_oldest()
                await self._accept(job)
                return evicted
            if self._closed:
                return job
            await self._accept(job)
            return None

    async def get(self) -> AudioJob | None:
        """Take the oldest waiter. ``None`` once the queue is closed and empty."""
        while not self._waiting:
            if self._closed:
                return None
            self._not_empty.clear()
            await self._not_empty.wait()
        job = self._waiting.pop(0)
        self._space.set()
        if not self._waiting:
            self._not_empty.clear()
        await self._restore(job)
        return job

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Stop accepting jobs and wake every waiting producer and consumer."""
        self._closed = True
        self._not_empty.set()
        self._space.set()

    def cleanup(self) -> None:
        """Drop leftover PCM files and, for a self-made directory, the directory. Idempotent."""
        for job in self._waiting:
            self._discard_pcm(job)
        self._waiting.clear()
        if self._owned_spill and self.spill_dir is not None and self.spill_dir.is_dir():
            for leftover in self.spill_dir.glob("*.pcm"):
                leftover.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                self.spill_dir.rmdir()

    def describe(self) -> dict[str, Any]:
        oldest = self.oldest_media_ts()
        return {
            "depth": self.qsize(),
            "max_depth": self.max_depth,
            "capacity": self.capacity,
            "memory_size": self.memory_size,
            "spill_size": self.spill_size,
            "in_ram": self._in_ram(),
            "spilled": self._spilled(),
            "dropped": self.dropped,
            "stale_evicted": self.stale_evicted,
            "oldest_media_ts": None if oldest is None else round(oldest, 3),
        }
