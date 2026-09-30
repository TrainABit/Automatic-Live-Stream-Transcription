"""The audio job queue: RAM plus disk spill, eviction, replay back-pressure."""

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path

import pytest

from livestream_transcriber.audio.lane import (
    SPILL_DIR_PREFIX,
    AudioJob,
    AudioJobQueue,
    AudioLaneStats,
)
from tests.support.audio import chunk, tone


def job(ticket: int, ts: float | None = None, seconds: float = 2.5) -> AudioJob:
    start = ticket * 2.5 if ts is None else ts
    return AudioJob(
        ticket=ticket,
        chunk=chunk(start, tone(seconds), index=ticket),
        silent=False,
        enqueued_mono=0.0,
    )


def age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


# ------------------------------------------------------------------ spilling --


async def test_jobs_beyond_the_ram_budget_spill_and_come_back_intact(tmp_path):
    queue = AudioJobQueue(memory_size=1, spill_size=4, spill_dir=tmp_path)
    jobs = [job(i) for i in range(3)]
    original = jobs[2].chunk.pcm
    for j in jobs:
        assert await queue.put(j, wait=False) is None
    assert [j.spill_path is not None for j in jobs] == [False, True, True]
    assert len(list(tmp_path.glob("*.pcm"))) == 2
    assert queue.describe()["in_ram"] == 1 and queue.describe()["spilled"] == 2
    got = [await queue.get() for _ in range(3)]
    assert got == jobs
    assert got[2].chunk.pcm == original and not got[2].spill_lost
    assert not list(tmp_path.glob("*.pcm"))


async def test_a_spill_does_not_blank_a_chunk_someone_else_holds(tmp_path):
    queue = AudioJobQueue(memory_size=1, spill_size=2, spill_dir=tmp_path)
    await queue.put(job(0), wait=False)
    second = job(1)
    shared = second.chunk
    await queue.put(second, wait=False)
    assert second.spill_path is not None
    assert shared.pcm != b"" and shared.peak_dbfs() > -60  # the recorder still sees the audio
    assert second.chunk.pcm == b""
    assert second.media_end == pytest.approx(2.5 + 2.5)  # span survives the spill


async def test_a_vanished_spill_file_is_reported_not_raised(tmp_path):
    queue = AudioJobQueue(memory_size=1, spill_size=4, spill_dir=tmp_path)
    first, second = job(0), job(1)
    await queue.put(first, wait=False)
    await queue.put(second, wait=False)
    assert second.spill_path is not None
    second.spill_path.unlink()
    assert (await queue.get()) is first
    got = await queue.get()
    assert got is second and got.spill_lost is True and got.chunk.pcm == b""


def test_the_default_spill_directory_is_a_private_temp_dir():
    queue = AudioJobQueue(memory_size=1, spill_size=2)
    try:
        assert queue.spill_dir is not None
        assert queue.spill_dir.name.startswith(SPILL_DIR_PREFIX) and SPILL_DIR_PREFIX == "lst-stt-"
        assert queue.spill_dir.is_dir()
    finally:
        queue.cleanup()
    assert queue.spill_dir is not None and not queue.spill_dir.exists()


def test_without_spill_there_is_no_directory():
    queue = AudioJobQueue(memory_size=2, spill_size=0)
    assert queue.spill_dir is None and queue.capacity == 2
    queue.cleanup()


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError):
        AudioJobQueue(memory_size=0)
    with pytest.raises(ValueError):
        AudioJobQueue(spill_size=-1)


# ------------------------------------------------------------ orphan pruning --


def test_the_queue_prunes_its_own_stale_spill_files(tmp_path):
    orphan = tmp_path / "17_42_0.pcm"
    orphan.write_bytes(b"\x00" * 80_000)
    age(orphan, 3 * 3600)
    fresh = tmp_path / "18_43_0.pcm"
    fresh.write_bytes(b"\x00" * 80_000)
    stranger = tmp_path / "notes.pcm"  # not the queue's naming scheme
    stranger.write_bytes(b"keep me")
    age(stranger, 3 * 3600)
    queue = AudioJobQueue(memory_size=1, spill_size=4, spill_dir=tmp_path)
    assert not orphan.exists()
    assert fresh.exists()  # could belong to a live queue
    assert stranger.exists()
    assert queue.orphans_pruned == 1


# ---------------------------------------------------------------- live vs replay --


async def test_live_mode_drops_the_oldest_waiter_when_full(tmp_path):
    queue = AudioJobQueue(memory_size=1, spill_size=1, spill_dir=tmp_path)
    jobs = [job(i) for i in range(3)]
    assert await queue.put(jobs[0], wait=False) is None
    assert await queue.put(jobs[1], wait=False) is None
    evicted = await queue.put(jobs[2], wait=False)
    assert evicted is jobs[0]
    assert queue.dropped == 1 and queue.qsize() == 2
    assert (await queue.get()) is jobs[1]
    assert (await queue.get()) is jobs[2]
    assert not list(tmp_path.glob("*.pcm"))


async def test_replay_mode_waits_instead_of_dropping():
    queue = AudioJobQueue(memory_size=1, spill_size=0)
    first, second = job(0), job(1)
    await queue.put(first, wait=True)
    putter = asyncio.create_task(queue.put(second, wait=True))
    await asyncio.sleep(0.05)
    assert not putter.done() and queue.dropped == 0
    assert (await queue.get()) is first
    assert await asyncio.wait_for(putter, 5) is None
    assert (await queue.get()) is second and queue.dropped == 0


async def test_get_waits_for_a_job_and_close_wakes_it():
    queue = AudioJobQueue(memory_size=1, spill_size=0)
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0.02)
    assert not getter.done()
    queue.close()
    assert await asyncio.wait_for(getter, 5) is None
    assert queue.closed
    assert await queue.put(job(0), wait=False) is not None  # refused once closed


async def test_close_releases_a_blocked_producer():
    queue = AudioJobQueue(memory_size=1, spill_size=0)
    await queue.put(job(0), wait=True)
    blocked = job(1)
    putter = asyncio.create_task(queue.put(blocked, wait=True))
    await asyncio.sleep(0.02)
    queue.close()
    assert await asyncio.wait_for(putter, 5) is blocked


async def test_stale_jobs_are_evicted_by_cutoff_without_reading_the_spill(tmp_path):
    queue = AudioJobQueue(memory_size=1, spill_size=4, spill_dir=tmp_path)
    jobs = [job(i) for i in range(4)]  # spans 0-2.5, 2.5-5, 5-7.5, 7.5-10
    for j in jobs:
        await queue.put(j, wait=False)
    gone = queue.evict_older_than(5.0)  # strictly before: a span ending at 5.0 stays
    assert gone == jobs[:1]
    gone = queue.evict_older_than(7.5)
    assert gone == jobs[1:2]
    assert queue.stale_evicted == 2 and queue.qsize() == 2
    assert len(list(tmp_path.glob("*.pcm"))) == 2
    assert queue.evict_older_than(0.0) == []
    assert len(queue.evict_older_than(math.inf)) == 2 and queue.empty()
    assert not list(tmp_path.glob("*.pcm"))


async def test_cleanup_removes_leftover_files_and_is_idempotent(tmp_path):
    queue = AudioJobQueue(memory_size=1, spill_size=3, spill_dir=tmp_path)
    for i in range(3):
        await queue.put(job(i), wait=False)
    queue.cleanup()
    queue.cleanup()
    assert queue.empty() and not list(tmp_path.glob("*.pcm"))
    assert tmp_path.exists()  # a directory the caller gave is not removed


async def test_describe_reports_depth_and_oldest_media_time():
    queue = AudioJobQueue(memory_size=2, spill_size=0)
    assert queue.describe()["oldest_media_ts"] is None
    await queue.put(job(3), wait=False)
    await queue.put(job(4), wait=False)
    state = queue.describe()
    assert (state["depth"], state["max_depth"], state["capacity"]) == (2, 2, 2)
    assert state["oldest_media_ts"] == 7.5


async def test_futures_follow_the_running_loop():
    j = job(0)
    j.ensure_futures(asyncio.get_running_loop())
    events: list[str] = []
    j.mark_ingested(events)
    assert j.transcribed is not None and j.transcribed.done()
    assert j.ingested is not None and (await j.ingested) is events
    j.mark_ingested([])  # a second call is a no-op, not an InvalidStateError


def test_lane_stats_describe_averages_over_real_calls_only():
    stats = AudioLaneStats(
        started=6, completed=2, failed=1, empty=3, transcription_seconds=3.0, e2e_seconds=12.0
    )
    state = stats.describe()
    assert state["mean_transcription_s"] == 1.0
    assert state["mean_e2e_s"] == 2.0
    assert state["stt_calls"] == 3
    assert AudioLaneStats().describe()["mean_transcription_s"] is None
