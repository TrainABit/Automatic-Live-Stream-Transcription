"""The transcription pipeline: audio chunks in, transcripts, rule hits and alerts out.

::

    on_audio ─> AudioJobQueue ─> STT workers ─> MediaOrderGate ─> SpeechStream ─> sinks
                (RAM + spill)    (thread pool)  (media order)     (dedup, stitch)   │
                                                                                    v
                                                              rules ─> notifications

Each arrow is a place where something can be slow, so each one is decoupled:

* capture never waits for a transcription. ``on_audio`` announces the chunk in
  media order and hands it to a bounded queue; live capture drops the *oldest*
  waiting chunk when that is full (counted, logged), replay waits instead;
* transcription runs on a dedicated thread pool. Results come back in any order,
  and the :class:`~..audio.ordering.MediaOrderGate` releases them strictly in media
  order, so the stitcher and the rules only ever see the recording's order;
* rule evaluation (which may call an LLM) and notification delivery (network I/O)
  run in their own stages behind small queues. A slow webhook delays alerts, never
  transcripts;
* an :class:`~.overload.OverloadGuard` pauses transcription when it cannot keep up
  with a live stream and probes for recovery.

Shutdown is :meth:`TranscriptionPipeline.drain`: let everything already accepted
finish (or abandon it), in the order the stages depend on each other.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import time
from collections import deque
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audio.lane import (
    DEFAULT_STT_QUEUE_SIZE,
    DEFAULT_STT_SPILL_CHUNKS,
    DEFAULT_STT_WORKERS,
    AudioJob,
    AudioJobQueue,
    AudioLaneStats,
)
from ..audio.ordering import MediaOrderGate, Slot
from ..audio.speech import IngestResult, SpeechStream
from ..audio.timing import align_match_timing, sessionize_transcript_timestamps
from ..logging_setup import get_logger
from ..models import AudioChunk
from ..notify.base import Event
from ..outputs.base import CompositeSink, TranscriptSegment, TranscriptSink
from ..process_priority import apply_thread_nice
from ..resilience.stt_health import SttOutageMonitor
from ..rules.engine import RuleEngine, RuleHit, TextSegment
from ..stt.base import SILENCE_DBFS, Transcriber, Transcript, transcribe_chunk
from ..stt.factory import set_fallback_enabled, transcriber_health
from ..textnorm import norm_token
from .overload import REASON_OUTAGE, OverloadConfig, OverloadGuard

if TYPE_CHECKING:
    from ..config import Settings
    from ..notify.dispatcher import NotificationDispatcher
    from ..stream.auto_resume import AlertFn

log = get_logger(__name__)

__all__ = ["PipelineOptions", "TranscriptionPipeline", "utterance_delta"]

# Why a job ended without a transcript. Kept as plain strings: they end up in logs.
REASON_QUEUE_DROPPED = "queue_dropped"
REASON_SPILL_LOST = "spill_lost"
REASON_PAUSED = "stt_paused"
REASON_FAILED = "stt_failed"

# Words of the previous utterance handed to the rules together with a new one, so a
# phrase that a chunk boundary cut in two ("new" | "release") still matches.
CARRY_WORDS = 8

# How long the rule and notification stages get to finish once transcription is done.
DEFAULT_FLUSH_SECONDS = 20.0

_STOP = object()


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Tunables of the pipeline, independent of how they were configured."""

    workers: int = DEFAULT_STT_WORKERS
    queue_size: int = DEFAULT_STT_QUEUE_SIZE
    spill_chunks: int = DEFAULT_STT_SPILL_CHUNKS
    spill_dir: Path | None = None
    skip_silence: bool = True
    """Never send digital silence or room tone to the provider."""
    silence_dbfs: float = SILENCE_DBFS
    lossless: bool = False
    """Replay and files: ``on_audio`` waits for room instead of dropping the oldest chunk."""
    overload: OverloadConfig | None = None
    """Lag/drop guard for live capture; ``None`` disables it (lossless runs)."""
    nice: int = 0
    """Niceness of the STT worker threads; 0 leaves priority alone."""
    outage_seconds: float = 300.0
    health_lag_seconds: float = 30.0
    health_drop_ratio: float = 0.05
    notify_queue_size: int = 1024
    retry_interval_seconds: float = 60.0
    """How often stored, undelivered events are retried."""
    flush_seconds: float = DEFAULT_FLUSH_SECONDS

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if self.spill_chunks < 0:
            raise ValueError("spill_chunks must be >= 0")

    @classmethod
    def from_settings(cls, settings: Settings, *, live: bool) -> PipelineOptions:
        """Options for a live run (drop-oldest, guarded) or a finite one (lossless)."""
        return cls(
            workers=settings.stt_workers,
            queue_size=settings.stt_queue_size,
            spill_chunks=settings.stt_spill_chunks,
            lossless=not live,
            overload=OverloadConfig.from_settings(settings) if live else None,
            nice=settings.stt_nice,
            outage_seconds=settings.stt_outage_seconds,
            health_lag_seconds=settings.stt_health_lag_seconds,
            health_drop_ratio=settings.stt_health_drop_ratio,
        )


@dataclass(slots=True)
class _SpeechStats:
    """What happened to transcripts after they were put in media order."""

    transcripts: int = 0
    utterances: int = 0
    duplicates: int = 0
    revisions: int = 0
    gaps: int = 0
    """Chunks in media order that produced no transcript because STT failed or dropped them."""
    hits: int = 0
    events: int = 0
    notify_failures: int = 0
    sink_errors: int = 0

    def describe(self) -> dict[str, int]:
        return {
            "transcripts": self.transcripts,
            "utterances": self.utterances,
            "duplicates_suppressed": self.duplicates,
            "revisions": self.revisions,
            "gaps": self.gaps,
            "rule_hits": self.hits,
            "events": self.events,
            "notify_failures": self.notify_failures,
            "sink_errors": self.sink_errors,
        }


@dataclass(slots=True)
class _RuleWork:
    segment: TranscriptSegment
    transcript: Transcript
    carry: str


@dataclass(slots=True)
class _Stage:
    """A queue, and the task that consumes it."""

    queue: asyncio.Queue[Any]
    task: asyncio.Task[None] | None = None


def utterance_delta(previous: str | None, merged: str) -> str:
    """The words of ``merged`` that the ``previous`` utterance did not already say.

    The stream stitcher reports an extension or overlap as a *revision* of the
    previous utterance. Downstream consumers (subtitle cues, rule matching) must not
    see the shared words twice, so they get the new words only. When ``previous``
    is a word-for-word prefix of ``merged`` that is everything after it; otherwise
    (a re-worded repeat) it is the words whose normalised form ``previous`` lacks.
    """
    if not previous:
        return merged.strip()
    merged_words = merged.split()
    prev_keys = [key for key in (norm_token(w) for w in previous.split()) if key]
    consumed = 0
    matched = 0
    for word in merged_words:
        key = norm_token(word)
        consumed += 1
        if not key:
            continue
        if matched < len(prev_keys) and key == prev_keys[matched]:
            matched += 1
            if matched == len(prev_keys):
                return " ".join(merged_words[consumed:])
        else:
            break
    known = set(prev_keys)
    return " ".join(w for w in merged_words if (k := norm_token(w)) and k not in known)


class TranscriptionPipeline:
    """Wire capture, speech-to-text, rules and notifications together.

    ``transcriber=None`` switches speech off: chunks are counted and dropped, which is
    what ``lst record`` wants. ``sinks`` receive final utterances in media order.
    ``rules`` and ``dispatcher`` are optional; with rules but no dispatcher, hits are
    only counted and passed to ``on_hit``.
    """

    def __init__(
        self,
        transcriber: Transcriber | None,
        *,
        options: PipelineOptions | None = None,
        sinks: Sequence[TranscriptSink] = (),
        rules: RuleEngine | None = None,
        dispatcher: NotificationDispatcher | None = None,
        alert: AlertFn | None = None,
        source_url: str | None = None,
        session_id: int | None = None,
        on_hit: Callable[[RuleHit], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transcriber = transcriber
        self.options = options or PipelineOptions()
        self.sink = CompositeSink(list(sinks))
        self.rules = rules
        self.dispatcher = dispatcher
        self.alert = alert
        self.source_url = source_url
        self.session_id = session_id
        self.on_hit = on_hit
        self._clock = clock

        opts = self.options
        self.stats = AudioLaneStats()
        self.speech_stats = _SpeechStats()
        self.order: MediaOrderGate[Transcript] = MediaOrderGate()
        self.speech = SpeechStream()
        self.guard: OverloadGuard | None = (
            OverloadGuard(opts.overload, clock=clock) if opts.overload is not None else None
        )
        self._outage = SttOutageMonitor(opts.outage_seconds)
        self._queue = AudioJobQueue(opts.queue_size, opts.spill_chunks, opts.spill_dir)
        self._jobs: dict[int, AudioJob] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._in_flight = 0
        self._ingest_lock = asyncio.Lock()
        self._drain_lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._audio_head: float | None = None
        self._drained = False
        self._closed_intake = False
        self._carry: deque[str] = deque(maxlen=CARRY_WORDS)
        self._carry_end = -math.inf
        self._rules_stage: _Stage | None = None
        self._notify_stage: _Stage | None = None
        self._last_degraded: float | None = None
        self._cost_usd = 0.0
        self._skipped_paused = 0
        self._background: set[asyncio.Task[None]] = set()
        self.last_error: str | None = None

    # ------------------------------------------------------------------ intake

    async def on_audio(self, chunk: AudioChunk, *, wait: bool = False) -> list[TranscriptSegment]:
        """Accept one captured chunk. Transcription happens off this call.

        Capture passes ``wait=False`` so a slow provider can never stall ffmpeg. With
        ``wait=True`` the call returns once this chunk's transcript (and every earlier
        one) has been written to the sinks, and returns the utterances it produced.
        """
        if self._closed_intake:
            return []
        now = self._clock()
        self.stats.produced += 1
        self.stats.last_media_ts = chunk.media_ts
        self.stats.last_progress_mono = now
        self.stats.last_progress_wallclock = chunk.wallclock
        end = chunk.ts + chunk.duration
        self._audio_head = end if self._audio_head is None else max(self._audio_head, end)
        if self.transcriber is None:
            return []

        probe = False
        guard = self.guard
        if guard is not None and guard.paused:
            # Speech is switched off: nothing is transcribed except, now and then,
            # one audible chunk that asks whether the provider is back.
            if not guard.take_probe(audible=chunk.peak_dbfs() >= self.options.silence_dbfs):
                self._skipped_paused += 1
                return []
            probe = True
        else:
            self._release_fallback_hold()

        job = await self._submit(chunk, probe=probe)
        await self._check_overload()
        if not wait or job.ingested is None:
            return []
        result: list[TranscriptSegment] = await job.ingested
        return result

    async def _submit(self, chunk: AudioChunk, *, probe: bool) -> AudioJob:
        """Announce the chunk in media order, then enqueue it for a worker.

        ``MediaOrderGate.submit`` is synchronous and happens before the first await, so
        chunks are announced in the order they entered this method. Capture calls it
        from one task, which makes announcement order media order.
        """
        loop = asyncio.get_running_loop()
        self._ensure_workers()
        opts = self.options
        ticket = self.order.submit(chunk.ts, chunk.ts + chunk.duration)
        silent = opts.skip_silence and chunk.peak_dbfs() < opts.silence_dbfs
        job = AudioJob(
            ticket=ticket,
            chunk=chunk,
            silent=silent,
            enqueued_mono=self._clock(),
            probe=probe,
        )
        job.ensure_futures(loop)
        self._jobs[ticket] = job
        self.stats.queued += 1
        if silent:
            # Nothing to transcribe. The ticket still has to close so later chunks
            # are released, but the job does not need a place in the queue.
            log.debug("stt skip silence", extra=chunk.describe())
            self.stats.empty += 1
            await self._finish(job, transcript=None, unavailable=False, dropped=False)
            return job

        evicted = await self._queue.put(job, wait=opts.lossless)
        self.stats.max_queue_depth = max(self.stats.max_queue_depth, self._queue.qsize())
        if evicted is not None:
            await self._finish(
                evicted,
                transcript=None,
                unavailable=True,
                dropped=True,
                reason=REASON_QUEUE_DROPPED,
            )
        return job

    def _ensure_workers(self) -> None:
        self._workers = [task for task in self._workers if not task.done()]
        if self._workers or self._closed_intake:
            return
        if self._executor is None:
            nice = self.options.nice
            self._executor = ThreadPoolExecutor(
                max_workers=self.options.workers,
                thread_name_prefix="lst-stt",
                initializer=functools.partial(apply_thread_nice, nice) if nice else None,
            )
        loop = asyncio.get_running_loop()
        for index in range(self.options.workers):
            self._workers.append(loop.create_task(self._worker(), name=f"stt-{index}"))

    # ------------------------------------------------------------- transcribing

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                return
            self._in_flight += 1
            try:
                await self._transcribe(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A bug in the bookkeeping must not leave the media-order ticket open:
                # every later chunk would wait behind it forever.
                log.exception("stt worker failed", extra={"start": round(job.media_start, 2)})
                await self._finish(
                    job, transcript=None, unavailable=True, dropped=False, reason=REASON_FAILED
                )
            finally:
                self._in_flight -= 1

    async def _transcribe(self, job: AudioJob) -> None:
        if job.spill_lost:
            # Its PCM is gone (a pruned or deleted spill file): nothing to transcribe,
            # but the ticket must close.
            await self._finish(
                job,
                transcript=None,
                unavailable=True,
                dropped=True,
                reason=REASON_SPILL_LOST,
            )
            return
        transcriber = self.transcriber
        assert transcriber is not None
        self.stats.started += 1
        loop = asyncio.get_running_loop()
        started = self._clock()
        transcript: Transcript | None = None
        unavailable = False
        try:
            transcript = await loop.run_in_executor(
                self._executor,
                functools.partial(transcribe_chunk, transcriber, job.chunk, skip_silence=False),
            )
        except Exception as exc:
            log.exception("stt call raised", extra={"start": round(job.media_start, 2)})
            self.last_error = f"{type(exc).__name__}: {exc}"
            unavailable = True
        elapsed = self._clock() - started
        self.stats.transcription_seconds += elapsed

        if unavailable or (transcript is not None and transcript.unavailable):
            unavailable, transcript = True, None
            self.stats.failed += 1
        elif transcript is None or not transcript.text.strip():
            transcript = None
            self.stats.empty += 1
        else:
            self.stats.completed += 1

        degraded = transcript is not None and transcript.degraded
        if degraded:
            self._last_degraded = self._clock()
        if unavailable or degraded:
            if self._outage.record_failure(self._clock()):
                self._spawn(self._on_outage(), name="stt-outage")
        else:
            self._outage.record_success()

        if job.probe and self.guard is not None:
            healthy = not unavailable and elapsed <= self._probe_budget(job)
            if healthy:
                self._resume(job)
        await self._finish(job, transcript=transcript, unavailable=unavailable, dropped=False)

    def _probe_budget(self, job: AudioJob) -> float:
        """How long a probe may take and still count as healthy.

        One request is one noisy sample, so the bar is generous: the probe passes if its
        answer would still be useful, within half the lag limit (or one chunk length per
        worker when the lag guard is off). A provider that answers in time but is too
        slow on average is caught by the lag guard again, and the relapse back-off keeps
        that from flapping.
        """
        chunk_s = max(1.0, job.media_end - job.media_start)
        limit = self.options.overload.max_lag_seconds if self.options.overload else 0.0
        return max(chunk_s * self.options.workers, limit / 2.0)

    async def _finish(
        self,
        job: AudioJob,
        *,
        transcript: Transcript | None,
        unavailable: bool,
        dropped: bool,
        reason: str | None = None,
    ) -> None:
        """Close a job: account for it, release it in media order and ingest what is released."""
        now = self._clock()
        guard = self.guard
        if job.probe and guard is not None:
            # Every way a probe can end passes through here, so the next probe is never
            # blocked by one that ended without a verdict (evicted, overflowed, spill lost).
            guard.probe_ended(no_verdict=dropped)
        if dropped:
            self.stats.dropped += 1
            reason = reason or REASON_QUEUE_DROPPED
            log.warning(
                "audio chunk dropped before transcription",
                extra={
                    "reason": reason,
                    "start": round(job.media_start, 2),
                    "dropped_total": self.stats.dropped,
                },
            )
        if guard is not None and not guard.paused and reason != REASON_PAUSED:
            guard.record_outcome(dropped)
        self.stats.e2e_seconds += now - job.enqueued_mono
        done_ts = job.chunk.media_ts
        latest = self.stats.latest_completed_media_ts
        if latest is None or done_ts > latest:
            self.stats.latest_completed_media_ts = done_ts
        job.mark_transcribed()

        released_by_ticket: dict[int, list[TranscriptSegment]] = {}
        async with self._ingest_lock:
            released = self.order.complete(
                job.ticket,
                transcript,
                chunk=job.chunk,
                unavailable=unavailable,
                dropped=dropped,
                reason=reason,
            )
            for slot in released:
                released_by_ticket[slot.ticket] = await self._ingest(slot)
        for ticket, segments in released_by_ticket.items():
            done = self._jobs.pop(ticket, None)
            if done is not None:
                done.mark_ingested(segments)

    # ---------------------------------------------------------------- ingesting

    async def _ingest(self, slot: Slot[Transcript]) -> list[TranscriptSegment]:
        """One chunk's result, in media order, whatever order it came back in."""
        stats = self.speech_stats
        if slot.extra.get("unavailable") or slot.extra.get("dropped"):
            stats.gaps += 1
            return []
        transcript = slot.payload
        if transcript is None:
            return []
        transcript = sessionize_transcript_timestamps(transcript)
        stats.transcripts += 1
        if transcript.cost_usd:
            self._cost_usd += transcript.cost_usd
        log.debug("transcript received", extra=transcript.describe())

        previous = self.speech.normalized[-1].text if self.speech.normalized else None
        result: IngestResult = self.speech.ingest(transcript)
        if result.transcript is None:
            stats.duplicates += 1
            log.debug("duplicate transcript suppressed", extra=transcript.describe())
            return []
        text = utterance_delta(previous, result.transcript.text) if result.revision else None
        if result.revision:
            stats.revisions += 1
            if not text:
                stats.duplicates += 1
                return []
        segment = TranscriptSegment.from_transcript(
            # A delta is only the new words: the provider's segment times describe the whole
            # chunk, so they must not be attached to it.
            replace(transcript, text=text, segments=None)
            if text is not None
            else result.transcript,
            language=transcript.language,
        )
        stats.utterances += 1
        try:
            await asyncio.to_thread(self.sink.write, segment)
        except Exception:
            stats.sink_errors += 1
            log.exception("writing an utterance failed")
        self._queue_rules(segment, transcript)
        return [segment]

    def _queue_rules(self, segment: TranscriptSegment, transcript: Transcript) -> None:
        carry = ""
        if segment.start - self._carry_end <= self.speech.max_gap:
            carry = " ".join(self._carry)
        self._carry.extend(segment.text.split())
        self._carry_end = segment.end
        if self.rules is None:
            return
        stage = self._ensure_rules_stage()
        stage.queue.put_nowait(_RuleWork(segment, transcript, carry))

    # -------------------------------------------------------------- rules stage

    def _ensure_rules_stage(self) -> _Stage:
        if self._rules_stage is None:
            stage = _Stage(asyncio.Queue())
            stage.task = asyncio.get_running_loop().create_task(
                self._rules_loop(stage.queue), name="rules"
            )
            self._rules_stage = stage
        return self._rules_stage

    async def _rules_loop(self, queue: asyncio.Queue[Any]) -> None:
        while True:
            work = await queue.get()
            if work is _STOP:
                return
            try:
                await self._evaluate(work)
            except Exception:
                log.exception("rule evaluation failed")

    async def _evaluate(self, work: _RuleWork) -> None:
        rules = self.rules
        if rules is None:
            return
        seg = work.segment
        body = seg.text.strip()
        text = f"{work.carry} {body}" if work.carry else body
        offset = len(work.carry) + 1 if work.carry else 0
        hits = await rules.evaluate_async(
            TextSegment(
                text=text,
                start=seg.start,
                end=seg.end,
                language=seg.language,
                confidence=seg.confidence,
            )
        )
        for hit in hits:
            # A match that lies wholly in the carried words was already reported when
            # those words were new.
            if hit.span[1] <= offset:
                continue
            start, end = align_match_timing(
                hit.start, hit.end, work.transcript, matched_text=hit.matched_text
            )
            hit = replace(hit, start=start, end=end)
            self.speech_stats.hits += 1
            log.info(
                "rule hit",
                extra={"rule": hit.rule_id, "severity": hit.severity.value, "at": round(start, 2)},
            )
            if self.on_hit is not None:
                try:
                    self.on_hit(hit)
                except Exception:
                    log.exception("on_hit callback failed")
            if self.dispatcher is not None:
                event = Event.from_hit(hit, source_url=self.source_url, session_id=self.session_id)
                await self._ensure_notify_stage().queue.put(event)

    # ----------------------------------------------------------- notify stage

    def _ensure_notify_stage(self) -> _Stage:
        if self._notify_stage is None:
            stage = _Stage(asyncio.Queue(self.options.notify_queue_size))
            stage.task = asyncio.get_running_loop().create_task(
                self._notify_loop(stage.queue), name="notify"
            )
            self._notify_stage = stage
        return self._notify_stage

    async def _notify_loop(self, queue: asyncio.Queue[Any]) -> None:
        dispatcher = self.dispatcher
        assert dispatcher is not None
        interval = self.options.retry_interval_seconds
        retry = dispatcher.store is not None and interval > 0
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), interval if retry else None)
            except TimeoutError:
                # Idle: a good moment to retry what an earlier run or a transient
                # failure left undelivered.
                with contextlib.suppress(Exception):
                    await dispatcher.retry_pending_async()
                continue
            if event is _STOP:
                return
            try:
                result = await dispatcher.dispatch_async(event)
            except Exception:
                log.exception("dispatching an event failed", extra={"event_id": event.event_id})
                self.speech_stats.notify_failures += 1
                continue
            if not result.duplicate:
                self.speech_stats.events += 1
            if result.failed:
                self.speech_stats.notify_failures += 1

    # ---------------------------------------------------------------- overload

    def stt_lag_seconds(self) -> float | None:
        """How far the oldest chunk still owed a transcript trails the audio head.

        Zero when nothing is outstanding: a gap in the capture (no chunks arriving) is
        not STT falling behind. ``None`` before the first chunk. Measured against the
        audio head clock, so it works for audio-only sources. Read-only over a snapshot,
        so a health reporter may call it from another thread.
        """
        head = self._audio_head
        if head is None:
            return None
        oldest = min((job.media_start for job in list(self._jobs.values())), default=None)
        if oldest is None:
            return 0.0
        return max(0.0, head - oldest)

    def stt_drop_ratio(self) -> tuple[float | None, int]:
        """Dropped share of recently finished chunks, and how many chunks that covers."""
        if self.guard is None:
            return None, 0
        return self.guard.drop_ratio()

    async def _check_overload(self) -> None:
        guard = self.guard
        if guard is None or guard.paused or self.transcriber is None:
            return
        verdict = guard.evaluate(self.stt_lag_seconds())
        if verdict is not None:
            detail = verdict.describe()
            detail.pop("reason")
            await self._pause(verdict.reason, **detail)

    async def _pause(self, reason: str, **detail: Any) -> None:
        """Switch speech off until a healthy provider answers, dropping what is queued.

        Everything still waiting is behind by definition; it is dropped (counted, never
        silent) so the lane catches up with the live edge.
        """
        guard = self.guard
        if guard is None or guard.paused or self.transcriber is None:
            return
        health = transcriber_health(self.transcriber)
        fallback = health.get("fallback")
        fallback_active = bool(fallback and fallback.get("enabled"))
        interval = guard.pause(reason, fallback_active=fallback_active)
        # A fallback that cannot keep up would starve the rest of the process.
        set_fallback_enabled(self.transcriber, False)
        log.error(
            "stt paused: transcription is off until a provider answers again",
            extra={**detail, "reason": reason, "probe_every_s": round(interval, 1)},
        )
        await self._alert(
            "Speech-to-text paused",
            "Transcription cannot keep up with the stream"
            if reason != REASON_OUTAGE
            else "The speech-to-text provider has been failing",
            detail,
        )
        for job in self._queue.evict_older_than(math.inf):
            await self._finish(
                job, transcript=None, unavailable=True, dropped=True, reason=REASON_PAUSED
            )

    def _resume(self, probe: AudioJob) -> None:
        guard = self.guard
        if guard is None or not guard.paused:
            return
        hold = guard.resume()
        self._outage.record_success()
        if hold is None and self.transcriber is not None:
            set_fallback_enabled(self.transcriber, True)
        self._spawn(
            self._alert(
                "Speech-to-text resumed",
                "A provider answers again.",
                {"probe_at_s": round(probe.media_start, 1)},
            ),
            name="stt-resumed-alert",
        )

    def _spawn(self, coro: Coroutine[Any, Any, None], *, name: str) -> None:
        """Run ``coro`` in the background, keeping a reference so it is not collected."""
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _release_fallback_hold(self) -> None:
        guard = self.guard
        if guard is None or self.transcriber is None:
            return
        if guard.release_fallback_hold() and set_fallback_enabled(self.transcriber, True):
            log.warning("stt fallback back on after the post-overload hold")

    async def _on_outage(self) -> None:
        """The provider has failed for ``outage_seconds`` in a row.

        With a fallback still answering, the lane stays up (degraded) and only the alert
        goes out. Without one, live speech is paused, reversibly: a probe that gets an
        answer resumes it without a restart.
        """
        if self.transcriber is None:
            return
        outage_s = int(self.options.outage_seconds)
        recent = self._last_degraded
        if recent is not None and self._clock() - recent <= max(self.options.outage_seconds, 60.0):
            await self._alert(
                "Speech-to-text provider down",
                f"The primary provider has failed for {outage_s}s; the fallback is answering.",
                {"outage_s": outage_s},
            )
            return
        if self.guard is not None:
            await self._pause(REASON_OUTAGE, outage_s=outage_s)
        else:
            await self._alert(
                "Speech-to-text provider down",
                f"The provider has failed for {outage_s}s.",
                {"outage_s": outage_s},
            )

    async def _alert(self, title: str, body: str, detail: dict[str, Any]) -> None:
        if self.alert is None:
            return
        extra = ", ".join(f"{k}={v}" for k, v in detail.items() if v is not None)
        try:
            await self.alert(title, f"{body}\n{extra}" if extra else body)
        except Exception:
            log.exception("alert delivery failed", extra={"title": title})

    # -------------------------------------------------------------- introspection

    def summary(self) -> dict[str, Any]:
        """Counters for logs and the session record. Operational, never semantic."""
        out: dict[str, Any] = {
            "audio": {**self.stats.describe(), "skipped_while_paused": self._skipped_paused},
            "queue": self._queue.describe(),
            "order": self.order.describe(),
            "speech": self.speech_stats.describe(),
            "cost_usd": round(self._cost_usd, 6),
        }
        if self.rules is not None:
            out["rules"] = self.rules.describe()
        if self.guard is not None:
            out["overload"] = self.guard.describe()
        return out

    def health(self) -> dict[str, Any]:
        """A small, stable dict for the heartbeat: health, its reasons and the STT gauges."""
        lag = self.stt_lag_seconds()
        ratio, samples = self.stt_drop_ratio()
        reasons: list[str] = []
        opts = self.options
        paused = self.guard is not None and self.guard.paused
        if paused:
            reasons.append("stt_paused")
        if opts.health_lag_seconds > 0 and lag is not None and lag > opts.health_lag_seconds:
            reasons.append("stt_lag")
        if (
            opts.health_drop_ratio > 0
            and ratio is not None
            and samples >= 4
            and ratio > opts.health_drop_ratio
        ):
            reasons.append("stt_drops")
        outage_started = self._outage.describe().get("outage_started")
        outage_s = None if outage_started is None else self._clock() - float(outage_started)
        if outage_s is not None:
            reasons.append("stt_failing")
        out: dict[str, Any] = {
            "health": "degraded" if reasons else "ok",
            "health_reasons": reasons,
            "stt_lag_s": None if lag is None else round(lag, 1),
            "stt_drop_ratio": None if ratio is None else round(ratio, 3),
            "stt_paused": paused,
            "stt_outage_s": None if outage_s is None else round(outage_s, 1),
            "queue_depth": self._queue.qsize(),
            "chunks": self.stats.produced,
            "transcribed": self.stats.completed,
            "dropped": self.stats.dropped,
            "utterances": self.speech_stats.utterances,
            "rule_hits": self.speech_stats.hits,
        }
        if self.transcriber is not None:
            stt = transcriber_health(self.transcriber)
            out["stt_cost_usd"] = stt.get("cost_usd")
            out["cloud_successes"] = stt.get("cloud_successes")
        return out

    # ------------------------------------------------------------------ shutdown

    async def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every accepted chunk has been ingested. True unless ``timeout`` hit."""
        jobs = [job.ingested for job in list(self._jobs.values()) if job.ingested is not None]
        if not jobs:
            return True
        try:
            await asyncio.wait_for(asyncio.gather(*jobs), timeout)
        except TimeoutError:
            return False
        return True

    async def drain(self, *, wait: bool = True, timeout: float | None = None) -> None:
        """Finish or abandon everything accepted so far, in dependency order. Idempotent.

        1. stop taking audio;
        2. ``wait=True``: let queued chunks be transcribed and ingested (up to
           ``timeout``); ``wait=False`` (or a timeout) cancels the STT workers instead,
           so a wedged provider call cannot hold shutdown hostage;
        3. flush the rule and notification stages: whatever was already transcribed
           still gets its alerts, within ``flush_seconds``;
        4. release the thread pool and any spilled audio.
        """
        async with self._drain_lock:
            if self._drained:
                return
            self._closed_intake = True
            abandon = not wait
            if wait and not await self.wait_idle(timeout):
                log.warning(
                    "stt did not finish within the drain budget; abandoning the rest",
                    extra={"timeout_s": timeout, "outstanding": len(self._jobs)},
                )
                abandon = True
            self._queue.close()
            workers, self._workers = self._workers, []
            if abandon:
                for worker in workers:
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._fail_outstanding()
            if self._background:
                await asyncio.gather(*self._background, return_exceptions=True)
            await self._flush_stages()
            self._queue.cleanup()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            self._drained = True
            log.info("pipeline drained", extra=self.speech_stats.describe())

    def _fail_outstanding(self) -> None:
        """Complete leftover futures so nobody waiting on a job hangs after a cancel."""
        for ticket, job in list(self._jobs.items()):
            if job.transcribed is not None and not job.transcribed.done():
                job.transcribed.set_result(None)
            if job.ingested is not None and not job.ingested.done():
                job.ingested.set_result([])
            self._jobs.pop(ticket, None)

    async def _flush_stages(self) -> None:
        """Stop the rule stage, then the notification stage it feeds, each within the budget."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.options.flush_seconds
        # Read the notify stage only after the rule stage is done: it is created lazily by
        # the first hit, which may be one of the last ones the rule stage processes.
        await self._stop_stage(self._rules_stage, deadline)
        await self._stop_stage(self._notify_stage, deadline)

    async def _stop_stage(self, stage: _Stage | None, deadline: float) -> None:
        if stage is None or stage.task is None:
            return
        loop = asyncio.get_running_loop()
        if not stage.task.done():
            # The queue is bounded: with a consumer stuck on a hung endpoint even the stop
            # marker can block, so it is inside the budget too.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stage.queue.put(_STOP), max(0.1, deadline - loop.time()))
        done, _pending = await asyncio.wait({stage.task}, timeout=max(0.1, deadline - loop.time()))
        if not done:
            log.warning(
                "stage did not finish within the flush budget; cancelling it",
                extra={"stage": stage.task.get_name(), "pending": stage.queue.qsize()},
            )
            stage.task.cancel()
            await asyncio.gather(stage.task, return_exceptions=True)
