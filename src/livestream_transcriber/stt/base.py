"""Speech-to-text core: the transcript type, the provider protocol and shared plumbing.

Contract every provider follows
-------------------------------
``transcribe(pcm, sample_rate, *, start, end)`` is synchronous and blocking; the
pipeline runs it on a worker thread. ``pcm`` is mono signed 16-bit little-endian
audio, ``start``/``end`` are the chunk's span on the session clock.

* ``None`` means *no speech*: silence, a chunk that is too short, an empty answer.
  It is a normal outcome and never a failure.
* A :class:`Transcript` with ``status == STT_UNAVAILABLE`` (and empty text) means
  the provider *failed*: network down, breaker open, model missing. Callers can
  tell "nothing was said" from "we could not listen".
* Segment and word times inside a transcript are relative to the start of the
  chunk. :func:`livestream_transcriber.audio.timing.sessionize_transcript_timestamps`
  moves them onto the session clock.

The module also holds the offline stand-ins (:class:`NullTranscriber`,
:class:`MockTranscriber`, :class:`FixtureTranscriber`) and the two pieces of
failure handling every cloud provider shares: :class:`ConditionLog` (log a
recurring problem once, then as periodic counts) and :class:`CircuitBreaker`.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..logging_setup import get_logger
from ..models import AudioChunk
from ..netutil import NonJsonBody

__all__ = [
    "MIN_AUDIO_SECONDS",
    "SILENCE_DBFS",
    "STT_OK",
    "STT_UNAVAILABLE",
    "CircuitBreaker",
    "ConditionLog",
    "FixtureTranscriber",
    "LoadBackoff",
    "MockTranscriber",
    "NullTranscriber",
    "Transcriber",
    "Transcript",
    "bad_body",
    "body_snippet",
    "normalise_timed_items",
    "pcm_is_silent",
    "pcm_to_wav",
    "too_short",
    "transcribe_chunk",
    "unavailable",
]

log = get_logger(__name__)

STT_OK = "ok"
STT_UNAVAILABLE = "STT_UNAVAILABLE"

#: Peak level below which a chunk counts as silence for the energy gate.
SILENCE_DBFS = -50.0
#: Providers are not called for less audio than this: Whisper-style models
#: hallucinate on fragments, and a paid request for 200 ms is a waste.
MIN_AUDIO_SECONDS = 0.5

# A failing provider logs once, then one summary line per this many seconds
# while it keeps failing, never one line per chunk (a dead cloud endpoint would
# otherwise write several warnings every 2.5 s).
LOG_SUMMARY_SECONDS = 300.0
# A condition counts as over only after this long without a new occurrence, so
# a provider that fails every other chunk is one episode, not two lines per chunk.
LOG_QUIET_SECONDS = 60.0

DEFAULT_BREAKER_FAILURES = 3
DEFAULT_BREAKER_COOLDOWN_SECONDS = 30.0
DEFAULT_BREAKER_MAX_COOLDOWN_SECONDS = 600.0


# --------------------------------------------------------------------------- #
# Transcript and protocol
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Transcript:
    """The text of one audio chunk (or of several chunks, once stitched)."""

    start: float
    """Start of the audio span on the session clock, in seconds."""
    end: float
    text: str
    confidence: float | None = None
    status: str = STT_OK
    segments: list[dict[str, Any]] | None = None
    """Provider segments ``{"start", "end", "text", ...}``, chunk-relative until sessionized."""
    words: list[dict[str, Any]] | None = None
    """Provider words ``{"start", "end", "text", ...}``, chunk-relative until sessionized."""
    provider_latency: float | None = None
    """Seconds the provider took to answer this chunk."""
    cost_usd: float | None = None
    model: str | None = None
    provider: str | None = None
    """Registry name of the backend that produced the text (``local``, ``openai``, ...)."""
    language: str | None = None
    """Language the provider reports (or was asked for), when known."""
    degraded: bool = False
    """True when a fallback provider produced this because the primary failed."""
    quota_exceeded: bool = False
    """True when the provider refused the call for lack of credit or quota (HTTP 402)."""

    @property
    def unavailable(self) -> bool:
        return self.status == STT_UNAVAILABLE

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def describe(self) -> dict[str, Any]:
        """Compact, JSON-friendly summary for logs and outputs."""
        out: dict[str, Any] = {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.status != STT_OK:
            out["status"] = self.status
        if self.provider:
            out["provider"] = self.provider
        if self.model:
            out["model"] = self.model
        if self.language:
            out["language"] = self.language
        if self.degraded:
            out["degraded"] = True
        if self.quota_exceeded:
            out["quota_exceeded"] = True
        return out


@runtime_checkable
class Transcriber(Protocol):
    """Anything that turns a chunk of PCM into a :class:`Transcript`."""

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None: ...


def unavailable(start: float, end: float, *, model: str | None = None, **extra: Any) -> Transcript:
    """The transcript that means "the provider failed" (never "no speech")."""
    return Transcript(start=start, end=end, text="", status=STT_UNAVAILABLE, model=model, **extra)


def too_short(pcm: bytes, sample_rate: int) -> bool:
    """True when ``pcm`` holds less than :data:`MIN_AUDIO_SECONDS` of audio."""
    return len(pcm) < int(sample_rate * MIN_AUDIO_SECONDS) * 2


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #


def pcm_is_silent(pcm: bytes, *, threshold_dbfs: float = SILENCE_DBFS) -> bool:
    """Energy gate: True when the peak of ``pcm`` is below ``threshold_dbfs``.

    Cheap and dependency-free. It skips digital silence and room tone; it will
    also skip very quiet real speech, which is why the threshold is a knob and
    providers with built-in voice activity detection do not need it.
    """
    if len(pcm) < 2:
        return True
    samples = np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2)
    peak = int(np.abs(samples.astype(np.int32)).max())
    if peak == 0:
        return True
    return bool(20.0 * np.log10(peak / 32768.0) < threshold_dbfs)


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap mono s16le PCM in a WAV container (what the cloud APIs accept)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe_chunk(
    transcriber: Transcriber,
    chunk: AudioChunk,
    *,
    skip_silence: bool = True,
    silence_dbfs: float = SILENCE_DBFS,
) -> Transcript | None:
    """Transcribe one :class:`AudioChunk` on the session clock, skipping silence."""
    if skip_silence and chunk.peak_dbfs() < silence_dbfs:
        return None
    return transcriber.transcribe(
        chunk.pcm, chunk.sample_rate, start=chunk.ts, end=chunk.ts + chunk.duration
    )


# --------------------------------------------------------------------------- #
# Response helpers shared by the cloud providers
# --------------------------------------------------------------------------- #


def bad_body(data: Any) -> str | None:
    """Why a 2xx body is not a transcription response; ``None`` when it is one.

    :mod:`netutil` hands a body that was not JSON (an HTML error page, a proxy
    banner) back as :class:`NonJsonBody`, and an empty body as ``{}``. Read as a
    transcription response the first is a transcript of the error page and the
    second a paid "no speech": billed, cached, counted as a success. Each is a
    failed call instead.
    """
    if isinstance(data, NonJsonBody) or not isinstance(data, dict):
        return "non_json_body"
    if not data:
        return "empty_body"
    return None


def body_snippet(data: Any) -> str:
    """A short single-line excerpt of a response body, for log lines."""
    raw = data.raw if isinstance(data, NonJsonBody) else str(data)
    return " ".join(raw.split())[:120]


def normalise_timed_items(
    items: Any, *, text_keys: tuple[str, ...] = ("text", "word")
) -> list[dict[str, Any]] | None:
    """Reduce provider segment/word records to ``{"start", "end", "text", ...}``.

    Times are kept as the provider gave them and are never invented: a record
    without a numeric time keeps ``None``. Optional quality fields
    (``confidence``, ``avg_logprob``, ``no_speech_prob``, ``speaker``) survive.
    """
    if not isinstance(items, list) or not items:
        return None
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rec: dict[str, Any] = {}
        for key in ("start", "end"):
            value = item.get(key)
            rec[key] = (
                float(value)
                if isinstance(value, int | float) and not isinstance(value, bool)
                else None
            )
        rec["text"] = next(
            (v for k in text_keys if isinstance(v := item.get(k), str) and v.strip()), ""
        )
        for extra in ("confidence", "avg_logprob", "no_speech_prob", "speaker"):
            if extra in item:
                rec[extra] = item[extra]
        out.append(rec)
    return out or None


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


class LoadBackoff:
    """Spacing between attempts to load a local model after a failure.

    A first-run download that failed for a moment (no network, a full disk that was
    cleaned up) must not disable speech-to-text until the process restarts, and
    retrying on every chunk would hammer the failing resource. After a failure the
    load is refused for ``initial`` seconds, then twice that, up to ``maximum``.
    """

    def __init__(
        self,
        initial: float = 30.0,
        maximum: float = 600.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.initial = initial
        self.maximum = maximum
        self._clock = clock
        self._delay = initial
        self._retry_at: float | None = None

    @property
    def blocked(self) -> bool:
        """True while a retry is not yet due."""
        return self._retry_at is not None and self._clock() < self._retry_at

    @property
    def failing(self) -> bool:
        """True once a load failed and has not succeeded since."""
        return self._retry_at is not None

    def failed(self) -> None:
        self._retry_at = self._clock() + self._delay
        self._delay = min(self._delay * 2, self.maximum)

    def succeeded(self) -> None:
        self._retry_at = None
        self._delay = self.initial


class ConditionLog:
    """Log a recurring condition once when it starts, then as periodic counts.

    ``hit`` records one occurrence: the first of an episode is logged with its
    details, later ones are counted and reported in one summary line every
    ``summary_seconds``. ``ok`` ends the episode (logged at INFO) once nothing
    has hit for ``quiet_seconds``. Thread-safe: transcribers run on worker threads.
    """

    def __init__(
        self,
        message: str,
        *,
        logger: logging.Logger | None = None,
        level: int = logging.WARNING,
        summary_seconds: float = LOG_SUMMARY_SECONDS,
        quiet_seconds: float = LOG_QUIET_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.message = message
        self.logger = logger or log
        self.level = level
        self.summary_seconds = float(summary_seconds)
        self.quiet_seconds = float(quiet_seconds)
        self._clock = clock
        self.active = False
        self.total = 0
        """Occurrences in the current (or last) episode."""
        self.episodes = 0
        self._unreported = 0
        self._since = 0.0
        self._last_line = 0.0
        self._last_hit = 0.0
        self._lock = threading.Lock()

    def hit(self, **extra: Any) -> None:
        line: str | None = None
        with self._lock:
            now = self._clock()
            self._last_hit = now
            if not self.active:
                self.active = True
                self.episodes += 1
                self.total = 1
                self._unreported = 0
                self._since = now
                self._last_line = now
                line = "start"
            else:
                self.total += 1
                self._unreported += 1
                if now - self._last_line >= self.summary_seconds:
                    line = "summary"
            count = self._unreported
            if line == "summary":
                self._unreported = 0
                self._last_line = now
            total, since = self.total, now - self._since
        if line == "start":
            self.logger.log(self.level, self.message, extra=extra)
        elif line == "summary":
            self.logger.log(
                self.level,
                f"{self.message} (still occurring)",
                extra={
                    **extra,
                    "occurrences": count,
                    "episode_total": total,
                    "episode_s": round(since, 1),
                },
            )

    def ok(self, **extra: Any) -> bool:
        """End the episode once it has been quiet long enough. True if it ended."""
        with self._lock:
            if not self.active:
                return False
            now = self._clock()
            if now - self._last_hit < self.quiet_seconds:
                return False
            self.active = False
            total, since = self.total, now - self._since
        self.logger.info(
            f"{self.message} - recovered",
            extra={**extra, "occurrences": total, "episode_s": round(since, 1)},
        )
        return True


class CircuitBreaker:
    """Per-provider circuit breaker for cloud STT calls.

    ``closed`` passes every call. ``failure_threshold`` consecutive failures
    (timeouts, 5xx, network errors, exhausted 429 retries, malformed bodies) or
    one HTTP 402 open it: calls are refused without a request. After the
    cool-down one probe is let through (``half_open``); success closes the
    breaker, failure reopens it with the cool-down doubled, up to
    ``max_cooldown_seconds``. Only state changes are logged.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = DEFAULT_BREAKER_FAILURES,
        cooldown_seconds: float = DEFAULT_BREAKER_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = DEFAULT_BREAKER_MAX_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.base_cooldown = max(0.0, float(cooldown_seconds))
        self.max_cooldown = max(self.base_cooldown, float(max_cooldown_seconds))
        self._clock = clock
        self._state = self.CLOSED
        self._cooldown = self.base_cooldown
        self._open_until = 0.0
        self._probe_started = 0.0
        self.consecutive_failures = 0
        self.last_failure: str | None = None
        self.opened = 0
        self.short_circuited = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """May a request go out now? Refusals are counted, never logged."""
        with self._lock:
            if self._state == self.CLOSED:
                return True
            now = self._clock()
            if self._state == self.OPEN:
                if now < self._open_until:
                    self.short_circuited += 1
                    return False
                self._state = self.HALF_OPEN
            elif now - self._probe_started < max(self._cooldown, 1.0):
                # One probe at a time; the others wait for its verdict.
                self.short_circuited += 1
                return False
            # else: the last probe never reported (its caller was abandoned).
            self._probe_started = now
            cooldown = self._cooldown
        log.info(
            "stt circuit half-open; probing provider",
            extra={"provider": self.name, "cooldown_s": round(cooldown, 1)},
        )
        return True

    def record_success(self) -> None:
        with self._lock:
            was = self._state
            self._state = self.CLOSED
            self.consecutive_failures = 0
            self._cooldown = self.base_cooldown
        if was != self.CLOSED:
            log.warning(
                "stt circuit closed; provider answering again", extra={"provider": self.name}
            )

    def record_failure(self, kind: str, *, immediate: bool = False) -> None:
        with self._lock:
            self.consecutive_failures += 1
            self.last_failure = kind
            now = self._clock()
            if self._state == self.HALF_OPEN:
                # The probe failed: back off exponentially.
                self._cooldown = min(max(self._cooldown * 2.0, 1.0), self.max_cooldown)
            elif self._state == self.OPEN or not (
                immediate or self.consecutive_failures >= self.failure_threshold
            ):
                return
            else:
                self._cooldown = self.base_cooldown
            self._state = self.OPEN
            self._open_until = now + self._cooldown
            self.opened += 1
            cooldown, failures = self._cooldown, self.consecutive_failures
        log.warning(
            "stt circuit open; provider paused",
            extra={
                "provider": self.name,
                "failure": kind,
                "consecutive_failures": failures,
                "cooldown_s": round(cooldown, 1),
            },
        )

    def describe(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            return {
                "state": self._state,
                "consecutive_failures": self.consecutive_failures,
                "last_failure": self.last_failure,
                "cooldown_s": round(self._cooldown, 1),
                "retry_in_s": (
                    round(max(0.0, self._open_until - now), 1) if self._state == self.OPEN else None
                ),
                "opened": self.opened,
                "short_circuited": self.short_circuited,
            }


# --------------------------------------------------------------------------- #
# Offline stand-ins
# --------------------------------------------------------------------------- #


class NullTranscriber:
    """Speech-to-text switched off. Never invents speech."""

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        return None


class MockTranscriber:
    """Deterministic STT for tests and demos. Never touches the network.

    ``text`` is returned for every chunk (``{n}`` is replaced by the call
    number, which keeps consecutive chunks distinct for the stitcher). A
    ``script`` is consumed one item per call instead: a string, ``None`` for
    "no speech", or a :class:`Transcript` whose text, status and extras are
    kept while its span is replaced by the chunk's. Every call is recorded in
    ``calls`` as ``(start, end, n_bytes)``.
    """

    def __init__(
        self,
        text: str | None = None,
        script: Sequence[str | Transcript | None] | None = None,
    ) -> None:
        self.text = text
        self.script = list(script or [])
        self.calls: list[tuple[float, float, int]] = []
        self.model = "mock"

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        self.calls.append((start, end, len(pcm)))
        if self.script:
            item = self.script.pop(0)
            if item is None:
                return None
            if isinstance(item, Transcript):
                return replace(item, start=start, end=end)
            return Transcript(start=start, end=end, text=item, confidence=1.0)
        if self.text:
            text = self.text.replace("{n}", str(len(self.calls)))
            return Transcript(start=start, end=end, text=text, confidence=1.0)
        return None


def _first_key(row: dict[str, Any], primary: str, alias: str) -> Any:
    """``row[primary]``, or ``row[alias]`` when only the alias is present."""
    return row[primary] if primary in row else row[alias if alias in row else primary]


class FixtureTranscriber:
    """Replay recorded transcripts. A chunk gets the utterance it overlaps most in time."""

    def __init__(self, utterances: Sequence[Transcript]) -> None:
        self.utterances = list(utterances)
        self.model = "fixture"

    @classmethod
    def from_jsonl(cls, path: str | Path) -> FixtureTranscriber:
        """Load ``{"start", "end", "text", "confidence"?}`` rows; ``#`` lines are comments.

        Times are what a chunk is matched against, so a row needs a ``start`` and an
        ``end`` after it; anything else raises :class:`ValueError` naming the line rather
        than becoming an utterance that no chunk ever overlaps.
        """
        rows: list[Transcript] = []
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("expected a JSON object")
                start = float(_first_key(row, "start", "start_ts"))
                end = float(_first_key(row, "end", "end_ts"))
                conf = row.get("confidence")
                confidence = None if conf is None else float(conf)
            except (ValueError, TypeError, KeyError) as exc:
                detail = (
                    f"missing {exc.args[0]!r} (rows need start and end)"
                    if isinstance(exc, KeyError)
                    else str(exc)
                )
                raise ValueError(f"{where}: invalid fixture row: {detail}") from exc
            if end <= start:
                raise ValueError(f"{where}: end ({end}) must be after start ({start})")
            text = str(row.get("text") or "").strip()
            if text:
                rows.append(Transcript(start=start, end=end, text=text, confidence=confidence))
        return cls(rows)

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        best: Transcript | None = None
        best_overlap = 0.0
        for item in self.utterances:
            overlap = min(end, item.end) - max(start, item.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = item
        if best is None:
            return None
        return replace(best)
