"""Live capture: resolve, run ffmpeg, emit time-stamped audio chunks.

Failure model
-------------
One ffmpeg run is a *segment*. Anything that ends a run (network drop, the platform
rotating the URL, ffmpeg dying) ends the segment; a supervisor then re-resolves and
starts the next one after a jittered exponential back-off. Consumers see one continuous
chunk iterator across all of that, with ``ts`` staying monotonic and the gap visible in
the logs and in :class:`SegmentInfo`.

Audio-only means the health signal is audio too. A watchdog per segment notices when
chunks stop arriving and ends the segment; it does not depend on any other track.

A live HLS playlist that is still listed but no longer extended is *held*: ffmpeg stays
stopped and only the playlist's live edge is polled until it moves, the stream stops
being live, or the fallback selector moves elsewhere. This needs the playlist's edge, so
it is available when the opt-in HLS window is on.

Finite sources (a local file, a recording, a VOD) take the lossless path: the queue
applies backpressure instead of dropping, so no audio is lost however slow the consumer
is, and a failure ends the capture with an error instead of retrying from the start.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from ..logging_setup import get_logger
from ..models import AudioChunk, CaptureStats, SegmentInfo, StreamInfo
from ..redact import redact_text
from .base import StreamError, StreamNotLiveError, StreamResolutionError, StreamSource
from .fallback import short_error
from .ffmpeg import FFmpegAudioPipe, FFmpegSpec, is_http_url
from .hls_window import (
    USER_AGENT,
    HlsInput,
    fetch_live_edge,
    make_playlist_fetcher,
    prepare_hls_input,
)
from .queues import DropOldestQueue
from .resolver import recently_resolved, resolve_stream

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

__all__ = [
    "CaptureOptions",
    "LiveStreamSource",
    "ReselectCallback",
    "await_capture_tasks",
    "playlist_stalled",
    "should_attempt_reselect",
]

# After this many consecutive empty segments, ask ``reselect`` even if the URL still
# resolves (a stale playlist, a dead DVR window).
_RESELECT_EMPTY_STREAK = 2
# How long ffmpeg gets to exit on its own once its output hit EOF.
_EXIT_GRACE_S = 5.0
# How far past the requested duration a live segment may run before it is cut off by the
# wall clock. ffmpeg's own ``-t`` limit normally ends it first; this bounds a stalled feed.
_DURATION_GRACE_S = 3.0
# Reconnect reason after ``stale_playlist_stalls`` audio stalls on a frozen edge.
_STALE_PLAYLIST = "stale_playlist"
# While a frozen playlist is held, the selector and a re-resolve are asked once per
# this many ``reconnect_max_delay`` intervals.
_HOLD_RECHECK_DELAYS = 4


class _ResolveArgs(TypedDict):
    """The options that shape a resolve; they are also the resolver's cache key."""

    format_selector: str
    cookiefile: str | Path | None
    proxy: str | None


ReselectCallback = Callable[[str, str, int], Awaitable[str | None]]
"""``await reselect(current_url, reason, attempt)``: a new URL to switch to, or None."""


@dataclass(slots=True)
class _Supervision:
    """What the supervisor carries from one segment to the next."""

    info: StreamInfo
    live_session: bool
    delay: float
    """The reconnect back-off ceiling; doubles per failure, resets after a healthy segment."""
    playable: bool = True
    """False while a newly chosen source has no resolvable media URL yet."""
    attempt: int = 0
    empty_streak: int = 0
    """Consecutive segments that delivered no audio."""
    frozen_stalls: int = 0
    """Consecutive stalls on a playlist whose live edge never moved."""
    previous_edge: int | None = None
    """The newest playlist edge any earlier segment saw."""
    last_reason: str = "start"
    segment_started: float | None = None
    """Start of the last segment: the resolve that fed it is not reused."""
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class CaptureOptions:
    """Everything the capture tunes, independent of how it is configured.

    The defaults mirror the ``capture_*`` fields of :class:`~..config.Settings`;
    :meth:`from_settings` builds one from a loaded configuration.
    """

    sample_rate: int = 16000
    live_chunk_seconds: float = 2.5
    file_chunk_seconds: float = 5.0
    queue_size: int = 32
    """Live chunk queue: drop-oldest, so a slow consumer never stalls ffmpeg."""
    file_queue_size: int = 8
    """Finite-source queue: lossless, the producer waits for room."""
    reconnect_initial_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    max_reconnect_attempts: int = 0
    """Give up after this many consecutive reconnects; 0 retries forever."""
    resolve_cache_seconds: float = 20.0
    audio_stall_seconds: float = 30.0
    """A live segment with no chunk for this long is restarted; 0 disables."""
    stale_playlist_stalls: int = 2
    stream_format: str = "bestaudio/best"
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_loglevel: str = "warning"
    hls_window: bool = False
    hls_live_start_index: int = -3

    def __post_init__(self) -> None:
        if not 8000 <= self.sample_rate <= 48000:
            raise ValueError("sample_rate must be between 8000 and 48000")
        if self.live_chunk_seconds <= 0 or self.file_chunk_seconds <= 0:
            raise ValueError("chunk seconds must be > 0")
        if self.queue_size < 1 or self.file_queue_size < 1:
            raise ValueError("queue sizes must be >= 1")
        if self.reconnect_initial_delay < 0 or self.reconnect_max_delay < 0:
            raise ValueError("reconnect delays must be >= 0")
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be >= 0")

    @classmethod
    def from_settings(cls, settings: Settings) -> CaptureOptions:
        return cls(
            sample_rate=settings.capture_sample_rate,
            live_chunk_seconds=settings.capture_live_chunk_seconds,
            file_chunk_seconds=settings.capture_file_chunk_seconds,
            queue_size=settings.capture_queue_size,
            file_queue_size=settings.capture_file_queue_size,
            reconnect_initial_delay=settings.capture_reconnect_initial_delay,
            reconnect_max_delay=settings.capture_reconnect_max_delay,
            max_reconnect_attempts=settings.capture_max_reconnect_attempts,
            resolve_cache_seconds=settings.capture_resolve_cache_seconds,
            audio_stall_seconds=settings.capture_audio_stall_seconds,
            stale_playlist_stalls=settings.capture_stale_playlist_stalls,
            stream_format=settings.capture_stream_format,
            ffmpeg_binary=settings.capture_ffmpeg_binary,
            ffmpeg_loglevel=settings.capture_ffmpeg_loglevel,
            hls_window=settings.capture_hls_window,
            hls_live_start_index=settings.capture_hls_live_start_index,
        )

    def chunk_seconds_for(self, *, live: bool) -> float:
        return self.live_chunk_seconds if live else self.file_chunk_seconds

    def chunk_bytes_for(self, *, live: bool) -> int:
        """Bytes per chunk: whole 16-bit samples, at least one."""
        samples = max(1, round(self.sample_rate * self.chunk_seconds_for(live=live)))
        return samples * 2


def should_attempt_reselect(
    *,
    reason: str,
    is_live: bool,
    empty_streak: int = 0,
    empty_streak_limit: int = _RESELECT_EMPTY_STREAK,
) -> bool:
    """Whether a live run should re-run source selection before the next resolve.

    A brief ffmpeg drop that already produced data is not a loss of the source. A
    clean live ``stream_ended``, a frozen playlist (``stale_playlist``) or a streak of
    empty reconnects is.
    """
    if is_live and reason in ("stream_ended", _STALE_PLAYLIST):
        return True
    return empty_streak >= empty_streak_limit


def playlist_stalled(
    reason: str,
    first_edge: int | None,
    last_edge: int | None,
    previous_edge: int | None,
) -> bool:
    """An audio stall on a live playlist that did not move at all.

    Neither during the segment (``first_edge`` to ``last_edge``, both from the remote
    playlist behind the HLS window) nor since the previous segment ended. Without a
    window there is no edge and nothing is claimed.
    """
    if reason != "audio_stalled" or first_edge is None or last_edge is None:
        return False
    if last_edge > first_edge:
        return False
    return previous_edge is None or last_edge <= previous_edge


async def await_capture_tasks(
    pumps: asyncio.Future[Any],
    watchdog: asyncio.Task[str | None],
    *,
    stop_pipe: Callable[[], Awaitable[None]],
    unblock_timeout: float = 8.0,
    on_stuck: Callable[[str], None] | None = None,
) -> str | None:
    """Wait until the pump finishes or the watchdog aborts the segment.

    Gathering the pump *and* the watchdog deadlocks when ffmpeg exits or the pipe
    stalls: the pump ends (or blocks in a read) while the watchdog keeps sleeping, and
    the supervisor never logs that the segment ended.

    The watchdog only returns a verdict. ffmpeg is stopped here, after the verdict is
    recorded, so the segment is filed by that verdict and not by whatever exit code the
    stopped process then reports.

    If the pump does not unblock within ``unblock_timeout`` of the pipe being stopped,
    ``on_stuck`` is told (an operator may want to restart the process) and the pump is
    cancelled so the supervisor can go on.
    """
    done, _pending = await asyncio.wait({pumps, watchdog}, return_when=asyncio.FIRST_COMPLETED)
    abort_reason: str | None = None
    if watchdog in done and not watchdog.cancelled():
        err = watchdog.exception()
        if err is not None:
            raise err
        result = watchdog.result()
        if result:
            abort_reason = result
            await stop_pipe()
            if not pumps.done():
                finished, _still = await asyncio.wait({pumps}, timeout=unblock_timeout)
                if not finished:
                    log.error(
                        "capture pump stuck after stall; cancelling it",
                        extra={"reason": result},
                    )
                    if on_stuck is not None:
                        on_stuck(result)
                    pumps.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await pumps
                    return result
    if not pumps.done():
        await pumps
    elif not pumps.cancelled():
        err = pumps.exception()
        if err is not None:
            raise err
    return abort_reason


class LiveStreamSource(StreamSource):
    """A :class:`StreamSource` backed by a URL or file: live, VOD or local."""

    def __init__(
        self,
        url: str,
        options: CaptureOptions | None = None,
        *,
        duration: float | None = None,
        input_seek: float | None = None,
        cookiefile: str | Path | None = None,
        http_proxy: str | None = None,
        reselect: ReselectCallback | None = None,
        realtime: bool = False,
        lossless: bool | None = None,
        on_stuck: Callable[[str], None] | None = None,
        on_hold: Callable[[bool], None] | None = None,
    ) -> None:
        """
        ``realtime`` reads a file at its native rate (``-re``), to simulate a live feed.
        ``lossless`` forces (True) or forbids (False) backpressure; by default a
        finite source is lossless and a live one drops the oldest chunk when full.
        ``on_stuck(reason)`` is called if a stalled capture cannot be unblocked.
        ``on_hold(True/False)`` brackets a hold on a frozen playlist, when no audio is
        expected (a supervisor watchdog elsewhere may want to know).
        """
        self.url = url
        self.options = options or CaptureOptions()
        self._duration = duration
        self._input_seek = input_seek
        self._cookiefile = cookiefile
        self._http_proxy = http_proxy
        self.reselect = reselect
        self._realtime = realtime
        self._lossless_override = lossless
        self.on_stuck = on_stuck
        self.on_hold = on_hold
        # Set by a caller that opens this URL because a probe found it live: connect()
        # then refuses a resolve that is not live.
        self.require_live = False
        self.on_fatal: Callable[[Exception], None] | None = None
        self._fatal_notified = False

        self._lossless = bool(lossless)
        self._audio_q: DropOldestQueue[AudioChunk] = DropOldestQueue(self.options.queue_size)
        self._stats = CaptureStats()
        self._segments: list[SegmentInfo] = []
        self._info: StreamInfo | None = None
        self._pipe: FFmpegAudioPipe | None = None
        # (at start, at end) live-edge media sequence of the last segment's remote
        # playlist; (None, None) without an HLS window.
        self._segment_edges: tuple[int | None, int | None] = (None, None)
        self._supervisor: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()
        self._session_start = 0.0
        self._session_audio_samples = 0
        self._first_data = asyncio.Event()
        # Consecutive failed re-resolves (see _try_resolve).
        self._resolve_failures = 0

    @classmethod
    def from_settings(cls, url: str, settings: Settings, **kwargs: Any) -> LiveStreamSource:
        """Build a source from a loaded configuration (cookies and proxy included)."""
        proxy = settings.capture_proxy.get_secret_value() if settings.capture_proxy else None
        return cls(
            url,
            CaptureOptions.from_settings(settings),
            cookiefile=settings.capture_cookies_file,
            http_proxy=proxy,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # StreamSource
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> CaptureStats:
        self._stats.audio_chunks_dropped = self._audio_q.dropped
        return self._stats

    @property
    def segments(self) -> list[SegmentInfo]:
        return list(self._segments)

    @property
    def info(self) -> StreamInfo | None:
        return self._info

    @property
    def lossless(self) -> bool:
        """Whether the queue applies backpressure (decided at :meth:`connect`)."""
        return self._lossless

    async def connect(self) -> StreamInfo:
        """Resolve the URL once (so failures surface immediately) and start capture.

        A source probe has usually just resolved this URL. The resolver remembers that
        answer for ``resolve_cache_seconds``, keyed by everything that shapes it, so a
        session start costs no extraction of its own.
        """
        if self._supervisor is not None:
            raise RuntimeError("already connected")

        info = await resolve_stream(
            self.url, **self._resolve_args(), max_age=self.options.resolve_cache_seconds
        )
        log.info(
            "resolved stream",
            extra={**info.describe(), "cookies": "set" if self._cookiefile else "unset"},
        )
        if self.require_live and not info.is_live:
            # A probe said live, this resolve says the broadcast is over. Capturing it
            # as a finite recording would push the whole post-live DVR through STT at
            # full decode speed.
            raise StreamNotLiveError(
                f"{self.url} was listed live but resolves as a recording "
                "(the broadcast is over); not capturing it"
            )
        self._info = info
        self._lossless = (
            not info.is_live if self._lossless_override is None else self._lossless_override
        )
        if not info.is_live:
            log.info("source is not live; capturing it as a finite recording")
        # The consumer starts reading after connect(), so the queue can still be
        # chosen here, once the resolve has said whether the source is finite.
        size = self.options.file_queue_size if self._lossless else self.options.queue_size
        self._audio_q = DropOldestQueue(size)

        self._session_start = time.time()
        self._supervisor = asyncio.create_task(self._supervise(info), name="capture-supervisor")
        return info

    def get_audio(self) -> AsyncIterator[AudioChunk]:
        return self._audio_q.__aiter__()

    async def close(self) -> None:
        if self._closing.is_set() and self._supervisor is None:
            return
        self._closing.set()
        if self._supervisor is not None:
            self._supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor
            self._supervisor = None
        if self._pipe is not None:
            await self._pipe.stop()
            self._pipe = None
        self._close_queue()
        log.info("capture closed", extra=self.stats.describe())

    def _close_queue(self, error: Exception | None = None) -> None:
        """Close the chunk queue. The first non-None error wins."""
        if error is not None:
            self._notify_fatal(error)
        self._audio_q.close(error)

    def _notify_fatal(self, error: Exception) -> None:
        if self._fatal_notified:
            return
        self._fatal_notified = True
        callback = self.on_fatal
        if callback is None:
            return
        try:
            callback(error)
        except Exception:
            log.exception("stream-lost callback failed")

    async def wait_for_first_data(self, timeout: float | None = None) -> bool:
        """Block until the first audio chunk arrives. For CLI and tests."""
        try:
            await asyncio.wait_for(self._first_data.wait(), timeout)
            return True
        except TimeoutError:
            return False

    # ------------------------------------------------------------------ #
    # resolving and reselecting
    # ------------------------------------------------------------------ #

    async def _closed_within(self, seconds: float) -> bool:
        """Sleep ``seconds``; True as soon as close() is called."""
        try:
            await asyncio.wait_for(self._closing.wait(), seconds)
            return True
        except TimeoutError:
            return False

    def _resolve_args(self) -> _ResolveArgs:
        return {
            "format_selector": self.options.stream_format,
            "cookiefile": self._cookiefile,
            "proxy": self._http_proxy,
        }

    async def _try_resolve(
        self, *, max_age: float, since: float | None = None
    ) -> StreamInfo | None:
        try:
            info = await resolve_stream(
                self.url, **self._resolve_args(), max_age=max_age, since=since
            )
        except StreamResolutionError as exc:
            # A reconnect loop through a bot check or an outage fails here on every
            # attempt with the same few KB of extractor text. The first failure of a
            # streak is a WARNING with the error's head; the repeats are DEBUG, and
            # the streak's length is on the next success.
            self._resolve_failures += 1
            log.log(
                logging.WARNING if self._resolve_failures == 1 else logging.DEBUG,
                "re-resolve failed",
                extra={"error": short_error(str(exc)), "failures": self._resolve_failures},
            )
            return None
        self._info = info
        extra = info.describe()
        if self._resolve_failures:
            extra["after_failures"] = self._resolve_failures
            self._resolve_failures = 0
        log.info("re-resolved stream", extra=extra)
        return info

    async def _apply_reselect(self, reason: str, attempt: int) -> bool:
        callback = self.reselect
        if callback is None:
            return False
        try:
            new_url = await callback(self.url, reason, attempt)
        except Exception:
            log.exception("source reselect failed")
            return False
        if not new_url or new_url == self.url:
            return False
        log.warning(
            "switching capture url",
            extra={"from_url": self.url, "to_url": new_url, "after": reason, "attempt": attempt},
        )
        self.url = new_url
        return True

    async def _next_stream_info(
        self,
        *,
        reason: str,
        is_live: bool,
        empty_streak: int,
        attempt: int,
        not_before: float | None = None,
    ) -> tuple[StreamInfo | None, bool]:
        """Resolve the next segment. May switch URL via ``reselect``.

        Returns ``(info, switched)``. ``info is None`` means the resolve failed; if
        ``switched`` is also True the caller must not reuse the previous source's
        media URL.

        At most one extraction of our own per iteration, and at most one reselect
        (whose probes resolve every candidate). Right after a reselect the URL was
        just probed, so the resolve is a cache hit.

        ``not_before``: a remembered resolve is reused only if it was made at or after
        this time, the start of the segment that just ended. The answer that fed that
        segment is never handed to the next one: a segment that failed within seconds
        (a 403 on a signed manifest) would otherwise be retried on exactly the URL that
        just failed.

        A live run whose URL now resolves as *not* live gets the reselect (reason
        ``live_ended``) if it has not had one. If the selector keeps the URL, its own
        probe of it is the newer answer and wins; only when that one is not live either
        (or there is none) does the returned info stay not-live, and the supervisor ends.
        """
        asked_at = time.time()
        max_age = self.options.resolve_cache_seconds
        reselected = False
        switched = False
        if should_attempt_reselect(reason=reason, is_live=is_live, empty_streak=empty_streak):
            switched = await self._apply_reselect(reason, attempt)
            reselected = True

        info = await self._try_resolve(max_age=max_age, since=not_before)
        if info is not None and is_live and not info.is_live and not reselected:
            # The broadcast may be over (the platform serves its recording now), or
            # this one answer was a blip. The selector probes the URL again.
            switched = await self._apply_reselect("live_ended", attempt)
            if switched:
                return self._recent_info(max_age, since=not_before), True
            # A probe that says live outranks our older not-live answer: one odd reply
            # must not end a live capture.
            fresher = self._recent_info(math.inf, since=asked_at, newer_than=info)
            return fresher or info, False
        if info is not None or reselected:
            return info, switched

        switched = await self._apply_reselect(reason, attempt)
        if switched:
            # A second network resolve waits for the next (backed-off) round.
            return self._recent_info(max_age, since=not_before), True
        return None, switched

    def _recent_info(
        self,
        max_age: float,
        *,
        since: float | None = None,
        newer_than: StreamInfo | None = None,
    ) -> StreamInfo | None:
        """The selector's answer for the current URL (just probed), if any.

        ``since``/``newer_than``: only an answer resolved at or after that time, or
        after that answer, counts.
        """
        info = recently_resolved(self.url, **self._resolve_args(), max_age=max_age, since=since)
        if info is None:
            return None
        if newer_than is not None and (
            info is newer_than or info.resolved_at < newer_than.resolved_at
        ):
            return None
        self._info = info
        log.info("re-resolved stream", extra=info.describe())
        return info

    # ------------------------------------------------------------------ #
    # supervision
    # ------------------------------------------------------------------ #

    async def _supervise(self, initial: StreamInfo) -> None:
        """Keep capture running: one ffmpeg segment after another, until the source is done.

        A segment ends when ffmpeg exits, the audio stalls or the duration is reached. The
        loop then decides whether that was the end of the source, and if not, reconnects
        with full-jitter back-off, possibly onto another URL.
        """
        state = _Supervision(
            info=initial,
            live_session=initial.is_live,
            delay=self.options.reconnect_initial_delay,
        )
        deadline = self._session_start + self._duration if self._duration is not None else None
        try:
            while not self._closing.is_set():
                if deadline is not None and time.time() >= deadline:
                    log.info("requested duration reached")
                    break
                if state.playable and not await self._capture_segment(state, deadline):
                    break
                if not await self._prepare_reconnect(state, deadline):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.error = StreamError(f"capture supervisor crashed: {exc}")
            state.error.__cause__ = exc
            log.exception("capture supervisor crashed")
        finally:
            self._close_queue(state.error)

    async def _capture_segment(self, state: _Supervision, deadline: float | None) -> bool:
        """Run one segment and file its outcome in ``state``. False ends the capture."""
        opts = self.options
        segment = SegmentInfo(
            index=len(self._segments),
            started_at=time.time(),
            offset=max(0.0, time.time() - self._session_start),
            audio_sample_base=self._session_audio_samples,
        )
        self._segments.append(segment)
        self._stats.segments += 1
        state.segment_started = segment.started_at

        remaining = None if deadline is None else max(0.5, deadline - time.time())
        self._segment_edges = (None, None)
        reason = await self._run_segment(state.info, segment, remaining)
        segment.ended_at = time.time()
        segment.reason = reason
        state.last_reason = reason
        first_edge, last_edge = self._segment_edges

        log.info(
            "capture segment ended",
            extra={
                "segment": segment.index,
                "reason": reason,
                "duration": round(segment.ended_at - segment.started_at, 1),
                "audio_chunks": segment.audio_chunks,
                "playlist_edge": last_edge,
            },
        )

        if self._closing.is_set() or reason == "duration_reached":
            return False
        if not state.info.is_live:
            # A finite source is read once. A clean end is the end of the file; anything
            # else is a failure, and retrying would replay it from the start and
            # duplicate what was already emitted.
            if reason != "stream_ended":
                state.error = StreamError(f"could not read {self.url}: {reason}")
                log.error("finite source failed", extra={"reason": reason})
            else:
                log.info("finite source finished")
            return False

        # A frozen-but-listed playlist: every reconnect replays the same last seconds,
        # stalls, and "produced data", so the back-off never grows. Count it instead.
        if playlist_stalled(reason, first_edge, last_edge, state.previous_edge):
            state.frozen_stalls += 1
        else:
            state.frozen_stalls = 0
        if last_edge is not None:
            state.previous_edge = max(state.previous_edge or last_edge, last_edge)
        stall_limit = opts.stale_playlist_stalls
        if state.live_session and stall_limit and state.frozen_stalls >= stall_limit:
            log.warning(
                "live playlist stopped advancing",
                extra={"stalls": state.frozen_stalls, "playlist_edge": last_edge},
            )
            state.last_reason = _STALE_PLAYLIST

        # A segment that produced data means the stream is healthy and this was a
        # transient drop: reset the back-off.
        if segment.audio_chunks:
            state.attempt = 0
            state.empty_streak = 0
            state.delay = opts.reconnect_initial_delay
        else:
            state.empty_streak += 1
        return True

    async def _prepare_reconnect(self, state: _Supervision, deadline: float | None) -> bool:
        """Back off, then pick the stream to capture next. False ends the capture."""
        opts = self.options
        state.attempt += 1
        self._stats.reconnects += 1
        limit = opts.max_reconnect_attempts
        if limit and state.attempt > limit:
            state.error = StreamError(f"gave up after {state.attempt - 1} reconnects")
            log.error("giving up", extra={"attempts": state.attempt - 1})
            return False

        # Full jitter, so an outage on the platform side does not produce a synchronised
        # retry storm from many watchers. A small floor keeps a lucky draw from becoming
        # a hot loop.
        sleep_for = max(0.01, random.uniform(0.0, state.delay))
        log.warning(
            "reconnecting",
            extra={
                "attempt": state.attempt,
                "delay": round(sleep_for, 1),
                "after": state.last_reason,
            },
        )
        if await self._closed_within(sleep_for):
            return False  # close() won the race
        state.delay = min(state.delay * 2, opts.reconnect_max_delay)

        nxt, switched = await self._next_stream_info(
            reason=state.last_reason,
            is_live=state.info.is_live,
            empty_streak=state.empty_streak,
            attempt=state.attempt,
            not_before=state.segment_started,
        )
        if nxt is not None and state.live_session and not nxt.is_live:
            # Playing the post-live recording would push hours of old audio through STT
            # at full decode speed.
            log.warning(
                "ending capture: stream is no longer live",
                extra={"url": self.url, "title": nxt.title},
            )
            return False
        if switched:
            state.frozen_stalls = 0
            state.previous_edge = None
        elif state.last_reason == _STALE_PLAYLIST:
            # The selector kept this source, but its playlist is not moving. Stop replaying
            # the same seconds, and do not end the capture either: a supervisor would only
            # restart it into the same source. Wait for the edge to move.
            held = await self._hold_frozen(
                nxt or state.info, state.previous_edge, state.attempt, deadline
            )
            if held is None:
                return False
            nxt, switched = held
            state.frozen_stalls = 0
            state.delay = opts.reconnect_initial_delay
            state.last_reason = "switched" if switched else "playlist_moved"
            if switched:
                state.previous_edge = None
        if nxt is not None:
            state.info = nxt
            state.playable = True
        elif switched:
            # New source chosen but not yet resolvable: do not keep playing the previous
            # source's stale media URL.
            state.playable = False
        # else: keep the old URL; it sometimes still works, and if it does not the next
        # loop iteration backs off further.
        return True

    async def _hold_frozen(
        self,
        info: StreamInfo,
        frozen_edge: int | None,
        attempt: int,
        deadline: float | None,
    ) -> tuple[StreamInfo | None, bool] | None:
        """Wait, with ffmpeg stopped, for a frozen live playlist to move again.

        The platform still lists the stream, but its playlist stopped growing: the
        encoder is gone and may come back. Reconnecting would replay the last few
        seconds through a stall, over and over, and ending the capture would make a
        process supervisor restart it into the same source.

        Nothing is decoded or transcribed meanwhile. A poll is one GET of the playlist,
        backed off up to ``reconnect_max_delay``, and capture resumes once its live edge
        moves. Every ``_HOLD_RECHECK_DELAYS`` times ``reconnect_max_delay`` the selector
        is asked and the URL re-resolved: a switch resumes on the new source; a resolve
        that is no longer live ends capture; a fresh resolve whose playlist lists
        another edge (a new broadcast behind the same URL) resumes.

        Returns ``(info, switched)`` to go on with, or None to end capture.
        """
        log.warning(
            "live playlist frozen; holding capture until it moves",
            extra={"url": self.url, "playlist_edge": frozen_edge},
        )
        self._notify_hold(True)
        try:
            return await self._await_playlist_motion(info, frozen_edge, attempt, deadline)
        finally:
            self._notify_hold(False)

    def _notify_hold(self, holding: bool) -> None:
        if self.on_hold is None:
            return
        try:
            self.on_hold(holding)
        except Exception:
            log.exception("hold callback failed")

    async def _await_playlist_motion(
        self,
        info: StreamInfo,
        frozen_edge: int | None,
        attempt: int,
        deadline: float | None,
    ) -> tuple[StreamInfo | None, bool] | None:
        opts = self.options
        fetcher = make_playlist_fetcher(self._http_proxy)
        recheck_s = _HOLD_RECHECK_DELAYS * opts.reconnect_max_delay
        started = last_check = time.monotonic()
        delay = opts.reconnect_initial_delay
        # ``info`` comes from a resolve made during this hold: its playlist URL is new,
        # so any other edge (not only a higher one) means it moved.
        fresh = False
        while True:
            if await self._closed_within(max(0.01, random.uniform(0.5 * delay, delay))):
                return None
            delay = min(delay * 2, opts.reconnect_max_delay)
            if deadline is not None and time.time() >= deadline:
                return info, False
            if time.monotonic() - last_check >= recheck_s:
                last_check = time.monotonic()
                attempt += 1
                nxt, switched = await self._next_stream_info(
                    reason=_STALE_PLAYLIST,
                    is_live=True,
                    empty_streak=0,
                    attempt=attempt,
                    not_before=time.time(),
                )
                if self._closing.is_set():
                    return None
                if switched:
                    return nxt, True
                if nxt is not None and not nxt.is_live:
                    log.warning(
                        "ending capture: stream is no longer live",
                        extra={"url": self.url, "title": nxt.title},
                    )
                    return None
                if nxt is not None and nxt is not info:
                    info, fresh = nxt, True
            try:
                edge = await asyncio.to_thread(fetch_live_edge, info.media_url or info.url, fetcher)
            except Exception as exc:
                log.debug("frozen playlist poll failed", extra={"error": type(exc).__name__})
                continue
            if frozen_edge is None or edge > frozen_edge or (fresh and edge != frozen_edge):
                log.warning(
                    "live playlist moving again; resuming capture",
                    extra={
                        "playlist_edge": edge,
                        "frozen_edge": frozen_edge,
                        "held_s": round(time.monotonic() - started, 1),
                    },
                )
                return info, False
            fresh = False

    # ------------------------------------------------------------------ #
    # one ffmpeg run
    # ------------------------------------------------------------------ #

    def _ffmpeg_spec(self, info: StreamInfo, hls: HlsInput, remaining: float | None) -> FFmpegSpec:
        opts = self.options
        agent = next((v for k, v in info.headers.items() if k.lower() == "user-agent"), None)
        if agent is None and info.is_live and is_http_url(hls.url):
            agent = USER_AGENT
        return FFmpegSpec(
            url=hls.url,
            sample_rate=opts.sample_rate,
            binary=opts.ffmpeg_binary,
            loglevel=opts.ffmpeg_loglevel,
            # A slim window already sits on the live edge; -live_start_index on that
            # short list would leave only a few seconds before the demuxer stalls.
            # Keep it for a remote playlist only.
            hls_live_start_index=(
                opts.hls_live_start_index if info.is_live and not hls.windowed else None
            ),
            input_seek=self._input_seek,
            duration=remaining,
            realtime=self._realtime,
            user_agent=agent,
            headers=info.headers,
            http_proxy=self._http_proxy,
        )

    async def _run_segment(
        self, info: StreamInfo, segment: SegmentInfo, remaining: float | None
    ) -> str:
        """Run one ffmpeg process to completion. Returns why it stopped."""
        opts = self.options
        source = info.media_url or info.url
        if opts.hls_window:
            hls = await prepare_hls_input(source, is_live=info.is_live, proxy=self._http_proxy)
        else:
            hls = HlsInput(source)
        pipe = FFmpegAudioPipe(self._ffmpeg_spec(info, hls, remaining))
        self._pipe = pipe
        try:
            await pipe.start()
        except Exception as exc:
            await hls.aclose()
            self._pipe = None
            log.error("ffmpeg failed to start", extra={"error": str(exc)})
            return "ffmpeg_start_failed"
        except BaseException:
            await hls.aclose()
            raise

        log.info(
            "capture segment started",
            extra={"segment": segment.index, "sample_rate": opts.sample_rate},
        )
        pump = asyncio.create_task(
            self._pump_audio(pipe, segment, is_live=info.is_live), name=f"audio-{segment.index}"
        )
        watchdog = asyncio.create_task(
            self._watch_segment_audio(segment, is_live=info.is_live, remaining=remaining),
            name=f"audio-watchdog-{segment.index}",
        )
        abort_reason: str | None = None
        try:
            abort_reason = await await_capture_tasks(
                pump, watchdog, stop_pipe=pipe.stop, on_stuck=self.on_stuck
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("capture pump failed", extra={"segment": segment.index})
            return "pump_error"
        finally:
            # The pump at EOF means ffmpeg is exiting by itself: let it, so a clean end
            # is recorded with its own exit code, not our SIGTERM.
            natural_end = (
                abort_reason is None
                and pump.done()
                and not pump.cancelled()
                and pump.exception() is None
            )
            watchdog.cancel()
            pump.cancel()
            await asyncio.gather(pump, watchdog, return_exceptions=True)
            try:
                # The exit code exists only once stop() has reaped ffmpeg.
                await pipe.stop(grace=_EXIT_GRACE_S if natural_end else 0.0)
            finally:
                rc = pipe.returncode
                tail = pipe.stderr_tail
                diag = dict(pipe.diagnostics)
                self._pipe = None
                self._segment_edges = hls.live_edges()
                await hls.aclose()
                log.info("ffmpeg capture diagnostics", extra={"segment": segment.index, **diag})
                if tail and (rc not in (0, None) or abort_reason) and not self._closing.is_set():
                    log.warning(
                        "ffmpeg stderr",
                        extra={"segment": segment.index, "stderr": redact_text(tail[-800:])},
                    )
                elif tail:
                    log.debug(
                        "ffmpeg output",
                        extra={"segment": segment.index, "stderr": redact_text(tail[-800:])},
                    )

        if abort_reason:
            return abort_reason
        seg_seconds = (self._session_audio_samples - segment.audio_sample_base) / opts.sample_rate
        if remaining is not None and seg_seconds >= remaining - 0.5:
            return "duration_reached"
        if rc == 0:
            return "stream_ended"
        return f"ffmpeg_exit_{rc}"

    async def _watch_segment_audio(
        self, segment: SegmentInfo, *, is_live: bool, remaining: float | None = None
    ) -> str | None:
        """Say when a live segment must be stopped from the outside.

        Two verdicts, both returned before anything is stopped (:func:`await_capture_tasks`
        then stops ffmpeg):

        * ``"audio_stalled"``: no chunk arrived for ``audio_stall_seconds``, counted from
          the start of the segment so one that never produces anything is caught too.
          Off for lossless sources, where a slow consumer legitimately stops chunks.
        * ``"duration_reached"``: the wall clock passed the requested duration (plus a short
          grace). ffmpeg's ``-t`` counts media time, so a feed that stalls near the end would
          otherwise outlive the limit by the whole stall threshold.
        """
        stall_s = self.options.audio_stall_seconds
        watch_stall = is_live and stall_s > 0 and not self._lossless
        cutoff = (
            time.monotonic() + remaining + _DURATION_GRACE_S
            if is_live and remaining is not None
            else None
        )
        if not watch_stall and cutoff is None:
            return None
        poll_s = min(5.0, max(0.05, stall_s / 6)) if watch_stall else 1.0
        last_at = time.monotonic()
        seen = 0
        while True:
            wait = poll_s if cutoff is None else min(poll_s, max(0.05, cutoff - time.monotonic()))
            await asyncio.sleep(wait)
            now = time.monotonic()
            if cutoff is not None and now >= cutoff:
                log.info("requested duration reached", extra={"segment": segment.index})
                return "duration_reached"
            if not watch_stall:
                continue
            if segment.audio_chunks > seen:
                seen = segment.audio_chunks
                last_at = now
            if now - last_at >= stall_s:
                log.warning(
                    "audio stalled; restarting ffmpeg segment",
                    extra={
                        "segment": segment.index,
                        "audio_chunks": segment.audio_chunks,
                        "stall_s": round(now - last_at, 1),
                    },
                )
                return "audio_stalled"

    # ------------------------------------------------------------------ #
    # pump
    # ------------------------------------------------------------------ #

    async def _pump_audio(
        self, pipe: FFmpegAudioPipe, segment: SegmentInfo, *, is_live: bool
    ) -> None:
        """Re-chunk ffmpeg's PCM into fixed-length, time-stamped chunks."""
        rate = self.options.sample_rate
        target = self.options.chunk_bytes_for(live=is_live)
        buf = bytearray()
        last_emit = time.monotonic()

        async def emit(payload: bytes) -> None:
            nonlocal last_emit
            now = time.monotonic()
            gap = now - last_emit
            last_emit = now
            index = self._stats.audio_chunks_emitted
            if index and gap > 10.0:
                log.warning(
                    "audio capture gap",
                    extra={"gap_s": round(gap, 2), "index": index, "segment": segment.index},
                )
            # Timestamps come from counting samples. The session sample total survives
            # ffmpeg restarts; within a segment it is exact, and a partial tail chunk
            # (the last words before EOF) is stamped from the same offset.
            seg_samples = self._session_audio_samples - segment.audio_sample_base
            media_ts = seg_samples / rate
            chunk = AudioChunk(
                index=index,
                segment=segment.index,
                media_ts=media_ts,
                ts=segment.offset + media_ts,
                wallclock=time.time(),
                sample_rate=rate,
                pcm=payload,
            )
            self._session_audio_samples += len(payload) // 2
            segment.audio_chunks += 1
            self._stats.audio_chunks_emitted += 1
            self._stats.audio_seconds += chunk.duration
            self._first_data.set()
            if self._lossless:
                await self._audio_q.put_wait(chunk)
            elif not self._audio_q.put(chunk):
                dropped = self._audio_q.dropped
                # Dropped audio is lost speech, so this is loud; but a consumer that is
                # persistently behind would log on every chunk, so it thins out.
                if dropped in (1, 10) or dropped % 100 == 0:
                    log.error(
                        "audio queue overflow: transcription will have gaps",
                        extra={"dropped_total": dropped},
                    )

        async for data in pipe.read_audio():
            buf.extend(data)
            while len(buf) >= target:
                await emit(bytes(buf[:target]))
                del buf[:target]

        # Flush a partial tail so the last words of a segment are not lost.
        if len(buf) >= 2:
            await emit(bytes(buf[: len(buf) // 2 * 2]))
