"""The process-level loop: wait for a stream, capture it, transcribe it, and go again.

One :class:`SessionRunner` is one ``lst run`` (or ``record`` / ``replay``). It owns
everything that outlives a single capture session:

* signal handling: the first SIGINT/SIGTERM stops gracefully (in-flight speech-to-text
  is abandoned, outputs and the recording are finalised), the second exits at once;
* the source-availability state (probe back-off, status monitor, health) and the
  heartbeat, so an idle process still proves it is alive;
* the database and the notification dispatcher, so events not yet delivered when one
  session ended are retried in the next.

A *session* is one connection to one source. ``run`` loops: optionally wait until a
candidate is live (``wait_until_live``), connect, drive audio into a fresh
:class:`~.engine.TranscriptionPipeline` until the source ends, then either exit or,
for a live source with resuming enabled, go back to waiting. Failing over between
candidates mid-session is the job of :mod:`.reselect`.

Exit codes are part of the command line contract, see :class:`ExitCode`.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..config import ConfigError, Settings
from ..heartbeat import ProcessHeartbeat
from ..logging_setup import get_logger
from ..models import AudioChunk, StreamInfo
from ..notify.base import Event, make_event_id
from ..notify.factory import build_dispatcher, missing_targets
from ..outputs import (
    CompositeSink,
    ConsoleSink,
    JsonlSink,
    SqliteSink,
    SrtSink,
    TranscriptSink,
    VttSink,
)
from ..redact import redact_text, redact_url
from ..rules import LlmMatcher, RuleEngine, RuleSet, build_llm_matcher
from ..rules.model import Severity
from ..source_health import HealthWatch, SourceHealth, TrackedStatusMonitor
from ..store import Database
from ..stream import (
    LiveSourceSelector,
    LiveStreamSource,
    ProbeBackoff,
    Recorder,
    RecordingError,
    ReplayStreamSource,
    SourceCandidate,
    SourceSelection,
    StreamError,
    StreamNotLiveError,
    StreamResolutionError,
    StreamSource,
)
from ..stream.auto_resume import notify_stream_transition, until_stopped, wait_until_live
from ..stream.fallback import STATE_LIVE, format_selection_lines, short_error
from ..stt.base import Transcriber
from ..stt.factory import build_transcriber, close_transcriber, provider_problem
from ..stt.wrappers import SttSpend
from ..systemd_notify import SystemdNotifier
from .engine import PipelineOptions, TranscriptionPipeline
from .reselect import attach_reselect

if TYPE_CHECKING:
    from ..notify.dispatcher import NotificationDispatcher

log = get_logger(__name__)

__all__ = [
    "ExitCode",
    "SessionPlan",
    "SessionReport",
    "SessionRunner",
    "build_sinks",
    "is_remote_url",
]

# Refused opens in a row (listed live, resolved as over) before they raise an alert: one
# or two are the live listing lagging a stream end.
NOT_LIVE_OPENS_BEFORE_ALERT = 3
# A chunk this recent means media is flowing.
DELIVERING_WITHIN_SECONDS = 10.0
_REMOTE_SCHEMES = ("http://", "https://", "rtmp://", "rtmps://", "rtsp://", "udp://", "srt://")


class ExitCode(IntEnum):
    """The command line's exit codes."""

    OK = 0
    CONFIG = 2
    NOT_LIVE = 3
    NO_DATA = 4
    STREAM_LOST = 5
    INTERRUPTED = 130


def is_remote_url(url: str) -> bool:
    """True for a network URL; False for a local path (which is never probed or resumed)."""
    return url.lower().startswith(_REMOTE_SCHEMES)


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """What to run. The command line builds one; tests build them directly."""

    url: str
    """A stream URL, media URL or file for ``live`` and ``record``; a recording directory
    for ``replay``."""
    mode: Literal["live", "record", "replay"] = "live"
    fallbacks: tuple[str, ...] = ()
    """Backup sources, in order of preference after ``url``."""
    duration: float | None = None
    """Stop capturing after this many seconds of media."""
    record_dir: Path | None = None
    """Write the raw audio here (``record`` mode, or ``run --record``)."""
    replay_speed: float = 0.0
    """``replay`` only: 1.0 is real time, 0 is unthrottled and deterministic."""
    realtime: bool = False
    """Read a file at its native rate, to simulate a live feed."""
    resume: bool | None = None
    """Wait for the stream to come back after it ends. ``None`` follows the settings."""
    transcribe: bool = True
    """False captures without speech-to-text (``lst record``)."""
    out_dir: Path | None = None
    """Where transcript files go; ``None`` uses ``settings.out_dir``."""
    note: str | None = None
    mock_fixtures: Path | None = None
    """JSONL transcripts for the ``mock`` provider (demos and tests)."""
    console: bool = True
    """Print transcript lines to stdout."""


@dataclass(slots=True)
class SessionReport:
    """What one capture session did, for tests and the final log line."""

    url: str
    session_id: int | None = None
    chunks: int = 0
    seconds: float = 0.0
    utterances: int = 0
    hits: int = 0
    stream_lost: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Progress:
    """Capture progress, shared by the consumer, the heartbeat and the hold watch."""

    started: float = field(default_factory=time.monotonic)
    last_chunk: float | None = None
    chunks: int = 0
    seconds: float = 0.0

    def note(self, chunk: AudioChunk) -> None:
        self.chunks += 1
        self.seconds += chunk.duration
        self.last_chunk = time.monotonic()

    def idle_seconds(self) -> float:
        reference = self.last_chunk if self.last_chunk is not None else self.started
        return time.monotonic() - reference

    def delivering(self) -> bool:
        return self.last_chunk is not None and (
            time.monotonic() - self.last_chunk <= DELIVERING_WITHIN_SECONDS
        )


@dataclass(slots=True)
class _SessionParts:
    """What one capture session owns, closed together when it ends."""

    pipeline: TranscriptionPipeline
    sinks: CompositeSink
    transcriber: Transcriber | None
    recorder: Recorder | None
    live: bool
    """A real-time source, as opposed to a file or replay that is read losslessly."""

    def close_outputs(self) -> None:
        self.sinks.close()
        if self.transcriber is not None:
            close_transcriber(self.transcriber)


@dataclass(slots=True)
class _Outcome:
    """How a capture session ended: an exit code, or ``resume`` to wait for the stream again."""

    code: int | None = None
    resume: bool = False
    selection: SourceSelection | None = None


def build_sinks(
    settings: Settings,
    *,
    out_dir: Path,
    stem: str,
    db: Database | None,
    session_id: int | None,
    console: bool,
    color: bool | None,
) -> list[TranscriptSink]:
    """The transcript destinations of one session: console, JSONL, SRT, VTT and SQLite."""
    sinks: list[TranscriptSink] = []
    if console:
        sinks.append(ConsoleSink(color=color))
    sinks.append(JsonlSink(out_dir / f"{stem}.jsonl", append=False))
    sinks.append(SrtSink(out_dir / f"{stem}.srt"))
    sinks.append(VttSink(out_dir / f"{stem}.vtt"))
    if db is not None:
        sinks.append(SqliteSink(db, session_id))
    return sinks


_TRANSCRIPT_SUFFIXES = (".jsonl", ".srt", ".vtt")


def free_transcript_stem(out_dir: Path, session_number: int = 1) -> str:
    """The file stem for a session's transcript files, without touching earlier output.

    ``transcript`` for the first session, ``transcript-2`` for the next, and so on. A
    stem whose files already exist (a restarted process writing into the same
    directory) is skipped, so a supervisor restart never overwrites what was captured
    before it.
    """
    number = max(1, session_number)
    while True:
        stem = "transcript" if number == 1 else f"transcript-{number}"
        if not any((out_dir / f"{stem}{suffix}").exists() for suffix in _TRANSCRIPT_SUFFIXES):
            return stem
        number += 1


class AlertRouter:
    """Send operator-facing messages (stream online/offline, failover, STT trouble).

    Every alert is already logged where it is raised; this forwards it to the
    configured network notifiers (webhook, Telegram). The console needs no copy.
    """

    def __init__(self, dispatcher: NotificationDispatcher | None) -> None:
        self._dispatcher = dispatcher

    async def __call__(self, title: str, body: str) -> None:
        dispatcher = self._dispatcher
        if dispatcher is None:
            return
        now = time.time()
        event = Event(
            event_id=make_event_id("stream-alert", title, now, Severity.WARNING, scope=str(now)),
            rule_id="stream-alert",
            text=body,
            matched_text=title,
            start=0.0,
            end=0.0,
            wallclock=now,
            severity=Severity.WARNING,
        )
        for name, notifier in dispatcher.notifiers.items():
            if name == "console":
                continue
            try:
                await asyncio.to_thread(notifier.send, event)
            except Exception as exc:
                log.warning(
                    "alert delivery failed",
                    extra={"target": name, "error": redact_text(f"{type(exc).__name__}: {exc}")},
                )


class SessionRunner:
    """Run a plan to completion and return an :class:`ExitCode`."""

    def __init__(
        self,
        settings: Settings,
        plan: SessionPlan,
        *,
        ruleset: RuleSet | None = None,
        color: bool | None = None,
        stop: asyncio.Event | None = None,
        install_signals: bool = True,
        source_factory: Callable[[str], StreamSource] | None = None,
        selector: LiveSourceSelector | None = None,
        transcriber_factory: Callable[[Settings], Transcriber | None] | None = None,
        systemd: SystemdNotifier | None = None,
        backoff: ProbeBackoff | None = None,
    ) -> None:
        self.settings = settings
        self.plan = plan
        self.ruleset = ruleset
        self.color = color
        self.stop = stop if stop is not None else asyncio.Event()
        self.install_signals = install_signals
        self._source_factory = source_factory
        self._selector = selector
        self._transcriber_factory = transcriber_factory
        self._backoff = backoff
        self.systemd = systemd if systemd is not None else SystemdNotifier()
        self.reports: list[SessionReport] = []
        self.pipeline: TranscriptionPipeline | None = None
        """The most recent session's pipeline (kept for inspection after the run)."""
        self._spend = SttSpend()
        self._sessions = 0
        # Opens in a row that were refused (listed live, resolved as over) or failed.
        self._opens_refused = 0
        self._open_failures = 0
        self._db: Database | None = None
        self._dispatcher: NotificationDispatcher | None = None
        self._alerts = AlertRouter(None)
        self._llm: LlmMatcher | None = None

    # ------------------------------------------------------------------ signals

    def _request_stop(self, signame: str) -> None:
        """First signal: stop gracefully. Second: leave now."""
        if self.stop.is_set():
            log.warning("second %s received; exiting now", signame)
            raise SystemExit(int(ExitCode.INTERRUPTED))
        log.info("%s received; shutting down cleanly (repeat to exit at once)", signame)
        self.stop.set()

    @contextlib.contextmanager
    def _signals(self) -> Any:
        installed: list[signal.Signals] = []
        if self.install_signals:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.add_signal_handler(sig, self._request_stop, sig.name)
                    installed.append(sig)
        try:
            yield
        finally:
            loop = asyncio.get_running_loop()
            for sig in installed:
                with contextlib.suppress(Exception):
                    loop.remove_signal_handler(sig)

    # ---------------------------------------------------------------------- run

    async def run(self) -> int:
        """Run until the plan is done, the stream is gone for good, or a signal arrives."""
        try:
            self._preflight()
        except ConfigError as exc:
            log.error("cannot start: %s", exc)
            return int(ExitCode.CONFIG)
        plan, settings = self.plan, self.settings
        if (
            plan.record_dir is not None
            and plan.record_dir.is_dir()
            and any(plan.record_dir.iterdir())
        ):
            log.error("refusing to overwrite the recording directory %s", plan.record_dir)
            return int(ExitCode.CONFIG)

        settings.ensure_dirs()
        (plan.out_dir or settings.out_dir).mkdir(parents=True, exist_ok=True)
        self._db = Database(settings.database_path)
        try:
            self._dispatcher = build_dispatcher(
                settings, self._db, ruleset=self.ruleset, color=self.color
            )
        except ConfigError as exc:
            self._db.close()
            log.error("cannot start: %s", exc)
            return int(ExitCode.CONFIG)
        self._alerts = AlertRouter(self._dispatcher)
        if self.ruleset is not None:
            for rule_id, targets in missing_targets(
                self.ruleset, self._dispatcher.notifiers
            ).items():
                log.warning(
                    "rule names a notifier that is not configured",
                    extra={"rule": rule_id, "targets": list(targets)},
                )
            self._llm = build_llm_matcher(settings, self.ruleset)

        source_health = SourceHealth()
        status = TrackedStatusMonitor(self._alerts, source_health)
        backoff = self._backoff or ProbeBackoff.from_settings(settings)
        resume = self._resume_enabled()
        heartbeat = ProcessHeartbeat(
            source=source_health,
            backoff=backoff,
            watch=HealthWatch(self._alerts, stt_outage_seconds=settings.stt_outage_seconds),
            idle_beats=resume,
            memory_limit_mb=settings.memory_limit_mb,
        )
        background: list[asyncio.Task[None]] = []
        try:
            with self._signals():
                if settings.heartbeat_seconds > 0:
                    background.append(
                        asyncio.create_task(
                            heartbeat.run(self.stop, settings.heartbeat_seconds),
                            name="process-heartbeat",
                        )
                    )
                if self.systemd.enabled:
                    self.systemd.ready(f"transcribing {redact_url(plan.url)}")
                    background.append(
                        asyncio.create_task(
                            self.systemd.watchdog_loop(self.stop), name="systemd-watchdog"
                        )
                    )
                try:
                    return await self._loop(
                        status=status, backoff=backoff, heartbeat=heartbeat, resume=resume
                    )
                except ConfigError as exc:
                    # Raised when a session builds its speech-to-text chain: a provider
                    # that could not start is a configuration problem, not a crash.
                    log.error("cannot start: %s", exc)
                    return int(ExitCode.CONFIG)
        finally:
            self.systemd.set_active(False)
            heartbeat.detach()
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            if self._db is not None:
                self._db.close()
                self._db = None

    def _preflight(self) -> None:
        """Fail before any capture when the configuration cannot work."""
        plan, settings = self.plan, self.settings
        if plan.mode == "replay":
            if not Path(plan.url).is_dir():
                raise ConfigError(f"recording directory not found: {plan.url}")
        elif not is_remote_url(plan.url) and not Path(plan.url).exists():
            raise ConfigError(f"file not found: {plan.url}")
        if plan.transcribe and self._transcriber_factory is None:
            settings.validate_stt()
            for name in dict.fromkeys((settings.stt_provider, settings.stt_fallback)):
                if name == "none" and name != settings.stt_provider:
                    continue
                problem = provider_problem(name, settings)
                if problem is not None:
                    raise ConfigError(f"STT provider {name!r} cannot start: {problem}")

    def _resume_enabled(self) -> bool:
        plan = self.plan
        if plan.mode != "live" or plan.duration is not None:
            return False
        if plan.resume is not None:
            return plan.resume and is_remote_url(plan.url)
        return self.settings.resume_auto and is_remote_url(plan.url)

    def _build_selector(self, *, needed: bool) -> LiveSourceSelector | None:
        if self._selector is not None:
            return self._selector
        plan, settings = self.plan, self.settings
        if plan.mode != "live" or not is_remote_url(plan.url) or not needed:
            return None
        candidates = [SourceCandidate("primary", plan.url, settings.stt_language)]
        candidates.extend(
            SourceCandidate(f"fallback-{n}", url, settings.stt_language)
            for n, url in enumerate(plan.fallbacks, start=1)
        )
        proxy = settings.capture_proxy.get_secret_value() if settings.capture_proxy else None
        cookies = str(settings.capture_cookies_file) if settings.capture_cookies_file else None
        return LiveSourceSelector(
            candidates,
            format_selector=settings.capture_stream_format,
            cookiefile=cookies,
            proxy=proxy,
        )

    # -------------------------------------------------------------------- loop

    async def _loop(
        self,
        *,
        status: TrackedStatusMonitor,
        backoff: ProbeBackoff,
        heartbeat: ProcessHeartbeat,
        resume: bool,
    ) -> int:
        plan, settings, stop = self.plan, self.settings, self.stop
        selector = self._build_selector(needed=resume or bool(plan.fallbacks))
        url = plan.url
        selection: SourceSelection | None = None
        wait_first = False

        if selector is not None and plan.fallbacks:
            # Several candidates: ask which one is live before opening anything.
            selection = await until_stopped(selector.select(), stop)
            if selection is None:
                return int(ExitCode.INTERRUPTED)
            for line in format_selection_lines(selection):
                log.info("source selection", extra={"line": line})
            if selection.capture_allowed and selection.selected_url:
                url = selection.selected_url
            elif resume:
                wait_first = True
            else:
                log.error("no candidate source is live")
                return int(ExitCode.NOT_LIVE)

        resumed = False
        while True:
            if wait_first:
                assert selector is not None
                selection = await wait_until_live(
                    selector,
                    stop,
                    status=status,
                    backoff=backoff,
                    initial=selection,
                    settings=settings,
                )
                if selection is None:
                    return int(ExitCode.INTERRUPTED if stop.is_set() else ExitCode.NOT_LIVE)
                assert selection.selected_url
                url = selection.selected_url
                wait_first, resumed = False, True

            outcome = await self._session(
                url,
                selection,
                selector=selector,
                status=status,
                backoff=backoff,
                heartbeat=heartbeat,
                resume=resume,
                resumed=resumed,
            )
            if outcome.code is not None:
                return outcome.code
            assert outcome.resume
            wait_first, selection = True, outcome.selection

    async def _session(
        self,
        url: str,
        selection: SourceSelection | None,
        *,
        selector: LiveSourceSelector | None,
        status: TrackedStatusMonitor,
        backoff: ProbeBackoff,
        heartbeat: ProcessHeartbeat,
        resume: bool,
        resumed: bool,
    ) -> _Outcome:
        """Open one source, drive it to its end and finish. Returns how it ended."""
        plan, stop = self.plan, self.stop
        source = self._make_source(url)
        if isinstance(source, LiveStreamSource):
            # A probe found this URL live: a resolve that says otherwise is a failed
            # open, not a recording to capture.
            source.require_live = selection is not None and selection.capture_allowed

        try:
            info = await until_stopped(source.connect(), stop)
        except StreamNotLiveError as exc:
            self._opens_refused += 1
            log.warning(
                "listed live, but the broadcast is over",
                extra={"url": redact_url(url), "refused_opens": self._opens_refused},
            )
            await source.close()
            if resume and not stop.is_set():
                delay = backoff.record_not_live()
                if self._opens_refused == NOT_LIVE_OPENS_BEFORE_ALERT and self._open_failures == 0:
                    await self._alerts(
                        "Stream would not open",
                        f"listed live {self._opens_refused} times in a row, but every resolve says "
                        f"the broadcast is over: {short_error(redact_text(str(exc)))}\n"
                        f"next probe in {delay:.0f} s",
                    )
                return _Outcome(resume=True)
            return _Outcome(code=int(ExitCode.NOT_LIVE))
        except (StreamResolutionError, RecordingError) as exc:
            log.error(
                "could not open the source",
                extra={"url": redact_url(url), "error": redact_text(str(exc))},
            )
            await source.close()
            if isinstance(exc, RecordingError):
                return _Outcome(code=int(ExitCode.CONFIG))
            if resume and not stop.is_set():
                delay = backoff.record_failure()
                self._open_failures += 1
                if self._open_failures == 1:
                    await self._alerts(
                        "Stream would not open",
                        f"the capture could not open it: {short_error(redact_text(str(exc)))}\n"
                        f"next probe in {delay:.0f} s",
                    )
                return _Outcome(resume=True)
            return _Outcome(code=int(ExitCode.NOT_LIVE))
        if info is None or stop.is_set():
            await source.close()
            return _Outcome(code=int(ExitCode.INTERRUPTED))
        self._opens_refused = 0
        self._open_failures = 0

        if resumed:
            log.info(
                "capture session starting after the stream came online",
                extra={"title": info.title, "url": redact_url(url)},
            )
        watcher = selector if selector is not None and info.is_live else None
        if selection is not None:
            backoff.observe(selection)
            await status.observe(selection, detail=info.title or redact_url(url))
        elif info.is_live and plan.mode == "live":
            await notify_stream_transition(self._alerts, online=True, detail=info.title or url)
            status.assume(STATE_LIVE)
        return await self._drive_session(
            source,
            info,
            url,
            selection,
            watcher=watcher,
            status=status,
            backoff=backoff,
            heartbeat=heartbeat,
            resume=resume,
        )

    def _make_source(self, url: str) -> StreamSource:
        plan = self.plan
        if self._source_factory is not None:
            return self._source_factory(url)
        if plan.mode == "replay":
            return ReplayStreamSource(url, speed=plan.replay_speed, deterministic=True)
        return LiveStreamSource.from_settings(
            url,
            self.settings,
            duration=plan.duration,
            realtime=plan.realtime,
            on_stuck=self._on_stuck,
            on_hold=lambda held: self.systemd.set_active(not held),
        )

    def _on_stuck(self, reason: str) -> None:
        log.error("capture was stuck and was restarted", extra={"reason": reason})

    def _new_transcriber(self, url: str, selector: LiveSourceSelector | None) -> Transcriber | None:
        plan, settings = self.plan, self.settings
        if not plan.transcribe:
            return None
        language = selector.language_for(url) if selector is not None else None
        if language:
            settings = settings.with_overrides(stt_language=language)
        if self._transcriber_factory is not None:
            return self._transcriber_factory(settings)
        return build_transcriber(
            settings,
            fixtures=plan.mock_fixtures,
            budget_spend=self._spend,
            cache_dir=settings.stt_cache_dir,
        )

    # ----------------------------------------------------------- one session

    async def _drive_session(
        self,
        source: StreamSource,
        info: StreamInfo,
        url: str,
        selection: SourceSelection | None,
        *,
        watcher: LiveSourceSelector | None,
        status: TrackedStatusMonitor,
        backoff: ProbeBackoff,
        heartbeat: ProcessHeartbeat,
        resume: bool,
    ) -> _Outcome:
        """Run one capture session from an opened source to its end, and say what happens next."""
        self._sessions += 1
        progress = _Progress()
        report = SessionReport(url=redact_url(url))
        self.reports.append(report)
        session_end = asyncio.Event()
        stream_lost: StreamError | None = None
        abandoned = False

        async with contextlib.AsyncExitStack() as cleanup:
            cleanup.push_async_callback(self._close_source, source)
            parts = await self._open_session(source, info, url, watcher, report)
            hold = self._start_hold_watch(source, watcher, session_end, status, backoff, progress)
            started = time.monotonic()

            def summary() -> dict[str, Any]:
                return {
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "audio_chunks": progress.chunks,
                    "audio_seconds": round(progress.seconds, 1),
                    "delivering": progress.delivering(),
                    **parts.pipeline.health(),
                }

            self.systemd.set_active(True)
            if info.is_live:
                heartbeat.attach(summary)
            try:
                abandoned = await self._drive(
                    source, parts.pipeline, parts.recorder, progress, session_end=session_end
                )
            except StreamError as exc:
                # The capture failed, but what it delivered before that is still worth
                # transcribing: only a stop or a crash abandons in-flight work.
                stream_lost = exc
            except BaseException:
                abandoned = True
                raise
            finally:
                self.systemd.set_active(False)
                heartbeat.detach()
                if hold is not None:
                    hold.cancel()
                    await asyncio.gather(hold, return_exceptions=True)
                await self._wind_down(
                    parts,
                    source,
                    report,
                    progress,
                    abandoned=abandoned,
                    ending=session_end.is_set() or stream_lost is not None,
                )

        return await self._session_outcome(
            report,
            progress,
            stream_lost=stream_lost,
            source_offline=session_end.is_set(),
            watcher=watcher,
            resume=resume,
        )

    async def _open_session(
        self,
        source: StreamSource,
        info: StreamInfo,
        url: str,
        watcher: LiveSourceSelector | None,
        report: SessionReport,
    ) -> _SessionParts:
        """Create everything one session owns: its record, recorder, outputs and pipeline."""
        plan, settings = self.plan, self.settings
        db = self._db
        assert db is not None
        out_dir = plan.out_dir or settings.out_dir
        session_id = db.start_session(
            url=redact_url(url),
            info=info,
            mode="file" if plan.mode == "live" and not is_remote_url(url) else plan.mode,
            recording_dir=str(plan.record_dir) if plan.record_dir else None,
            config=settings.redacted_summary(),
        )
        report.session_id = session_id
        recorder = self._open_recorder(source, info)
        sinks: CompositeSink | None = None
        transcriber: Transcriber | None = None
        try:
            # Capture-only (`lst record`) has nothing to write: no empty transcript files.
            sinks = CompositeSink(
                build_sinks(
                    settings,
                    out_dir=out_dir,
                    stem=free_transcript_stem(out_dir, self._sessions),
                    db=db,
                    session_id=session_id,
                    console=plan.console,
                    color=self.color,
                )
                if plan.transcribe
                else []
            )
            transcriber = self._new_transcriber(url, watcher)
            live = info.is_live and not bool(getattr(source, "lossless", False))
            rules = (
                RuleEngine(self.ruleset, llm=self._llm)
                if self.ruleset is not None and transcriber is not None
                else None
            )
            pipeline = TranscriptionPipeline(
                transcriber,
                options=PipelineOptions.from_settings(settings, live=live),
                sinks=[sinks],
                rules=rules,
                dispatcher=self._dispatcher,
                alert=self._alerts,
                source_url=url,
                session_id=session_id,
            )
            self.pipeline = pipeline
            if self._dispatcher is not None:
                await self._dispatcher.retry_pending_async()
        except BaseException:
            if sinks is not None:
                sinks.close()
            if transcriber is not None:
                close_transcriber(transcriber)
            raise
        return _SessionParts(pipeline, sinks, transcriber, recorder, live)

    def _open_recorder(self, source: StreamSource, info: StreamInfo) -> Recorder | None:
        plan, settings = self.plan, self.settings
        if plan.record_dir is None:
            return None
        recorder = Recorder(
            plan.record_dir,
            sample_rate=settings.capture_sample_rate,
            chunk_seconds=(
                source.options.chunk_seconds_for(live=info.is_live)
                if isinstance(source, LiveStreamSource)
                else None
            ),
            stream_info=info,
            note=plan.note,
        )
        recorder.open()
        return recorder

    def _start_hold_watch(
        self,
        source: StreamSource,
        watcher: LiveSourceSelector | None,
        session_end: asyncio.Event,
        status: TrackedStatusMonitor,
        backoff: ProbeBackoff,
        progress: _Progress,
    ) -> asyncio.Task[None] | None:
        """Let a live source ask the selector on each reconnect; returns the hold-watch task."""
        if watcher is None or not isinstance(source, LiveStreamSource):
            return None
        settings = self.settings
        watch_hold = attach_reselect(
            source,
            watcher,
            session_end=session_end,
            status=status,
            backoff=backoff,
            media_idle_seconds=progress.idle_seconds,
            hold_seconds=settings.resume_uncertain_hold_seconds,
            reuse_seconds=settings.resume_probe_interval_seconds,
            alert=self._alerts,
            on_end=lambda: self.systemd.set_active(False),
        )
        return asyncio.create_task(watch_hold(), name="session-hold")

    async def _wind_down(
        self,
        parts: _SessionParts,
        source: StreamSource,
        report: SessionReport,
        progress: _Progress,
        *,
        abandoned: bool,
        ending: bool,
    ) -> None:
        """Finish or abandon what the session accepted, close its outputs, save its record.

        ``ending`` is true when the source went away (offline or lost): the drain is then
        bounded, because whatever is still queued will not get more audio behind it.
        """
        settings = self.settings
        db = self._db
        assert db is not None
        # Everything already accepted is finished (or abandoned) before the outputs
        # close, so the last utterances are not lost.
        drain_timeout = settings.resume_end_settle_seconds if parts.live else None
        await parts.pipeline.drain(
            wait=not (abandoned or self.stop.is_set()),
            timeout=drain_timeout if ending else None,
        )
        parts.close_outputs()
        report.chunks = progress.chunks
        report.seconds = progress.seconds
        report.utterances = parts.pipeline.speech_stats.utterances
        report.hits = parts.pipeline.speech_stats.hits
        report.stats = {**source.stats.describe(), **parts.pipeline.summary()}
        self._finish_records(db, report, source, parts.recorder)

    async def _session_outcome(
        self,
        report: SessionReport,
        progress: _Progress,
        *,
        stream_lost: StreamError | None,
        source_offline: bool,
        watcher: LiveSourceSelector | None,
        resume: bool,
    ) -> _Outcome:
        """Turn how a session ended into an exit code, or a request to wait and resume."""
        if self.stop.is_set():
            return _Outcome(code=int(ExitCode.INTERRUPTED))
        if stream_lost is not None:
            report.stream_lost = True
            log.error("stream lost", extra={"error": redact_text(str(stream_lost))})
            await self._alerts("Stream lost", redact_text(str(stream_lost)))
            return _Outcome(code=int(ExitCode.STREAM_LOST))
        audio = report.stats.get("audio", {})
        log.info(
            "session summary",
            extra={
                "chunks": report.chunks,
                "audio_s": round(report.seconds, 1),
                "utterances": report.utterances,
                "rule_hits": report.hits,
                "dropped": audio.get("dropped", 0),
                "skipped_while_paused": audio.get("skipped_while_paused", 0),
                "gaps": report.stats.get("speech", {}).get("gaps", 0),
            },
        )
        if resume and watcher is not None and (source_offline or progress.chunks):
            # A stream end is a session end, not a process end: exiting would leave the
            # next start to the supervisor and lose the resume back-off state.
            log.info(
                "stream session ended; waiting for the stream to come back",
                extra={"source_offline": source_offline},
            )
            return _Outcome(
                resume=True,
                selection=watcher.last_selection if source_offline else None,
            )
        # Otherwise: opened, delivered nothing, and nobody saw it go offline is a broken capture.
        if progress.chunks == 0:
            log.error("no audio was captured")
            return _Outcome(code=int(ExitCode.NO_DATA))
        if self.plan.record_dir is not None:
            log.info("replay it with: lst replay %s", self.plan.record_dir)
        return _Outcome(code=int(ExitCode.OK))

    async def _close_source(self, source: StreamSource) -> None:
        with contextlib.suppress(Exception):
            await source.close()

    def _finish_records(
        self,
        db: Database,
        report: SessionReport,
        source: StreamSource,
        recorder: Recorder | None,
    ) -> None:
        """Persist the session's end state. Each step is independent: one failing must
        not lose the others."""
        segments = getattr(source, "segments", None)
        if report.session_id is not None:
            if segments:
                with contextlib.suppress(Exception):
                    db.record_segments(report.session_id, segments)
            try:
                db.finish_session(report.session_id, report.stats)
            except Exception:
                log.exception("could not finish the session record")
        if recorder is not None:
            try:
                recorder.close(segments or None)
            except Exception:
                log.exception("could not finalise the recording")

    async def _drive(
        self,
        source: StreamSource,
        pipeline: TranscriptionPipeline,
        recorder: Recorder | None,
        progress: _Progress,
        *,
        session_end: asyncio.Event,
    ) -> bool:
        """Feed the source's audio to the pipeline until it ends. True if work was abandoned.

        Ends when the source runs dry, a stop is requested (in-flight speech-to-text is
        abandoned so shutdown cannot hang on a wedged provider), or ``session_end`` is set:
        the stream went away under a process that keeps running, so the source is closed
        and the audio already captured gets ``resume_end_settle_seconds`` to be transcribed.
        """
        stop = self.stop
        consumer = asyncio.create_task(
            self._consume(source, pipeline, recorder, progress), name="consume-audio"
        )
        stopper = asyncio.create_task(stop.wait(), name="stop")
        ender = asyncio.create_task(session_end.wait(), name="session-end")
        abandoned = False
        try:
            done, _ = await asyncio.wait(
                {consumer, stopper, ender}, return_when=asyncio.FIRST_COMPLETED
            )
            if consumer in done:
                await consumer
            elif stopper in done:
                abandoned = True
                await source.close()
            else:
                await source.close()
                settle = self.settings.resume_end_settle_seconds
                done, _ = await asyncio.wait(
                    {consumer, stopper}, timeout=settle, return_when=asyncio.FIRST_COMPLETED
                )
                if consumer in done:
                    await consumer
                else:
                    abandoned = True
                    log.warning(
                        "session tail still transcribing after the settle budget; abandoning it",
                        extra={"settle_s": settle},
                    )
        except BaseException:
            abandoned = True
            raise
        finally:
            for task in (consumer, stopper, ender):
                task.cancel()
            await asyncio.gather(consumer, stopper, ender, return_exceptions=True)
        return abandoned

    async def _consume(
        self,
        source: StreamSource,
        pipeline: TranscriptionPipeline,
        recorder: Recorder | None,
        progress: _Progress,
    ) -> None:
        recording = recorder
        async for chunk in source.get_audio():
            progress.note(chunk)
            self.systemd.note_progress()
            if recording is not None:
                try:
                    await recording.add_audio(chunk)
                except Exception:
                    # A full disk must not end the transcription; say so once and go on.
                    log.exception("recording failed; continuing without it")
                    recording = None
            await pipeline.on_audio(chunk, wait=False)
