"""Ordered fallback between several candidate sources.

A run can be given a primary URL and any number of backups (a second channel that
simulcasts, a mirror, a lower-quality feed). :class:`LiveSourceSelector` probes all
candidates concurrently, :func:`decide_source` turns the answers into one decision,
and :func:`maybe_failover` tells a running capture when to move.

Selection rules, in order:

1. Exactly one candidate live: capture it, even if another probe failed.
2. Several live: the first in list order. Order is the operator's preference.
3. None live: capture is blocked, and *why* decides how the caller waits. A bot check
   or a probe error hides the answer, so the caller backs off; a plain "offline" is a
   conclusive answer.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from ..logging_setup import get_logger
from ..models import StreamInfo
from .base import StreamResolutionError
from .resolver import resolve_stream

log = get_logger(__name__)

__all__ = [
    "BLOCKED_BOT",
    "BLOCKED_NONE_LIVE",
    "BLOCKED_UNCERTAIN",
    "STATE_BOT_CHECK",
    "STATE_LIVE",
    "STATE_OFFLINE",
    "STATE_UNCERTAIN",
    "STATUS_BOT_BLOCKED",
    "STATUS_LIVE",
    "STATUS_OFFLINE",
    "STATUS_PROBE_ERROR",
    "LiveSourceSelector",
    "ProbeFn",
    "ProbeResult",
    "SourceCandidate",
    "SourceSelection",
    "classify_probe_error",
    "decide_source",
    "format_selection_lines",
    "maybe_failover",
    "probe_candidate",
    "short_error",
]

# What a single probe found.
STATUS_LIVE = "LIVE"
STATUS_OFFLINE = "OFFLINE"
STATUS_PROBE_ERROR = "PROBE_ERROR"
STATUS_BOT_BLOCKED = "BOT_BLOCKED"

# Why capture is blocked when no candidate is confirmed live.
BLOCKED_NONE_LIVE = "no candidate is currently live"
BLOCKED_UNCERTAIN = "live status is uncertain"
BLOCKED_BOT = "the platform is asking for a bot check"

# Availability states, one per thing an operator would act on differently.
# ``SourceSelection.state`` maps every selection onto exactly one of them.
STATE_LIVE = "live"
STATE_OFFLINE = "offline"
STATE_BOT_CHECK = "bot_check"
STATE_UNCERTAIN = "uncertain"

_OFFLINE_MARKERS = (
    "not currently live",
    "live event will begin",
    "premiere will begin",
    "is offline",
)
_BOT_MARKERS = (
    "confirm you're not a bot",
    "sign in to confirm",
)

# How much of a probe error reaches a log line. The full text is extractor noise
# repeated on every probe; the head says what went wrong.
_LOG_ERROR_CHARS = 160


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """One place a stream might be live: a display name, a URL and its language."""

    name: str
    url: str
    language: str | None = None
    """STT language hint for this source; ``None`` lets the model detect it."""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    probe_start: float
    probe_end: float
    status: str
    is_live: bool
    stream_id: str | None
    url: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SourceSelection:
    probe_results: tuple[ProbeResult, ...]
    selected_name: str | None
    selected_url: str | None
    selected_stream_id: str | None
    selection_reason: str
    blocked: str | None
    capture_allowed: bool

    @property
    def state(self) -> str:
        """``live``, ``offline``, ``bot_check`` or ``uncertain``."""
        if self.capture_allowed:
            return STATE_LIVE
        if self.blocked == BLOCKED_BOT:
            return STATE_BOT_CHECK
        if self.blocked == BLOCKED_UNCERTAIN:
            return STATE_UNCERTAIN
        return STATE_OFFLINE

    @property
    def probe_failed(self) -> bool:
        """No candidate is confirmed live and at least one probe could not tell.

        Back-off applies here: repeating extractions every 30 s from a flagged
        address is what keeps a bot check in force.
        """
        return self.state in (STATE_BOT_CHECK, STATE_UNCERTAIN)


ProbeFn = Callable[..., Awaitable[ProbeResult]]


def _normalize_url(url: str) -> str:
    """Compare URLs ignoring a trailing slash and the case of scheme and host."""
    parts = urlsplit(url.strip())
    head = f"{parts.scheme}://{parts.netloc}".lower()
    tail = parts.path.rstrip("/") + (f"?{parts.query}" if parts.query else "")
    return head + tail


def classify_probe_error(message: str) -> str:
    """Map a resolver error text onto a probe status."""
    text = (message or "").lower().replace("\u2019", "'")  # typographic apostrophe
    if any(marker in text for marker in _BOT_MARKERS):
        return STATUS_BOT_BLOCKED
    if any(marker in text for marker in _OFFLINE_MARKERS):
        return STATUS_OFFLINE
    return STATUS_PROBE_ERROR


def _stream_id(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"(?:v=|/live/|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    return match.group(1) if match else None


def short_error(message: str | None, limit: int = _LOG_ERROR_CHARS) -> str | None:
    """A probe error cut to one log line: URLs dropped, whitespace folded."""
    if not message:
        return message
    text = " ".join(re.sub(r"https?://\S+", "<url>", message).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _select(rows: tuple[ProbeResult, ...], row: ProbeResult, reason: str) -> SourceSelection:
    return SourceSelection(
        probe_results=rows,
        selected_name=row.name,
        selected_url=row.url,
        selected_stream_id=row.stream_id,
        selection_reason=reason,
        blocked=None,
        capture_allowed=True,
    )


def _blocked(rows: tuple[ProbeResult, ...], reason: str, blocked: str) -> SourceSelection:
    return SourceSelection(
        probe_results=rows,
        selected_name=None,
        selected_url=None,
        selected_stream_id=None,
        selection_reason=reason,
        blocked=blocked,
        capture_allowed=False,
    )


def decide_source(probes: Sequence[ProbeResult]) -> SourceSelection:
    """Pick a candidate to capture.

    A confirmed live candidate is captured even when another probe failed or
    several are live. Refusal stays for the case where no candidate is confirmed
    live (offline, bot check, or a probe error).
    """
    rows = tuple(probes)
    live_rows = [row for row in rows if row.status == STATUS_LIVE and row.url]
    if len(live_rows) == 1:
        chosen = live_rows[0]
        peer_failed = any(
            other.status in {STATUS_PROBE_ERROR, STATUS_BOT_BLOCKED}
            for other in rows
            if other is not chosen
        )
        reason = "live_despite_probe_failure" if peer_failed else "exactly_one_live"
        return _select(rows, chosen, reason)
    if len(live_rows) > 1:
        # Candidate order is the operator's preference: stable, and logged.
        return _select(rows, live_rows[0], "multiple_live_prefer_listed_order")

    if any(row.status == STATUS_BOT_BLOCKED for row in rows):
        # A bot check hides whether that candidate is live. Keep refusing to
        # capture, but do not file it under a generic probe error: that would make a
        # day-long block look like a transient network blip. A bot check next to a
        # probe error is still a bot check, since the address is flagged and that is
        # what an operator acts on; filing it as "uncertain" would make a block that
        # answers bot-check and error in turn flip between two states unalerted.
        return _blocked(rows, "bot_check", BLOCKED_BOT)
    if any(row.status == STATUS_PROBE_ERROR for row in rows):
        return _blocked(rows, "live_status_uncertain", BLOCKED_UNCERTAIN)
    return _blocked(rows, "no_candidate_currently_live", BLOCKED_NONE_LIVE)


def format_selection_lines(selection: SourceSelection) -> list[str]:
    lines = [f"{row.name}: {row.status}" for row in selection.probe_results]
    if selection.selected_name:
        lines.append(f"selected: {selection.selected_name}")
    elif selection.blocked:
        lines.append(f"blocked: {selection.blocked}")
    return lines


def maybe_failover(current_url: str, selection: SourceSelection) -> tuple[str | None, str | None]:
    """``(new_url, alert_text)`` when the capture should move, else ``(None, None)``.

    It moves only if the URL being captured is no longer live, so two live
    candidates do not make it flap. Initial selection still prefers list order.
    """
    if not selection.capture_allowed or not selection.selected_url:
        return None, None
    current = _normalize_url(current_url)
    if current == _normalize_url(selection.selected_url):
        return None, None
    still_live = {
        _normalize_url(row.url)
        for row in selection.probe_results
        if row.status == STATUS_LIVE and row.url
    }
    if current in still_live:
        return None, None
    lines = format_selection_lines(selection)
    alert = "Source failover\n\n" + "\n".join(lines) + f"\n\nreason={selection.selection_reason}"
    return selection.selected_url, alert


async def probe_candidate(
    candidate: SourceCandidate,
    format_selector: str,
    *,
    cookiefile: str | None = None,
    proxy: str | None = None,
) -> ProbeResult:
    """Resolve one candidate and report whether it is live."""
    started = time.time()
    try:
        info: StreamInfo = await resolve_stream(
            candidate.url,
            format_selector=format_selector,
            cookiefile=cookiefile,
            proxy=proxy,
        )
    except StreamResolutionError as exc:
        return ProbeResult(
            name=candidate.name,
            probe_start=started,
            probe_end=time.time(),
            status=classify_probe_error(str(exc)),
            is_live=False,
            stream_id=None,
            url=None,
            error=str(exc),
        )
    live = info.is_live
    return ProbeResult(
        name=candidate.name,
        probe_start=started,
        probe_end=time.time(),
        status=STATUS_LIVE if live else STATUS_OFFLINE,
        is_live=live,
        # A channel-level URL names no broadcast; the resolve says which one it is.
        stream_id=(info.stream_id or _stream_id(info.url)) if live else None,
        url=candidate.url if live else None,
        error=None,
    )


class LiveSourceSelector:
    """Probe an ordered list of candidates and decide which one to capture."""

    def __init__(
        self,
        candidates: Sequence[SourceCandidate],
        *,
        format_selector: str = "bestaudio/best",
        cookiefile: str | None = None,
        proxy: str | None = None,
        probe: ProbeFn | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("at least one source candidate is required")
        self.candidates: tuple[SourceCandidate, ...] = tuple(candidates)
        self.format_selector = format_selector
        self.cookiefile = cookiefile
        self.proxy = proxy
        self._probe = probe or probe_candidate
        self.last_selection: SourceSelection | None = None
        self._last_status: dict[str, str] = {}

    def candidate_for(self, url: str | None) -> SourceCandidate | None:
        """The candidate that ``url`` belongs to."""
        if not url:
            return None
        wanted = _normalize_url(url)
        for candidate in self.candidates:
            if _normalize_url(candidate.url) == wanted:
                return candidate
        return None

    def language_for(self, url: str | None) -> str | None:
        """STT language hint of the candidate ``url`` belongs to, if it has one."""
        candidate = self.candidate_for(url)
        return candidate.language if candidate else None

    def status_of(self, selection: SourceSelection, url: str | None) -> str | None:
        """What ``selection`` found for the candidate ``url`` belongs to."""
        candidate = self.candidate_for(url)
        if candidate is None:
            return None
        for row in selection.probe_results:
            if row.name == candidate.name:
                return row.status
        return None

    async def _probe_one(self, candidate: SourceCandidate) -> ProbeResult:
        started = time.time()
        try:
            return await self._probe(
                candidate,
                self.format_selector,
                cookiefile=self.cookiefile,
                proxy=self.proxy,
            )
        except Exception as exc:
            # A probe that blows up must not take the other probes' answers with
            # it. It is reported as "could not tell", which backs the caller off.
            log.warning(
                "source probe crashed",
                extra={"candidate": candidate.name, "error": type(exc).__name__},
            )
            return ProbeResult(
                name=candidate.name,
                probe_start=started,
                probe_end=time.time(),
                status=STATUS_PROBE_ERROR,
                is_live=False,
                stream_id=None,
                url=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def select(self) -> SourceSelection:
        """Probe every candidate at once and decide.

        One after another, N candidates would cost N times the slowest extraction;
        the probes are independent, so they run concurrently.
        """
        results = await asyncio.gather(*(self._probe_one(c) for c in self.candidates))
        for row in results:
            # An idle process probes every candidate every 30 s for hours. Only a
            # change of a candidate's status is news; a repeat is debug noise.
            changed = self._last_status.get(row.name) != row.status
            self._last_status[row.name] = row.status
            log.log(
                logging.INFO if changed else logging.DEBUG,
                "source probe",
                extra={
                    "candidate": row.name,
                    "status": row.status,
                    "stream_id": row.stream_id,
                    "error": short_error(row.error),
                },
            )
        selection = decide_source(results)
        self.last_selection = selection
        return selection
