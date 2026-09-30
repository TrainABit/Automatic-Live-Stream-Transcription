"""In-session source reselection: what a running capture does when its supervisor asks.

The capture supervisor calls ``reselect(current_url, reason, attempt)`` when a
reconnect suggests the source may be gone (a live stream that ended, a frozen
playlist, a streak of empty reconnects). The callback re-runs source selection and
answers with what the session should do:

* a candidate is live: keep capturing, or fail over to it (return its URL);
* every candidate is confirmed offline, or the captured one is offline and none is
  live: end the *session*, not the process, so the idle loop can wait for the stream
  to return;
* a bot check or probe error: the answer is unknown, and the signed media URLs the
  session already holds usually still work, so the session is kept. Only once no media
  has arrived for ``hold_seconds`` does the capture count as failed and end.

Probes back off here exactly as in the idle loop. Within one reconnect (the supervisor
asks before its resolve and again after a failed one) the answer is reused for up to
``reuse_seconds``: a second extraction seconds after the first says nothing new.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Coroutine
from typing import Any

from ..logging_setup import get_logger
from ..stream.auto_resume import AlertFn, ProbeBackoff, SourceStatusMonitor
from ..stream.fallback import (
    STATE_OFFLINE,
    STATUS_OFFLINE,
    LiveSourceSelector,
    SourceSelection,
    format_selection_lines,
    maybe_failover,
)
from ..stream.source import LiveStreamSource

log = get_logger(__name__)

__all__ = ["HOLD_CHECK_SECONDS", "attach_reselect"]

# How often a session whose live status is unknown checks how long it has gone
# without media.
HOLD_CHECK_SECONDS = 5.0


def attach_reselect(
    source: LiveStreamSource,
    selector: LiveSourceSelector,
    *,
    session_end: asyncio.Event,
    status: SourceStatusMonitor,
    backoff: ProbeBackoff,
    media_idle_seconds: Callable[[], float],
    hold_seconds: float,
    reuse_seconds: float,
    alert: AlertFn | None = None,
    on_end: Callable[[], None] | None = None,
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Install the reselect callback on ``source``; return the hold watch to run per session.

    The hold watch enforces ``hold_seconds`` on a timer: reconnects (the only time the
    callback runs) can be more than a minute apart, so the limit cannot rely on them.
    """
    reselector = _Reselector(
        source,
        selector,
        session_end=session_end,
        status=status,
        backoff=backoff,
        media_idle_seconds=media_idle_seconds,
        hold_seconds=hold_seconds,
        reuse_seconds=reuse_seconds,
        alert=alert,
        on_end=on_end,
    )
    source.reselect = reselector.reselect
    return reselector.watch_hold


class _Reselector:
    """The state and decisions behind one session's reselect callback."""

    def __init__(
        self,
        source: LiveStreamSource,
        selector: LiveSourceSelector,
        *,
        session_end: asyncio.Event,
        status: SourceStatusMonitor,
        backoff: ProbeBackoff,
        media_idle_seconds: Callable[[], float],
        hold_seconds: float,
        reuse_seconds: float,
        alert: AlertFn | None,
        on_end: Callable[[], None] | None,
    ) -> None:
        self.source = source
        self.selector = selector
        self.session_end = session_end
        self.status = status
        self.backoff = backoff
        self.media_idle_seconds = media_idle_seconds
        self.hold_seconds = hold_seconds
        self.reuse_seconds = reuse_seconds
        self.alert = alert
        self.on_end = on_end
        # The last in-session answer while it leaves the live status unknown (a bot check
        # or probe error, nothing confirmed live); None once a candidate is live.
        self.unknown: SourceSelection | None = None
        # The last round this session probed: for which reconnect ((reason, attempt), as
        # the supervisor passes them), when, and its answer.
        self.asked: tuple[tuple[str, int], float, SourceSelection] | None = None

    # ------------------------------------------------------------ side effects

    async def notify(self, title: str, body: str, **fields: object) -> None:
        log.warning("stream alert", extra={"title": title, **fields})
        if self.alert is None:
            return
        try:
            await self.alert(title, body)
        except Exception:
            log.exception("alert delivery failed", extra={"title": title})

    async def end_session(self, why: str, selection: SourceSelection, reason: str) -> None:
        if self.session_end.is_set():
            return
        log.info(
            "%s; ending capture session",
            why,
            extra={"reason": selection.selection_reason, "after": reason},
        )
        self.session_end.set()
        if self.on_end is not None:
            self.on_end()
        # This runs inside the capture supervisor. Closing the source cancels the
        # supervisor right here, before it can re-resolve a stream that is gone; the
        # shield keeps that cancel from reaching close() itself.
        await asyncio.shield(asyncio.ensure_future(self.source.close()))

    async def end_if_starved(self, selection: SourceSelection, reason: str) -> bool:
        idle = self.media_idle_seconds()
        if idle < self.hold_seconds or self.session_end.is_set():
            return False
        await self.notify(
            "Capture session ended",
            f"no media for {idle:.0f} s and the live status is unknown\n"
            + "\n".join(format_selection_lines(selection)),
            kind="session_end",
            after=reason,
        )
        await self.end_session("no media while the live status is unknown", selection, reason)
        return True

    # --------------------------------------------------------------- callbacks

    async def reselect(self, current: str, reason: str, attempt: int) -> str | None:
        """The supervisor's question: which URL should capture use now (None keeps it)?"""
        selection, held = await self._answer(reason, attempt)
        # Confirmed gone: every candidate offline, or none live and the captured one
        # offline while another's probe failed (a stream end during a partial block).
        # Only a failed probe of the captured candidate itself leaves its status unknown.
        gone = (
            not held
            and not selection.capture_allowed
            and (
                selection.state == STATE_OFFLINE
                or self.selector.status_of(selection, current) == STATUS_OFFLINE
            )
        )
        await self._report(selection, reason, gone=gone)
        if selection.capture_allowed:
            self.unknown = None
            new_url, failover_text = maybe_failover(current, selection)
            if failover_text:
                await self.notify("Source failover", failover_text, kind="failover", after=reason)
            return new_url
        if gone:
            what = "every candidate" if len(self.selector.candidates) > 1 else "the source"
            if selection.state != STATE_OFFLINE:
                what = "the captured source"
            await self.end_session(f"{what} offline", selection, reason)
            return None
        self.unknown = selection
        if await self.end_if_starved(selection, reason):
            return None
        log.warning(
            "live status unknown; keeping the capture session",
            extra={
                "state": selection.state,
                "after": reason,
                "media_idle_s": round(self.media_idle_seconds(), 1),
                "hold_s": self.hold_seconds,
            },
        )
        return None

    async def watch_hold(self) -> None:
        tick = min(HOLD_CHECK_SECONDS, self.hold_seconds / 4)
        while not self.session_end.is_set():
            await asyncio.sleep(tick)
            if self.unknown is not None:
                with contextlib.suppress(Exception):
                    await self.end_if_starved(self.unknown, "no_media")

    # ---------------------------------------------------------------- internals

    async def _answer(self, reason: str, attempt: int) -> tuple[SourceSelection, bool]:
        """The selection to act on, and whether the probe back-off held a fresh one back."""
        key = (reason, attempt)
        asked = self.asked
        if (
            asked is not None
            and asked[0] == key
            and time.monotonic() - asked[1] <= self.reuse_seconds
        ):
            log.debug(
                "source reselect reuses this reconnect's probe",
                extra={"state": asked[2].state, "after": reason, "attempt": attempt},
            )
            return asked[2], False
        last = self.selector.last_selection
        if not self.backoff.due() and last is not None:
            log.info(
                "source reselect held by probe back-off",
                extra={
                    "state": last.state,
                    "next_probe_s": round(self.backoff.remaining(), 1),
                    "after": reason,
                    "attempt": attempt,
                },
            )
            return last, True
        selection = await self.selector.select()
        self.asked = (key, time.monotonic(), selection)
        self.backoff.observe(selection)
        for line in format_selection_lines(selection):
            log.info("source reselect", extra={"line": line, "after": reason, "attempt": attempt})
        return selection, False

    async def _report(self, selection: SourceSelection, reason: str, *, gone: bool) -> None:
        """Tell the status monitor (and the operator) what this probe found."""
        detail = f"after={reason}"
        if gone and selection.state != STATE_OFFLINE:
            detail += "; the captured source is offline, session ended"
        reported = await self.status.observe(
            selection, detail=detail, in_session=not gone, immediate=gone
        )
        if gone and not reported:
            # The state had been reported already (a probe error earlier in this
            # session); the session end itself is still news.
            await self.notify(
                "Capture session ended",
                f"{detail}\n" + "\n".join(format_selection_lines(selection)),
                kind="session_end",
                after=reason,
            )
