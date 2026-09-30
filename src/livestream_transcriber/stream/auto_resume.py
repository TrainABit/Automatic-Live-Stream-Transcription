"""Wait for a stream to come (back) online and report availability changes.

The pieces, from the bottom up:

* :func:`interruptible_sleep` and :func:`until_stopped` make every wait cancellable
  by a stop event, so a shutdown never waits out a probe or a back-off;
* :class:`ProbeBackoff` spaces probes: normal cadence while answers are conclusive,
  exponential once a bot check or probe error hides them;
* :class:`SourceStatusMonitor` turns a stream of probe rounds into *alerts*, once per
  real change instead of once per round;
* :func:`wait_until_live` ties them together into the idle loop of a session.

Alerts leave through a plain ``async (title, body)`` callable, so this module knows
nothing about which notifier (webhook, chat bot, log) is configured.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from ..logging_setup import get_logger
from .fallback import (
    STATE_BOT_CHECK,
    STATE_LIVE,
    STATE_OFFLINE,
    STATE_UNCERTAIN,
    STATUS_BOT_BLOCKED,
    STATUS_PROBE_ERROR,
    LiveSourceSelector,
    SourceSelection,
    format_selection_lines,
)

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

__all__ = [
    "AlertFn",
    "ProbeBackoff",
    "SourceStatusMonitor",
    "interruptible_sleep",
    "notify_stream_transition",
    "until_stopped",
    "wait_until_live",
]

T = TypeVar("T")

AlertFn = Callable[[str, str], Awaitable[None]]
"""``await alert(title, body)``: deliver one operator-facing message."""

_STATE_TITLES = {
    STATE_LIVE: "Stream online",
    STATE_OFFLINE: "Stream offline",
    STATE_BOT_CHECK: "Bot check in force",
    STATE_UNCERTAIN: "Stream status uncertain",
}

# 2**32 base intervals is already far past any sane cap; bounding the exponent keeps
# a week-long block from overflowing the float.
_MAX_BACKOFF_EXPONENT = 32

# Probe rounds a change between two not-live states must hold before it is alerted.
# An intermittent block answers bot-check, offline, bot-check, ... round after round;
# each flip is not news.
_CONFIRM_ROUNDS = 2

# Rounds in a row without a bot check that end a bot-check episode. One clean round
# in the middle of an intermittent block is not its end (the probe back-off keeps its
# level for the same two rounds).
_EPISODE_CLEAR_ROUNDS = 2


def _has_bot_check(selection: SourceSelection) -> bool:
    return any(row.status == STATUS_BOT_BLOCKED for row in selection.probe_results)


def _has_failed_probe(selection: SourceSelection) -> bool:
    return any(
        row.status in (STATUS_BOT_BLOCKED, STATUS_PROBE_ERROR) for row in selection.probe_results
    )


async def interruptible_sleep(seconds: float, stop: asyncio.Event) -> bool:
    """Sleep up to *seconds* unless *stop* is set. Returns True if stopped."""
    if seconds <= 0:
        return stop.is_set()
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except TimeoutError:
        return stop.is_set()


async def until_stopped(aw: Awaitable[T], stop: asyncio.Event) -> T | None:
    """Await ``aw`` unless ``stop`` is set first; then cancel it, return None.

    A probe round is several extractions, each up to its socket timeout; a shutdown
    signal must not wait them out. A cancelled round starts no further extraction;
    one already running in its worker thread cannot be interrupted and ends on its
    own, unread.
    """
    task = asyncio.ensure_future(aw)
    stopper = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        raise
    finally:
        stopper.cancel()
    if task.done():
        return task.result()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return None


async def _emit(alert: AlertFn | None, title: str, body: str, *, kind: str, **fields: Any) -> bool:
    """The one exit for stream-lifecycle alerts.

    Always a WARNING line; plus the configured callable, whose failure must never
    break the capture loop that raised the alert.
    """
    log.warning(
        "stream alert",
        extra={
            "kind": kind,
            "title": title,
            "alert": " | ".join(filter(None, body.splitlines())),
            **fields,
        },
    )
    if alert is None:
        return False
    try:
        await alert(title, body)
    except Exception:
        log.exception("alert delivery failed", extra={"kind": kind})
        return False
    return True


async def notify_stream_transition(
    alert: AlertFn | None,
    *,
    online: bool,
    selection: SourceSelection | None = None,
    detail: str | None = None,
) -> None:
    """A one-off online/offline alert for a session without source selection."""
    lines = [detail] if detail else []
    if selection is not None:
        lines.extend(format_selection_lines(selection))
    title = _STATE_TITLES[STATE_LIVE if online else STATE_OFFLINE]
    await _emit(alert, title, "\n".join(lines), kind="stream_status", online=online)


class SourceStatusMonitor:
    """Alert when the source availability *state* changes.

    The state is one of live / offline / bot_check / uncertain. Alerting only on
    online/offline flips would hide a bot check that started while nothing was live:
    blocked and offline are both "not capturable". Idle and in-session probes feed the
    same monitor, so a session that ends into a state already reported does not
    report it twice.

    A change to or from ``live`` is alerted at once. A change between two not-live
    states is alerted once it held for ``confirm_rounds`` probe rounds: an
    intermittent block that flips between bot check and offline every round would
    otherwise page twice a minute.

    A bot check is also tracked as an *episode*: it begins with a round that hits one
    and ends after a few rounds in a row without one. An episode is alerted once, as
    soon as ``confirm_rounds`` of its rounds hit the bot check, whether or not they
    were consecutive. A live candidate whose peer hit the bot check is alerted once
    per episode too; the state stays ``live`` because capture is allowed.
    """

    def __init__(self, alert: AlertFn | None = None, *, confirm_rounds: int = _CONFIRM_ROUNDS):
        self.alert = alert
        self.confirm_rounds = max(1, int(confirm_rounds))
        self.state: str | None = None
        """The state last reported (or assumed)."""
        self._pending: str | None = None
        self._pending_rounds = 0
        self._last_round: SourceSelection | None = None
        # The current bot-check episode: whether it was alerted, how many of its
        # rounds hit the bot check, and the clean rounds in a row since.
        self._bot_reported = False
        self._bot_rounds = 0
        self._clean_rounds = 0

    def assume(self, state: str) -> None:
        """Take ``state`` as reported already, so it is not reported twice."""
        self.state = state

    async def observe(
        self,
        selection: SourceSelection,
        *,
        detail: str | None = None,
        next_probe_s: float | None = None,
        in_session: bool = False,
        immediate: bool = False,
    ) -> bool:
        """Alert if ``selection`` changes the state. Returns whether it did.

        The same selection observed again (a session's last round handed on to the
        idle loop, a held answer) is not another round. ``immediate`` skips the
        confirmation rounds: the caller acts on this answer at once.
        """
        state = selection.state
        new_round = selection is not self._last_round
        self._last_round = selection
        has_bot = _has_bot_check(selection)
        if new_round:
            self._count_episode(has_bot)
        if state == self.state:
            self._pending, self._pending_rounds = None, 0
            if new_round and selection.capture_allowed:
                await self._peer_bot_check(selection, detail=detail)
            return False
        settled = self._settled(state, new_round=new_round)
        if not (immediate or settled or self._bot_episode_due(state)):
            return False
        previous, self.state = self.state, state
        self._pending, self._pending_rounds = None, 0
        self._bot_reported = self._bot_reported or has_bot
        # Only live and offline decide ``online``. A bot check or an uncertain probe
        # says neither: inside a session the capture goes on on its media URLs, so it
        # must not read as offline. None leaves the field undecided.
        online = True if state == STATE_LIVE else False if state == STATE_OFFLINE else None
        log.info(
            "stream availability transition",
            extra={
                "online": online,
                "state": state,
                "previous": previous,
                "reason": selection.selection_reason,
                "blocked": selection.blocked,
            },
        )
        lines = [detail] if detail else []
        lines.extend(format_selection_lines(selection))
        if previous is not None:
            lines.append(f"was: {previous}")
        if in_session and selection.probe_failed:
            lines.append("capture continues on the current media URLs")
        elif next_probe_s is not None and not selection.capture_allowed:
            lines.append(f"next probe in {next_probe_s:.0f} s")
        await _emit(
            self.alert,
            _STATE_TITLES[state],
            "\n".join(lines),
            kind="stream_status",
            state=state,
            previous=previous,
            reason=selection.selection_reason,
        )
        return True

    def _count_episode(self, has_bot: bool) -> None:
        if has_bot:
            self._bot_rounds += 1
            self._clean_rounds = 0
            return
        self._clean_rounds += 1
        if self._clean_rounds >= _EPISODE_CLEAR_ROUNDS:
            self._bot_rounds = 0
            self._bot_reported = False

    def _bot_episode_due(self, state: str) -> bool:
        """An unreported bot-check episode reached ``confirm_rounds`` rounds."""
        return (
            state == STATE_BOT_CHECK
            and not self._bot_reported
            and self._bot_rounds >= self.confirm_rounds
        )

    def _settled(self, state: str, *, new_round: bool) -> bool:
        """Whether a change to ``state`` is due for an alert now."""
        if self.state is None or STATE_LIVE in (state, self.state):
            return True
        if new_round:
            if self._pending == state:
                self._pending_rounds += 1
            else:
                self._pending, self._pending_rounds = state, 1
        if self._pending == state and self._pending_rounds >= self.confirm_rounds:
            return True
        if new_round:
            log.info(
                "stream availability change not confirmed yet",
                extra={"state": state, "reported": self.state, "rounds": self._pending_rounds},
            )
        return False

    async def _peer_bot_check(self, selection: SourceSelection, *, detail: str | None) -> None:
        """One alert when a live candidate's peer probe hits the bot check.

        Capture is allowed, so the state stays ``live``; but the address is flagged,
        and the next stream on the other candidate may not be seen.
        """
        if self._bot_reported or not _has_bot_check(selection):
            return
        self._bot_reported = True
        lines = [detail] if detail else []
        lines.extend(format_selection_lines(selection))
        lines.append(f"capture continues on {selection.selected_name}")
        await _emit(
            self.alert,
            _STATE_TITLES[STATE_BOT_CHECK],
            "\n".join(lines),
            kind="peer_bot_check",
            state=STATE_LIVE,
            reason=selection.selection_reason,
        )


class ProbeBackoff:
    """Spacing between source probes.

    The base interval while probes are conclusive. Every further round in which a bot
    check or probe error hides the answer doubles it, up to ``max_s``; the first
    conclusive round resets it. Repeating extractions every 30 s from a flagged
    address is how a bot check stays in force.

    A conclusive round drops the interval to the base at once (a stream that starts
    right after a block is seen within one base interval), but the level reached is
    kept until two clean rounds in a row (no probe failed at all) have passed. A block
    that comes straight back, the pattern of an intermittent one, carries on doubling
    from there instead of starting over every other round.
    """

    def __init__(
        self,
        base_s: float,
        max_s: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_s = float(base_s)
        self.max_s = max(self.base_s, float(base_s if max_s is None else max_s))
        self.failures = 0
        # The level a relapse resumes from, and the clean rounds since.
        self._relapse = 0
        self._clean = 0
        self._clock = clock
        self._last_at: float | None = None
        self._last_selection: SourceSelection | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ProbeBackoff:
        return cls(settings.resume_probe_interval_seconds, settings.resume_backoff_max_seconds)

    @property
    def delay(self) -> float:
        """Seconds from the last probe round to the next one."""
        if self.failures <= 1:
            # One failed round is often a blip: retry at the normal cadence.
            return self.base_s
        exponent = min(self.failures - 1, _MAX_BACKOFF_EXPONENT)
        return min(self.max_s, self.base_s * 2.0**exponent)

    def observe(self, selection: SourceSelection) -> float:
        """Account one probe round and return the delay before the next.

        A selection already accounted for (the round that ended a session is handed
        on to the idle loop) is not counted twice.
        """
        if selection is self._last_selection:
            return self.delay
        self._last_selection = selection
        return self._account(failed=selection.probe_failed, clean=not _has_failed_probe(selection))

    def record_failure(self) -> float:
        """A failed attempt outside a probe round (a live stream that would not
        resolve for capture). Counts toward the back-off like one."""
        return self._account(failed=True, clean=False)

    def record_not_live(self) -> float:
        """A connect refused because the resolve says the broadcast is over.

        A conclusive answer, like an offline round: no back-off, but the next probe
        still waits a full interval from now (no hot loop while the live listing lags
        the stream end)."""
        return self._account(failed=False, clean=False)

    def due(self) -> bool:
        """Whether an in-session probe may run now.

        Always true while probes are conclusive: a reconnect that asks is rare and
        worth an answer. While backing off, only once the delay is over.
        """
        return self.failures == 0 or self.remaining() <= 0

    def remaining(self) -> float:
        """Seconds until the next probe round is due (0 if none ran yet)."""
        if self._last_at is None:
            return 0.0
        return max(0.0, self._last_at + self.delay - self._clock())

    def _account(self, *, failed: bool, clean: bool) -> float:
        before = self.delay
        self._last_at = self._clock()
        if failed:
            self.failures = max(self.failures, self._relapse) + 1
            self._clean = 0
        else:
            # Conclusive (a candidate may be live while another's probe failed:
            # capture goes ahead, but that round is not clean).
            self._relapse = max(self._relapse, self.failures)
            self.failures = 0
            self._clean = self._clean + 1 if clean else 0
            if self._clean >= 2:
                self._relapse = 0
        after = self.delay
        if after != before:
            log.info(
                "source probe interval changed",
                extra={"interval_s": after, "failed_rounds": self.failures},
            )
        return after


async def wait_until_live(
    selector: LiveSourceSelector,
    stop: asyncio.Event,
    *,
    alert: AlertFn | None = None,
    status: SourceStatusMonitor | None = None,
    backoff: ProbeBackoff | None = None,
    stable_seconds: float = 0.0,
    initial: SourceSelection | None = None,
    settings: Settings | None = None,
) -> SourceSelection | None:
    """Probe the candidates until one is capturable, or *stop* is set.

    ``initial`` is a selection the caller already holds (the start-up probe, or the
    one that ended the last session). It is acted on without asking the platform
    again; the next probe waits for the back-off interval. Without one, the first
    probe still waits out a back-off already running (a session held through a bot
    check, a stream that would not open).

    ``stable_seconds``: a candidate must stay live that long before it is returned,
    so a stream that flaps at start-up does not open a session per flap. With
    ``settings``, the back-off and stable window come from its ``resume_*`` fields
    unless passed explicitly.
    """
    if backoff is None:
        backoff = (
            ProbeBackoff.from_settings(settings)
            if settings is not None
            else ProbeBackoff(30.0, 900.0)
        )
    if settings is not None and stable_seconds == 0.0:
        stable_seconds = settings.resume_online_stable_seconds
    status = status if status is not None else SourceStatusMonitor(alert)
    stable_s = max(0.0, float(stable_seconds))
    stable_deadline: float | None = None
    selection = initial
    while not stop.is_set():
        if selection is None:
            # The one place this loop waits: until the last round (here, in a
            # session, or a failed connect) is ``delay`` old.
            wait_s = backoff.remaining()
            if wait_s > 0 and await interruptible_sleep(wait_s, stop):
                break
            selection = await until_stopped(selector.select(), stop)
            if selection is None or stop.is_set():
                # Shutdown while probing: do not open a session.
                break
        delay = backoff.observe(selection)
        await status.observe(selection, next_probe_s=delay)
        if selection.capture_allowed:
            if stable_s <= 0:
                return selection
            now = time.monotonic()
            if stable_deadline is None:
                stable_deadline = now + stable_s
            elif now >= stable_deadline:
                return selection
        else:
            stable_deadline = None
        # State changes are logged (and alerted) by the monitor; a repeat of the same
        # answer every 30 s for hours is debug detail.
        log.debug(
            "no candidate capturable yet; probing again",
            extra={"state": selection.state, "interval_s": delay},
        )
        selection = None
    return None
