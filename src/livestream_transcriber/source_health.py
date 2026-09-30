"""Source-selection health: what the heartbeat reports and what pages an operator.

Two small pieces of process-wide state, fed by every probe round the process acts
on (the idle loop, a session start and an in-session reselect all pass through the
same :class:`~.stream.auto_resume.SourceStatusMonitor`):

* :class:`SourceHealth` keeps the last selection and how long the probes have been
  unable to tell whether anything is live;
* :class:`HealthWatch` raises an alert for conditions no single component edge-triggers
  by itself: speech-to-text failing for a long time, or the source blocked for a long
  time. Each alerts once per episode and re-arms when the episode is over, never per
  heartbeat tick.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .logging_setup import get_logger
from .stream.auto_resume import AlertFn, ProbeBackoff, SourceStatusMonitor
from .stream.fallback import (
    STATUS_BOT_BLOCKED,
    SourceSelection,
    format_selection_lines,
)

log = get_logger(__name__)

__all__ = [
    "SOURCE_HEALTH_FIELDS",
    "HealthWatch",
    "SourceHealth",
    "TrackedStatusMonitor",
]

SOURCE_HEALTH_FIELDS = (
    "selection_state",
    "selection_reason",
    "selection_blocked_s",
    "bot_check",
    "next_probe_s",
    "last_probe_age_s",
)
"""The source fields of the heartbeat. Names are stable and ``None`` means unknown.

``selection_state``      the last probe round's answer: ``live``, ``offline``,
                         ``bot_check`` or ``uncertain``; ``selection_reason`` says why.
``selection_blocked_s``  seconds the probes have been unable to tell whether anything
                         is live (a bot check or probe errors round after round);
                         ``None`` while the answer is conclusive, and while a live
                         capture's media flows, which is an answer too.
``bot_check``            the last round hit a bot check on at least one candidate.
``next_probe_s``         seconds until the probe loop asks again (it backs off while
                         blocked).
``last_probe_age_s``     seconds since the last probe round the process acted on. The
                         heartbeat is its own task, so a probe loop that stopped
                         probing would still beat ``capture=idle``; this is its trace.
"""


class SourceHealth:
    """Source-selection state for the heartbeat and the "source blocked" alert."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.selection: SourceSelection | None = None
        self.blocked_since: float | None = None
        self.observed_at: float | None = None

    def observe(self, selection: SourceSelection) -> None:
        if selection is self.selection:
            return  # the same round handed on (a held answer, a session end)
        self.selection = selection
        self.observed_at = self._clock()
        if selection.probe_failed:
            if self.blocked_since is None:
                self.blocked_since = self._clock()
        else:
            self.blocked_since = None

    def capture_delivering(self) -> None:
        """A live capture's media is flowing: that answers what a probe would.

        Without this, one inconclusive in-session round (a bot check on a reconnect,
        after which the session carries on with the media URLs it holds) would keep the
        blocked clock running with no probe behind it to stop it.
        """
        self.blocked_since = None

    def blocked_seconds(self) -> float | None:
        if self.blocked_since is None:
            return None
        return max(0.0, self._clock() - self.blocked_since)

    def fields(self, backoff: ProbeBackoff | None = None) -> dict[str, Any]:
        """:data:`SOURCE_HEALTH_FIELDS`."""
        selection = self.selection
        blocked = self.blocked_seconds()
        return {
            "selection_state": None if selection is None else selection.state,
            "selection_reason": None if selection is None else selection.selection_reason,
            "selection_blocked_s": None if blocked is None else round(blocked, 1),
            "bot_check": (
                None
                if selection is None
                else any(row.status == STATUS_BOT_BLOCKED for row in selection.probe_results)
            ),
            "next_probe_s": None if backoff is None else round(backoff.remaining(), 1),
            "last_probe_age_s": (
                None
                if self.observed_at is None
                else round(max(0.0, self._clock() - self.observed_at), 1)
            ),
        }


class TrackedStatusMonitor(SourceStatusMonitor):
    """The process's status monitor, which also feeds :class:`SourceHealth`."""

    def __init__(self, alert: AlertFn | None, source: SourceHealth) -> None:
        super().__init__(alert)
        self.source = source

    async def observe(self, selection: SourceSelection, **kwargs: Any) -> bool:
        self.source.observe(selection)
        return await super().observe(selection, **kwargs)


class HealthWatch:
    """Alerts for long-running trouble, checked on every heartbeat.

    * **speech-to-text failing**: the pipeline's ``stt_outage_s`` reached
      ``stt_outage_seconds``. The episode belongs to the process, not the session, so
      an outage that outlasts several sessions pages once; it re-arms only after the
      provider answers (the outage clock clears).
    * **source blocked**: the probes could not tell whether anything is live for
      ``source_blocked_seconds``. Re-armed by a conclusive round or by media flowing.

    A threshold of 0 turns that check off.
    """

    def __init__(
        self,
        alert: AlertFn | None,
        *,
        stt_outage_seconds: float = 0.0,
        source_blocked_seconds: float = 0.0,
    ) -> None:
        self.alert = alert
        self.stt_outage_seconds = max(0.0, float(stt_outage_seconds))
        self.source_blocked_seconds = max(0.0, float(source_blocked_seconds))
        self.stt_alerted = False
        self.blocked_alerted = False

    async def check(
        self,
        *,
        capture: dict[str, Any] | None,
        source: SourceHealth,
        next_probe_s: float | None = None,
    ) -> None:
        if capture is not None:
            await self._check_stt(capture)
        await self._check_source(source, next_probe_s)

    async def _send(self, title: str, body: str, **fields: Any) -> None:
        log.warning(
            "health alert",
            extra={"title": title, "alert": " | ".join(filter(None, body.splitlines())), **fields},
        )
        if self.alert is None:
            return
        try:
            await self.alert(title, body)
        except Exception:
            log.exception("alert delivery failed", extra={"title": title})

    async def _check_stt(self, capture: dict[str, Any]) -> None:
        outage = capture.get("stt_outage_s")
        if outage is None:
            self.stt_alerted = False
            return
        if (
            self.stt_outage_seconds <= 0
            or self.stt_alerted
            or float(outage) < self.stt_outage_seconds
        ):
            return
        self.stt_alerted = True
        paused = capture.get("stt_paused")
        await self._send(
            "Speech-to-text is failing",
            f"No transcript for {float(outage) / 60:.0f} min because the provider keeps "
            f"failing. Transcription paused: {'yes' if paused else 'no'}.",
            kind="stt_outage",
            outage_s=round(float(outage), 1),
        )

    async def _check_source(self, source: SourceHealth, next_probe_s: float | None) -> None:
        blocked = source.blocked_seconds()
        if blocked is None:
            self.blocked_alerted = False
            return
        if (
            self.source_blocked_seconds <= 0
            or self.blocked_alerted
            or blocked < self.source_blocked_seconds
        ):
            return
        self.blocked_alerted = True
        selection = source.selection
        lines = [
            f"For {blocked / 60:.0f} min no probe could tell whether a source is live; "
            "a stream that starts now may be missed."
        ]
        if selection is not None:
            lines.extend(format_selection_lines(selection))
        if next_probe_s is not None:
            lines.append(f"next probe in {next_probe_s:.0f} s")
        await self._send(
            "Source blocked",
            "\n".join(lines),
            kind="source_blocked",
            state=None if selection is None else selection.state,
            blocked_s=round(blocked, 1),
        )
