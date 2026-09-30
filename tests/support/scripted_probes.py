"""Scripted candidate probes for source-selection tests. No network."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from livestream_transcriber.stream.fallback import (
    STATUS_BOT_BLOCKED,
    STATUS_LIVE,
    STATUS_OFFLINE,
    STATUS_PROBE_ERROR,
    LiveSourceSelector,
    ProbeResult,
    SourceCandidate,
)

BOT = STATUS_BOT_BLOCKED
OFF = STATUS_OFFLINE
ERR = STATUS_PROBE_ERROR
LIVE = STATUS_LIVE

PRIMARY = SourceCandidate("primary", "https://streams.example.test/primary/live", "en")
BACKUP = SourceCandidate("backup", "https://streams.example.test/backup/live", "de")

BOT_ERROR = (
    "yt-dlp failed on https://streams.example.test/primary/live: "
    "ERROR: [site] abc: Sign in to confirm you're not a bot. "
    + "Use the cookies option for authentication. "
    * 12
)


def probe_row(candidate: SourceCandidate, status: str) -> ProbeResult:
    live = status == LIVE
    error = {BOT: BOT_ERROR, ERR: "yt-dlp failed: connection timed out"}.get(status)
    return ProbeResult(
        candidate.name,
        0.0,
        0.0,
        status,
        live,
        "stream-1" if live else None,
        candidate.url if live else None,
        error=error,
    )


class ScriptedProbe:
    """Answers each probe round with the next ``(primary, backup)`` status pair.

    Once the script is used up the last round repeats, so a loop that probes a
    little more than a test planned for does not crash. ``on_round`` is called with
    the number of finished rounds after each.
    """

    def __init__(
        self,
        *rounds: tuple[str, str],
        on_round: Callable[[int], None] | None = None,
    ) -> None:
        self.rounds = list(rounds)
        self.calls = 0
        self.rounds_used = 0
        self.on_round = on_round
        self._current: tuple[str, str] | None = None

    async def __call__(
        self, candidate: SourceCandidate, format_selector: str, **_kw: Any
    ) -> ProbeResult:
        self.calls += 1
        if candidate is PRIMARY or self._current is None:
            index = min(self.rounds_used, len(self.rounds) - 1)
            self._current = self.rounds[index]
            self.rounds_used += 1
        status = self._current[0] if candidate is PRIMARY else self._current[1]
        row = probe_row(candidate, status)
        if candidate is BACKUP and self.on_round is not None:
            self.on_round(self.rounds_used)
        return row


def scripted_selector(
    *rounds: tuple[str, str], **kwargs: Any
) -> tuple[LiveSourceSelector, ScriptedProbe]:
    probe = ScriptedProbe(*rounds, **kwargs)
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector="best", probe=probe)
    return selector, probe


class AlertRecorder:
    """An ``alert`` callable that records what was sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def __call__(self, title: str, body: str) -> None:
        self.sent.append((title, body))

    @property
    def titles(self) -> list[str]:
        return [title for title, _body in self.sent]

    @property
    def bodies(self) -> list[str]:
        return [body for _title, body in self.sent]
