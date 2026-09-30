"""Decide when speech-to-text has fallen too far behind, and when it is back.

A transcriber that answers slower than the stream plays never catches up: its
queue grows until memory or disk spill runs out, and every transcript arrives
minutes late. Rather than let that happen silently, the guard watches two
signals over a sliding window and *pauses* transcription when either one says
the lane cannot keep up:

* **lag**: how far the oldest chunk still owed a transcript trails the audio
  head, sustained for a whole window (a single slow request is noise);
* **drop ratio**: the share of chunks that were dropped (queue overflow) among
  those finished in the window, given enough samples to mean anything.

While paused, nothing is sent to the provider except an occasional *probe*: one
audible chunk per interval that asks "are you back?". A healthy answer resumes
transcription. A relapse shortly after resuming doubles the probe interval, so
a provider that is only half working costs one pause/resume pair per growing
interval rather than one per minute.

The guard is deliberately free of asyncio and of the pipeline: it takes numbers
in and returns verdicts, with an injectable clock, so every branch is testable
without sleeping.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..logging_setup import get_logger

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

__all__ = ["Overload", "OverloadConfig", "OverloadGuard"]

REASON_LAG = "lag"
REASON_DROPS = "drops"
REASON_OUTAGE = "outage"


@dataclass(frozen=True, slots=True)
class OverloadConfig:
    """Thresholds of the guard. A zero limit disables that signal."""

    max_lag_seconds: float = 45.0
    max_drop_ratio: float = 0.25
    window_seconds: float = 120.0
    probe_interval_seconds: float = 60.0
    min_samples: int = 12
    """Fewer finished chunks than this in the window say nothing about a ratio."""
    stable_seconds: float = 600.0
    """A pause sooner than this after resuming counts as a relapse."""
    max_probe_seconds: float = 600.0
    max_fallback_hold_seconds: float = 3600.0

    @classmethod
    def from_settings(cls, settings: Settings) -> OverloadConfig:
        return cls(
            max_lag_seconds=settings.stt_max_lag_seconds,
            max_drop_ratio=settings.stt_max_drop_ratio,
            window_seconds=settings.stt_overload_window_seconds,
            probe_interval_seconds=settings.stt_recovery_probe_seconds,
        )


@dataclass(frozen=True, slots=True)
class Overload:
    """A verdict: why the lane should be paused, with the numbers behind it."""

    reason: str
    lag_seconds: float | None
    drop_ratio: float | None

    def describe(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "lag_s": None if self.lag_seconds is None else round(self.lag_seconds, 1),
            "drop_ratio": None if self.drop_ratio is None else round(self.drop_ratio, 3),
        }


class OverloadGuard:
    """Pause and probe-based recovery state for one speech-to-text lane."""

    def __init__(
        self,
        config: OverloadConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or OverloadConfig()
        self._clock = clock
        self.paused = False
        self.pause_reason: str | None = None
        self.pauses = 0
        self.resumes = 0
        self.relapses = 0
        self.probe_interval = self.config.probe_interval_seconds
        self._outcomes: deque[tuple[float, bool]] = deque()
        self._lag_since: float | None = None
        self._last_probe = -math.inf
        self._probe_pending = False
        self._switched_on: float | None = None
        self._fallback_suspect = False
        self._fallback_hold_until: float | None = None

    # ------------------------------------------------------------- measuring

    def record_outcome(self, dropped: bool) -> None:
        """Note one finished chunk (``dropped`` = it never reached the provider)."""
        if self.paused:
            return
        now = self._clock()
        self._outcomes.append((now, dropped))
        horizon = now - self.config.window_seconds
        while self._outcomes and self._outcomes[0][0] < horizon:
            self._outcomes.popleft()

    def drop_ratio(self, now: float | None = None) -> tuple[float | None, int]:
        """Dropped share of the chunks finished inside the window, and how many there were.

        Read-only over a snapshot, so a health reporter on another thread may call it
        while the loop appends outcomes.
        """
        now = self._clock() if now is None else now
        horizon = now - self.config.window_seconds
        recent = [dropped for at, dropped in list(self._outcomes) if at >= horizon]
        if not recent:
            return None, 0
        return sum(recent) / len(recent), len(recent)

    def evaluate(self, lag_seconds: float | None) -> Overload | None:
        """Whether the lane should be paused now, given the current ``lag_seconds``."""
        if self.paused:
            return None
        cfg = self.config
        now = self._clock()
        lag_sustained = False
        if (
            cfg.max_lag_seconds > 0
            and lag_seconds is not None
            and lag_seconds > cfg.max_lag_seconds
        ):
            if self._lag_since is None:
                self._lag_since = now
            lag_sustained = now - self._lag_since >= cfg.window_seconds
        else:
            self._lag_since = None
        ratio, samples = self.drop_ratio(now)
        dropping = (
            cfg.max_drop_ratio > 0
            and ratio is not None
            and samples >= cfg.min_samples
            and ratio > cfg.max_drop_ratio
        )
        if lag_sustained:
            return Overload(REASON_LAG, lag_seconds, ratio)
        if dropping:
            return Overload(REASON_DROPS, lag_seconds, ratio)
        return None

    # ------------------------------------------------------ pause and recover

    def pause(self, reason: str, *, fallback_active: bool = False) -> float:
        """Enter the paused state and return the probe interval that applies.

        ``fallback_active`` says a local fallback was answering when the lane
        overloaded; that fallback is then suspected of being the slow part and is
        held off for a while after recovery (see :meth:`resume`).
        """
        now = self._clock()
        self.paused = True
        self.pause_reason = reason
        self.pauses += 1
        self._lag_since = None
        self._outcomes.clear()
        self._last_probe = now
        self._probe_pending = False
        # Relapsing soon after the last switch-on means the last probe was too
        # optimistic: probe less often each time.
        if self._switched_on is not None and now - self._switched_on < self.config.stable_seconds:
            self.relapses += 1
        else:
            self.relapses = 0
        cfg = self.config
        self.probe_interval = min(
            cfg.probe_interval_seconds * 2.0**self.relapses,
            max(cfg.probe_interval_seconds, cfg.max_probe_seconds),
        )
        if reason in (REASON_LAG, REASON_DROPS) and fallback_active:
            self._fallback_suspect = True
        self._fallback_hold_until = None
        log.warning(
            "stt paused",
            extra={
                "reason": reason,
                "probe_every_s": round(self.probe_interval, 1),
                "relapses": self.relapses,
            },
        )
        return self.probe_interval

    def take_probe(self, *, audible: bool) -> bool:
        """While paused: is this the chunk that tests whether speech-to-text recovered?

        Silence proves nothing about a provider, so only an audible chunk qualifies.
        """
        if not self.paused or self._probe_pending or not audible:
            return False
        now = self._clock()
        if now - self._last_probe < self.probe_interval:
            return False
        self._last_probe = now
        self._probe_pending = True
        return True

    def probe_ended(self, *, no_verdict: bool) -> None:
        """A probe finished. One that reached no provider proved nothing: probe again at once."""
        if not self._probe_pending:
            return
        self._probe_pending = False
        if no_verdict:
            self._last_probe = -math.inf

    def resume(self) -> float | None:
        """A probe came back healthy. Returns how long a suspect fallback stays off, if any."""
        if not self.paused:
            return None
        now = self._clock()
        reason = self.pause_reason
        self.paused = False
        self.pause_reason = None
        self._lag_since = None
        self._outcomes.clear()
        self._probe_pending = False
        self._switched_on = now
        self.resumes += 1
        hold: float | None = None
        if self._fallback_suspect:
            hold = min(
                self.config.stable_seconds * 2.0**self.relapses,
                self.config.max_fallback_hold_seconds,
            )
            self._fallback_hold_until = now + hold
        log.warning(
            "stt healthy again; transcription resumed",
            extra={"was": reason, "fallback_hold_s": None if hold is None else round(hold, 1)},
        )
        return hold

    def release_fallback_hold(self) -> bool:
        """True once the post-overload hold on the fallback is over (the caller re-enables it)."""
        if self._fallback_hold_until is None:
            return False
        now = self._clock()
        if now < self._fallback_hold_until:
            return False
        self._fallback_hold_until = None
        self._fallback_suspect = False
        # A relapse from here counts against the fallback's switch-on.
        self._switched_on = now
        return True

    def describe(self) -> dict[str, Any]:
        ratio, samples = self.drop_ratio()
        hold = (
            None
            if self._fallback_hold_until is None
            else round(max(0.0, self._fallback_hold_until - self._clock()), 1)
        )
        return {
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "pauses": self.pauses,
            "resumes": self.resumes,
            "relapses": self.relapses,
            "probe_every_s": round(self.probe_interval, 1),
            "drop_ratio": None if ratio is None else round(ratio, 3),
            "drop_window_chunks": samples,
            "fallback_hold_s": hold,
        }
