"""Detect a sustained speech-to-text outage, for alerting and lane shutdown."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SttOutageMonitor"]


@dataclass
class SttOutageMonitor:
    """Track consecutive STT failures in wall-clock time.

    The caller supplies ``now`` so the monitor stays deterministic in tests and
    independent of any particular clock.
    """

    outage_seconds: float = 300.0
    _outage_started: float | None = field(default=None, init=False, repr=False)
    _alerted: bool = field(default=False, init=False, repr=False)

    def record_success(self) -> None:
        self._outage_started = None
        self._alerted = False

    def record_failure(self, now: float) -> bool:
        """Return True exactly once per outage, when it exceeds the threshold."""
        if self._outage_started is None:
            self._outage_started = now
            return False
        if self._alerted:
            return False
        if now - self._outage_started >= self.outage_seconds:
            self._alerted = True
            return True
        return False

    def describe(self) -> dict[str, float | bool | None]:
        return {
            "outage_started": self._outage_started,
            "alerted": self._alerted,
            "outage_seconds": self.outage_seconds,
        }
