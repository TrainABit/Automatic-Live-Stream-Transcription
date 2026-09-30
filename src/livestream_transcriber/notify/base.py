"""The event that gets delivered, the notifier contract and the rate limiter."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

from ..logging_setup import get_logger
from ..redact import redact_url
from ..rules.engine import RuleHit
from ..rules.model import Severity

log = get_logger(__name__)

__all__ = [
    "Event",
    "Notifier",
    "NotifyError",
    "RateLimitedNotifier",
    "RateLimiter",
    "Verdict",
    "format_event",
    "format_stream_time",
    "make_event_id",
]


class NotifyError(RuntimeError):
    """Delivery failed for a reason that may go away. The event stays pending."""


def make_event_id(
    rule_id: str, key: str, start: float, severity: Severity | str, *, scope: str = ""
) -> str:
    """A deterministic id: the same alert always hashes to the same value.

    The id is built from the rule, the dedup key, the severity and the stream
    time in milliseconds, so replaying a recording into the same session cannot
    notify twice, while an escalation (different severity) is a new event. The
    time is not coarser on purpose: a rule with no cooldown must be able to fire
    twice for one key within a few seconds. ``scope`` (the session id) keeps two
    sessions of one stream, whose clocks both start at zero, from colliding.
    """
    millis = round(start * 1000)
    raw = f"{scope}|{rule_id}|{key}|{Severity(severity).value}|{millis}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class Event:
    """A rule hit, ready to be stored and delivered."""

    event_id: str
    rule_id: str
    text: str
    """Text around the match."""
    matched_text: str
    start: float
    """Stream time in seconds."""
    end: float
    wallclock: float
    severity: Severity = Severity.INFO
    source_url: str | None = None
    session_id: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    targets: tuple[str, ...] = ()
    """Which notifiers should receive it (``console``, ``webhook``, ``telegram``)."""

    @classmethod
    def from_hit(
        cls,
        hit: RuleHit,
        *,
        source_url: str | None = None,
        session_id: int | None = None,
        wallclock: float | None = None,
    ) -> Event:
        scope = "" if session_id is None else str(session_id)
        extra: dict[str, Any] = {}
        if hit.description:
            extra["description"] = hit.description
        if hit.groups:
            extra["groups"] = dict(hit.groups)
        if hit.upgrade:
            extra["upgrade"] = True
        if hit.reason:
            extra["reason"] = hit.reason
        return cls(
            event_id=make_event_id(hit.rule_id, hit.key, hit.start, hit.severity, scope=scope),
            rule_id=hit.rule_id,
            text=hit.context,
            matched_text=hit.matched_text,
            start=hit.start,
            end=hit.end,
            wallclock=time.time() if wallclock is None else wallclock,
            severity=hit.severity,
            source_url=redact_url(source_url) if source_url else None,
            session_id=session_id,
            extra=extra,
            targets=hit.notify,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "rule_id": self.rule_id,
            "text": self.text,
            "matched_text": self.matched_text,
            "start": self.start,
            "end": self.end,
            "wallclock": self.wallclock,
            "severity": self.severity.value,
            "source_url": self.source_url,
            "session_id": self.session_id,
            "extra": dict(self.extra),
            "targets": list(self.targets),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Event:
        return cls(
            event_id=str(data["event_id"]),
            rule_id=str(data["rule_id"]),
            text=str(data.get("text", "")),
            matched_text=str(data.get("matched_text", "")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            wallclock=float(data.get("wallclock", 0.0)),
            severity=Severity(data.get("severity", "info")),
            source_url=data.get("source_url"),
            session_id=data.get("session_id"),
            extra=dict(data.get("extra") or {}),
            targets=tuple(data.get("targets") or ()),
        )


class Notifier(Protocol):
    """One delivery channel.

    ``send`` has three outcomes, and callers rely on telling them apart:

    * returns ``True``: delivered now;
    * returns ``False``: deliberately not sent and never will be (a repeat
      swallowed by a guard); do not retry;
    * raises :class:`NotifyError`: delivery failed, keep the event pending.
    """

    name: str

    def send(self, event: Event) -> bool: ...


def format_stream_time(seconds: float) -> str:
    """``H:MM:SS`` for stream time, so alerts point at a place in the recording."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_event(event: Event, *, max_chars: int | None = None) -> str:
    """A plain-text, multi-line rendering shared by the chat-style notifiers."""
    lines = [f"[{event.severity.value.upper()}] {event.rule_id}"]
    description = event.extra.get("description")
    if description:
        lines.append(str(description))
    lines.append(f"Matched: {event.matched_text}")
    if event.text and event.text != event.matched_text:
        lines.append(f"Text: {event.text}")
    where = f"At {format_stream_time(event.start)}"
    if event.source_url:
        where += f" in {event.source_url}"
    lines.append(where)
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


# ----------------------------------------------------------------------- rate limit


class Verdict(Enum):
    ALLOW = "allow"
    REPEAT = "repeat"
    """The same text was just sent; sending it again would only be noise."""
    LIMITED = "limited"
    """The token bucket is empty; try again once it refills."""


class RateLimiter:
    """A repeat guard plus a token bucket.

    Alerts are edge-triggered upstream, but a misbehaving rule set can still
    produce a flood. The repeat guard drops the same text sent twice in a row
    within ``repeat_seconds``; the bucket allows ``burst`` alerts at once and
    refills over ``refill_seconds``. What was held back is counted and can be
    reported on the next alert that goes out (:meth:`take_held_back`).
    """

    def __init__(
        self,
        *,
        repeat_seconds: float = 30.0,
        burst: int = 20,
        refill_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repeat_seconds = max(0.0, float(repeat_seconds))
        self.burst = max(1, int(burst))
        self.refill_seconds = max(1.0, float(refill_seconds))
        self._clock = clock
        self._tokens = float(self.burst)
        self._refilled_at = clock()
        self._last_key: str | None = None
        self._last_at = -float("inf")
        self._held_back = 0
        self.allowed = 0
        self.suppressed = 0
        self._lock = threading.Lock()

    def acquire(self, key: str) -> Verdict:
        """Ask to send ``key``. On ``ALLOW`` a token is spent until :meth:`refund`."""
        with self._lock:
            now = self._clock()
            if key == self._last_key and now - self._last_at < self.repeat_seconds:
                self.suppressed += 1
                return Verdict.REPEAT
            self._refill(now)
            if self._tokens < 1.0:
                self.suppressed += 1
                self._held_back += 1
                return Verdict.LIMITED
            self._tokens -= 1.0
            self._last_key, self._last_at = key, now
            self.allowed += 1
            return Verdict.ALLOW

    def refund(self, key: str) -> None:
        """Undo an ``ALLOW`` whose delivery failed, so a retry is not called a repeat."""
        with self._lock:
            self._tokens = min(float(self.burst), self._tokens + 1.0)
            self.allowed = max(0, self.allowed - 1)
            if self._last_key == key:
                self._last_key, self._last_at = None, -float("inf")

    def take_held_back(self) -> int:
        """How many alerts the bucket held back since the last call; resets the count."""
        with self._lock:
            held, self._held_back = self._held_back, 0
            return held

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._refilled_at)
        self._refilled_at = now
        self._tokens = min(
            float(self.burst), self._tokens + elapsed * self.burst / self.refill_seconds
        )

    def describe(self) -> dict[str, int]:
        return {"allowed": self.allowed, "suppressed": self.suppressed}


class RateLimitedNotifier:
    """Wrap a notifier with a :class:`RateLimiter`.

    A repeat is dropped (``False``: not retried). An empty bucket raises
    :class:`NotifyError`, so the event stays pending and goes out once the
    budget refills. If the inner send fails the token is refunded.
    """

    def __init__(self, inner: Notifier, limiter: RateLimiter | None = None) -> None:
        self.inner = inner
        self.limiter = limiter or RateLimiter()
        self.name = inner.name

    def send(self, event: Event) -> bool:
        key = f"{event.rule_id}|{event.matched_text}|{event.text}"
        verdict = self.limiter.acquire(key)
        if verdict is Verdict.REPEAT:
            log.info(
                "repeated alert dropped", extra={"notifier": self.name, "rule_id": event.rule_id}
            )
            return False
        if verdict is Verdict.LIMITED:
            raise NotifyError(f"{self.name}: rate limit reached, will retry")
        held = self.limiter.take_held_back()
        if held:
            event = replace(event, extra={**event.extra, "held_back": held})
        try:
            return self.inner.send(event)
        except BaseException:
            self.limiter.refund(key)
            raise
