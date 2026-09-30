"""The process heartbeat: proof of life through sessions and idle time.

Every ``interval`` seconds it logs ``process heartbeat`` and writes one plain
``LST_HEARTBEAT key=value ...`` line to stderr. The line is distinct on every tick
and flushed immediately, so a supervisor (journald, a container log, a shell
``grep``) can tell a live process from a hung one without parsing structured logs.

While a session runs (:meth:`ProcessHeartbeat.attach`) the line carries that
session's summary; between sessions it says ``capture=idle`` together with the
source-selection state. The probe loop logs repeated answers at DEBUG, so without
this an idle process would write nothing for hours and look hung.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import Callable
from typing import Any, TextIO

from . import __version__
from .logging_setup import get_logger
from .resilience.memory import current_rss_mb, memory_pressure_level
from .source_health import SOURCE_HEALTH_FIELDS, HealthWatch, SourceHealth
from .stream.auto_resume import ProbeBackoff

log = get_logger(__name__)

__all__ = ["HEARTBEAT_FIELDS", "ProcessHeartbeat", "format_fields"]

HEARTBEAT_FIELDS = (
    "capture",
    "health",
    "health_reasons",
    "stt_lag_s",
    "stt_drop_ratio",
    "stt_paused",
    "queue_depth",
    "dropped",
    "rss_mb",
)
"""Fields of the heartbeat line beyond the fixed head. Names are stable; ``None`` is left out."""

SummaryFn = Callable[[], dict[str, Any]]


def format_fields(summary: dict[str, Any], names: tuple[str, ...]) -> str:
    """`` key=value`` for each of ``names`` present in ``summary``; one token per value."""
    parts: list[str] = []
    for name in names:
        value = summary.get(name)
        if value is None:
            continue
        if isinstance(value, list | tuple):
            value = ",".join(str(v) for v in value) or "-"
        elif isinstance(value, bool):
            value = str(value).lower()
        parts.append(f" {name}={str(value).replace(' ', '_')}")
    return "".join(parts)


class ProcessHeartbeat:
    """One per ``lst run``: beats through sessions and idle time."""

    def __init__(
        self,
        *,
        source: SourceHealth | None = None,
        backoff: ProbeBackoff | None = None,
        watch: HealthWatch | None = None,
        idle_beats: bool = False,
        memory_limit_mb: float = 0.0,
        out: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source if source is not None else SourceHealth(clock=clock)
        self.backoff = backoff
        self.watch = watch if watch is not None else HealthWatch(None)
        self.idle_beats = idle_beats
        self.memory_limit_mb = memory_limit_mb
        self._out = out
        self._clock = clock
        self._summary: SummaryFn | None = None
        self._session_started: float | None = None
        self._pressure = 0
        self.started = clock()
        self.tick = 0

    def attach(self, summary: SummaryFn) -> None:
        """A session started: beat with its summary until :meth:`detach`."""
        self._summary = summary
        self._session_started = self._clock()

    def detach(self) -> None:
        """The session ended. Drops the reference so nothing of it outlives it while idle."""
        self._summary = None
        self._session_started = None

    def _idle_summary(self) -> dict[str, Any]:
        source = self.source.fields(self.backoff)
        reasons: list[str] = []
        if source["bot_check"]:
            reasons.append("source_bot_check")
        if source["selection_state"] == "uncertain":
            reasons.append("source_uncertain")
        return {
            "capture": "idle",
            "health": "degraded" if reasons else "ok",
            "health_reasons": reasons,
            **source,
        }

    def _memory(self, summary: dict[str, Any]) -> None:
        rss = current_rss_mb()
        if rss is None:
            return
        summary["rss_mb"] = round(rss)
        if self.memory_limit_mb <= 0:
            return
        level = memory_pressure_level(rss, limit_mb=self.memory_limit_mb)
        if level > self._pressure:
            log.warning(
                "memory use is high",
                extra={"rss_mb": round(rss), "limit_mb": self.memory_limit_mb, "level": level},
            )
        self._pressure = level
        if level:
            summary.setdefault("health_reasons", []).append("memory")
            summary["health"] = "degraded"

    async def beat(self) -> None:
        """One tick: the log line, the stderr line and the health checks."""
        summarise = self._summary
        if summarise is None and not self.idle_beats:
            return
        self.tick += 1
        capture: dict[str, Any] | None = None
        if summarise is not None:
            capture = summarise()
            if capture.get("delivering"):
                self.source.capture_delivering()
            summary = {"capture": "live", **capture, **self.source.fields(self.backoff)}
        else:
            summary = self._idle_summary()
        self._memory(summary)
        uptime = round(self._clock() - self.started, 1)
        log.info(
            "process heartbeat",
            extra={"tick": self.tick, "uptime_s": uptime, **summary},
        )
        out = self._out or sys.stderr
        line = (
            f"LST_HEARTBEAT tick={self.tick} uptime_s={uptime} version={__version__}"
            f"{format_fields(summary, (*HEARTBEAT_FIELDS, *SOURCE_HEALTH_FIELDS))}\n"
        )
        with contextlib.suppress(OSError, ValueError):
            out.write(line)
            out.flush()
        try:
            await self.watch.check(
                capture=capture,
                source=self.source,
                next_probe_s=summary.get("next_probe_s"),
            )
        except Exception:
            log.exception("health check failed")

    async def run(self, stop: asyncio.Event, interval: float) -> None:
        """Beat every ``interval`` seconds until ``stop`` is set.

        A beat that raises is logged and the next one runs: this task is the process's
        proof of life between sessions, and it is awaited only when the process ends.
        """
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), interval)
                return
            except TimeoutError:
                try:
                    await self.beat()
                except Exception:
                    log.exception("process heartbeat failed", extra={"tick": self.tick})
