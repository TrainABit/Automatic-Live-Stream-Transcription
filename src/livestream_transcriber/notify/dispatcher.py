"""Fan an event out to its targets: persist first, then notify, retry what failed.

The ordering is the point of this module. Notifying first and storing after
loses an alert if the process dies in between, and storing first without a
retry loses it whenever a delivery fails. So:

1. ``insert_event`` commits the row (a duplicate ``event_id`` ends the story);
2. each target is tried once and every success is recorded per target;
3. only when no target is left failing does the event get ``notified_at``;
4. :meth:`NotificationDispatcher.retry_pending` picks up everything else,
   including events from a previous run, and repeats only the failed targets.

A target named by a rule but not configured (say ``telegram`` without a bot
token) is logged once and treated as handled, otherwise one missing credential
would keep every event pending forever.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..logging_setup import get_logger
from ..redact import redact_text
from .base import Event, Notifier, NotifyError

if TYPE_CHECKING:
    from ..store.database import Database

log = get_logger(__name__)

__all__ = ["DispatchResult", "NotificationDispatcher"]


@dataclass(slots=True)
class DispatchResult:
    event_id: str
    delivered: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    """Targets that were not configured, or that declined the event on purpose."""
    failed: dict[str, str] = field(default_factory=dict)
    duplicate: bool = False
    """The event id was already known; nothing was sent."""

    @property
    def ok(self) -> bool:
        return not self.failed


class NotificationDispatcher:
    def __init__(
        self,
        notifiers: Mapping[str, Notifier],
        *,
        store: Database | None = None,
        default_targets: tuple[str, ...] = ("console",),
        max_attempts: int = 5,
        memory_size: int = 2048,
    ) -> None:
        self.notifiers = dict(notifiers)
        self.store = store
        self.default_targets = default_targets
        self.max_attempts = max(1, max_attempts)
        self._memory_size = max(1, memory_size)
        # Without a store, remember what was sent so a duplicate is still dropped.
        self._sent: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._known: OrderedDict[str, None] = OrderedDict()
        self._warned: set[str] = set()
        # Events being delivered right now, from ``dispatch`` or a retry. A retry pass
        # that ran while an event was between "stored" and "delivered" would otherwise
        # pick it up as pending and send it a second time.
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ dispatch

    def dispatch(self, event: Event) -> DispatchResult:
        """Store, then deliver. Blocking: notifiers do network I/O."""
        if not self._claim(event.event_id):
            log.debug("event already being delivered", extra={"event_id": event.event_id})
            return DispatchResult(event.event_id, duplicate=True)
        try:
            if not self._register(event):
                log.debug("duplicate event ignored", extra={"event_id": event.event_id})
                return DispatchResult(event.event_id, duplicate=True)
            return self._deliver(event)
        finally:
            self._release(event.event_id)

    async def dispatch_async(self, event: Event) -> DispatchResult:
        return await asyncio.to_thread(self.dispatch, event)

    def retry_pending(self, limit: int = 50) -> list[DispatchResult]:
        """Retry stored events that are not fully delivered. Needs a store."""
        if self.store is None:
            return []
        results: list[DispatchResult] = []
        for row in self.store.pending_notifications(limit, max_attempts=self.max_attempts):
            try:
                event = Event.from_dict(json.loads(row["payload_json"]))
            except (ValueError, KeyError, TypeError) as exc:
                log.error("stored event unreadable", extra={"event_id": row["event_id"]})
                self.store.record_notify_failure(row["event_id"], f"unreadable payload: {exc}")
                continue
            if not self._claim(event.event_id):
                continue  # a live dispatch is on it
            try:
                results.append(self._deliver(event))
            finally:
                self._release(event.event_id)
        return results

    async def retry_pending_async(self, limit: int = 50) -> list[DispatchResult]:
        return await asyncio.to_thread(self.retry_pending, limit)

    # ------------------------------------------------------------------ internals

    def _claim(self, event_id: str) -> bool:
        """True when the caller now owns delivery of ``event_id``."""
        with self._lock:
            if event_id in self._in_flight:
                return False
            self._in_flight.add(event_id)
            return True

    def _release(self, event_id: str) -> None:
        with self._lock:
            self._in_flight.discard(event_id)

    def _register(self, event: Event) -> bool:
        """True when the event is new and should be delivered."""
        if self.store is not None:
            return self.store.insert_event(event)
        with self._lock:
            if event.event_id in self._known:
                return False
            self._known[event.event_id] = None
            while len(self._known) > self._memory_size:
                self._known.popitem(last=False)
        return True

    def _already_delivered(self, event: Event) -> set[str]:
        if self.store is not None:
            return self.store.delivered_targets(event.event_id)
        with self._lock:
            return {t for (eid, t) in self._sent if eid == event.event_id}

    def _remember(self, event: Event, target: str) -> None:
        if self.store is not None:
            self.store.mark_delivered(event.event_id, target)
            return
        with self._lock:
            self._sent[(event.event_id, target)] = None
            while len(self._sent) > self._memory_size:
                self._sent.popitem(last=False)

    def _deliver(self, event: Event) -> DispatchResult:
        result = DispatchResult(event.event_id)
        done = self._already_delivered(event)
        for target in event.targets or self.default_targets:
            if target in done:
                continue
            notifier = self.notifiers.get(target)
            if notifier is None:
                self._warn_unconfigured(target)
                result.skipped.append(target)
                continue
            try:
                sent = notifier.send(event)
            except NotifyError as exc:
                result.failed[target] = redact_text(str(exc))
                continue
            except Exception as exc:  # a broken notifier must not take the pipeline down
                log.exception("notifier crashed", extra={"target": target})
                result.failed[target] = redact_text(f"{type(exc).__name__}: {exc}")
                continue
            if sent:
                self._remember(event, target)
                result.delivered.append(target)
            else:
                result.skipped.append(target)
        self._finish(event, result)
        return result

    def _finish(self, event: Event, result: DispatchResult) -> None:
        if self.store is None:
            if result.failed:
                log.warning(
                    "delivery failed and no store is configured to retry it",
                    extra={"event_id": event.event_id, "failed": sorted(result.failed)},
                )
            return
        if result.failed:
            summary = "; ".join(f"{t}: {e}" for t, e in sorted(result.failed.items()))
            attempts = self.store.record_notify_failure(event.event_id, summary)
            log.warning(
                "notification pending retry",
                extra={"event_id": event.event_id, "attempts": attempts, "error": summary},
            )
        else:
            self.store.mark_notified(event.event_id)

    def _warn_unconfigured(self, target: str) -> None:
        if target in self._warned:
            return
        self._warned.add(target)
        log.warning("rule targets a notifier that is not configured", extra={"target": target})
