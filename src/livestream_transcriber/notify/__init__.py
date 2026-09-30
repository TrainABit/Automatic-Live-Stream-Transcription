"""Notifiers that deliver rule hits."""

from .base import (
    Event,
    Notifier,
    NotifyError,
    RateLimitedNotifier,
    RateLimiter,
    Verdict,
    format_event,
    make_event_id,
)
from .console import ConsoleNotifier
from .dispatcher import DispatchResult, NotificationDispatcher
from .factory import build_dispatcher, build_notifiers, missing_targets
from .telegram import TelegramNotifier
from .webhook import WebhookNotifier

__all__ = [
    "ConsoleNotifier",
    "DispatchResult",
    "Event",
    "NotificationDispatcher",
    "Notifier",
    "NotifyError",
    "RateLimitedNotifier",
    "RateLimiter",
    "TelegramNotifier",
    "Verdict",
    "WebhookNotifier",
    "build_dispatcher",
    "build_notifiers",
    "format_event",
    "make_event_id",
    "missing_targets",
]
