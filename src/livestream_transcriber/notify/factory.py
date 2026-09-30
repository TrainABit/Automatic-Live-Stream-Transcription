"""Build notifiers and the dispatcher from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

from ..config import ConfigError, Settings
from .base import Notifier, RateLimitedNotifier, RateLimiter
from .console import ConsoleNotifier
from .dispatcher import NotificationDispatcher
from .telegram import TelegramNotifier
from .webhook import WebhookNotifier

if TYPE_CHECKING:
    from ..rules.model import RuleSet
    from ..store.database import Database

__all__ = ["build_dispatcher", "build_notifiers", "missing_targets"]


def build_notifiers(
    settings: Settings,
    *,
    stream: TextIO | None = None,
    color: bool | None = None,
    rate_limit: bool = True,
) -> dict[str, Notifier]:
    """The notifiers the settings enable, keyed by target name.

    ``console`` is always present. ``webhook`` needs ``notify_webhook_url``;
    ``telegram`` needs both a bot token and a chat id. Half a Telegram
    configuration is an error rather than a silently absent channel.
    Network notifiers get a :class:`RateLimiter` each unless ``rate_limit`` is off.
    """
    notifiers: dict[str, Notifier] = {"console": ConsoleNotifier(stream, color=color)}

    if settings.notify_webhook_url is not None:
        token = settings.notify_webhook_token
        webhook: Notifier = WebhookNotifier(
            settings.notify_webhook_url.get_secret_value(),
            fmt=settings.notify_webhook_format,
            token=token.get_secret_value() if token else None,
            timeout=settings.notify_timeout_seconds,
        )
        notifiers["webhook"] = _guard(webhook, rate_limit)

    has_token = settings.notify_telegram_token is not None
    has_chat = bool(settings.notify_telegram_chat_id)
    if has_token != has_chat:
        raise ConfigError(
            "Telegram needs both notify_telegram_token and notify_telegram_chat_id "
            "(LST_NOTIFY_TELEGRAM_TOKEN, LST_NOTIFY_TELEGRAM_CHAT_ID)"
        )
    if settings.notify_telegram_token is not None and settings.notify_telegram_chat_id:
        telegram: Notifier = TelegramNotifier(
            settings.notify_telegram_token.get_secret_value(),
            settings.notify_telegram_chat_id,
            timeout=settings.notify_timeout_seconds,
        )
        notifiers["telegram"] = _guard(telegram, rate_limit)
    return notifiers


def _guard(notifier: Notifier, enabled: bool) -> Notifier:
    return RateLimitedNotifier(notifier, RateLimiter()) if enabled else notifier


def build_dispatcher(
    settings: Settings,
    store: Database | None = None,
    *,
    ruleset: RuleSet | None = None,
    stream: TextIO | None = None,
    color: bool | None = None,
) -> NotificationDispatcher:
    """A dispatcher wired to the configured notifiers and, optionally, a store."""
    default = ruleset.default_notify if ruleset is not None else ("console",)
    return NotificationDispatcher(
        build_notifiers(settings, stream=stream, color=color),
        store=store,
        default_targets=default,
    )


def missing_targets(ruleset: RuleSet, notifiers: dict[str, Notifier]) -> dict[str, tuple[str, ...]]:
    """Enabled rules that name a target which is not configured, for a startup warning."""
    problems: dict[str, tuple[str, ...]] = {}
    for rule in ruleset.enabled_rules:
        absent = tuple(t for t in rule.notify if t not in notifiers)
        if absent:
            problems[rule.id] = absent
    return problems
