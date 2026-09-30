"""Deliver alerts to an HTTP webhook.

Three payload shapes cover the common cases without a plugin system:

``json``
    The event as a JSON object, for your own service.
``slack``
    ``{"text": ...}`` in Slack's mrkdwn, accepted by Slack incoming webhooks and
    by most Slack-compatible servers.
``discord``
    ``{"content": ...}`` with mentions disabled, so a transcript that contains
    "@everyone" cannot ping a channel.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from ..logging_setup import get_logger
from ..netutil import HttpError, post_json
from ..redact import redact_text
from .base import Event, NotifyError, format_event

log = get_logger(__name__)

__all__ = ["WEBHOOK_FORMATS", "WebhookNotifier", "build_payload"]

WEBHOOK_FORMATS = ("json", "slack", "discord")
_DISCORD_LIMIT = 2000
_SLACK_LIMIT = 3000


def _slack_escape(text: str) -> str:
    """Slack treats ``&``, ``<`` and ``>`` as control characters (``<!channel>`` pings)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_payload(event: Event, fmt: str = "json") -> dict[str, Any]:
    """The request body for ``event`` in one of :data:`WEBHOOK_FORMATS`."""
    if fmt == "json":
        return event.to_dict()
    if fmt == "slack":
        head, _, rest = _slack_escape(format_event(event, max_chars=_SLACK_LIMIT)).partition("\n")
        return {"text": f"*{head}*" + (f"\n{rest}" if rest else "")}
    if fmt == "discord":
        head, _, rest = format_event(event, max_chars=_DISCORD_LIMIT - 8).partition("\n")
        return {
            "content": f"**{head}**" + (f"\n{rest}" if rest else ""),
            "allowed_mentions": {"parse": []},
        }
    raise ValueError(f"unknown webhook format {fmt!r}; choose one of {', '.join(WEBHOOK_FORMATS)}")


def _display_url(url: str) -> str:
    """Scheme and host only: webhook URLs carry their secret in the path or query."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or '?'}/…"


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, HttpError):
        return exc.status == 429 or exc.status >= 500
    return isinstance(exc, OSError)  # timeouts, refused or reset connections


class WebhookNotifier:
    """POST each event to ``url``, retrying transient failures a few times."""

    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        fmt: str = "json",
        token: str | None = None,
        timeout: float = 10.0,
        retries: int = 2,
        backoff_seconds: float = 0.5,
        post: Callable[..., dict[str, Any]] = post_json,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if fmt not in WEBHOOK_FORMATS:
            raise ValueError(f"unknown webhook format {fmt!r}")
        self._url = url
        self.fmt = fmt
        self._token = token
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff_seconds = backoff_seconds
        self._post = post
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"WebhookNotifier(url={_display_url(self._url)!r}, fmt={self.fmt!r})"

    def send(self, event: Event) -> bool:
        payload = build_payload(event, self.fmt)
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        last = ""
        for attempt in range(self.retries + 1):
            try:
                self._post(self._url, payload, headers=headers, timeout=self.timeout)
            except (HttpError, OSError) as exc:
                last = redact_text(str(exc))
                log.warning(
                    "webhook delivery failed",
                    extra={
                        "url": _display_url(self._url),
                        "attempt": attempt + 1,
                        "event_id": event.event_id,
                        "error": last,
                    },
                )
                if not _is_transient(exc):
                    break
                if attempt < self.retries:
                    self._sleep(self.backoff_seconds * (2**attempt))
            else:
                return True
        raise NotifyError(f"webhook: {last}")
