"""Deliver alerts through the Telegram Bot API."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

from ..logging_setup import get_logger
from ..netutil import HttpError, NonJsonBody, post_json
from ..redact import redact_text
from .base import Event, NotifyError, format_stream_time

log = get_logger(__name__)

__all__ = ["TELEGRAM_API", "TelegramNotifier", "format_html"]

TELEGRAM_API = "https://api.telegram.org"
_MESSAGE_LIMIT = 4096  # characters Telegram accepts in one message
_MATCH_LIMIT = 300
_CONTEXT_LIMIT = 1500


def format_html(event: Event) -> str:
    """The alert as Telegram HTML. All user-visible text is escaped."""

    def esc(text: str) -> str:
        return html.escape(text, quote=False)

    def build(matched_limit: int, context_limit: int) -> str:
        lines = [f"<b>[{esc(event.severity.value.upper())}] {esc(event.rule_id)}</b>"]
        description = event.extra.get("description")
        if description:
            lines.append(f"<i>{esc(str(description))}</i>")
        lines.append(f"Matched: <code>{esc(event.matched_text[:matched_limit])}</code>")
        if event.text and event.text != event.matched_text:
            lines.append(esc(event.text[:context_limit]))
        where = f"At {format_stream_time(event.start)}"
        if event.source_url:
            where += f" in {esc(event.source_url)}"
        lines.append(where)
        return "\n".join(lines)

    matched_limit, context_limit = _MATCH_LIMIT, _CONTEXT_LIMIT
    message = build(matched_limit, context_limit)
    # Escaping can multiply the length; shrink the free-text parts until it fits.
    while len(message) > _MESSAGE_LIMIT and context_limit > 40:
        matched_limit, context_limit = matched_limit // 2, context_limit // 2
        message = build(matched_limit, context_limit)
    return message[:_MESSAGE_LIMIT]


class TelegramNotifier:
    """Send alerts to one chat with a bot token."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout: float = 10.0,
        api_base: str = TELEGRAM_API,
        post: Callable[..., dict[str, Any]] = post_json,
    ) -> None:
        self._token = token
        self.chat_id = str(chat_id)
        self.timeout = timeout
        self._api_base = api_base.rstrip("/")
        self._post = post

    def __repr__(self) -> str:
        return f"TelegramNotifier(chat_id={self.chat_id!r})"

    def send(self, event: Event) -> bool:
        url = f"{self._api_base}/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": format_html(event),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            body = self._post(url, payload, timeout=self.timeout)
        except HttpError as exc:
            detail = _api_error(exc.body) or f"HTTP {exc.status}"
            log.warning(
                "telegram delivery failed", extra={"event_id": event.event_id, "error": detail}
            )
            raise NotifyError(f"telegram: {redact_text(detail)}") from exc
        except OSError as exc:
            detail = redact_text(str(exc))
            log.warning(
                "telegram delivery failed", extra={"event_id": event.event_id, "error": detail}
            )
            raise NotifyError(f"telegram: {detail}") from exc
        if isinstance(body, NonJsonBody) or body.get("ok") is False:
            raise NotifyError(
                f"telegram: {redact_text(str(body.get('description') or 'unexpected reply'))}"
            )
        return True


def _api_error(raw: str) -> str:
    """The ``description`` of a Bot API error body, if it has one."""
    try:
        data = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    detail = str(data.get("description") or "")
    retry_after = (data.get("parameters") or {}).get("retry_after")
    return f"{detail} (retry after {retry_after}s)" if retry_after else detail
