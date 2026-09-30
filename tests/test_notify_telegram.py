"""Telegram delivery with the HTTP layer mocked."""

from __future__ import annotations

from typing import Any

import pytest

from livestream_transcriber.netutil import HttpError, NonJsonBody
from livestream_transcriber.notify import Event, NotifyError, TelegramNotifier
from livestream_transcriber.notify.telegram import format_html
from livestream_transcriber.rules import Severity

TOKEN = "123456:your-telegram-bot-token"

EVENT = Event(
    event_id="abc",
    rule_id="giveaway",
    text='we run a "giveaway" <b>today</b> & more',
    matched_text="giveaway",
    start=3725.0,
    end=3727.0,
    wallclock=0.0,
    severity=Severity.CRITICAL,
    source_url="https://example.com/live",
    extra={"description": "A <i>giveaway</i>"},
)


class Post:
    def __init__(self, reply: Any = None) -> None:
        self.reply = {"ok": True, "result": {}} if reply is None else reply
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def __call__(self, url: str, payload: dict[str, Any], **kw: Any) -> Any:
        self.calls.append((url, payload, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_sends_html_to_the_bot_api() -> None:
    post = Post()
    notifier = TelegramNotifier(TOKEN, "-100999", timeout=7, post=post)
    assert notifier.send(EVENT) is True
    url, payload, kwargs = post.calls[0]
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert payload["chat_id"] == "-100999"
    assert payload["parse_mode"] == "HTML"
    assert payload["disable_web_page_preview"] is True
    assert kwargs["timeout"] == 7


def test_message_is_escaped_and_points_at_stream_time() -> None:
    text = format_html(EVENT)
    assert text.splitlines()[0] == "<b>[CRITICAL] giveaway</b>"
    assert "<i>A &lt;i&gt;giveaway&lt;/i&gt;</i>" in text
    assert "&lt;b&gt;today&lt;/b&gt; &amp; more" in text
    assert "<code>giveaway</code>" in text
    assert "At 1:02:05 in https://example.com/live" in text
    assert "<b>today</b>" not in text


def test_long_messages_are_shrunk_to_the_limit_without_breaking_markup() -> None:
    huge = Event.from_dict({**EVENT.to_dict(), "text": "&<" * 4000, "matched_text": "<" * 4000})
    text = format_html(huge)
    assert len(text) <= 4096
    assert text.count("<code>") == text.count("</code>") == 1
    assert text.count("<b>") == text.count("</b>") == 1


def test_api_errors_raise_notify_error_with_the_description() -> None:
    body = '{"ok": false, "description": "Bad Request: chat not found"}'
    notifier = TelegramNotifier(TOKEN, "1", post=Post(HttpError(400, body)))
    with pytest.raises(NotifyError, match="chat not found"):
        notifier.send(EVENT)


def test_rate_limit_reply_mentions_retry_after() -> None:
    body = '{"ok": false, "description": "Too Many Requests", "parameters": {"retry_after": 12}}'
    notifier = TelegramNotifier(TOKEN, "1", post=Post(HttpError(429, body)))
    with pytest.raises(NotifyError, match="retry after 12s"):
        notifier.send(EVENT)


def test_network_errors_raise_notify_error() -> None:
    notifier = TelegramNotifier(TOKEN, "1", post=Post(TimeoutError("timed out")))
    with pytest.raises(NotifyError, match="timed out"):
        notifier.send(EVENT)


@pytest.mark.parametrize(
    "reply", [{"ok": False, "description": "nope"}, NonJsonBody(text="<html>")]
)
def test_a_200_that_is_not_ok_is_a_failure(reply: Any) -> None:
    with pytest.raises(NotifyError):
        TelegramNotifier(TOKEN, "1", post=Post(reply)).send(EVENT)


def test_the_bot_token_never_appears_in_errors_or_repr() -> None:
    err = HttpError(
        401,
        "Unauthorized for https://api.telegram.org/bot123456:ABCdefGHIjklMNOpqrSTUvwxYZ_0123456789/sendMessage",
    )
    notifier = TelegramNotifier(TOKEN, "1", post=Post(err))
    assert TOKEN not in repr(notifier)
    with pytest.raises(NotifyError) as info:
        notifier.send(EVENT)
    assert "ABCdefGHIjklMNOpqrSTUvwxYZ_0123456789" not in str(info.value)
