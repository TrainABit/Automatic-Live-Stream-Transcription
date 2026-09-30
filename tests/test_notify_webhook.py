"""Webhook delivery against a real HTTP server on loopback."""

from __future__ import annotations

import json
import logging

import pytest

from livestream_transcriber.netutil import HttpError
from livestream_transcriber.notify import Event, NotifyError, WebhookNotifier
from livestream_transcriber.notify.webhook import build_payload
from livestream_transcriber.rules import Severity
from tests.support.loopback import serve

EVENT = Event(
    event_id="abc123",
    rule_id="giveaway",
    text="we run a giveaway today",
    matched_text="giveaway",
    start=75.0,
    end=77.0,
    wallclock=1_700_000_000.0,
    severity=Severity.WARNING,
    source_url="https://example.com/live",
    session_id=2,
    extra={"description": "A giveaway"},
    targets=("webhook",),
)


def no_sleep(_: float) -> None:
    return None


class TestPayloads:
    def test_json_is_the_event_itself(self) -> None:
        payload = build_payload(EVENT, "json")
        assert payload["event_id"] == "abc123"
        assert payload["severity"] == "warning"
        assert payload["start"] == 75.0
        json.dumps(payload)  # serialisable

    def test_slack_uses_text_with_a_bold_title_and_escapes_control_characters(self) -> None:
        evil = Event.from_dict({**EVENT.to_dict(), "text": "hey <!channel> & <@U1>"})
        payload = build_payload(evil, "slack")
        assert set(payload) == {"text"}
        assert payload["text"].startswith("*[WARNING] giveaway*")
        assert "<!channel>" not in payload["text"]
        assert "&lt;!channel&gt; &amp;" in payload["text"]

    def test_discord_uses_content_and_disables_mentions(self) -> None:
        evil = Event.from_dict({**EVENT.to_dict(), "text": "@everyone " + "x" * 5000})
        payload = build_payload(evil, "discord")
        assert payload["content"].startswith("**[WARNING] giveaway**")
        assert len(payload["content"]) <= 2000
        assert payload["allowed_mentions"] == {"parse": []}

    def test_unknown_format_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown webhook format"):
            build_payload(EVENT, "teams")
        with pytest.raises(ValueError):
            WebhookNotifier("http://x", fmt="teams")


class TestDelivery:
    def test_posts_json_with_bearer_token(self) -> None:
        with serve() as server:
            notifier = WebhookNotifier(server.url + "/hook", token="tok-123", sleep=no_sleep)
            assert notifier.send(EVENT) is True
        (request,) = server.requests
        assert request["path"] == "/hook"
        assert request["headers"]["Authorization"] == "Bearer tok-123"
        assert request["headers"]["Content-Type"] == "application/json"
        assert server.json_bodies()[0]["event_id"] == "abc123"

    def test_no_token_means_no_authorization_header(self) -> None:
        with serve() as server:
            WebhookNotifier(server.url).send(EVENT)
        assert "Authorization" not in server.requests[0]["headers"]

    def test_transient_errors_are_retried_with_backoff(self) -> None:
        sleeps: list[float] = []
        with serve([(503, b"busy"), (429, b"slow down"), (200, b"ok")]) as server:
            notifier = WebhookNotifier(
                server.url, retries=2, backoff_seconds=0.5, sleep=sleeps.append
            )
            assert notifier.send(EVENT) is True
        assert len(server.requests) == 3
        assert sleeps == [0.5, 1.0]

    def test_gives_up_after_the_retries_and_raises_notify_error(self) -> None:
        with serve([(500, b"nope")]) as server:
            notifier = WebhookNotifier(server.url, retries=1, sleep=no_sleep)
            with pytest.raises(NotifyError, match="HTTP 500"):
                notifier.send(EVENT)
        assert len(server.requests) == 2

    def test_client_errors_are_not_retried(self) -> None:
        with serve([(404, b"no such hook")]) as server:
            notifier = WebhookNotifier(server.url, retries=3, sleep=no_sleep)
            with pytest.raises(NotifyError, match="HTTP 404"):
                notifier.send(EVENT)
        assert len(server.requests) == 1

    def test_connection_errors_are_retried_then_raised(self) -> None:
        calls: list[int] = []

        def refuse(*a: object, **k: object) -> dict[str, object]:
            calls.append(1)
            raise ConnectionRefusedError("refused")

        notifier = WebhookNotifier(
            "http://127.0.0.1:9/hook", retries=2, post=refuse, sleep=no_sleep
        )
        with pytest.raises(NotifyError, match="refused"):
            notifier.send(EVENT)
        assert len(calls) == 3

    def test_non_json_success_body_is_still_success(self) -> None:
        with serve([(200, b"ok")]) as server:
            assert WebhookNotifier(server.url).send(EVENT) is True

    def test_url_with_a_secret_is_redacted_in_logs_and_repr(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret_url = "https://hooks.example.test/services/T000/B000/SUPERSECRETTOKEN123456"

        def fail(*a: object, **k: object) -> dict[str, object]:
            raise HttpError(400, "bad")

        notifier = WebhookNotifier(secret_url, post=fail, sleep=no_sleep)
        with caplog.at_level(logging.DEBUG), pytest.raises(NotifyError) as info:
            notifier.send(EVENT)
        assert "SUPERSECRETTOKEN123456" not in repr(notifier)
        assert "SUPERSECRETTOKEN123456" not in str(info.value)
        for record in caplog.records:
            assert "SUPERSECRETTOKEN123456" not in record.getMessage()
            assert "SUPERSECRETTOKEN123456" not in str(getattr(record, "url", ""))
