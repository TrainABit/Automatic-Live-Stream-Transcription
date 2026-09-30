"""Persist-before-notify, per-target retry and the factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from livestream_transcriber.config import ConfigError, Settings
from livestream_transcriber.notify import (
    ConsoleNotifier,
    Event,
    NotificationDispatcher,
    NotifyError,
    RateLimitedNotifier,
    TelegramNotifier,
    WebhookNotifier,
    build_dispatcher,
    build_notifiers,
    missing_targets,
)
from livestream_transcriber.rules import RuleSet, Severity
from livestream_transcriber.store import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "lst.db")
    yield database  # type: ignore[misc]
    database.close()


def make_event(event_id: str = "e1", targets: tuple[str, ...] = ("a", "b"), **kw: object) -> Event:
    base: dict[str, object] = {
        "event_id": event_id,
        "rule_id": "r",
        "text": "some text",
        "matched_text": "text",
        "start": 1.0,
        "end": 2.0,
        "wallclock": 0.0,
        "severity": Severity.INFO,
        "targets": targets,
    }
    base.update(kw)
    return Event(**base)  # type: ignore[arg-type]


class Fake:
    """A notifier that fails a set number of times, or answers False."""

    def __init__(
        self, name: str, *, fail: int = 0, result: bool = True, db: Database | None = None
    ) -> None:
        self.name = name
        self.fail = fail
        self.result = result
        self.db = db
        self.sent: list[str] = []
        self.row_existed_when_sent: list[bool] = []

    def send(self, event: Event) -> bool:
        if self.db is not None:
            self.row_existed_when_sent.append(self.db.get_event(event.event_id) is not None)
        self.sent.append(event.event_id)
        if self.fail > 0:
            self.fail -= 1
            raise NotifyError(f"{self.name} unavailable")
        return self.result


class TestDispatch:
    def test_the_event_is_stored_before_any_notifier_is_called(self, db: Database) -> None:
        a = Fake("a", db=db)
        dispatcher = NotificationDispatcher({"a": a}, store=db)
        result = dispatcher.dispatch(make_event(targets=("a",)))
        assert a.row_existed_when_sent == [True]
        assert result.delivered == ["a"] and result.ok
        assert db.is_notified("e1")
        assert db.pending_notifications() == []

    def test_duplicates_are_not_sent_twice(self, db: Database) -> None:
        a = Fake("a")
        dispatcher = NotificationDispatcher({"a": a}, store=db)
        dispatcher.dispatch(make_event(targets=("a",)))
        again = dispatcher.dispatch(make_event(targets=("a",)))
        assert again.duplicate and a.sent == ["e1"]

    def test_duplicates_are_dropped_without_a_store_too(self) -> None:
        a = Fake("a")
        dispatcher = NotificationDispatcher({"a": a})
        dispatcher.dispatch(make_event(targets=("a",)))
        assert dispatcher.dispatch(make_event(targets=("a",))).duplicate
        assert a.sent == ["e1"]

    def test_a_failed_target_leaves_the_event_pending(self, db: Database) -> None:
        a, b = Fake("a"), Fake("b", fail=1)
        dispatcher = NotificationDispatcher({"a": a, "b": b}, store=db)
        result = dispatcher.dispatch(make_event())
        assert result.delivered == ["a"] and list(result.failed) == ["b"] and not result.ok
        row = db.get_event("e1")
        assert row is not None
        assert row["notified_at"] is None and row["notify_attempts"] == 1
        assert "b unavailable" in row["notify_error"]
        assert [r["event_id"] for r in db.pending_notifications()] == ["e1"]

    def test_retry_repeats_only_the_targets_that_failed(self, db: Database) -> None:
        a, b = Fake("a"), Fake("b", fail=1)
        dispatcher = NotificationDispatcher({"a": a, "b": b}, store=db)
        dispatcher.dispatch(make_event())
        (result,) = dispatcher.retry_pending()
        assert result.delivered == ["b"]
        assert a.sent == ["e1"]  # not repeated
        assert b.sent == ["e1", "e1"]
        assert db.is_notified("e1") and db.pending_notifications() == []
        assert dispatcher.retry_pending() == []

    def test_a_retry_pass_does_not_resend_an_event_that_is_being_delivered(
        self, db: Database
    ) -> None:
        dispatcher = NotificationDispatcher({}, store=db)
        retried: list[int] = []

        class Retrying(Fake):
            def send(self, event: Event) -> bool:
                # The row is stored but not yet marked delivered: exactly what a retry
                # pass running on another thread would find and pick up.
                retried.append(len(dispatcher.retry_pending()))
                return super().send(event)

        a = Retrying("a")
        dispatcher.notifiers["a"] = a
        dispatcher.dispatch(make_event(targets=("a",)))
        assert retried == [0]
        assert a.sent == ["e1"]

    def test_pending_events_survive_a_restart(self, db: Database, tmp_path: Path) -> None:
        first = NotificationDispatcher({"a": Fake("a", fail=1)}, store=db)
        first.dispatch(make_event(targets=("a",)))
        db.close()

        reopened = Database(tmp_path / "lst.db")
        try:
            healthy = Fake("a")
            second = NotificationDispatcher({"a": healthy}, store=reopened)
            (result,) = second.retry_pending()
            assert result.ok and healthy.sent == ["e1"]
            assert reopened.is_notified("e1")
        finally:
            reopened.close()

    def test_retries_stop_after_max_attempts(self, db: Database) -> None:
        a = Fake("a", fail=99)
        dispatcher = NotificationDispatcher({"a": a}, store=db, max_attempts=3)
        dispatcher.dispatch(make_event(targets=("a",)))  # attempt 1
        assert len(dispatcher.retry_pending()) == 1  # attempt 2
        assert len(dispatcher.retry_pending()) == 1  # attempt 3
        assert dispatcher.retry_pending() == []
        assert len(a.sent) == 3

    def test_a_notifier_that_declines_is_not_a_failure(self, db: Database) -> None:
        dispatcher = NotificationDispatcher({"a": Fake("a", result=False)}, store=db)
        result = dispatcher.dispatch(make_event(targets=("a",)))
        assert result.skipped == ["a"] and result.ok
        assert db.is_notified("e1")

    def test_an_unconfigured_target_is_skipped_and_does_not_block(self, db: Database) -> None:
        dispatcher = NotificationDispatcher({"a": Fake("a")}, store=db)
        result = dispatcher.dispatch(make_event(targets=("a", "telegram")))
        assert result.skipped == ["telegram"] and result.delivered == ["a"]
        assert db.is_notified("e1")

    def test_a_crashing_notifier_is_contained(self, db: Database) -> None:
        class Broken:
            name = "a"

            def send(self, event: Event) -> bool:
                raise RuntimeError("bug")

        result = NotificationDispatcher({"a": Broken(), "b": Fake("b")}, store=db).dispatch(
            make_event()
        )
        assert "RuntimeError" in result.failed["a"] and result.delivered == ["b"]

    def test_events_without_targets_use_the_defaults(self) -> None:
        a = Fake("a")
        dispatcher = NotificationDispatcher({"a": a}, default_targets=("a",))
        dispatcher.dispatch(make_event(targets=()))
        assert a.sent == ["e1"]

    def test_errors_stored_with_the_event_do_not_leak_secrets(self, db: Database) -> None:
        class Leaky:
            name = "a"

            def send(self, event: Event) -> bool:
                raise NotifyError(
                    "HTTP 401 for https://api.telegram.org/bot123456:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/x"
                )

        NotificationDispatcher({"a": Leaky()}, store=db).dispatch(make_event(targets=("a",)))
        row = db.get_event("e1")
        assert row is not None and "AAAAAAAAAAAA" not in row["notify_error"]

    async def test_async_dispatch(self, db: Database) -> None:
        a = Fake("a")
        dispatcher = NotificationDispatcher({"a": a}, store=db)
        result = await dispatcher.dispatch_async(make_event(targets=("a",)))
        assert result.ok and a.sent == ["e1"]
        assert await dispatcher.retry_pending_async() == []

    def test_retry_without_a_store_is_a_no_op(self) -> None:
        assert NotificationDispatcher({}).retry_pending() == []

    def test_an_unreadable_stored_payload_does_not_stop_the_retry_loop(self, db: Database) -> None:
        db.insert_event(make_event("bad", targets=("a",)))
        db._conn.execute("UPDATE events SET payload_json='{oops' WHERE event_id='bad'")
        db.insert_event(make_event("good", start=5.0, targets=("a",)))
        a = Fake("a")
        results = NotificationDispatcher({"a": a}, store=db).retry_pending()
        assert [r.event_id for r in results] == ["good"]
        row = db.get_event("bad")
        assert row is not None and row["notify_attempts"] == 1


class TestFactory:
    def test_console_is_always_available(self) -> None:
        notifiers = build_notifiers(Settings())
        assert list(notifiers) == ["console"]
        assert isinstance(notifiers["console"], ConsoleNotifier)

    def test_webhook_and_telegram_are_built_from_settings_and_rate_limited(self) -> None:
        settings = Settings(
            notify_webhook_url=SecretStr("https://hooks.example.test/abc"),
            notify_webhook_format="slack",
            notify_webhook_token=SecretStr("tok"),
            notify_telegram_token=SecretStr("123456:your-telegram-bot-token"),
            notify_telegram_chat_id="42",
        )
        notifiers = build_notifiers(settings)
        assert set(notifiers) == {"console", "webhook", "telegram"}
        hook, chat = notifiers["webhook"], notifiers["telegram"]
        assert isinstance(hook, RateLimitedNotifier) and isinstance(hook.inner, WebhookNotifier)
        assert hook.inner.fmt == "slack"
        assert isinstance(chat, RateLimitedNotifier) and isinstance(chat.inner, TelegramNotifier)

    def test_rate_limiting_can_be_turned_off(self) -> None:
        settings = Settings(notify_webhook_url=SecretStr("https://hooks.example.test/abc"))
        assert isinstance(build_notifiers(settings, rate_limit=False)["webhook"], WebhookNotifier)

    def test_half_a_telegram_configuration_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="both"):
            build_notifiers(
                Settings(notify_telegram_token=SecretStr("123456:your-telegram-bot-token"))
            )
        with pytest.raises(ConfigError, match="both"):
            build_notifiers(Settings(notify_telegram_chat_id="42"))

    def test_missing_targets_lists_rules_that_cannot_be_delivered(self) -> None:
        rules = RuleSet.from_yaml(
            """
rules:
  - {id: a, type: keyword, keywords: [x], notify: [console, webhook]}
  - {id: b, type: keyword, keywords: [y], notify: [console]}
  - {id: c, type: keyword, keywords: [z], notify: [telegram, webhook], enabled: false}
"""
        )
        assert missing_targets(rules, build_notifiers(Settings())) == {"a": ("webhook",)}

    def test_build_dispatcher_uses_the_rules_default_targets(self, db: Database) -> None:
        rules = RuleSet.from_yaml(
            "defaults: {notify: [webhook]}\nrules: [{id: a, type: keyword, keywords: [x]}]"
        )
        dispatcher = build_dispatcher(Settings(), db, ruleset=rules)
        assert dispatcher.default_targets == ("webhook",) and dispatcher.store is db
