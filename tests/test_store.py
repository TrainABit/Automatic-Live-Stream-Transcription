"""SQLite round trips: sessions, segments, transcripts, events and pending notifications."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from livestream_transcriber.models import SegmentInfo, StreamInfo
from livestream_transcriber.notify import Event
from livestream_transcriber.rules import Severity
from livestream_transcriber.store import SCHEMA_VERSION, Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "sub" / "lst.db")
    yield database  # type: ignore[misc]
    database.close()


def make_event(
    event_id: str = "e1", *, start: float = 10.0, session_id: int | None = None
) -> Event:
    return Event(
        event_id=event_id,
        rule_id="giveaway",
        text="a giveaway today",
        matched_text="giveaway",
        start=start,
        end=start + 2,
        wallclock=1_700_000_000.0,
        severity=Severity.WARNING,
        source_url="https://example.com/live",
        session_id=session_id,
        extra={"description": "Giveaway"},
        targets=("console", "webhook"),
    )


def test_opens_in_wal_mode_and_creates_parent_directories(db: Database) -> None:
    assert db.path.exists()
    mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_session_lifecycle(db: Database) -> None:
    info = StreamInfo(url="https://example.com/live", title="Show", channel="Chan", is_live=True)
    sid = db.start_session(url=info.url, info=info, mode="live", config={"stt": "mock"})
    (row,) = db.list_sessions()
    assert (row["id"], row["title"], row["channel"], row["is_live"], row["mode"]) == (
        sid, "Show", "Chan", 1, "live",
    )  # fmt: skip
    assert row["ended_at"] is None
    db.finish_session(sid, {"segments": 3})
    (row,) = db.list_sessions()
    assert row["ended_at"] is not None and '"segments": 3' in row["stats_json"]


def test_segments_are_upserted_by_index(db: Database) -> None:
    sid = db.start_session(url="u")
    first = SegmentInfo(index=0, started_at=1.0, offset=0.0, audio_chunks=5)
    db.record_segments(sid, [first])
    first.ended_at, first.audio_chunks, first.reason = 9.0, 40, "eof"
    db.record_segments(sid, [first, SegmentInfo(index=1, started_at=10.0, offset=9.5)])
    rows = db.list_segments(sid)
    assert [(r["seg_index"], r["audio_chunks"], r["reason"]) for r in rows] == [
        (0, 40, "eof"),
        (1, 0, None),
    ]


def test_transcripts_round_trip_and_recent_returns_the_tail_in_order(db: Database) -> None:
    sid = db.start_session(url="u")
    for i in range(5):
        db.insert_transcript(
            sid, start=i, end=i + 1, text=f"line {i}", language="en", provider="mock",
            model="m", latency=0.25, confidence=0.9, wallclock=100.0 + i,
        )  # fmt: skip
    db.insert_transcript(None, start=0, end=1, text="orphan")
    rows = db.recent_transcripts(3, session_id=sid)
    assert [r["text"] for r in rows] == ["line 2", "line 3", "line 4"]
    first = db.recent_transcripts(10, session_id=sid)[0]
    assert (first["language"], first["provider"], first["latency"]) == ("en", "mock", 0.25)
    assert len(db.recent_transcripts(100)) == 6


def test_unicode_text_survives(db: Database) -> None:
    db.insert_transcript(None, start=0, end=1, text="Willkommen zum Stream, schön dass ihr da seid")
    assert "schön" in db.recent_transcripts(1)[0]["text"]


def test_insert_event_is_idempotent_and_durable(db: Database, tmp_path: Path) -> None:
    sid = db.start_session(url="u")
    assert db.insert_event(make_event(session_id=sid)) is True
    assert db.insert_event(make_event(session_id=sid)) is False
    # A second connection sees the row: it is committed, not pending in a transaction.
    other = sqlite3.connect(db.path)
    assert other.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    other.close()
    row = db.get_event("e1")
    assert row is not None
    assert (row["rule_id"], row["severity"], row["ts_start"], row["notified_at"]) == (
        "giveaway", "warning", 10.0, None,
    )  # fmt: skip
    assert Event.from_dict(__import__("json").loads(row["payload_json"])) == make_event(
        session_id=sid
    )


def test_list_events_filters_and_orders(db: Database) -> None:
    a, b = db.start_session(url="a"), db.start_session(url="b")
    db.insert_event(make_event("late", start=50, session_id=a))
    db.insert_event(make_event("early", start=5, session_id=a))
    db.insert_event(make_event("other", start=1, session_id=b))
    assert [r["event_id"] for r in db.list_events(a)] == ["early", "late"]
    assert [r["event_id"] for r in db.list_events()] == ["other", "early", "late"]
    assert len(db.list_events(rule_id="nope")) == 0
    assert len(db.list_events(limit=2)) == 2


def test_pending_notifications_lifecycle(db: Database) -> None:
    for i, start in enumerate((30.0, 10.0, 20.0)):
        db.insert_event(make_event(f"e{i}", start=start))
    assert [r["event_id"] for r in db.pending_notifications()] == ["e1", "e2", "e0"]

    assert db.mark_notified("e1", when=111.0) is True
    assert db.mark_notified("e1", when=222.0) is False  # never rewrites history
    assert db.get_event("e1")["notified_at"] == 111.0  # type: ignore[index]
    assert db.is_notified("e1") and not db.is_notified("e0")
    assert [r["event_id"] for r in db.pending_notifications()] == ["e2", "e0"]
    assert [r["event_id"] for r in db.pending_notifications(limit=1)] == ["e2"]


def test_failures_are_counted_and_capped(db: Database) -> None:
    db.insert_event(make_event("e1"))
    assert db.record_notify_failure("e1", "boom") == 1
    assert db.record_notify_failure("e1", "boom again") == 2
    row = db.get_event("e1")
    assert row is not None and row["notify_attempts"] == 2 and row["notify_error"] == "boom again"
    assert len(db.pending_notifications(max_attempts=3)) == 1
    assert db.pending_notifications(max_attempts=2) == []
    db.mark_notified("e1")
    row = db.get_event("e1")
    assert row is not None and row["notify_error"] is None


def test_deliveries_are_recorded_per_target(db: Database) -> None:
    db.insert_event(make_event("e1"))
    assert db.mark_delivered("e1", "console") is True
    assert db.mark_delivered("e1", "console") is False
    assert db.delivered_targets("e1") == {"console"}
    assert db.delivered_targets("missing") == set()


def test_reopening_keeps_data_and_pending_state(tmp_path: Path) -> None:
    path = tmp_path / "lst.db"
    first = Database(path)
    first.insert_event(make_event("e1"))
    first.close()
    with Database(path) as again:
        assert [r["event_id"] for r in again.pending_notifications()] == ["e1"]
        assert again.counts()["events"] == 1


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(RuntimeError, match="schema"):
        Database(path)


def test_a_corrupt_file_is_moved_aside(tmp_path: Path) -> None:
    path = tmp_path / "lst.db"
    path.write_bytes(b"this is not a sqlite database" * 100)
    with Database(path) as fresh:
        assert fresh.counts()["events"] == 0
    assert list(tmp_path.glob("lst.db.corrupt-*"))


def test_usable_from_several_threads(db: Database) -> None:
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(20):
                db.insert_transcript(None, start=i, end=i + 1, text=f"t{n}-{i}")
                db.insert_event(make_event(f"e{n}-{i}", start=float(i)))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert db.counts()["transcripts"] == 80 and db.counts()["events"] == 80
