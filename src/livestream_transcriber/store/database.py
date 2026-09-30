"""SQLite storage for sessions, transcripts and rule events.

Design notes:

* **One writer, WAL.** Writes are small and infrequent. WAL mode lets you open
  the file in another tool while a run is in progress without blocking it.
* **Durable before notify.** An event is inserted, and therefore committed
  (the connection is in autocommit mode), before any notifier is called. A
  crash between the two leaves a row with ``notified_at IS NULL`` that
  :meth:`Database.pending_notifications` returns after a restart.
* **Per-target deliveries.** An event may go to several targets. Each success is
  recorded in ``deliveries`` so a retry only repeats the targets that failed.
* **Thread-safe.** The connection is shared and guarded by a lock: the
  pipeline writes from the event loop while notifications are delivered from
  worker threads.
* **Additive schema.** Tables are created with ``IF NOT EXISTS``; ``user_version``
  records the schema generation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_setup import get_logger
from ..models import SegmentInfo, StreamInfo
from ..resilience.storage import open_sqlite_connection

if TYPE_CHECKING:
    from ..notify.base import Event

log = get_logger(__name__)

__all__ = ["SCHEMA_VERSION", "Database"]

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    url           TEXT NOT NULL,
    title         TEXT,
    channel       TEXT,
    is_live       INTEGER NOT NULL DEFAULT 0,
    mode          TEXT NOT NULL DEFAULT 'live',
    recording_dir TEXT,
    config_json   TEXT,
    stats_json    TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seg_index    INTEGER NOT NULL,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    offset_s     REAL NOT NULL,
    audio_chunks INTEGER NOT NULL DEFAULT 0,
    reason       TEXT,
    UNIQUE (session_id, seg_index)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    start_s    REAL NOT NULL,
    end_s      REAL NOT NULL,
    text       TEXT NOT NULL,
    language   TEXT,
    provider   TEXT,
    model      TEXT,
    latency    REAL,
    confidence REAL,
    wallclock  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    rule_id         TEXT NOT NULL,
    ts_start        REAL NOT NULL,
    ts_end          REAL NOT NULL,
    wallclock       REAL NOT NULL,
    text            TEXT NOT NULL,
    matched_text    TEXT NOT NULL,
    severity        TEXT NOT NULL,
    source_url      TEXT,
    payload_json    TEXT NOT NULL,
    notified_at     REAL,
    notify_attempts INTEGER NOT NULL DEFAULT 0,
    notify_error    TEXT
);

CREATE TABLE IF NOT EXISTS deliveries (
    event_id     TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    target       TEXT NOT NULL,
    delivered_at REAL NOT NULL,
    PRIMARY KEY (event_id, target)
);

CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id, start_s);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_start);
CREATE INDEX IF NOT EXISTS idx_events_pending ON events(notified_at) WHERE notified_at IS NULL;
"""


def _dump(value: Any) -> str | None:
    return json.dumps(value, default=str, ensure_ascii=False) if value else None


class Database:
    """A thin, synchronous wrapper around one SQLite file."""

    def __init__(self, path: str | Path, *, recover: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn = open_sqlite_connection(self.path, recover=recover, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            found = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if found > SCHEMA_VERSION:
                self._conn.close()
                raise RuntimeError(
                    f"{self.path} has schema {found}; this build supports up to {SCHEMA_VERSION}"
                )
            self._conn.executescript(SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._conn

    # ------------------------------------------------------------------ sessions

    def start_session(
        self,
        *,
        url: str,
        info: StreamInfo | None = None,
        mode: str = "live",
        recording_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> int:
        with self._cursor() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (started_at, url, title, channel, is_live, mode, recording_dir, config_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    url,
                    info.title if info else None,
                    info.channel if info else None,
                    int(bool(info and info.is_live)),
                    mode,
                    recording_dir,
                    _dump(config),
                ),
            )
        session_id = int(cur.lastrowid or 0)
        log.debug("session opened", extra={"session_id": session_id, "mode": mode})
        return session_id

    def finish_session(self, session_id: int, stats: dict[str, Any] | None = None) -> None:
        with self._cursor() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=?, stats_json=? WHERE id=?",
                (time.time(), _dump(stats), session_id),
            )

    def list_sessions(self, limit: int = 10) -> list[sqlite3.Row]:
        with self._cursor() as conn:
            return conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()

    def record_segments(self, session_id: int, segments: Iterable[SegmentInfo]) -> None:
        """Insert or update capture segments (one uninterrupted ffmpeg run each)."""
        rows = [
            (session_id, s.index, s.started_at, s.ended_at, s.offset, s.audio_chunks, s.reason)
            for s in segments
        ]
        with self._cursor() as conn:
            conn.executemany(
                """INSERT INTO segments
                   (session_id, seg_index, started_at, ended_at, offset_s, audio_chunks, reason)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(session_id, seg_index) DO UPDATE SET
                     ended_at=excluded.ended_at,
                     audio_chunks=excluded.audio_chunks,
                     reason=excluded.reason""",
                rows,
            )

    def list_segments(self, session_id: int) -> list[sqlite3.Row]:
        with self._cursor() as conn:
            return conn.execute(
                "SELECT * FROM segments WHERE session_id=? ORDER BY seg_index", (session_id,)
            ).fetchall()

    # --------------------------------------------------------------- transcripts

    def insert_transcript(
        self,
        session_id: int | None,
        *,
        start: float,
        end: float,
        text: str,
        language: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        latency: float | None = None,
        confidence: float | None = None,
        wallclock: float | None = None,
    ) -> int:
        with self._cursor() as conn:
            cur = conn.execute(
                """INSERT INTO transcripts
                   (session_id, start_s, end_s, text, language, provider, model,
                    latency, confidence, wallclock)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    start,
                    end,
                    text,
                    language,
                    provider,
                    model,
                    latency,
                    confidence,
                    time.time() if wallclock is None else wallclock,
                ),
            )
        return int(cur.lastrowid or 0)

    def recent_transcripts(
        self, limit: int = 50, session_id: int | None = None
    ) -> list[sqlite3.Row]:
        """The latest ``limit`` transcript rows, oldest first."""
        where, args = ("WHERE session_id=?", (session_id,)) if session_id is not None else ("", ())
        with self._cursor() as conn:
            rows = conn.execute(
                f"SELECT * FROM transcripts {where} ORDER BY id DESC LIMIT ?", (*args, limit)
            ).fetchall()
        return rows[::-1]

    # -------------------------------------------------------------------- events

    def insert_event(self, event: Event, session_id: int | None = None) -> bool:
        """Persist an event. Returns ``False`` when its ``event_id`` already exists.

        The row is committed when this returns, which is what allows the caller to
        notify afterwards without risking a lost alert.
        """
        sid = session_id if session_id is not None else event.session_id
        with self._cursor() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, session_id, rule_id, ts_start, ts_end, wallclock, text,
                    matched_text, severity, source_url, payload_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.event_id,
                    sid,
                    event.rule_id,
                    event.start,
                    event.end,
                    event.wallclock,
                    event.text,
                    event.matched_text,
                    event.severity.value,
                    event.source_url,
                    json.dumps(event.to_dict(), default=str, ensure_ascii=False),
                ),
            )
        return cur.rowcount == 1

    def get_event(self, event_id: str) -> sqlite3.Row | None:
        with self._cursor() as conn:
            row: sqlite3.Row | None = conn.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        return row

    def list_events(
        self,
        session_id: int | None = None,
        *,
        rule_id: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        args: list[Any] = []
        if session_id is not None:
            clauses.append("session_id=?")
            args.append(session_id)
        if rule_id is not None:
            clauses.append("rule_id=?")
            args.append(rule_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        tail = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            args.append(limit)
        with self._cursor() as conn:
            return conn.execute(
                f"SELECT * FROM events {where} ORDER BY ts_start, rowid{tail}", args
            ).fetchall()

    # ------------------------------------------------------ notification lifecycle

    def mark_delivered(self, event_id: str, target: str, when: float | None = None) -> bool:
        """Record that ``target`` received the event. Idempotent; returns True when new."""
        with self._cursor() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO deliveries (event_id, target, delivered_at) VALUES (?,?,?)",
                (event_id, target, time.time() if when is None else when),
            )
        return cur.rowcount == 1

    def delivered_targets(self, event_id: str) -> set[str]:
        with self._cursor() as conn:
            rows = conn.execute("SELECT target FROM deliveries WHERE event_id=?", (event_id,))
            return {str(r["target"]) for r in rows}

    def mark_notified(self, event_id: str, when: float | None = None) -> bool:
        """Mark the event as fully handled.

        It never moves an existing timestamp, so a retry that races a successful
        send cannot rewrite history. Returns True when this call set it.
        """
        with self._cursor() as conn:
            cur = conn.execute(
                "UPDATE events SET notified_at=?, notify_error=NULL "
                "WHERE event_id=? AND notified_at IS NULL",
                (time.time() if when is None else when, event_id),
            )
        return bool(cur.rowcount)

    def record_notify_failure(self, event_id: str, error: str) -> int:
        """Count a failed delivery attempt; returns the new attempt total."""
        with self._cursor() as conn:
            conn.execute(
                "UPDATE events SET notify_attempts=notify_attempts+1, notify_error=? "
                "WHERE event_id=? AND notified_at IS NULL",
                (error[:500], event_id),
            )
            row = conn.execute(
                "SELECT notify_attempts FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def is_notified(self, event_id: str) -> bool:
        row = self.get_event(event_id)
        return row is not None and row["notified_at"] is not None

    def pending_notifications(
        self, limit: int = 50, *, max_attempts: int | None = None
    ) -> list[sqlite3.Row]:
        """Events whose alert never fully landed, oldest first.

        ``max_attempts`` skips events that already failed that many times, so a
        permanently broken target cannot be retried forever.
        """
        clause = "AND notify_attempts < ?" if max_attempts is not None else ""
        args: tuple[Any, ...] = (max_attempts, limit) if max_attempts is not None else (limit,)
        with self._cursor() as conn:
            return conn.execute(
                f"""SELECT * FROM events
                    WHERE notified_at IS NULL {clause}
                    ORDER BY ts_start, rowid LIMIT ?""",
                args,
            ).fetchall()

    # --------------------------------------------------------------------- misc

    def counts(self) -> dict[str, int]:
        """Row counts per table, handy for a final run summary."""
        tables = ("sessions", "segments", "transcripts", "events")
        with self._cursor() as conn:
            return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables}
