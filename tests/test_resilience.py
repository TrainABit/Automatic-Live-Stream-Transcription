from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from livestream_transcriber.resilience.memory import current_rss_mb, memory_pressure_level
from livestream_transcriber.resilience.storage import (
    open_sqlite_connection,
    recover_corrupt_sqlite,
)
from livestream_transcriber.resilience.stt_health import SttOutageMonitor

# ------------------------------------------------------------------ memory


def test_rss_is_reported_on_this_platform():
    rss = current_rss_mb()
    assert rss is not None
    assert rss > 1.0


@pytest.mark.parametrize(
    ("rss", "limit", "level"),
    [
        (100, 1000, 0),
        (849, 1000, 0),
        (850, 1000, 1),
        (999, 1000, 1),
        (1000, 1000, 2),
        (5000, 1000, 2),
        (5000, 0, 0),  # a non-positive limit disables the check
        (5000, -1, 0),
    ],
)
def test_memory_pressure_levels(rss: float, limit: float, level: int):
    assert memory_pressure_level(rss, limit_mb=limit) == level


# ----------------------------------------------------------------- storage


def _seed(path: Path) -> None:
    conn = open_sqlite_connection(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.close()


def test_open_creates_parent_directories_and_returns_rows_by_name(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "a.db"
    conn = open_sqlite_connection(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (7)")
    assert conn.execute("SELECT x FROM t").fetchone()["x"] == 7
    conn.close()


def test_a_healthy_database_is_left_alone(tmp_path: Path):
    path = tmp_path / "a.db"
    _seed(path)
    conn = open_sqlite_connection(path)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    conn.close()
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_a_corrupt_database_is_moved_aside_and_replaced(tmp_path: Path):
    path = tmp_path / "a.db"
    path.write_bytes(b"this is not a sqlite database" * 100)
    Path(str(path) + "-wal").write_bytes(b"stale wal")
    conn = open_sqlite_connection(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.close()
    backups = list(tmp_path.glob("a.db.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes().startswith(b"this is not")
    assert not Path(str(path) + "-wal").exists()


def test_recover_false_raises_instead_of_moving_the_file(tmp_path: Path):
    path = tmp_path / "a.db"
    path.write_bytes(b"garbage" * 200)
    with pytest.raises(sqlite3.DatabaseError):
        open_sqlite_connection(path, recover=False)
    assert path.exists()
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_a_locked_database_is_not_mistaken_for_a_corrupt_one(tmp_path: Path):
    path = tmp_path / "a.db"
    _seed(path)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(sqlite3.OperationalError):
            open_sqlite_connection(path, timeout=0.05)
    finally:
        holder.close()
    assert path.exists()
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_recover_corrupt_sqlite_of_a_missing_file_is_a_noop(tmp_path: Path):
    assert recover_corrupt_sqlite(tmp_path / "missing.db") is None


# --------------------------------------------------------------- stt health


def test_outage_alerts_exactly_once_after_the_threshold():
    monitor = SttOutageMonitor(outage_seconds=300.0)
    assert monitor.record_failure(1000.0) is False  # outage starts
    assert monitor.record_failure(1200.0) is False
    assert monitor.record_failure(1300.0) is True  # threshold reached
    assert monitor.record_failure(1400.0) is False  # already alerted
    assert monitor.describe()["alerted"] is True


def test_success_ends_an_outage_and_rearms_the_alert():
    monitor = SttOutageMonitor(outage_seconds=10.0)
    monitor.record_failure(0.0)
    assert monitor.record_failure(11.0) is True
    monitor.record_success()
    assert monitor.describe() == {
        "outage_started": None,
        "alerted": False,
        "outage_seconds": 10.0,
    }
    assert monitor.record_failure(100.0) is False
    assert monitor.record_failure(111.0) is True
