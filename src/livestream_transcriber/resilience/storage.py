"""SQLite open with integrity check and corruption recovery."""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger(__name__)

__all__ = ["open_sqlite_connection", "recover_corrupt_sqlite"]


#: SQLite's messages for a file that is damaged or is not a database at all.
_CORRUPTION_MARKERS = ("malformed", "not a database")


def _is_corruption(exc: sqlite3.DatabaseError) -> bool:
    """True only for damage to the file itself.

    "database is locked" or "unable to open database file" are operational problems:
    moving the user's database aside because another process holds it would lose data
    that is perfectly fine.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _CORRUPTION_MARKERS)


def _integrity_ok(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.DatabaseError as exc:
        if not _is_corruption(exc):
            raise
        return False
    return row is not None and str(row[0]).lower() == "ok"


def recover_corrupt_sqlite(path: Path) -> Path | None:
    """Move a corrupt database aside, with its WAL/SHM files removed.

    Returns the backup path, or ``None`` when there was nothing to move.
    """
    path = Path(path)
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".corrupt-{stamp}")
    shutil.move(str(path), str(backup))
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    log.error("sqlite corrupt; moved aside", extra={"path": str(path), "backup": str(backup)})
    return backup


def _connect(path: Path, timeout: float, check_same_thread: bool) -> sqlite3.Connection:
    # isolation_level=None: autocommit, transactions are explicit where needed.
    conn = sqlite3.connect(
        path, isolation_level=None, timeout=timeout, check_same_thread=check_same_thread
    )
    conn.row_factory = sqlite3.Row
    return conn


def open_sqlite_connection(
    path: str | Path,
    *,
    timeout: float = 10.0,
    recover: bool = True,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Connect and verify integrity, optionally recovering by renaming the file.

    Only a damaged file is moved aside; a locked or unreadable one raises, untouched. With
    ``recover=False`` a corrupt file raises :class:`sqlite3.DatabaseError` as well.
    ``check_same_thread=False`` lets a caller that serialises access itself (with a
    lock) share the connection between threads.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    try:
        conn = _connect(p, timeout, check_same_thread)
        healthy = not existed or _integrity_ok(conn)
    except sqlite3.DatabaseError as exc:
        if not _is_corruption(exc):
            raise
        healthy = False
        conn = None
    if healthy and conn is not None:
        return conn
    if conn is not None:
        conn.close()
    if not recover:
        raise sqlite3.DatabaseError(f"integrity check failed for {p}")
    recover_corrupt_sqlite(p)
    return _connect(p, timeout, check_same_thread)
