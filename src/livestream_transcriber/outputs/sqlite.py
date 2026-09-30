"""Store transcript segments in the SQLite database."""

from __future__ import annotations

from ..store.database import Database
from .base import TranscriptSegment

__all__ = ["SqliteSink"]


class SqliteSink:
    """Write each segment to the ``transcripts`` table of ``db``.

    The sink does not own the database: the session that opened it closes it.
    """

    def __init__(self, db: Database, session_id: int | None) -> None:
        self.db = db
        self.session_id = session_id

    def write(self, segment: TranscriptSegment) -> None:
        self.db.insert_transcript(
            self.session_id,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            language=segment.language,
            provider=segment.provider,
            model=segment.model,
            latency=segment.latency,
            confidence=segment.confidence,
            wallclock=segment.wallclock,
        )

    def close(self) -> None:
        return None
