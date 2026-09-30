"""Turn overlapping chunk transcripts into a clean, temporal speech stream.

Chunks (and rolling windows, if a provider uses them) can carry the same
sentence twice or cut one sentence in two::

    chunk 1: "we are going live in"
    chunk 2: "live in five minutes"

The stream keeps every raw transcript but only emits an utterance when it is new
speech, or a *revision* of the previous one (an extension or a stitch). A
revision replaces the previous utterance instead of adding a second one, so
downstream consumers (subtitle writers, rule matching, notifications) see each
sentence once.

Memory is bounded: only the last ``max_history`` transcripts are kept for
inspection, however long the stream runs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..stt.base import Transcript
from ..textnorm import norm_text, norm_token

__all__ = ["MIXED", "IngestResult", "SpeechStream", "stitch"]

#: Provider/model label of an utterance stitched from chunks of different providers.
MIXED = "mixed"

DEFAULT_MAX_HISTORY = 256


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _keyed(text: str) -> list[tuple[str, str]]:
    """``(surface word, comparison key)`` pairs; punctuation-only words are dropped."""
    pairs = [(word, norm_token(word)) for word in text.split()]
    return [(word, key) for word, key in pairs if key]


def stitch(left: str, right: str) -> str | None:
    """Merge two texts when a suffix of ``left`` is a prefix of ``right``.

    Words are compared case- and punctuation-insensitively, but the result keeps
    the original spelling: ``left`` up to the overlap, then the rest of ``right``.
    Returns ``None`` when the texts do not overlap.
    """
    a, b = _keyed(left), _keyed(right)
    if not a or not b:
        return None
    best = 0
    for n in range(1, min(len(a), len(b)) + 1):
        if [k for _, k in a[-n:]] == [k for _, k in b[:n]]:
            best = n
    if best == 0:
        return None
    return " ".join([w for w, _ in a] + [w for w, _ in b[best:]])


def _merged_label(left: str | None, right: str | None) -> str | None:
    """One provider/model name for a merged utterance, ``mixed`` if they differ."""
    if not left or not right:
        return left or right
    return left if left == right else MIXED


def _merge_timed(
    first: list[dict[str, Any]] | None, second: list[dict[str, Any]] | None
) -> list[dict[str, Any]] | None:
    """Union of two timed-record lists, without records both sides carry."""
    if not first or not second:
        return first or second
    seen = {(round(float(r.get("start") or 0.0), 2), str(r.get("text") or "")) for r in first}
    extra = [
        r
        for r in second
        if (round(float(r.get("start") or 0.0), 2), str(r.get("text") or "")) not in seen
    ]
    return sorted([*first, *extra], key=lambda r: float(r.get("start") or 0.0))


@dataclass(slots=True)
class IngestResult:
    """What one chunk transcript did to the stream."""

    transcript: Transcript | None
    """The new or revised utterance; ``None`` when the input added nothing."""
    revision: bool = False
    """True when ``transcript`` replaces the previous utterance."""


def _is_repeat(text: str, prev_text: str) -> bool:
    """True when ``text`` is the same words as, or a contiguous run inside, ``prev_text``.

    Comparing whole words matters: a plain substring test would treat "no" as a
    repeat of "we can not know" and silently drop a real utterance.
    """
    return text == prev_text or f" {text} " in f" {prev_text} "


class SpeechStream:
    """Dedup and stitch a sequence of chunk transcripts.

    ``max_gap`` is the silence (seconds) after which a new transcript is a new
    utterance no matter how similar its text is. ``jaccard_dup`` is the word-set
    similarity at which a transcript counts as a repeat of the previous one.
    """

    def __init__(
        self,
        *,
        max_gap: float = 2.5,
        jaccard_dup: float = 0.75,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.max_gap = max_gap
        self.jaccard_dup = jaccard_dup
        self.raw: deque[Transcript] = deque(maxlen=max_history)
        """Every transcript received, most recent ``max_history`` only."""
        self.normalized: deque[Transcript] = deque(maxlen=max_history)
        """Emitted utterances, most recent ``max_history`` only."""

    def ingest(self, transcript: Transcript) -> IngestResult:
        self.raw.append(transcript)
        text = norm_text(transcript.text)
        if not text:
            return IngestResult(None)

        if not self.normalized:
            self.normalized.append(transcript)
            return IngestResult(transcript)

        prev = self.normalized[-1]
        if transcript.start - prev.end > self.max_gap:
            self.normalized.append(transcript)
            return IngestResult(transcript)

        prev_text = norm_text(prev.text)
        if _is_repeat(text, prev_text):
            return IngestResult(None)

        stitched = stitch(prev.text, transcript.text)
        longer = len(text) > len(prev_text) + 1 and (
            prev_text in text or _jaccard(text, prev_text) >= self.jaccard_dup
        )
        if stitched or longer:
            merged = Transcript(
                start=min(prev.start, transcript.start),
                end=max(prev.end, transcript.end),
                text=stitched or transcript.text,
                confidence=transcript.confidence,
                segments=(
                    _merge_timed(prev.segments, transcript.segments)
                    if stitched
                    else transcript.segments or prev.segments
                ),
                words=(
                    _merge_timed(prev.words, transcript.words)
                    if stitched
                    else transcript.words or prev.words
                ),
                provider_latency=transcript.provider_latency,
                cost_usd=transcript.cost_usd,
                # Provenance survives the merge: a sentence stitched from
                # fallback chunks is still fallback speech, and one degraded
                # part makes the whole utterance degraded.
                model=_merged_label(prev.model, transcript.model),
                provider=_merged_label(prev.provider, transcript.provider),
                language=transcript.language or prev.language,
                degraded=prev.degraded or transcript.degraded,
            )
            self.normalized[-1] = merged
            return IngestResult(merged, revision=True)

        if _jaccard(text, prev_text) >= self.jaccard_dup:
            return IngestResult(None)

        self.normalized.append(transcript)
        return IngestResult(transcript)
