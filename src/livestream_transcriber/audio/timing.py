"""Map provider word/segment times onto the session clock and narrow matches.

Providers report segment and word times relative to the audio they were given
(the chunk). Everything downstream (subtitles, rule hits, the database) speaks
session time, so :func:`sessionize_transcript_timestamps` shifts them once, on
arrival. :func:`align_match_timing` then narrows the interval of a rule match
from "the whole chunk" to the words that actually matched.

Only native word or segment times can place a phrase inside a chunk. A
transcript without them keeps its whole chunk interval, which is the honest
statement of when the words were said; audio energy is no substitute (the
loudest 250 ms of a chunk is where the speaker was loud, not where they said it).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..stt.base import Transcript
from ..textnorm import nfc, norm_token

__all__ = ["align_match_timing", "sessionize_transcript_timestamps"]

# Set on a word/segment record once its ``start``/``end`` are on the session
# clock, so a transcript can pass through sessionizing more than once (on arrival,
# and again wherever a caller is unsure) without the chunk offset being added twice.
_CLOCK = "clock"
_SESSION_CLOCK = "session"

# Span reported for a match whose words carry no duration.
_MIN_SPAN = 0.35


def _time(value: Any) -> float | None:
    """``value`` as seconds when it is a real number (not a bool), else ``None``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _session_record(item: dict[str, Any], *, chunk_start: float) -> dict[str, Any]:
    """``item`` with ``start``/``end`` on the session clock, marked as such."""
    if item.get(_CLOCK) == _SESSION_CLOCK:
        return item
    rec = dict(item)
    # Providers report times relative to the audio they were given, so the chunk start
    # is the only offset there is. A value past the chunk end (a last word that runs a
    # little over) is still chunk-relative; guessing otherwise would put one record's
    # start and end on different clocks.
    for key in ("start", "end"):
        value = _time(rec.get(key))
        if value is not None:
            rec[key] = chunk_start + value
    rec[_CLOCK] = _SESSION_CLOCK
    return rec


def sessionize_transcript_timestamps(transcript: Transcript) -> Transcript:
    """Put a provider transcript on the session clock.

    The text is put in Unicode NFC (a provider's decomposed "u" plus combining
    diaeresis would otherwise never match a keyword written with a precomposed
    letter), and every native word/segment record gets session-clock
    ``start``/``end``. Idempotent: a record already on the session clock is left
    as it is, however early in the session it sits. The provider's own record
    dicts are never mutated.
    """
    text = nfc(transcript.text)
    if text != transcript.text:
        transcript = replace(transcript, text=text)
    if not transcript.words and not transcript.segments:
        return transcript
    chunk_start = transcript.start

    def shift(items: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not items:
            return items
        shifted = [
            _session_record(item, chunk_start=chunk_start)
            for item in items
            if isinstance(item, dict)
        ]
        return shifted or None

    return replace(transcript, words=shift(transcript.words), segments=shift(transcript.segments))


def _timed(items: list[dict[str, Any]] | None) -> list[tuple[str, float, float]]:
    """``(text, start, end)`` of every sessionized record that has both times."""
    out: list[tuple[str, float, float]] = []
    for item in items or ():
        start, end = _time(item.get("start")), _time(item.get("end"))
        if start is None or end is None:
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        if text:
            out.append((text, start, end))
    return out


def _word_window(
    words: list[tuple[str, float, float]], target: str, near: tuple[float, float]
) -> tuple[float, float] | None:
    """The shortest run of consecutive words whose letters contain ``target``.

    Words are compared through :func:`norm_token`, joined without spaces, so a
    match survives tokenisation differences ("e-mail" as one word or two). Among
    candidates an exact word-for-word match wins, then the one overlapping the
    ``near`` interval most (the same phrase can occur twice in a chunk), then
    the earliest.
    """
    tokens = [norm_token(text) for text, _, _ in words]
    best: tuple[tuple[int, float, float], tuple[float, float]] | None = None
    for i in range(len(tokens)):
        joined = ""
        for j in range(i, len(tokens)):
            joined += tokens[j]
            if target not in joined:
                continue
            # Trim from the left while the target is still contained.
            lo = i
            while lo < j and target in "".join(tokens[lo + 1 : j + 1]):
                lo += 1
            span = (words[lo][1], words[j][2])
            overlap = min(span[1], near[1]) - max(span[0], near[0])
            exact = int("".join(tokens[lo : j + 1]) == target)
            score = (exact, overlap, -span[0])
            if best is None or score > best[0]:
                best = (score, span)
            break
    return None if best is None else best[1]


def _segment_span(
    segments: list[tuple[str, float, float]], target: str, near: tuple[float, float]
) -> tuple[float, float] | None:
    """The segment containing ``target``, preferring the one overlapping ``near``."""
    best: tuple[tuple[float, float], tuple[float, float]] | None = None
    for text, start, end in segments:
        if target not in norm_token(text):
            continue
        overlap = min(end, near[1]) - max(start, near[0])
        score = (overlap, -start)
        if best is None or score > best[0]:
            best = (score, (start, end))
    return None if best is None else best[1]


def align_match_timing(
    match_start: float,
    match_end: float,
    transcript: Transcript,
    *,
    matched_text: str | None = None,
) -> tuple[float, float]:
    """Session-clock ``(start, end)`` of a match inside ``transcript``.

    ``match_start``/``match_end`` are the interval the caller would report
    without any narrowing (normally the transcript's own span). ``matched_text``
    is the text a rule matched: when the transcript carries native word times the
    result is the span of exactly those words, otherwise the containing segment,
    otherwise the given interval. The result is clamped to the transcript's span,
    gets a short minimum duration if the words have none, and is a pure function of its inputs.
    """
    lo, hi = transcript.start, transcript.end
    fallback = (max(lo, min(match_start, hi)), max(lo, min(match_end, hi)))
    target = norm_token(matched_text)
    if not target:
        return fallback

    spoken = sessionize_transcript_timestamps(transcript)
    near = (match_start, match_end)
    span = _word_window(_timed(spoken.words), target, near)
    if span is None:
        span = _segment_span(_timed(spoken.segments), target, near)
    if span is None:
        return fallback
    start = max(lo, min(span[0], hi))
    end = max(start, min(span[1], hi))
    if end <= start:
        end = min(hi, start + _MIN_SPAN)
    return start, end
