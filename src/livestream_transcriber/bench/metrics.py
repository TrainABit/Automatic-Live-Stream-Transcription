"""Accuracy and speed metrics for the benchmark.

* **WER / CER**: word (or character) error rate, ``(S + D + I) / N``, from a
  Levenshtein alignment of the *normalised* reference and hypothesis. Normalisation
  (:func:`~livestream_transcriber.textnorm.norm_text`: NFC, case folding, punctuation
  removed) keeps casing and punctuation, which providers choose freely, from being
  scored as mistakes. Counts, not just rates, are kept so results over several clips
  are combined by *micro-averaging* (sum the errors, divide by the total reference
  length) rather than by averaging per-clip rates, which lets a two-word clip count
  as much as a ten-minute one.
* **Hallucination check**: speech models invent text on quiet or empty audio
  ("Thanks for watching!", subtitle credits, a phrase repeated in a loop). The
  patterns here are generic; they flag a transcript, they do not prove one.
* **Latency**: mean, median, p95 and max seconds per request, and the real-time
  factor (processing time divided by audio time; below 1 keeps up with a live stream).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from ..textnorm import norm_text

__all__ = [
    "ErrorCounts",
    "aggregate",
    "cer",
    "edit_counts",
    "has_repetition",
    "is_boilerplate",
    "is_hallucination",
    "latency_stats",
    "percentile",
    "real_time_factor",
    "wer",
]


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    """Edit operations needed to turn the hypothesis into the reference."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """``errors / reference_length``.

        An empty reference has no defined rate: ``0.0`` when the hypothesis is empty too
        (nothing said, nothing heard), ``None`` when it is not (pure insertions).
        """
        if self.reference_length:
            return self.errors / self.reference_length
        return 0.0 if self.errors == 0 else None

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )

    def describe(self) -> dict[str, Any]:
        rate = self.rate
        return {
            "rate": None if rate is None else round(rate, 4),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
        }


def aggregate(counts: Iterable[ErrorCounts]) -> ErrorCounts:
    """Micro-average: the sum of the counts (see the module docstring)."""
    total = ErrorCounts()
    for item in counts:
        total = total + item
    return total


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Levenshtein alignment of two token sequences, as counts of S, D and I.

    Runs in O(len(reference) * len(hypothesis)) time and O(len(hypothesis)) memory.
    Tokens the two sequences share at both ends are peeled off first, which is most of a
    good transcript.
    """
    ref, hyp = list(reference), list(hypothesis)
    total = len(ref)
    head = 0
    while head < len(ref) and head < len(hyp) and ref[head] == hyp[head]:
        head += 1
    ref, hyp = ref[head:], hyp[head:]
    while ref and hyp and ref[-1] == hyp[-1]:
        ref.pop()
        hyp.pop()
    if not ref or not hyp:
        return ErrorCounts(0, len(ref), len(hyp), total)

    # Each cell: (cost, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, r in enumerate(ref, start=1):
        current = [(i, 0, i, 0)]
        for j, h in enumerate(hyp, start=1):
            if r == h:
                best = previous[j - 1]
            else:
                c, s, d, n = previous[j - 1]
                best = (c + 1, s + 1, d, n)
            c, s, d, n = previous[j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, n)
            c, s, d, n = current[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, n + 1)
            current.append(best)
        previous = current
    _, s, d, n = previous[-1]
    return ErrorCounts(s, d, n, total)


def _words(text: str) -> list[str]:
    return norm_text(text).split()


def _chars(text: str) -> list[str]:
    return list(norm_text(text))


def wer(reference: str, hypothesis: str) -> ErrorCounts:
    """Word error counts of ``hypothesis`` against ``reference`` (both normalised)."""
    return edit_counts(_words(reference), _words(hypothesis))


def cer(reference: str, hypothesis: str) -> ErrorCounts:
    """Character error counts; single spaces between words count as characters."""
    return edit_counts(_chars(reference), _chars(hypothesis))


# --------------------------------------------------------------------------- #
# Hallucinations
# --------------------------------------------------------------------------- #

# Phrases speech models emit on silence or noise, mostly learned from subtitle files.
_BOILERPLATE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"thanks? (you )?for watching",
        r"thank you for (your )?(watching|listening)",
        r"please (like|subscribe)|like and subscribe|don'?t forget to subscribe",
        r"subtitles? (by|from)|captions? (by|from)|translated by",
        r"untertitel(ung)? (von|des|der|im auftrag)",
        r"vielen dank f(ü|ue)r('s| das| ihre)? (das )?(zuschauen|zusehen|aufmerksamkeit)",
        r"bis zum n(ä|ae)chsten mal",
        r"see you (in the )?next (time|video|episode)",
        r"merci d'avoir regard(é|e)",
        r"gracias por ver",
    )
)
# Boilerplate is a hallucination when it is (nearly) all the chunk says. Inside a long
# real sentence the same words are just speech.
_BOILERPLATE_MAX_WORDS = 10


def is_boilerplate(text: str) -> bool:
    """True when ``text`` is a short stock phrase of the kind models make up on silence."""
    words = _words(text)
    if not words or len(words) > _BOILERPLATE_MAX_WORDS:
        return False
    return any(pattern.search(text) for pattern in _BOILERPLATE)


def has_repetition(text: str, *, min_repeats: int = 4, max_ngram: int = 6) -> bool:
    """True when some 1..``max_ngram``-word phrase repeats ``min_repeats`` times in a row.

    The other classic failure: the decoder gets stuck and loops on one phrase.
    """
    words = _words(text)
    for size in range(1, max_ngram + 1):
        if len(words) < size * min_repeats:
            break
        for start in range(len(words) - size * min_repeats + 1):
            unit = words[start : start + size]
            if all(
                words[start + k * size : start + (k + 1) * size] == unit
                for k in range(1, min_repeats)
            ):
                return True
    return False


def is_hallucination(text: str) -> bool:
    """A transcript that looks invented: stock boilerplate or a repetition loop."""
    return is_boilerplate(text) or has_repetition(text)


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile (``fraction`` in [0, 1]); ``None`` for no values.

    Nearest rank always returns a value that was actually measured, which is what a
    latency report should quote.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(len(ordered) * fraction)))
    return ordered[rank - 1]


def latency_stats(latencies: Sequence[float]) -> dict[str, float | int | None]:
    """Mean, median, p95 and max of per-request latencies, in seconds."""
    if not latencies:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(latencies),
        "mean": fmean(latencies),
        "p50": percentile(latencies, 0.5),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies),
    }


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float | None:
    """Processing time per second of audio; ``None`` without audio. Below 1 keeps up live."""
    if audio_seconds <= 0:
        return None
    return processing_seconds / audio_seconds
