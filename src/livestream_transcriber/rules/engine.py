"""Evaluate transcript segments against a rule set.

The engine has to answer one question well: *is this hit news?* A speaker who
says "giveaway" five times in a minute should raise one alert, and an STT
provider that re-emits the tail of the previous chunk must not raise another.
Three mechanisms cooperate:

1. **Cooldown.** After a rule fires for a key (the normalised match, or the
   values of its named groups), the same key is silent for the rule's
   ``cooldown_seconds``.
2. **Fuzzy dedup.** Within a shorter window, a hit whose surrounding text is
   near-identical to one already reported (token-set Jaccard or character
   ratio >= 0.85) is a repeat even when its key differs, which is what an
   overlapping re-transcription looks like.
3. **Escalation.** A hit of higher severity than the one that started the
   cooldown always passes. Rules that share a ``group`` share cooldown state,
   so a ``critical`` rule can escalate an earlier ``info`` alert for the same
   phrase.

All windows are measured in *stream time* (the segment's start), never wall
clock, so replaying a recording gives the same alerts as the live run. State is
bounded: an LRU of cooldown entries and a short deque of recent contexts per
group.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol

from ..logging_setup import get_logger
from ..textnorm import nfc, norm_text
from .matchers import LlmMatcher, Match, Matcher, build_matcher
from .model import Rule, RuleSet, RuleType, Severity

log = get_logger(__name__)

__all__ = [
    "RuleEngine",
    "RuleHit",
    "RuleStats",
    "SegmentLike",
    "TextSegment",
    "text_similarity",
]

SIMILARITY_THRESHOLD = 0.85
LLM_MIN_WORDS = 3
_KEY_MAX_CHARS = 120
_LLM_KEY = "llm"


class SegmentLike(Protocol):
    """What the engine needs from a transcript segment.

    ``language`` and ``confidence`` are optional attributes, read with
    ``getattr``; anything with ``text``, ``start`` and ``end`` works.
    """

    @property
    def text(self) -> str: ...

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A minimal segment, for tests and the ``lst rules test`` dry run."""

    text: str
    start: float = 0.0
    end: float = 0.0
    language: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RuleHit:
    """A rule that fired for a segment."""

    rule_id: str
    matched_text: str
    start: float
    """Stream time (seconds) at which the segment began."""
    end: float
    """Stream time (seconds) at which the segment ended."""
    context: str
    """The text around the match, trimmed to a readable length."""
    severity: Severity
    groups: Mapping[str, str] = field(default_factory=dict)
    """Named regex groups, or ``{"keyword": ...}`` for keyword rules."""
    span: tuple[int, int] = (0, 0)
    """Character offsets of the match in the NFC-normalised segment text."""
    key: str = ""
    """The dedup key; identical keys are the same alert."""
    notify: tuple[str, ...] = ()
    description: str = ""
    upgrade: bool = False
    """True when this hit escalated an earlier, lower-severity alert."""
    reason: str | None = None


@dataclass(slots=True)
class RuleStats:
    segments: int = 0
    hits: int = 0
    upgrades: int = 0
    suppressed_cooldown: int = 0
    suppressed_similar: int = 0
    llm_skipped: int = 0

    def describe(self) -> dict[str, int]:
        return {
            "segments": self.segments,
            "hits": self.hits,
            "upgrades": self.upgrades,
            "suppressed_cooldown": self.suppressed_cooldown,
            "suppressed_similar": self.suppressed_similar,
            "llm_skipped": self.llm_skipped,
        }


@dataclass(slots=True)
class _Fired:
    ts: float
    rank: int


@dataclass(slots=True)
class _Recent:
    seg: int
    ts: float
    rank: int
    tokens: frozenset[str]
    text: str


def text_similarity(a: str, b: str) -> float:
    """Similarity of two *normalised* strings in [0, 1].

    The larger of token-set Jaccard (robust to word order and repeated words)
    and the character-level ratio (robust to one extra or missing word).
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return max(jaccard, SequenceMatcher(None, a, b, autojunk=False).ratio())


class RuleEngine:
    """Stateful evaluator: feed it segments in stream order."""

    def __init__(
        self,
        ruleset: RuleSet,
        *,
        llm: LlmMatcher | None = None,
        fuzzy_window_seconds: float = 30.0,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_state: int = 1024,
        recent_per_group: int = 32,
        context_chars: int = 160,
    ) -> None:
        self.ruleset = ruleset
        self.llm = llm
        self.fuzzy_window_seconds = fuzzy_window_seconds
        self.similarity_threshold = similarity_threshold
        self.context_chars = max(20, context_chars)
        self.stats = RuleStats()
        self._max_state = max(1, max_state)
        self._recent_per_group = max(1, recent_per_group)
        self._cooldowns: OrderedDict[tuple[str, str], _Fired] = OrderedDict()
        self._recent: dict[str, deque[_Recent]] = {}
        self._lock = threading.Lock()
        self._llm_warned = False
        self._seg_no = 0
        # Most severe first: when two rules of one group match the same phrase in one
        # segment, the stronger one fires and the weaker is suppressed as a repeat.
        ordered = sorted(ruleset.enabled_rules, key=lambda r: -r.severity.rank)
        self._matchers: list[tuple[Rule, Matcher | None]] = [
            (rule, build_matcher(rule)) for rule in ordered
        ]

    # ---------------------------------------------------------------- evaluation

    def evaluate(self, segment: SegmentLike) -> list[RuleHit]:
        """Hits for one segment, in text order. Cheap unless an ``llm`` rule applies."""
        text = nfc(segment.text).strip()
        self.stats.segments += 1
        self._seg_no += 1
        if not text:
            return []
        ts = float(segment.start)
        language = _primary_language(getattr(segment, "language", None))
        confidence = getattr(segment, "confidence", None)

        hits: list[RuleHit] = []
        for rule, matcher in self._matchers:
            if rule.languages and language and language not in rule.languages:
                continue
            if (
                rule.min_confidence is not None
                and confidence is not None
                and confidence < rule.min_confidence
            ):
                continue
            hits.extend(self._evaluate_rule(rule, matcher, segment, text, ts))
        hits.sort(key=lambda h: h.span)
        return hits

    async def evaluate_async(self, segment: SegmentLike) -> list[RuleHit]:
        """:meth:`evaluate` off the event loop; ``llm`` rules make blocking HTTP calls."""
        import asyncio

        return await asyncio.to_thread(self.evaluate, segment)

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"rules": len(self._matchers), **self.stats.describe()}
        if self.llm is not None:
            info["llm"] = self.llm.describe()
        return info

    # ------------------------------------------------------------------ internals

    def _evaluate_rule(
        self,
        rule: Rule,
        matcher: Matcher | None,
        segment: SegmentLike,
        text: str,
        ts: float,
    ) -> list[RuleHit]:
        matches = self._find(rule, matcher, text, ts)
        rank = rule.severity.rank
        hits: list[RuleHit] = []
        seen_keys: set[str] = set()
        for match in matches:
            key = _match_key(rule, match)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            context = _context(text, match.start, match.end, self.context_chars)
            verdict = self._admit(rule, key, norm_text(context), ts, rank)
            if verdict == "cooldown":
                self.stats.suppressed_cooldown += 1
                continue
            if verdict == "similar":
                self.stats.suppressed_similar += 1
                continue
            self.stats.hits += 1
            if verdict == "upgrade":
                self.stats.upgrades += 1
            hits.append(
                RuleHit(
                    rule_id=rule.id,
                    matched_text=match.text,
                    start=ts,
                    end=float(segment.end),
                    context=context,
                    severity=rule.severity,
                    groups=dict(match.groups),
                    span=(match.start, match.end),
                    key=key,
                    notify=rule.notify,
                    description=rule.description,
                    upgrade=verdict == "upgrade",
                    reason=match.reason,
                )
            )
        return hits

    def _find(self, rule: Rule, matcher: Matcher | None, text: str, ts: float) -> list[Match]:
        if rule.type is not RuleType.LLM:
            return matcher.find(text) if matcher else []
        if self.llm is None:
            self.stats.llm_skipped += 1
            if not self._llm_warned:
                self._llm_warned = True
                log.warning(
                    "llm rules are configured but no endpoint is set; skipping them",
                    extra={"rule_id": rule.id},
                )
            return []
        if len(text.split()) < LLM_MIN_WORDS or self._recently_fired(rule, ts):
            return []  # not worth a network call: too short, or it could not alert anyway
        match = self.llm.judge(rule, text)
        return [match] if match else []

    def _recently_fired(self, rule: Rule, ts: float) -> bool:
        """Would a hit of this rule be suppressed right now? Then skip the LLM call."""
        if rule.cooldown_seconds <= 0:
            return False
        with self._lock:
            prev = self._cooldowns.get((rule.cooldown_group, _LLM_KEY))
        return (
            prev is not None
            and prev.rank >= rule.severity.rank
            and 0 <= ts - prev.ts < rule.cooldown_seconds
        )

    def _admit(self, rule: Rule, key: str, context: str, ts: float, rank: int) -> str:
        """Decide ``fire``, ``upgrade``, ``cooldown`` or ``similar`` and record a fire."""
        if rule.cooldown_seconds <= 0:
            return "fire"
        group = rule.cooldown_group
        window = min(rule.cooldown_seconds, self.fuzzy_window_seconds)
        with self._lock:
            blocker_rank = -1
            reason = ""
            prev = self._cooldowns.get((group, key))
            if prev is not None and 0 <= ts - prev.ts < rule.cooldown_seconds:
                blocker_rank, reason = prev.rank, "cooldown"
            tokens = frozenset(context.split())
            for r in self._recent.get(group, ()):
                # Hits of the same segment are distinct facts, not re-emissions.
                if r.seg == self._seg_no or not 0 <= ts - r.ts < window or r.rank <= blocker_rank:
                    continue
                if self._similar(tokens, context, r):
                    blocker_rank, reason = r.rank, reason or "similar"
            if blocker_rank >= rank:
                return reason
            verdict = "upgrade" if blocker_rank >= 0 else "fire"
            self._record(group, key, context, tokens, ts, rank)
            return verdict

    def _similar(self, tokens: frozenset[str], context: str, other: _Recent) -> bool:
        if tokens and other.tokens:
            jaccard = len(tokens & other.tokens) / len(tokens | other.tokens)
            if jaccard >= self.similarity_threshold:
                return True
        return text_similarity(context, other.text) >= self.similarity_threshold

    def _record(
        self, group: str, key: str, context: str, tokens: frozenset[str], ts: float, rank: int
    ) -> None:
        self._cooldowns[(group, key)] = _Fired(ts, rank)
        self._cooldowns.move_to_end((group, key))
        while len(self._cooldowns) > self._max_state:
            self._cooldowns.popitem(last=False)
        recent = self._recent.setdefault(group, deque(maxlen=self._recent_per_group))
        recent.append(_Recent(self._seg_no, ts, rank, tokens, context))


# ------------------------------------------------------------------------ helpers


def _primary_language(code: str | None) -> str | None:
    if not code:
        return None
    return code.strip().lower().replace("_", "-").split("-")[0] or None


def _match_key(rule: Rule, match: Match) -> str:
    """The identity of an alert: what makes two hits 'the same thing'."""
    if rule.type is RuleType.KEYWORD:
        return norm_text(match.groups.get("keyword", match.text))
    if rule.type is RuleType.REGEX and match.groups:
        return "|".join(f"{k}={norm_text(v)}" for k, v in sorted(match.groups.items()))
    if rule.type is RuleType.LLM:
        # A verdict has no stable identity, so the cooldown covers the rule as a whole.
        return _LLM_KEY
    return norm_text(match.text)[:_KEY_MAX_CHARS]


def _context(text: str, start: int, end: int, width: int) -> str:
    """A window of about ``width`` characters around the match, cut at word borders."""
    if len(text) <= width:
        return text
    pad = max(0, (width - (end - start)) // 2)
    lo, hi = max(0, start - pad), min(len(text), end + pad)
    if lo > 0:
        space = text.find(" ", lo, start)
        lo = space + 1 if space != -1 else lo
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        hi = space if space != -1 else hi
    snippet = text[lo:hi].strip()
    return ("…" if lo > 0 else "") + snippet + ("…" if hi < len(text) else "")
