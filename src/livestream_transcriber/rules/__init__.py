"""Keyword, regex and LLM rules evaluated against transcripts."""

from .engine import RuleEngine, RuleHit, RuleStats, SegmentLike, TextSegment, text_similarity
from .matchers import (
    KeywordMatcher,
    LlmMatcher,
    Match,
    RegexMatcher,
    build_llm_matcher,
    build_matcher,
)
from .model import (
    NOTIFY_TARGETS,
    LlmConfig,
    Rule,
    RulesError,
    RuleSet,
    RuleType,
    Severity,
)

__all__ = [
    "NOTIFY_TARGETS",
    "KeywordMatcher",
    "LlmConfig",
    "LlmMatcher",
    "Match",
    "RegexMatcher",
    "Rule",
    "RuleEngine",
    "RuleHit",
    "RuleSet",
    "RuleStats",
    "RuleType",
    "RulesError",
    "SegmentLike",
    "Severity",
    "TextSegment",
    "build_llm_matcher",
    "build_matcher",
    "text_similarity",
]
