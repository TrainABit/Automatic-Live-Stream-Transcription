"""Cooldown, escalation, fuzzy dedup and bounded state of the rule engine."""

from __future__ import annotations

from typing import Any

from livestream_transcriber.rules import (
    LlmMatcher,
    RuleEngine,
    RuleSet,
    TextSegment,
    text_similarity,
)


def ruleset(body: str) -> RuleSet:
    return RuleSet.from_yaml("version: 1\nrules:\n" + body)


GIVEAWAY = """
  - id: giveaway
    type: keyword
    keywords: [giveaway, "free copy"]
    severity: warning
    cooldown_seconds: 60
"""


def seg(text: str, start: float = 0.0, **kw: Any) -> TextSegment:
    return TextSegment(text, start=start, end=start + 2.0, **kw)


def ids(hits: list[Any]) -> list[str]:
    return [h.rule_id for h in hits]


class TestBasics:
    def test_hit_carries_the_fields_notifiers_need(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        (hit,) = engine.evaluate(seg("Today we run a Giveaway for you", start=12.5))
        assert hit.rule_id == "giveaway"
        assert hit.matched_text == "Giveaway"
        assert hit.severity.value == "warning"
        assert hit.context == "Today we run a Giveaway for you"
        assert (hit.start, hit.end) == (12.5, 14.5)
        assert hit.groups == {"keyword": "giveaway"}
        assert hit.notify == ("console",)
        assert "Today we run a Giveaway for you"[slice(*hit.span)] == "Giveaway"

    def test_empty_and_unmatched_text_yield_nothing(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        assert engine.evaluate(seg("")) == []
        assert engine.evaluate(seg("   ")) == []
        assert engine.evaluate(seg("nothing to see")) == []

    def test_regex_groups_reach_the_hit(self) -> None:
        engine = RuleEngine(
            ruleset(
                r"""
  - id: version
    type: regex
    pattern: 'v(?P<major>\d+)\.(?P<minor>\d+)'
"""
            )
        )
        (hit,) = engine.evaluate(seg("we ship v3.14 today"))
        assert hit.groups == {"major": "3", "minor": "14"}

    def test_hits_are_returned_in_text_order(self) -> None:
        engine = RuleEngine(
            ruleset(
                """
  - {id: late, type: keyword, keywords: [zebra], severity: critical}
  - {id: early, type: keyword, keywords: [apple]}
"""
            )
        )
        assert ids(engine.evaluate(seg("apple then zebra"))) == ["early", "late"]

    def test_language_filter_uses_the_primary_subtag_and_lets_unknown_pass(self) -> None:
        engine = RuleEngine(
            ruleset("  - {id: de-only, type: keyword, keywords: [hallo], languages: [de]}")
        )
        assert engine.evaluate(seg("hallo", language="en")) == []
        assert len(engine.evaluate(seg("hallo", start=100, language="de-DE"))) == 1
        assert len(engine.evaluate(seg("hallo", start=200, language=None))) == 1

    def test_min_confidence_filters_uncertain_segments(self) -> None:
        engine = RuleEngine(
            ruleset("  - {id: r, type: keyword, keywords: [go], min_confidence: 0.6}")
        )
        assert engine.evaluate(seg("go", confidence=0.3)) == []
        assert len(engine.evaluate(seg("go", start=100, confidence=0.9))) == 1
        assert len(engine.evaluate(seg("go", start=200, confidence=None))) == 1

    def test_disabled_rules_do_not_run(self) -> None:
        engine = RuleEngine(ruleset("  - {id: r, type: keyword, keywords: [go], enabled: false}"))
        assert engine.evaluate(seg("go")) == []

    def test_long_text_gets_a_trimmed_context_around_the_match(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY), context_chars=60)
        text = "blah " * 40 + "giveaway " + "more " * 40
        (hit,) = engine.evaluate(seg(text))
        assert len(hit.context) <= 70
        assert "giveaway" in hit.context
        assert hit.context.startswith("…") and hit.context.endswith("…")


class TestCooldown:
    def test_same_key_is_silent_until_the_cooldown_passes(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        assert len(engine.evaluate(seg("a giveaway", start=0))) == 1
        assert engine.evaluate(seg("another giveaway now", start=30)) == []
        assert engine.evaluate(seg("giveaway once more", start=59.9)) == []
        assert len(engine.evaluate(seg("yes, a giveaway", start=60))) == 1
        assert engine.stats.suppressed_cooldown == 2

    def test_key_is_the_normalised_keyword_not_the_spoken_form(self) -> None:
        engine = RuleEngine(
            ruleset(
                """
  - {id: r, type: keyword, keywords: [giveaway], whole_word: false, cooldown_seconds: 60}
"""
            )
        )
        assert len(engine.evaluate(seg("giveaway", start=0))) == 1
        assert engine.evaluate(seg("giveaways galore", start=10)) == []

    def test_different_keys_of_one_rule_are_independent(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        assert len(engine.evaluate(seg("a giveaway", start=0))) == 1
        assert len(engine.evaluate(seg("win a free copy", start=5))) == 1

    def test_regex_keys_follow_named_groups(self) -> None:
        engine = RuleEngine(
            ruleset(
                r"""
  - id: version
    type: regex
    pattern: 'v(?P<major>\d+)\.(?P<minor>\d+)'
    cooldown_seconds: 120
"""
            )
        )
        assert len(engine.evaluate(seg("v1.2 is out", start=0))) == 1
        assert engine.evaluate(seg("still talking about v1.2 here", start=10)) == []
        assert len(engine.evaluate(seg("but v2.0 is different", start=20))) == 1

    def test_two_facts_in_one_segment_both_fire(self) -> None:
        engine = RuleEngine(
            ruleset(
                r"""
  - {id: version, type: regex, pattern: 'v(?P<n>\d+)'}
"""
            )
        )
        hits = engine.evaluate(seg("v1 and v2 are both out"))
        assert [h.groups["n"] for h in hits] == ["1", "2"]

    def test_the_same_key_twice_in_one_segment_is_one_hit(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        assert len(engine.evaluate(seg("giveaway giveaway giveaway"))) == 1

    def test_zero_cooldown_fires_every_time(self) -> None:
        engine = RuleEngine(
            ruleset("  - {id: r, type: keyword, keywords: [go], cooldown_seconds: 0}")
        )
        assert all(len(engine.evaluate(seg("go", start=i))) == 1 for i in range(5))

    def test_cooldown_runs_on_stream_time_not_wall_clock(self) -> None:
        # Evaluating instantly, but with segment times 100 s apart, must fire twice.
        engine = RuleEngine(ruleset(GIVEAWAY))
        assert len(engine.evaluate(seg("giveaway", start=0))) == 1
        assert len(engine.evaluate(seg("giveaway", start=100))) == 1

    def test_a_restarted_stream_clock_does_not_freeze_old_entries(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        assert len(engine.evaluate(seg("giveaway", start=5000))) == 1
        assert len(engine.evaluate(seg("giveaway", start=1))) == 1

    def test_replaying_the_same_segments_gives_the_same_hits(self) -> None:
        segments = [seg("giveaway", start=t) for t in (0, 10, 70, 75, 200)]
        runs = []
        for _ in range(2):
            engine = RuleEngine(ruleset(GIVEAWAY))
            runs.append([len(engine.evaluate(s)) for s in segments])
        assert runs[0] == runs[1] == [1, 0, 1, 0, 1]


ESCALATION = """
  - {id: mention, type: keyword, keywords: [outage], severity: info, group: incident}
  - id: incident
    type: keyword
    keywords: [outage, "service down"]
    severity: critical
    group: incident
"""


class TestEscalation:
    def test_higher_severity_bypasses_a_running_cooldown(self) -> None:
        # Only the info rule sees the first segment; the critical one comes later.
        engine = RuleEngine(
            ruleset(
                """
  - {id: mention, type: keyword, keywords: [outage], severity: info, group: incident}
  - {id: incident, type: keyword, keywords: [outage], severity: critical, group: incident,
     languages: [de]}
"""
            )
        )
        (first,) = engine.evaluate(seg("an outage", start=0, language="en"))
        assert (first.rule_id, first.upgrade) == ("mention", False)
        (second,) = engine.evaluate(seg("an outage", start=5, language="de"))
        assert (second.rule_id, second.upgrade) == ("incident", True)
        assert engine.stats.upgrades == 1

    def test_lower_or_equal_severity_stays_suppressed_after_the_upgrade(self) -> None:
        engine = RuleEngine(ruleset(ESCALATION))
        engine.evaluate(seg("outage", start=0))
        assert engine.evaluate(seg("outage again", start=10)) == []

    def test_the_stronger_rule_wins_when_both_match_one_segment(self) -> None:
        engine = RuleEngine(ruleset(ESCALATION))
        assert ids(engine.evaluate(seg("an outage"))) == ["incident"]

    def test_escalation_changes_the_event_identity_downstream(self) -> None:
        from livestream_transcriber.notify import Event

        engine = RuleEngine(
            ruleset(
                """
  - {id: a, type: keyword, keywords: [x1], severity: info, group: g}
  - {id: b, type: keyword, keywords: [x1], severity: warning, group: g, languages: [de]}
"""
            )
        )
        (low,) = engine.evaluate(seg("x1", start=0, language="en"))
        (high,) = engine.evaluate(seg("x1", start=1, language="de"))
        assert Event.from_hit(low).event_id != Event.from_hit(high).event_id


class TestFuzzyDedup:
    def test_two_keywords_in_one_segment_are_two_facts(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        hits = engine.evaluate(seg("we are running a giveaway and a free copy today folks", 0))
        assert [h.groups["keyword"] for h in hits] == ["giveaway", "free copy"]
        assert engine.stats.suppressed_similar == 0

    def test_overlapping_reemission_with_an_extra_word_is_suppressed(self) -> None:
        engine = RuleEngine(
            ruleset(
                """
  - {id: r, type: keyword, keywords: [giveaway, gewinnspiel], cooldown_seconds: 300}
"""
            )
        )
        assert len(engine.evaluate(seg("the big giveaway starts right now everyone", 0))) == 1
        # Same words from an overlapping chunk, but the matched keyword differs.
        assert engine.evaluate(seg("the big gewinnspiel starts right now everyone", 4)) == []
        assert engine.stats.suppressed_similar == 1

    def test_dissimilar_context_is_not_suppressed(self) -> None:
        engine = RuleEngine(
            ruleset("  - {id: r, type: keyword, keywords: [alpha, beta], cooldown_seconds: 300}")
        )
        assert len(engine.evaluate(seg("alpha is the first letter of the greek alphabet", 0))) == 1
        assert len(engine.evaluate(seg("completely different sentence about beta testing", 5))) == 1

    def test_fuzzy_window_limits_how_far_back_similarity_reaches(self) -> None:
        engine = RuleEngine(
            ruleset("  - {id: r, type: keyword, keywords: [alpha, beta], cooldown_seconds: 300}"),
            fuzzy_window_seconds=10,
        )
        assert len(engine.evaluate(seg("say alpha to the whole wide world now", 0))) == 1
        assert len(engine.evaluate(seg("say beta to the whole wide world now", 60))) == 1

    def test_similarity_measure(self) -> None:
        assert text_similarity("a b c", "a b c") == 1.0
        assert text_similarity("", "a") == 0.0
        assert text_similarity("a b c d e f", "f e d c b a") == 1.0  # token sets
        assert text_similarity("new release is out today", "new release is out today folks") > 0.85
        assert text_similarity("apples and pears", "quarterly earnings call") < 0.5


class TestBoundedState:
    def test_cooldown_table_never_exceeds_max_state(self) -> None:
        engine = RuleEngine(
            ruleset(r"  - {id: v, type: regex, pattern: 'id(?P<n>\d+)', cooldown_seconds: 9999}"),
            max_state=50,
        )
        for i in range(500):
            engine.evaluate(seg(f"id{i}", start=float(i)))
        assert len(engine._cooldowns) == 50
        assert len(engine._recent["v"]) <= 32

    def test_evicted_keys_can_fire_again(self) -> None:
        engine = RuleEngine(
            ruleset(r"  - {id: v, type: regex, pattern: 'id(?P<n>\d+)', cooldown_seconds: 9999}"),
            max_state=3,
        )
        for i in range(10):
            engine.evaluate(seg(f"id{i}", start=float(i)))
        # Far outside the fuzzy window, only the cooldown table decides.
        assert len(engine.evaluate(seg("id0", start=1000))) == 1  # long evicted
        assert engine.evaluate(seg("id9", start=1001)) == []  # still remembered


LLM_RULES = """
  - {id: announce, type: llm, prompt: "Is a product announced?", cooldown_seconds: 120}
"""


class FakeLlm:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls = 0

    def judge(self, rule: Any, text: str) -> Any:
        from livestream_transcriber.rules import Match

        self.calls += 1
        return Match(0, len(text), text, {"reason": "r"}, reason="r") if self.answer else None

    def describe(self) -> dict[str, Any]:
        return {}


class TestLlmRules:
    def test_skipped_without_an_endpoint(self) -> None:
        engine = RuleEngine(ruleset(LLM_RULES))
        assert engine.evaluate(seg("we announce a brand new product today")) == []
        assert engine.stats.llm_skipped == 1

    def test_hit_carries_the_reason(self) -> None:
        llm = FakeLlm()
        engine = RuleEngine(ruleset(LLM_RULES), llm=llm)  # type: ignore[arg-type]
        (hit,) = engine.evaluate(seg("we announce a brand new product today"))
        assert hit.reason == "r" and hit.groups == {"reason": "r"}

    def test_cooldown_saves_the_network_call(self) -> None:
        llm = FakeLlm()
        engine = RuleEngine(ruleset(LLM_RULES), llm=llm)  # type: ignore[arg-type]
        engine.evaluate(seg("we announce a brand new product today", 0))
        engine.evaluate(seg("here is another announcement of a thing", 30))
        assert llm.calls == 1
        engine.evaluate(seg("and now a third announcement of stuff", 200))
        assert llm.calls == 2

    def test_very_short_segments_are_not_sent(self) -> None:
        llm = FakeLlm()
        engine = RuleEngine(ruleset(LLM_RULES), llm=llm)  # type: ignore[arg-type]
        engine.evaluate(seg("uh okay"))
        assert llm.calls == 0

    def test_a_failing_endpoint_never_raises(self) -> None:
        def broken(*a: Any, **k: Any) -> dict[str, Any]:
            raise OSError("down")

        llm = LlmMatcher(base_url="http://llm.test", model="m", post=broken)
        engine = RuleEngine(ruleset(LLM_RULES), llm=llm)
        assert engine.evaluate(seg("we announce a brand new product today")) == []

    async def test_async_evaluation_matches_sync(self) -> None:
        engine = RuleEngine(ruleset(GIVEAWAY))
        hits = await engine.evaluate_async(seg("a giveaway"))
        assert ids(hits) == ["giveaway"]
