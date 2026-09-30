"""Loading and validating the rules YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from livestream_transcriber.rules import RulesError, RuleSet, RuleType, Severity

REPO_ROOT = Path(__file__).resolve().parent.parent

MINIMAL = """
version: 1
rules:
  - id: hello
    type: keyword
    keywords: [hello]
"""


def problems(text: str) -> list[str]:
    with pytest.raises(RulesError) as info:
        RuleSet.from_yaml(text)
    return info.value.problems


def test_example_file_loads_and_matches_documented_shape() -> None:
    rules = RuleSet.load(REPO_ROOT / "rules.example.yaml")
    ids = [r.id for r in rules]
    assert "giveaway" in ids and "version-number" in ids
    llm = rules.get("product-announcement")
    assert llm is not None
    assert llm.type is RuleType.LLM and llm.enabled is False
    assert llm not in rules.enabled_rules
    assert rules.llm is not None and rules.llm.api_key_env == "OPENAI_API_KEY"
    giveaway = rules.get("giveaway")
    assert giveaway is not None
    assert giveaway.notify == ("console", "webhook")
    assert giveaway.severity is Severity.WARNING


def test_defaults_apply_to_rules_that_do_not_override_them() -> None:
    rules = RuleSet.from_yaml(
        """
version: 1
defaults: {cooldown_seconds: 15, notify: [console, webhook]}
rules:
  - {id: a, type: keyword, keywords: [x]}
  - {id: b, type: keyword, keywords: [y], cooldown_seconds: 0, notify: telegram}
"""
    )
    a, b = rules.rules
    assert (a.cooldown_seconds, a.notify) == (15.0, ("console", "webhook"))
    assert (b.cooldown_seconds, b.notify) == (0.0, ("telegram",))


def test_minimal_rule_gets_sensible_defaults() -> None:
    (rule,) = RuleSet.from_yaml(MINIMAL).rules
    assert rule.whole_word is True and rule.case_sensitive is False
    assert rule.severity is Severity.INFO
    assert rule.notify == ("console",)
    assert rule.languages == ()


def test_languages_are_reduced_to_primary_subtags() -> None:
    text = MINIMAL + "    languages: [de-DE, EN_us, de]\n"
    (rule,) = RuleSet.from_yaml(text).rules
    assert rule.languages == ("de", "en")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("- just a list", "top level must be a mapping"),
        ("version: 2\nrules: [{id: a, type: keyword, keywords: [x]}]", "unsupported"),
        ("version: 1\nrules: []", "non-empty list"),
        (
            "version: 1\nextra: 1\nrules: [{id: a, type: keyword, keywords: [x]}]",
            "unknown key 'extra'",
        ),
        ("rules: [{type: keyword, keywords: [x]}]", "'id' is required"),
        ("rules: [{id: a, type: fuzzy}]", "'type' must be one of"),
        ("rules: [{id: a, type: keyword}]", "'keywords' must be a non-empty list"),
        ("rules: [{id: a, type: keyword, keywords: ['!!!']}]", "no letters or digits"),
        ("rules: [{id: a, type: keyword, keywords: [x], keywrods: [y]}]", "unknown key 'keywrods'"),
        ("rules: [{id: a, type: regex, pattern: 'a(', }]", "invalid regular expression"),
        ("rules: [{id: a, type: regex, pattern: 'a*'}]", "matches the empty string"),
        ("rules: [{id: a, type: regex, pattern: x, keywords: [y]}]", "unknown key 'keywords'"),
        ("rules: [{id: a, type: llm}]", "'prompt'"),
        ("rules: [{id: a, type: keyword, keywords: [x], severity: urgent}]", "'severity'"),
        ("rules: [{id: a, type: keyword, keywords: [x], notify: [sms]}]", "unknown target 'sms'"),
        ("rules: [{id: a, type: keyword, keywords: [x], cooldown_seconds: -1}]", ">= 0.0"),
        (
            "rules: [{id: a, type: keyword, keywords: [x], cooldown_seconds: fast}]",
            "must be a number",
        ),
        ("rules: [{id: a, type: keyword, keywords: [x], min_confidence: 2}]", "between"),
        ("rules: [{id: a, type: keyword, keywords: [x], whole_word: maybe}]", "true or false"),
        ("rules: [{id: 'has space', type: keyword, keywords: [x]}]", "'id' is required"),
        (
            "rules: [{id: a, type: keyword, keywords: [x]}, {id: a, type: keyword, keywords: [y]}]",
            "duplicate id 'a'",
        ),
        ("llm: {base_url: ftp://x}\nrules: [{id: a, type: keyword, keywords: [x]}]", "http"),
    ],
)
def test_invalid_files_are_rejected_with_a_specific_message(text: str, expected: str) -> None:
    assert any(expected in p for p in problems(text)), problems(text)


def test_all_problems_are_reported_at_once() -> None:
    found = problems(
        """
rules:
  - {id: a, type: regex, pattern: 'a('}
  - {id: b, type: keyword}
  - {id: c, type: keyword, keywords: [x], severity: nope}
"""
    )
    assert len(found) == 3
    assert any("rules[0] (a)" in p for p in found)
    assert any("rules[1] (b)" in p for p in found)
    assert any("rules[2] (c)" in p for p in found)


def test_yaml_syntax_error_is_a_rules_error() -> None:
    assert "not valid YAML" in problems("rules: [unclosed")[0]


def test_missing_file_is_a_rules_error(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="cannot read file"):
        RuleSet.load(tmp_path / "nope.yaml")


def test_error_message_names_the_source_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("rules: []\n", encoding="utf-8")
    with pytest.raises(RulesError, match=r"bad\.yaml"):
        RuleSet.load(path)
