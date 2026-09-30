"""Rule definitions and their YAML schema.

A rules file is plain YAML so that non-programmers can edit it and a diff
tells the whole story::

    version: 1
    defaults:
      cooldown_seconds: 60
      notify: [console]
    rules:
      - id: giveaway
        type: keyword
        keywords: [giveaway, "free copy"]
        severity: warning
        notify: [console, webhook]

Loading is strict on purpose. A misspelt key or an invalid regex would
otherwise turn into a rule that silently never fires, which is the worst
failure mode for an alerting tool. Every problem in the file is collected and
reported at once, each with the path of the offending entry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "NOTIFY_TARGETS",
    "SCHEMA_VERSION",
    "LlmConfig",
    "Rule",
    "RuleSet",
    "RuleType",
    "RulesError",
    "Severity",
]

SCHEMA_VERSION = 1
NOTIFY_TARGETS: tuple[str, ...] = ("console", "webhook", "telegram")
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_TARGETS: tuple[str, ...] = ("console",)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_LEVEL_KEYS = {"version", "defaults", "llm", "rules"}
_DEFAULT_KEYS = {"cooldown_seconds", "notify"}
_LLM_KEYS = {"base_url", "model", "api_key_env", "timeout_seconds"}
_COMMON_KEYS = {
    "id",
    "type",
    "description",
    "case_sensitive",
    "languages",
    "cooldown_seconds",
    "min_confidence",
    "notify",
    "severity",
    "enabled",
    "group",
}
_TYPE_KEYS: dict[str, set[str]] = {
    "keyword": {"keywords", "whole_word"},
    "regex": {"pattern"},
    "llm": {"prompt"},
}


class RulesError(ValueError):
    """The rules file is invalid. ``problems`` lists every issue found."""

    def __init__(self, problems: list[str], *, source: str | None = None) -> None:
        self.problems = list(problems)
        self.source = source
        head = f"invalid rules file {source}" if source else "invalid rules"
        super().__init__(head + ":\n" + "\n".join(f"  - {p}" for p in self.problems))


class RuleType(StrEnum):
    KEYWORD = "keyword"
    REGEX = "regex"
    LLM = "llm"


class Severity(StrEnum):
    """How urgent a hit is. Ranked so that an escalation can be detected."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Optional ``llm:`` block. Settings from the environment take precedence."""

    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    """Name of the environment variable holding the API key (never the key itself)."""
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Rule:
    """One matching rule. Exactly one of ``keywords``, ``pattern`` or ``prompt`` applies."""

    id: str
    type: RuleType
    keywords: tuple[str, ...] = ()
    pattern: str | None = None
    prompt: str | None = None
    description: str = ""
    case_sensitive: bool = False
    whole_word: bool = True
    languages: tuple[str, ...] = ()
    """Primary language subtags this rule applies to; empty means every language."""
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    min_confidence: float | None = None
    notify: tuple[str, ...] = DEFAULT_TARGETS
    severity: Severity = Severity.INFO
    enabled: bool = True
    group: str | None = None
    """Rules sharing a group share cooldown state, so a stronger rule can escalate."""

    @property
    def cooldown_group(self) -> str:
        return self.group or self.id


@dataclass(frozen=True, slots=True)
class RuleSet:
    """A validated collection of rules."""

    rules: tuple[Rule, ...]
    version: int = SCHEMA_VERSION
    default_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    default_notify: tuple[str, ...] = DEFAULT_TARGETS
    llm: LlmConfig | None = None
    source: str | None = field(default=None, compare=False)

    def __iter__(self) -> Any:
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def enabled_rules(self) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if r.enabled)

    def get(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | Path) -> RuleSet:
        """Read and validate a YAML file; raise :class:`RulesError` on any problem."""
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise RulesError([f"cannot read file: {exc.strerror or exc}"], source=str(p)) from exc
        return cls.from_yaml(text, source=str(p))

    @classmethod
    def from_yaml(cls, text: str, *, source: str | None = None) -> RuleSet:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RulesError([f"not valid YAML: {exc}"], source=source) from exc
        return cls.from_mapping(data, source=source)

    @classmethod
    def from_mapping(cls, data: Any, *, source: str | None = None) -> RuleSet:
        problems: list[str] = []
        ruleset = _parse(data, problems)
        if problems or ruleset is None:
            raise RulesError(problems or ["empty rules file"], source=source)
        return RuleSet(
            rules=ruleset.rules,
            version=ruleset.version,
            default_cooldown_seconds=ruleset.default_cooldown_seconds,
            default_notify=ruleset.default_notify,
            llm=ruleset.llm,
            source=source,
        )


# ------------------------------------------------------------------------ parsing


def _parse(data: Any, problems: list[str]) -> RuleSet | None:
    if not isinstance(data, Mapping):
        problems.append("the top level must be a mapping with a 'rules' list")
        return None
    _reject_unknown(data, _TOP_LEVEL_KEYS, "top level", problems)

    version = data.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION or isinstance(version, bool):
        problems.append(
            f"version: unsupported {version!r}; this build reads version {SCHEMA_VERSION}"
        )

    cooldown, notify = _parse_defaults(data.get("defaults"), problems)
    llm = _parse_llm(data.get("llm"), problems)

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        problems.append("rules: must be a non-empty list")
        return None

    rules: list[Rule] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        rule = _parse_rule(raw, index, cooldown, notify, problems)
        if rule is None:
            continue
        if rule.id in seen:
            problems.append(f"rules[{index}]: duplicate id {rule.id!r}")
            continue
        seen.add(rule.id)
        rules.append(rule)

    return RuleSet(
        rules=tuple(rules),
        version=SCHEMA_VERSION,
        default_cooldown_seconds=cooldown,
        default_notify=notify,
        llm=llm,
    )


def _parse_defaults(raw: Any, problems: list[str]) -> tuple[float, tuple[str, ...]]:
    cooldown, notify = DEFAULT_COOLDOWN_SECONDS, DEFAULT_TARGETS
    if raw is None:
        return cooldown, notify
    if not isinstance(raw, Mapping):
        problems.append("defaults: must be a mapping")
        return cooldown, notify
    _reject_unknown(raw, _DEFAULT_KEYS, "defaults", problems)
    if "cooldown_seconds" in raw:
        cooldown = (
            _number(raw["cooldown_seconds"], "defaults.cooldown_seconds", problems, minimum=0.0)
            or 0.0
        )
    if "notify" in raw:
        notify = _targets(raw["notify"], "defaults.notify", problems) or DEFAULT_TARGETS
    return cooldown, notify


def _parse_llm(raw: Any, problems: list[str]) -> LlmConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        problems.append("llm: must be a mapping")
        return None
    _reject_unknown(raw, _LLM_KEYS, "llm", problems)
    values: dict[str, str | None] = {}
    for key in ("base_url", "model", "api_key_env"):
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            problems.append(f"llm.{key}: must be a non-empty string")
            value = None
        values[key] = value.strip() if isinstance(value, str) else None
    base_url = values["base_url"]
    if base_url and not base_url.lower().startswith(("http://", "https://")):
        problems.append("llm.base_url: must start with http:// or https://")
    timeout = None
    if "timeout_seconds" in raw:
        timeout = _number(raw["timeout_seconds"], "llm.timeout_seconds", problems, minimum=0.1)
    return LlmConfig(
        base_url=base_url,
        model=values["model"],
        api_key_env=values["api_key_env"],
        timeout_seconds=timeout,
    )


def _parse_rule(
    raw: Any,
    index: int,
    default_cooldown: float,
    default_notify: tuple[str, ...],
    problems: list[str],
) -> Rule | None:
    where = f"rules[{index}]"
    if not isinstance(raw, Mapping):
        problems.append(f"{where}: must be a mapping")
        return None

    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not _ID_PATTERN.match(rule_id):
        problems.append(f"{where}: 'id' is required (letters, digits, '.', '_' or '-')")
        return None
    where = f"rules[{index}] ({rule_id})"
    before = len(problems)

    try:
        rule_type = RuleType(str(raw.get("type", "")).lower())
    except ValueError:
        problems.append(f"{where}: 'type' must be one of keyword, regex, llm")
        return None
    _reject_unknown(raw, _COMMON_KEYS | _TYPE_KEYS[rule_type.value], where, problems)

    keywords: tuple[str, ...] = ()
    pattern: str | None = None
    prompt: str | None = None
    if rule_type is RuleType.KEYWORD:
        keywords = _keywords(raw.get("keywords"), where, problems)
    elif rule_type is RuleType.REGEX:
        pattern = _pattern(raw.get("pattern"), where, problems)
    else:
        prompt = _text(raw.get("prompt"), f"{where}: 'prompt'", problems)

    severity = Severity.INFO
    if "severity" in raw:
        try:
            severity = Severity(str(raw["severity"]).lower())
        except ValueError:
            problems.append(f"{where}: 'severity' must be one of info, warning, critical")

    cooldown = default_cooldown
    if "cooldown_seconds" in raw:
        parsed = _number(
            raw["cooldown_seconds"], f"{where}: 'cooldown_seconds'", problems, minimum=0.0
        )
        cooldown = default_cooldown if parsed is None else parsed
    min_conf = None
    if raw.get("min_confidence") is not None:
        min_conf = _number(
            raw["min_confidence"], f"{where}: 'min_confidence'", problems, minimum=0.0, maximum=1.0
        )
    notify = default_notify
    if "notify" in raw:
        notify = _targets(raw["notify"], f"{where}: 'notify'", problems) or default_notify

    flags = {}
    for key in ("case_sensitive", "whole_word", "enabled"):
        if key in raw:
            if not isinstance(raw[key], bool):
                problems.append(f"{where}: {key!r} must be true or false")
            else:
                flags[key] = raw[key]

    languages = _languages(raw.get("languages"), where, problems)
    group = raw.get("group")
    if group is not None and (not isinstance(group, str) or not _ID_PATTERN.match(group)):
        problems.append(f"{where}: 'group' must be a short identifier")
        group = None
    description = raw.get("description", "")
    if not isinstance(description, str):
        problems.append(f"{where}: 'description' must be a string")
        description = ""

    if len(problems) > before:
        return None
    return Rule(
        id=rule_id,
        type=rule_type,
        keywords=keywords,
        pattern=pattern,
        prompt=prompt,
        description=description.strip(),
        case_sensitive=flags.get("case_sensitive", False),
        whole_word=flags.get("whole_word", True),
        languages=languages,
        cooldown_seconds=float(cooldown),
        min_confidence=min_conf,
        notify=notify,
        severity=severity,
        enabled=flags.get("enabled", True),
        group=group,
    )


# ----------------------------------------------------------------- field helpers


def _reject_unknown(
    data: Mapping[Any, Any], allowed: set[str], where: str, problems: list[str]
) -> None:
    for key in data:
        if key not in allowed:
            problems.append(f"{where}: unknown key {key!r} (allowed: {', '.join(sorted(allowed))})")


def _number(
    value: Any,
    label: str,
    problems: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        problems.append(f"{label} must be a number")
        return None
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        problems.append(f"{label} must be {bounds}")
        return None
    return float(value)


def _text(value: Any, label: str, problems: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{label} is required and must be a non-empty string")
        return None
    return value.strip()


def _keywords(value: Any, where: str, problems: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        problems.append(f"{where}: 'keywords' must be a non-empty list of strings")
        return ()
    out: list[str] = []
    for item in value:
        if not isinstance(item, str | int | float) or isinstance(item, bool):
            problems.append(f"{where}: keyword {item!r} must be a string")
        elif not re.search(r"\w", str(item)):
            problems.append(f"{where}: keyword {item!r} has no letters or digits")
        else:
            out.append(str(item).strip())
    return tuple(out)


def _pattern(value: Any, where: str, problems: list[str]) -> str | None:
    text = _text(value, f"{where}: 'pattern'", problems)
    if text is None:
        return None
    try:
        compiled = re.compile(text)
    except re.error as exc:
        problems.append(f"{where}: invalid regular expression: {exc}")
        return None
    if compiled.match(""):
        problems.append(f"{where}: pattern matches the empty string and would fire on everything")
        return None
    return text


def _targets(value: Any, label: str, problems: list[str]) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list):
        problems.append(f"{label} must be a list of {', '.join(NOTIFY_TARGETS)}")
        return ()
    out: list[str] = []
    for item in items:
        name = str(item).strip().lower()
        if name not in NOTIFY_TARGETS:
            problems.append(
                f"{label}: unknown target {item!r} (choose from {', '.join(NOTIFY_TARGETS)})"
            )
        elif name not in out:
            out.append(name)
    return tuple(out)


def _languages(value: Any, where: str, problems: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(i, str) and i.strip() for i in items):
        problems.append(f"{where}: 'languages' must be a list of language codes such as [en, de]")
        return ()
    return tuple(dict.fromkeys(i.strip().lower().replace("_", "-").split("-")[0] for i in items))
