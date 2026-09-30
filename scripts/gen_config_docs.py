#!/usr/bin/env python3
"""Regenerate the configuration reference from ``config.py``.

Writes ``docs/configuration.md`` (a table of every ``LST_*`` setting) and
``.env.example`` (every key, commented out, with placeholders for secrets), so neither
can drift from the code. The description of a setting is the docstring under its field.

    python scripts/gen_config_docs.py            # rewrite both files
    python scripts/gen_config_docs.py --check    # exit 1 when they are out of date
"""

from __future__ import annotations

import argparse
import ast
import itertools
import sys
from pathlib import Path
from typing import Any

from pydantic import AliasChoices
from pydantic_core import PydanticUndefined

from livestream_transcriber.config import Settings

ROOT = Path(__file__).resolve().parent.parent
CONFIG_SOURCE = ROOT / "src" / "livestream_transcriber" / "config.py"

#: Section headings, by the prefix of the field name.
SECTIONS: list[tuple[str, str]] = [
    ("capture_", "Capture"),
    ("stt_", "Speech to text"),
    ("openai_", "Speech to text: cloud providers"),
    ("openrouter_", "Speech to text: cloud providers"),
    ("resume_", "Resuming after a stream ends"),
    ("notify_", "Notifications"),
    ("rules_", "Rules"),
]
FALLBACK_SECTION = "Paths and process"

SECRET_PLACEHOLDERS = {
    "openai_api_key": "your-openai-key",
    "openrouter_api_key": "your-openrouter-key",
    "notify_webhook_url": "https://example.com/your-webhook-path",
    "notify_webhook_token": "your-webhook-token",
    "notify_telegram_token": "123456:your-telegram-bot-token",
    "notify_telegram_chat_id": "your-chat-id",
    "capture_proxy": "http://proxy.example.com:3128",
    "rules_llm_api_key": "your-llm-key",
}


#: Descriptions for settings whose field has no docstring of its own.
EXTRA_DOCS: dict[str, str] = {
    "capture_reconnect_initial_delay": "First reconnect delay in seconds; grows with full jitter.",
    "capture_reconnect_max_delay": "Upper bound of the reconnect delay in seconds.",
    "capture_resolve_cache_seconds": "How long a resolved media URL is reused.",
    "capture_stale_playlist_stalls": "Audio stalls on a frozen HLS playlist before reselecting.",
    "capture_ffmpeg_binary": "ffmpeg executable to use.",
    "capture_ffmpeg_loglevel": "ffmpeg's own log level.",
    "stt_provider": "Speech-to-text provider: local, onnx, openai, openrouter, mock or none.",
    "stt_device": "Device for local inference (cpu or cuda).",
    "stt_compute_type": "Quantisation of the local model (int8, float16, ...).",
    "stt_threads": "CPU threads used by local inference; raise it on a multi-core machine.",
    "stt_word_timestamps": "Ask the provider for word-level timestamps where it supports them.",
    "stt_vad_filter": "Let the local model skip non-speech with its voice activity filter.",
    "stt_workers": "Concurrent transcription requests.",
    "stt_timeout_seconds": "Per-request timeout of a provider call.",
    "stt_breaker_failures": "Consecutive failures that open a provider's circuit breaker.",
    "stt_breaker_cooldown_seconds": "How long an open breaker waits before trying one request.",
    "stt_max_drop_ratio": "Overload guard: pause STT when this share of chunks was dropped.",
    "stt_overload_window_seconds": "Sliding window the overload guard looks at.",
    "stt_recovery_probe_seconds": "Interval of recovery probes while STT is paused.",
    "stt_health_lag_seconds": "Lag above which the heartbeat reports the run as degraded.",
    "stt_health_drop_ratio": "Drop ratio above which the heartbeat reports the run as degraded.",
    "openai_api_key": "OpenAI (or compatible) API key. OPENAI_API_KEY is accepted too.",
    "openai_base_url": "Base URL of the transcription API. OPENAI_BASE_URL is accepted too.",
    "openrouter_api_key": "OpenRouter API key. OPENROUTER_API_KEY is accepted too.",
    "openrouter_base_url": "Base URL of the OpenRouter API.",
    "resume_auto": "Keep probing after a stream ends or is offline, and resume when it returns.",
    "resume_probe_interval_seconds": "Interval between liveness probes while waiting.",
    "resume_backoff_max_seconds": "Upper bound of the probe interval when probes are inconclusive.",
    "notify_webhook_format": "Payload shape: json, slack or discord.",
    "notify_telegram_token": "Telegram bot token; needs the chat id as well.",
    "notify_telegram_chat_id": "Chat that receives Telegram alerts.",
    "notify_timeout_seconds": "Timeout of one notification request.",
    "rules_file": "Rules file used when --rules is not given.",
    "rules_llm_model": "Model for llm rules.",
    "rules_llm_api_key": "API key for llm rules.",
    "rules_llm_timeout_seconds": "Timeout of one llm rule request.",
    "out_dir": "Directory for transcript files (jsonl, srt, vtt) and the default database.",
    "recordings_dir": "Where lst record stores recordings.",
    "log_level": "DEBUG, INFO, WARNING or ERROR.",
}


def field_docs() -> dict[str, str]:
    """Field name -> the docstring that follows it in ``Settings``."""
    tree = ast.parse(CONFIG_SOURCE.read_text(encoding="utf-8"))
    docs: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "Settings"):
            continue
        body = node.body
        for current, following in itertools.pairwise(body):
            if (
                isinstance(current, ast.AnnAssign)
                and isinstance(current.target, ast.Name)
                and isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                docs[current.target.id] = " ".join(following.value.value.split())
    merged = {**EXTRA_DOCS, **docs}
    return {k: v.replace("``", "`") for k, v in merged.items()}


def env_names(name: str, info: Any) -> list[str]:
    alias = info.validation_alias
    if isinstance(alias, AliasChoices):
        return [str(choice) for choice in alias.choices]
    return [f"LST_{name.upper()}"]


def default_text(info: Any) -> str:
    default = info.default
    if default is PydanticUndefined or default is None:
        return "unset"
    if isinstance(default, bool):
        return "true" if default else "false"
    if isinstance(default, Path):
        return str(default).replace(str(Path.home()), "~")
    return str(default)


def is_secret(info: Any) -> bool:
    return "SecretStr" in str(info.annotation)


def section_of(name: str) -> str:
    for prefix, title in SECTIONS:
        if name.startswith(prefix):
            return title
    return FALLBACK_SECTION


def grouped() -> dict[str, list[tuple[str, Any]]]:
    groups: dict[str, list[tuple[str, Any]]] = {}
    for name, info in Settings.model_fields.items():
        groups.setdefault(section_of(name), []).append((name, info))
    return groups


def render_markdown(docs: dict[str, str]) -> str:
    lines = [
        "# Configuration reference",
        "",
        "Every setting is read from the environment (or a `.env` file) with the prefix `LST_`.",
        "Precedence, highest first: command line flags, process environment, `.env`, defaults.",
        "An invalid value is a startup error that names the variable, never its value.",
        "",
        "This file is generated by `scripts/gen_config_docs.py` from `config.py`.",
        "",
    ]
    for title, fields in grouped().items():
        lines += [f"## {title}", "", "| Variable | Default | Description |", "|---|---|---|"]
        for name, info in fields:
            names = " / ".join(f"`{n}`" for n in env_names(name, info))
            default = "*(secret)*" if is_secret(info) else f"`{default_text(info)}`"
            lines.append(f"| {names} | {default} | {docs.get(name, '').replace('|', '/')} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_env_example(docs: dict[str, str]) -> str:
    lines = [
        "# Copy to .env and uncomment what you need. Every value here is a placeholder.",
        "# Reference: docs/configuration.md (generated from src/livestream_transcriber/config.py).",
        "# Never commit a real .env file.",
        "",
        "# Read by docker-compose.yml only (not a setting of lst):",
        "# STREAM_URL=https://example.com/live.m3u8",
        "",
    ]
    for title, fields in grouped().items():
        lines += [f"# --- {title} " + "-" * max(3, 60 - len(title)), ""]
        for name, info in fields:
            primary = env_names(name, info)[0]
            if docs.get(name):
                lines.append(f"# {docs[name]}")
            value = SECRET_PLACEHOLDERS.get(name)
            if value is None:
                value = default_text(info)
                value = "" if value == "unset" else value
            lines.append(f"# {primary}={value}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail when the files are stale")
    args = parser.parse_args()
    docs = field_docs()
    targets = {
        ROOT / "docs" / "configuration.md": render_markdown(docs),
        ROOT / ".env.example": render_env_example(docs),
    }
    stale = [p for p, text in targets.items() if not p.exists() or p.read_text() != text]
    if args.check:
        for path in stale:
            print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1 if stale else 0
    for path, text in targets.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
