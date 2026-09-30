"""``lst doctor``: check that this machine can run the pipeline, and say what is missing.

Every check reports one of four statuses:

``pass``  works;
``warn``  works, but something is degraded (no JavaScript runtime for yt-dlp);
``fail``  the pipeline cannot run as configured (no ffmpeg, the selected STT provider
          cannot start, a half-configured notifier);
``info``  a fact worth knowing (other providers' readiness, whether a key is set).

The checks never print a secret (a key is only ever "set" or "unset"), never send audio
anywhere and have no side effects on disk. Only the optional stream check touches the
network, and only when a URL is given.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import __version__
from .ansi import paint, supports_color
from .config import CLOUD_STT_PROVIDERS, DEFAULT_ENV_FILE, STT_PROVIDERS, ConfigError, Settings
from .logging_setup import get_logger
from .redact import redact_text
from .rules import RulesError, RuleSet
from .stream.base import StreamResolutionError
from .stream.ffmpeg import ffmpeg_available
from .stream.resolver import js_runtime_status, resolve_stream
from .stt.factory import provider_problem

log = get_logger(__name__)

__all__ = ["Check", "DoctorReport", "Status", "run_doctor"]

Status = Literal["pass", "warn", "fail", "info"]

_STYLE = {"pass": "green", "warn": "yellow", "fail": "bold_red", "info": "dim"}
_MIN_SQLITE = (3, 24, 0)  # ``INSERT .. ON CONFLICT DO UPDATE``, used by the store


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(slots=True)
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    stream_failed: bool = False
    """The optional URL check failed; kept apart from environment failures."""

    def add(self, name: str, status: Status, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def exit_code(self) -> int:
        """0 healthy, 2 the environment or configuration cannot work, 3 the URL is not usable."""
        env_failures = [c for c in self.failures if c.name != "stream"]
        if env_failures:
            return 2
        return 3 if self.stream_failed else 0

    def render(self, *, color: bool | None = None, stream: object = None) -> str:
        use_color = supports_color(stream or sys.stdout, force=color)
        width = max((len(c.name) for c in self.checks), default=0)
        lines = [f"lst doctor (livestream-transcriber {__version__})"]
        for check in self.checks:
            label = paint(check.status.upper().ljust(4), _STYLE[check.status], use_color)
            lines.append(f"  {label}  {check.name.ljust(width)}  {check.detail}")
        problems = len(self.failures)
        warns = sum(1 for c in self.checks if c.status == "warn")
        lines.append("")
        if problems:
            lines.append(f"{problems} problem(s) to fix before `lst run` will work.")
        elif warns:
            lines.append(f"Ready, with {warns} warning(s).")
        else:
            lines.append("Everything needed is in place.")
        return "\n".join(lines)


def _ffmpeg_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    first = out.stdout.splitlines()[0] if out.stdout else ""
    return first.split(" Copyright")[0]


def _writable(path: Path) -> bool:
    """Whether ``path`` exists and is writable, or could be created under a writable parent."""
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)


def _check_stt(report: DoctorReport, settings: Settings) -> None:
    selected = settings.stt_provider
    try:
        settings.validate_stt()
        key_error = None
    except ConfigError as exc:
        key_error = str(exc)

    problem = provider_problem(selected, settings)
    if key_error is not None:
        report.add("stt", "fail", key_error)
    elif problem is not None:
        report.add("stt", "fail", f"{selected}: {problem}")
    else:
        detail = f"provider={selected}, model={settings.model_for(selected) or '-'}"
        if settings.stt_fallback != "none":
            detail += f", fallback={settings.stt_fallback}"
        report.add("stt", "pass", detail)
    if settings.stt_fallback not in ("none", selected):
        fallback_problem = provider_problem(settings.stt_fallback, settings)
        if fallback_problem is not None:
            report.add("stt fallback", "fail", f"{settings.stt_fallback}: {fallback_problem}")

    for name in STT_PROVIDERS:
        if name in ("none", "mock", selected):
            continue
        other = provider_problem(name, settings)
        report.add(f"stt {name}", "info", "ready" if other is None else other)
    for name in sorted(CLOUD_STT_PROVIDERS):
        report.add(f"{name} key", "info", "set" if settings.api_key_for(name) else "unset")


def _check_notifiers(report: DoctorReport, settings: Settings) -> None:
    webhook = settings.notify_webhook_url is not None
    report.add("webhook", "info", f"{'set' if webhook else 'unset'} (LST_NOTIFY_WEBHOOK_URL)")
    has_token = settings.notify_telegram_token is not None
    has_chat = bool(settings.notify_telegram_chat_id)
    if has_token != has_chat:
        report.add(
            "telegram",
            "fail",
            "needs both LST_NOTIFY_TELEGRAM_TOKEN and LST_NOTIFY_TELEGRAM_CHAT_ID",
        )
    else:
        report.add("telegram", "info", "set" if has_token else "unset")


def _check_rules(report: DoctorReport, settings: Settings, rules_file: Path | None) -> None:
    path = rules_file or settings.rules_file
    if path is None:
        report.add("rules", "info", "no rules file (pass --rules or set LST_RULES_FILE)")
        return
    try:
        ruleset = RuleSet.load(path)
    except RulesError as exc:
        report.add("rules", "fail", "; ".join(exc.problems[:3]))
    except OSError as exc:
        report.add("rules", "fail", f"cannot read {path}: {exc.strerror or exc}")
    else:
        report.add("rules", "pass", f"{path}: {len(ruleset)} rule(s)")


async def run_doctor(
    settings: Settings,
    *,
    url: str | None = None,
    env_file: str | Path | None = None,
    rules_file: Path | None = None,
) -> DoctorReport:
    """Run every check and return the report. Only the URL check uses the network."""
    report = DoctorReport()
    version = sys.version.split()[0]
    report.add(
        "python", "pass" if sys.version_info >= (3, 11) else "fail", f"{version} (needs 3.11+)"
    )

    binary = ffmpeg_available(settings.capture_ffmpeg_binary)
    if binary is None:
        report.add("ffmpeg", "fail", "not on PATH (macOS: brew install ffmpeg)")
    else:
        report.add("ffmpeg", "pass", f"{binary} ({_ffmpeg_version(binary) or 'version unknown'})")

    try:
        import yt_dlp

        report.add("yt-dlp", "pass", yt_dlp.version.__version__)
    except ImportError:
        report.add("yt-dlp", "fail", "not installed (pip install livestream-transcriber)")
    else:
        js_ok, js_detail = await asyncio.to_thread(js_runtime_status)
        report.add("js runtime", "pass" if js_ok else "warn", js_detail)

    if sqlite3.sqlite_version_info >= _MIN_SQLITE:
        report.add("sqlite", "pass", sqlite3.sqlite_version)
    else:
        report.add("sqlite", "fail", f"{sqlite3.sqlite_version} is too old (needs 3.24+)")

    env_path = Path(env_file) if env_file else Path(DEFAULT_ENV_FILE)
    report.add(
        "env file",
        "info",
        f"{env_path} loaded" if env_path.is_file() else f"{env_path} not found (using environment)",
    )

    _check_stt(report, settings)
    _check_notifiers(report, settings)
    _check_rules(report, settings, rules_file)

    for label, path in (("out dir", settings.out_dir), ("database", settings.database_path.parent)):
        ok = _writable(path)
        shown = settings.out_dir if label == "out dir" else settings.database_path
        report.add(
            label, "pass" if ok else "fail", f"{shown}" if ok else f"{shown} is not writable"
        )

    if url:
        await _check_stream(report, settings, url)
    else:
        report.add("stream", "info", "skipped (pass --url to resolve a source)")
    return report


async def _check_stream(report: DoctorReport, settings: Settings, url: str) -> None:
    if not url.lower().startswith(("http://", "https://", "rtmp://", "rtmps://")):
        exists = Path(url).exists()
        report.add(
            "stream",
            "pass" if exists else "fail",
            f"local file {url}" if exists else f"file not found: {url}",
        )
        report.stream_failed = not exists
        return
    proxy = settings.capture_proxy.get_secret_value() if settings.capture_proxy else None
    cookies = str(settings.capture_cookies_file) if settings.capture_cookies_file else None
    try:
        info = await resolve_stream(
            url,
            format_selector=settings.capture_stream_format,
            cookiefile=cookies,
            proxy=proxy,
        )
    except StreamResolutionError as exc:
        report.add("stream", "fail", redact_text(str(exc))[:200])
        report.stream_failed = True
        return
    title = info.title or "untitled"
    report.add("stream", "pass", f"{title} (live={'yes' if info.is_live else 'no'})")
