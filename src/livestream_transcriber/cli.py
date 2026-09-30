"""The ``lst`` command line.

::

    lst doctor  [--url URL]
    lst run     --url URL [--fallback URL ...] [--language de] [--stt local|onnx|openai|...]
                [--model M] [--rules rules.yaml] [--out out/] [--db out/lst.db]
                [--record DIR] [--duration S]
    lst record  NAME --url URL --duration 300 [--dir recordings]
    lst replay  recordings/NAME [--speed 0] [--stt ...] [--rules ...] [--out ...]
    lst bench   --clips clips/ --stt local,openai [--out bench.json]
    lst models fetch [--model parakeet-tdt-0.6b-v3]
    lst rules test rules.yaml "some text"

Global options (``--env-file``, ``--log-level``, ``--no-color``) are accepted before or
after the subcommand.

Exit codes: 0 ok, 1 a command-specific failure (a failed model download), 2 configuration
error, 3 not live, 4 no data, 5 stream lost, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .ansi import paint, supports_color
from .config import STT_PROVIDERS, ConfigError, Settings
from .logging_setup import get_logger, setup_logging
from .redact import redact_text

log = get_logger(__name__)

__all__ = ["build_parser", "main"]

Handler = Callable[[Settings, argparse.Namespace], int]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130

# Which flag overrides which setting, per command. A flag that was not given is ``None``
# and leaves the setting (environment, ``.env`` or default) alone. ``bench`` and the
# ``doctor``/``rules``/``models`` commands use ``--stt``, ``--rules`` and ``--model`` with
# their own meaning, so nothing maps for them.
_OVERRIDES: dict[str, dict[str, str]] = {
    "run": {
        "stt": "stt_provider",
        "model": "stt_model",
        "language": "stt_language",
        "out": "out_dir",
        "db": "db_path",
        "rules": "rules_file",
    },
    "replay": {
        "stt": "stt_provider",
        "model": "stt_model",
        "language": "stt_language",
        "out": "out_dir",
        "db": "db_path",
        "rules": "rules_file",
    },
    "record": {"stt": "stt_provider"},
}


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _global_options(*, top_level: bool) -> argparse.ArgumentParser:
    """The options every command accepts.

    The same options are attached to the top-level parser (with real defaults) and to
    every subparser (defaults suppressed), which is what lets ``lst --no-color run ...``
    and ``lst run ... --no-color`` both work without one overwriting the other.
    """
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("global options")

    def default(value: Any) -> Any:
        return value if top_level else argparse.SUPPRESS

    group.add_argument(
        "--env-file",
        metavar="FILE",
        default=default(None),
        help="read settings from this .env file (default: ./.env when present)",
    )
    group.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=default(None),
        help="DEBUG, INFO, WARNING or ERROR (default: LST_LOG_LEVEL or INFO)",
    )
    group.add_argument(
        "--no-color",
        action="store_true",
        default=default(False),
        help="never colour output (the NO_COLOR variable is honoured too)",
    )
    return parent


def _add_stt_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stt",
        choices=STT_PROVIDERS,
        metavar="PROVIDER",
        help=f"speech-to-text provider: {', '.join(STT_PROVIDERS)} (default: LST_STT_PROVIDER)",
    )
    parser.add_argument("--model", help="model name (default: the provider's default)")
    parser.add_argument(
        "--language", help="language code such as en or de (default: detect automatically)"
    )
    parser.add_argument(
        "--mock-fixtures",
        metavar="FILE",
        type=Path,
        help="with --stt mock: replay transcripts from a JSONL file (for demos and tests)",
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rules", metavar="FILE", type=Path, help="event rules (YAML)")
    parser.add_argument(
        "--out", metavar="DIR", type=Path, help="directory for transcript files (default: out/)"
    )
    parser.add_argument(
        "--db", metavar="FILE", type=Path, help="SQLite database (default: OUT/lst.db)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lst",
        description="Real-time transcription of live streams with pluggable speech-to-text, "
        "event rules and notifiers.",
        parents=[_global_options(top_level=True)],
    )
    parser.add_argument("--version", action="version", version=f"lst {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name: str, help_text: str, handler: Handler) -> argparse.ArgumentParser:
        sub = commands.add_parser(
            name,
            help=help_text,
            description=help_text,
            parents=[_global_options(top_level=False)],
        )
        sub.set_defaults(handler=handler)
        return sub

    doctor = command("doctor", "check the environment (and resolve a URL)", _cmd_doctor)
    doctor.add_argument("--url", help="also resolve this source")
    doctor.add_argument("--rules", metavar="FILE", type=Path, help="also validate this rules file")
    doctor.add_argument(
        "--show-config", action="store_true", help="print the settings (secrets hidden)"
    )

    run = command("run", "transcribe a live stream, URL or file", _cmd_run)
    run.add_argument("--url", required=True, help="stream URL, m3u8/media URL or local file")
    run.add_argument(
        "--fallback",
        action="append",
        default=[],
        metavar="URL",
        help="backup source, in order of preference (repeatable)",
    )
    _add_stt_options(run)
    _add_output_options(run)
    run.add_argument("--record", metavar="DIR", type=Path, help="also record the raw audio here")
    run.add_argument("--duration", type=float, metavar="SECONDS", help="stop after this much audio")
    run.add_argument(
        "--realtime", action="store_true", help="read a file at its native rate, like a live feed"
    )
    run.add_argument(
        "--no-resume",
        action="store_true",
        help="exit when the stream ends instead of waiting for it to come back",
    )

    record = command("record", "record a stream's audio for later replay", _cmd_record)
    record.add_argument("name", help="name of the recording (a directory inside --dir)")
    record.add_argument("--url", required=True, help="stream URL or file")
    record.add_argument("--duration", type=float, metavar="SECONDS", help="stop after this long")
    record.add_argument(
        "--dir", type=Path, metavar="DIR", help="recordings directory (default: recordings/)"
    )
    record.add_argument(
        "--stt",
        choices=STT_PROVIDERS,
        metavar="PROVIDER",
        help="also transcribe while recording (default: record only)",
    )
    record.add_argument("--note", help="free text stored in the recording's manifest")

    replay = command("replay", "run a recording through the pipeline again", _cmd_replay)
    replay.add_argument("recording", type=Path, help="a recording directory")
    replay.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="1 is real time; 0 (default) is unthrottled and deterministic",
    )
    _add_stt_options(replay)
    _add_output_options(replay)

    bench = command("bench", "measure accuracy and latency of STT providers", _cmd_bench)
    bench.add_argument("--clips", required=True, type=Path, help="directory of clips + references")
    bench.add_argument(
        "--stt",
        metavar="LIST",
        help="comma-separated providers, each optionally as provider:model "
        "(default: LST_STT_PROVIDER)",
    )
    bench.add_argument("--out", type=Path, metavar="FILE", help="write the JSON report here")
    bench.add_argument("--markdown", type=Path, metavar="FILE", help="write a markdown report here")
    bench.add_argument("--chunk-seconds", type=float, default=5.0, help="chunk length (default: 5)")
    bench.add_argument("--language", help="language code (default: detect automatically)")
    bench.add_argument(
        "--no-cold-pass", action="store_true", help="do not warm the model up before timing"
    )

    models = command("models", "manage local models", lambda _s, _a: _usage(models))
    models_sub = models.add_subparsers(dest="models_command", metavar="ACTION")
    fetch = models_sub.add_parser(
        "fetch",
        help="download an ONNX speech model (explicit opt-in)",
        parents=[_global_options(top_level=False)],
    )
    fetch.add_argument("--model", help="model name (default: parakeet-tdt-0.6b-v3)")
    fetch.set_defaults(handler=_cmd_models_fetch)

    rules = command("rules", "work with event rules", lambda _s, _a: _usage(rules))
    rules_sub = rules.add_subparsers(dest="rules_command", metavar="ACTION")
    test = rules_sub.add_parser(
        "test",
        help="dry-run rule matching against some text",
        parents=[_global_options(top_level=False)],
    )
    test.add_argument("rules_file", metavar="RULES", type=Path, help="rules file (YAML)")
    test.add_argument("text", help="the text to match")
    test.add_argument("--language", help="language code of the text")
    test.add_argument("--json", action="store_true", help="print hits as JSON")
    test.set_defaults(handler=_cmd_rules_test)
    return parser


def _usage(parser: argparse.ArgumentParser) -> int:
    parser.print_usage(sys.stderr)
    return EXIT_CONFIG


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _color(args: argparse.Namespace) -> bool | None:
    """``False`` when colour was refused; ``None`` lets the terminal decide."""
    return False if args.no_color else None


def _run_session(settings: Settings, args: argparse.Namespace, plan: Any) -> int:
    from .pipeline.session import SessionRunner
    from .rules import RulesError, RuleSet

    ruleset = None
    if settings.rules_file is not None:
        try:
            ruleset = RuleSet.load(settings.rules_file)
        except RulesError as exc:
            print(f"lst: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        except OSError as exc:
            print(f"lst: cannot read the rules file: {exc.strerror or exc}", file=sys.stderr)
            return EXIT_CONFIG
    runner = SessionRunner(settings, plan, ruleset=ruleset, color=_color(args))
    return int(asyncio.run(runner.run()))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    report = asyncio.run(
        run_doctor(settings, url=args.url, env_file=args.env_file, rules_file=args.rules)
    )
    print(report.render(color=_color(args)))
    if args.show_config:
        print()
        print(json.dumps(settings.redacted_summary(), indent=2, default=str))
    return report.exit_code


def _cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    from .pipeline.session import SessionPlan

    plan = SessionPlan(
        url=args.url,
        mode="live",
        fallbacks=tuple(args.fallback),
        duration=args.duration,
        record_dir=args.record,
        realtime=args.realtime,
        resume=False if args.no_resume else None,
        out_dir=settings.out_dir,
        mock_fixtures=args.mock_fixtures,
    )
    return _run_session(settings, args, plan)


def _cmd_record(settings: Settings, args: argparse.Namespace) -> int:
    from .pipeline.session import SessionPlan

    directory = (args.dir or settings.recordings_dir) / args.name
    plan = SessionPlan(
        url=args.url,
        mode="record",
        duration=args.duration,
        record_dir=directory,
        transcribe=args.stt is not None,
        out_dir=settings.out_dir,
        note=args.note,
        console=False,
    )
    return _run_session(settings, args, plan)


def _cmd_replay(settings: Settings, args: argparse.Namespace) -> int:
    from .pipeline.session import SessionPlan

    plan = SessionPlan(
        url=str(args.recording),
        mode="replay",
        replay_speed=args.speed,
        out_dir=settings.out_dir,
        mock_fixtures=args.mock_fixtures,
    )
    return _run_session(settings, args, plan)


def _cmd_bench(settings: Settings, args: argparse.Namespace) -> int:
    from .bench.runner import BenchError, parse_provider_specs, run_bench, write_report

    try:
        specs = parse_provider_specs(args.stt or settings.stt_provider)
        settings = settings.with_overrides(stt_language=args.language)
        report = run_bench(
            args.clips,
            specs,
            settings,
            chunk_seconds=args.chunk_seconds,
            cold_pass=not args.no_cold_pass,
            progress=lambda msg: print(f"  {msg}", file=sys.stderr),
        )
    except BenchError as exc:
        print(f"lst: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    print(report.to_markdown())
    write_report(report, args.out, args.markdown)
    for path in (args.out, args.markdown):
        if path is not None:
            print(f"wrote {path}", file=sys.stderr)
    return EXIT_OK if any(p.error is None for p in report.providers) else EXIT_CONFIG


def _cmd_models_fetch(settings: Settings, args: argparse.Namespace) -> int:
    from .stt.providers import sherpa_onnx

    name = args.model or settings.model_for("onnx")
    if name not in sherpa_onnx.ONNX_MODELS:
        known = ", ".join(sorted(sherpa_onnx.ONNX_MODELS))
        print(f"lst: unknown model {name!r}; known models: {known}", file=sys.stderr)
        return EXIT_CONFIG
    root = settings.stt_models_dir
    if sherpa_onnx.model_present(name, root):
        print(f"{name} is already in {root / name}")
        return EXIT_OK
    print(f"downloading {name} to {root / name}")
    try:
        path = sherpa_onnx.ensure_model(
            name, root, on_file=lambda fname: print(f"  fetching {fname}", flush=True)
        )
    except OSError as exc:
        print(f"lst: download failed: {redact_text(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"ready: {path}")
    return EXIT_OK


def _cmd_rules_test(settings: Settings, args: argparse.Namespace) -> int:
    from .rules import RuleEngine, RulesError, RuleSet, TextSegment

    try:
        ruleset = RuleSet.load(args.rules_file)
    except RulesError as exc:
        print(f"lst: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except OSError as exc:
        print(f"lst: cannot read {args.rules_file}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_CONFIG
    # ``llm`` rules are skipped here: a dry run must not make network calls.
    engine = RuleEngine(ruleset, llm=None)
    hits = engine.evaluate(TextSegment(text=args.text, language=args.language))
    if args.json:
        rows = [
            {
                "rule": h.rule_id,
                "severity": h.severity.value,
                "matched": h.matched_text,
                "groups": dict(h.groups),
                "notify": list(h.notify),
                "context": h.context,
            }
            for h in hits
        ]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return EXIT_OK
    if not hits:
        print(f"no rule matched ({len(ruleset.enabled_rules)} enabled rule(s) checked)")
        return EXIT_OK
    color = supports_color(sys.stdout, force=_color(args))
    for hit in hits:
        head = paint(f"{hit.severity.value.upper()} {hit.rule_id}", "bold_yellow", color)
        print(f"{head}: {hit.matched_text!r} -> {', '.join(hit.notify) or 'no targets'}")
        if hit.context and hit.context != hit.matched_text:
            print(f"  {hit.context}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Settings overridden by flags; flags that were not given are ``None`` and ignored."""
    mapping = _OVERRIDES.get(args.command, {})
    overrides: dict[str, Any] = {
        setting: getattr(args, flag) for flag, setting in mapping.items() if hasattr(args, flag)
    }
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level
    return overrides


def _config_error_text(exc: ConfigError, args: argparse.Namespace) -> str:
    """The error with ``LST_*`` names swapped for the flag that supplied the bad value.

    A validation error names the environment variable; when the value came from a flag
    (``--log-level bogus``), pointing at ``LST_LOG_LEVEL`` would send the user looking in
    the wrong place.
    """
    text = str(exc)
    flags = {setting: f"--{flag.replace('_', '-')}" for flag, setting in _flag_settings(args)}
    for setting, flag in flags.items():
        text = text.replace(f"LST_{setting.upper()}", flag)
    return redact_text(text)


def _flag_settings(args: argparse.Namespace) -> list[tuple[str, str]]:
    """``(flag, setting)`` for every flag the user actually passed."""
    pairs = [
        (flag, setting)
        for flag, setting in _OVERRIDES.get(args.command, {}).items()
        if getattr(args, flag, None) is not None
    ]
    if getattr(args, "log_level", None):
        pairs.append(("log_level", "log_level"))
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``lst``. Returns the exit code instead of raising :class:`SystemExit`."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_CONFIG

    color = _color(args)
    setup_logging(args.log_level or "INFO", color=color)
    try:
        settings = Settings.load(args.env_file, **_overrides(args))
    except ConfigError as exc:
        print(f"lst: {_config_error_text(exc, args)}", file=sys.stderr)
        return EXIT_CONFIG
    setup_logging(settings.log_level, color=color)

    handler: Handler = args.handler
    try:
        return int(handler(settings, args))
    except ConfigError as exc:
        print(f"lst: {redact_text(str(exc))}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
