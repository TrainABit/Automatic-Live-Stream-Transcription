"""Command line parsing, setting overrides, exit codes and the cheap subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from livestream_transcriber import __version__, cli
from livestream_transcriber.cli import build_parser, main
from livestream_transcriber.config import Settings
from livestream_transcriber.pipeline import session as session_module

EXAMPLE_RULES = Path(__file__).parent.parent / "rules.example.yaml"


class FakeRunner:
    """Stands in for :class:`SessionRunner`: remembers what it was asked to run."""

    instances: list[FakeRunner] = []
    exit_code = 0

    def __init__(self, settings: Settings, plan: Any, **kwargs: Any) -> None:
        self.settings = settings
        self.plan = plan
        self.kwargs = kwargs
        FakeRunner.instances.append(self)

    async def run(self) -> int:
        return FakeRunner.exit_code


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunner]:
    FakeRunner.instances = []
    FakeRunner.exit_code = 0
    monkeypatch.setattr(session_module, "SessionRunner", FakeRunner)
    return FakeRunner


class TestParser:
    def test_every_documented_command_parses(self) -> None:
        parser = build_parser()
        for argv in (
            ["doctor"],
            ["doctor", "--url", "https://example.com/live"],
            ["run", "--url", "https://example.com/live"],
            ["record", "demo", "--url", "https://example.com/live", "--duration", "300"],
            ["replay", "recordings/demo", "--speed", "0"],
            ["bench", "--clips", "clips/", "--stt", "local,openai"],
            ["models", "fetch", "--model", "parakeet-tdt-0.6b-v3"],
            ["rules", "test", "rules.yaml", "some text"],
        ):
            assert parser.parse_args(argv).handler is not None, argv

    def test_run_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--url", "u", "--fallback", "a", "--fallback", "b", "--language", "de",
                "--stt", "openai", "--model", "m", "--rules", "r.yaml", "--out", "o",
                "--db", "d.db", "--record", "rec", "--duration", "12.5",
            ]
        )  # fmt: skip
        assert args.url == "u"
        assert args.fallback == ["a", "b"]
        assert (args.language, args.stt, args.model) == ("de", "openai", "m")
        assert args.rules == Path("r.yaml")
        assert args.out == Path("o")
        assert args.db == Path("d.db")
        assert args.record == Path("rec")
        assert args.duration == 12.5

    def test_defaults_leave_settings_alone(self) -> None:
        args = build_parser().parse_args(["run", "--url", "u"])
        assert cli._overrides(args) == {
            "stt_provider": None,
            "stt_model": None,
            "stt_language": None,
            "out_dir": None,
            "db_path": None,
            "rules_file": None,
        }

    def test_global_options_work_before_and_after_the_command(self) -> None:
        parser = build_parser()
        before = parser.parse_args(["--no-color", "--log-level", "DEBUG", "doctor"])
        after = parser.parse_args(["doctor", "--no-color", "--log-level", "DEBUG"])
        for args in (before, after):
            assert args.no_color is True
            assert args.log_level == "DEBUG"

    def test_a_global_option_given_early_survives_the_subparser(self) -> None:
        args = build_parser().parse_args(["--env-file", "x.env", "run", "--url", "u"])
        assert args.env_file == "x.env"

    def test_unset_global_options_have_neutral_defaults(self) -> None:
        args = build_parser().parse_args(["run", "--url", "u"])
        assert (args.env_file, args.log_level, args.no_color) == (None, None, False)

    def test_replay_speed_defaults_to_unthrottled(self) -> None:
        assert build_parser().parse_args(["replay", "rec"]).speed == 0.0

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["run"],  # --url is required
            ["run", "--url", "u", "--stt", "whisperx"],  # unknown provider
            ["bench"],  # --clips is required
            ["record", "--url", "u"],  # name is required
            ["frobnicate"],
        ],
    )
    def test_invalid_command_lines_exit_2(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(argv) == 2
        assert capsys.readouterr().err

    def test_help_and_version_exit_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--help"]) == 0
        assert "doctor" in capsys.readouterr().out
        assert main(["--version"]) == 0
        assert __version__ in capsys.readouterr().out

    def test_every_provider_is_a_valid_choice(self) -> None:
        from livestream_transcriber.config import STT_PROVIDERS

        parser = build_parser()
        for provider in STT_PROVIDERS:
            assert parser.parse_args(["run", "--url", "u", "--stt", provider]).stt == provider

    def test_bare_models_and_rules_print_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["models"]) == 2
        assert main(["rules"]) == 2
        assert "usage" in capsys.readouterr().err


class TestSettingsPrecedence:
    def test_flags_override_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, fake_runner: type[FakeRunner]
    ) -> None:
        monkeypatch.setenv("LST_STT_PROVIDER", "openai")
        monkeypatch.setenv("LST_STT_MODEL", "whisper-1")
        assert main(["run", "--url", "u", "--stt", "mock", "--model", "tiny"]) == 0
        settings = fake_runner.instances[0].settings
        assert (settings.stt_provider, settings.stt_model) == ("mock", "tiny")

    def test_environment_applies_when_no_flag_is_given(
        self, monkeypatch: pytest.MonkeyPatch, fake_runner: type[FakeRunner]
    ) -> None:
        monkeypatch.setenv("LST_STT_PROVIDER", "mock")
        monkeypatch.setenv("LST_STT_LANGUAGE", "de")
        assert main(["run", "--url", "u"]) == 0
        settings = fake_runner.instances[0].settings
        assert (settings.stt_provider, settings.stt_language) == ("mock", "de")

    def test_env_file_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runner: type[FakeRunner]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        env = tmp_path / "custom.env"
        env.write_text("LST_STT_PROVIDER=mock\nLST_STT_LANGUAGE=fr\n")
        assert main(["--env-file", str(env), "run", "--url", "u"]) == 0
        assert fake_runner.instances[0].settings.stt_language == "fr"

    def test_a_missing_env_file_is_a_config_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--env-file", str(tmp_path / "nope.env"), "doctor"]) == 2
        assert "not found" in capsys.readouterr().err

    def test_an_invalid_setting_is_a_config_error_that_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LST_STT_WORKERS", "0")
        assert main(["doctor"]) == 2
        assert "LST_STT_WORKERS" in capsys.readouterr().err

    def test_an_invalid_log_level_is_a_config_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--log-level", "LOUD", "doctor"]) == 2
        err = capsys.readouterr().err
        assert "--log-level" in err  # the flag that carried the value, not LST_LOG_LEVEL
        assert "LST_LOG_LEVEL" not in err

    def test_a_bad_environment_value_still_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LST_LOG_LEVEL", "LOUD")
        assert main(["doctor"]) == 2
        assert "LST_LOG_LEVEL" in capsys.readouterr().err


class TestSessionCommands:
    def test_run_builds_the_plan(self, tmp_path: Path, fake_runner: type[FakeRunner]) -> None:
        rules = tmp_path / "rules.yaml"
        rules.write_text(EXAMPLE_RULES.read_text())
        code = main(
            [
                "run", "--url", "https://example.com/live", "--fallback", "https://example.com/b",
                "--stt", "mock", "--rules", str(rules), "--out", str(tmp_path / "o"),
                "--duration", "30", "--no-resume", "--realtime", "--no-color",
            ]
        )  # fmt: skip
        assert code == 0
        (runner,) = fake_runner.instances
        plan = runner.plan
        assert plan.url == "https://example.com/live"
        assert plan.fallbacks == ("https://example.com/b",)
        assert plan.duration == 30.0
        assert plan.mode == "live"
        assert plan.resume is False
        assert plan.realtime is True
        assert plan.out_dir == tmp_path / "o"
        assert runner.settings.out_dir == tmp_path / "o"
        assert runner.kwargs["ruleset"] is not None
        assert runner.kwargs["color"] is False

    def test_run_with_record_stays_in_live_mode(
        self, tmp_path: Path, fake_runner: type[FakeRunner]
    ) -> None:
        assert main(["run", "--url", "u", "--stt", "mock", "--record", str(tmp_path / "r")]) == 0
        plan = fake_runner.instances[0].plan
        # Recording alongside a run must keep fallback sources and auto-resume.
        assert plan.mode == "live"
        assert plan.record_dir == tmp_path / "r"

    def test_record_records_only_by_default(self, fake_runner: type[FakeRunner]) -> None:
        assert main(["record", "demo", "--url", "u", "--duration", "300"]) == 0
        plan = fake_runner.instances[0].plan
        assert plan.mode == "record"
        assert plan.record_dir == Path("recordings") / "demo"
        assert plan.transcribe is False
        assert plan.duration == 300.0

    def test_record_can_transcribe_and_use_another_directory(
        self, tmp_path: Path, fake_runner: type[FakeRunner]
    ) -> None:
        assert main(["record", "demo", "--url", "u", "--dir", str(tmp_path), "--stt", "mock"]) == 0
        plan = fake_runner.instances[0].plan
        assert plan.record_dir == tmp_path / "demo"
        assert plan.transcribe is True

    def test_replay_builds_the_plan(self, fake_runner: type[FakeRunner]) -> None:
        assert main(["replay", "recordings/demo", "--stt", "mock", "--speed", "2"]) == 0
        plan = fake_runner.instances[0].plan
        assert plan.mode == "replay"
        assert plan.url == str(Path("recordings/demo"))
        assert plan.replay_speed == 2.0

    def test_the_exit_code_of_the_run_is_returned(self, fake_runner: type[FakeRunner]) -> None:
        for code in (0, 3, 4, 5, 130):
            fake_runner.exit_code = code
            assert main(["run", "--url", "u", "--stt", "mock"]) == code

    def test_an_invalid_rules_file_is_a_config_error(
        self, tmp_path: Path, fake_runner: type[FakeRunner], capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 1\nrules:\n  - id: x\n    type: keyword\n")
        assert main(["run", "--url", "u", "--stt", "mock", "--rules", str(bad)]) == 2
        assert "x" in capsys.readouterr().err
        assert fake_runner.instances == []

    def test_a_missing_rules_file_is_a_config_error(
        self, tmp_path: Path, fake_runner: type[FakeRunner]
    ) -> None:
        assert (
            main(["run", "--url", "u", "--stt", "mock", "--rules", str(tmp_path / "no.yaml")]) == 2
        )

    def test_a_key_missing_for_the_provider_is_a_config_error_not_silence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["run", "--url", "https://example.com/x", "--stt", "openai", "--out", str(tmp_path)]
        )
        assert code == 2
        assert "LST_OPENAI_API_KEY" in capsys.readouterr().err + ""

    def test_a_missing_input_file_is_a_config_error(self, tmp_path: Path) -> None:
        assert main(["run", "--url", str(tmp_path / "gone.mp4"), "--stt", "mock"]) == 2

    def test_a_missing_recording_is_a_config_error(self, tmp_path: Path) -> None:
        assert main(["replay", str(tmp_path / "gone"), "--stt", "mock"]) == 2


class TestRulesTest:
    def test_a_matching_text_prints_the_hit(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            ["rules", "test", str(EXAMPLE_RULES), "we are running a giveaway tonight", "--no-color"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "giveaway" in out
        assert "WARNING" in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["rules", "test", str(EXAMPLE_RULES), "a new release is coming", "--json"])
        assert code == 0
        rows = json.loads(capsys.readouterr().out)
        assert [r["rule"] for r in rows] == ["new-release"]
        assert rows[0]["severity"] == "info"

    def test_no_match_says_so_and_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["rules", "test", str(EXAMPLE_RULES), "nothing to see here"]) == 0
        assert "no rule matched" in capsys.readouterr().out

    def test_language_restricted_rules_respect_the_language_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["rules", "test", str(EXAMPLE_RULES), "a new release", "--language", "de", "--json"])
        assert json.loads(capsys.readouterr().out) == []

    def test_a_broken_rules_file_lists_its_problems(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 1\nrules:\n  - id: r\n    type: regex\n    pattern: '('\n")
        assert main(["rules", "test", str(bad), "x"]) == 2
        assert "r" in capsys.readouterr().err

    def test_a_missing_file_is_a_config_error(self, tmp_path: Path) -> None:
        assert main(["rules", "test", str(tmp_path / "none.yaml"), "x"]) == 2

    def test_it_needs_no_keys_and_no_network(self, capsys: pytest.CaptureFixture[str]) -> None:
        # The example file declares an ``llm`` block; the dry run must not use it.
        assert main(["rules", "test", str(EXAMPLE_RULES), "hello"]) == 0


class TestModelsFetch:
    def test_unknown_model_is_a_config_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["models", "fetch", "--model", "no-such-model"]) == 2
        assert "known models" in capsys.readouterr().err

    def test_an_installed_model_is_not_downloaded_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from livestream_transcriber.stt.providers import sherpa_onnx

        name = sherpa_onnx.DEFAULT_MODEL
        base = tmp_path / name
        base.mkdir(parents=True)
        for filename in sherpa_onnx.ONNX_MODELS[name].files:
            (base / filename).write_bytes(b"x")
        monkeypatch.setenv("LST_STT_MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(
            sherpa_onnx, "ensure_model", lambda *a, **k: pytest.fail("must not download")
        )
        assert main(["models", "fetch"]) == 0
        assert "already" in capsys.readouterr().out

    def test_a_failed_download_is_exit_1_and_says_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from livestream_transcriber.stt.providers import sherpa_onnx

        def failing(*args: object, **kwargs: object) -> Path:
            raise OSError("connection reset")

        monkeypatch.setenv("LST_STT_MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(sherpa_onnx, "ensure_model", failing)
        assert main(["models", "fetch"]) == 1
        assert "connection reset" in capsys.readouterr().err

    def test_a_download_reports_each_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from livestream_transcriber.stt.providers import sherpa_onnx

        def fake_download(name: str, root: Path, *, on_file: Any = None, **_kw: object) -> Path:
            for filename in ("a.onnx", "tokens.txt"):
                on_file(filename)
            return root / name

        monkeypatch.setenv("LST_STT_MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(sherpa_onnx, "ensure_model", fake_download)
        assert main(["models", "fetch"]) == 0
        out = capsys.readouterr().out
        assert "fetching a.onnx" in out
        assert "ready" in out
