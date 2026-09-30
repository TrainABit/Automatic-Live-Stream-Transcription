"""``lst doctor``: what it checks, how it reports and which exit code it earns."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from livestream_transcriber import doctor as doctor_mod
from livestream_transcriber.cli import main
from livestream_transcriber.config import Settings
from livestream_transcriber.doctor import DoctorReport, run_doctor
from livestream_transcriber.models import StreamInfo
from livestream_transcriber.stream.base import StreamResolutionError

EXAMPLE_RULES = Path(__file__).parent.parent / "rules.example.yaml"


@pytest.fixture(autouse=True)
def _healthy_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ffmpeg and a JavaScript runtime exist, so results do not depend on the host."""
    monkeypatch.setattr(doctor_mod, "ffmpeg_available", lambda _binary: "/usr/bin/ffmpeg")
    monkeypatch.setattr(doctor_mod, "_ffmpeg_version", lambda _binary: "ffmpeg version 7.0")
    monkeypatch.setattr(doctor_mod, "js_runtime_status", lambda: (True, "deno 2.0"))


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {"stt_provider": "mock", "out_dir": tmp_path / "out"}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def check(report: DoctorReport, name: str):  # type: ignore[no-untyped-def]
    (found,) = [c for c in report.checks if c.name == name]
    return found


def run(settings: Settings, **kwargs):  # type: ignore[no-untyped-def]
    return asyncio.run(run_doctor(settings, **kwargs))


class TestEnvironment:
    def test_a_working_setup_passes(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path))
        assert report.exit_code == 0
        assert check(report, "ffmpeg").status == "pass"
        assert check(report, "stt").status == "pass"
        assert "Everything needed is in place" in report.render(color=False)

    def test_missing_ffmpeg_is_a_failure_with_a_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor_mod, "ffmpeg_available", lambda _binary: None)
        report = run(settings_for(tmp_path))
        assert check(report, "ffmpeg").status == "fail"
        assert "brew install ffmpeg" in check(report, "ffmpeg").detail
        assert report.exit_code == 2

    def test_a_missing_js_runtime_is_only_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor_mod, "js_runtime_status", lambda: (False, "none found"))
        report = run(settings_for(tmp_path))
        assert check(report, "js runtime").status == "warn"
        assert report.exit_code == 0
        assert "Ready, with 1 warning" in report.render(color=False)

    def test_an_unwritable_output_directory_fails(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        report = run(settings_for(tmp_path, out_dir=blocker / "out"))
        assert check(report, "out dir").status == "fail"
        assert report.exit_code == 2

    def test_nothing_is_created_on_disk(self, tmp_path: Path) -> None:
        run(settings_for(tmp_path))
        assert not (tmp_path / "out").exists()


class TestSpeechToText:
    def test_a_cloud_provider_without_a_key_names_the_variable(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path, stt_provider="openai"))
        stt = check(report, "stt")
        assert stt.status == "fail"
        assert "LST_OPENAI_API_KEY" in stt.detail
        assert report.exit_code == 2

    def test_the_local_provider_without_its_extra_says_how_to_install_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            doctor_mod,
            "provider_problem",
            lambda name, _s: "faster-whisper is not installed" if name == "local" else None,
        )
        report = run(settings_for(tmp_path, stt_provider="local"))
        assert check(report, "stt").status == "fail"
        assert "faster-whisper" in check(report, "stt").detail

    def test_keys_are_reported_as_set_or_unset_never_shown(self, tmp_path: Path) -> None:
        secret = "your-openai-key"
        report = run(settings_for(tmp_path, openai_api_key=secret))
        assert check(report, "openai key").detail == "set"
        assert check(report, "openrouter key").detail == "unset"
        assert secret not in report.render(color=False)

    def test_other_providers_are_listed_as_information(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path))
        assert {"stt local", "stt onnx", "stt openai", "stt openrouter"} <= {
            c.name for c in report.checks
        }
        assert all(c.status == "info" for c in report.checks if c.name.startswith("stt "))

    def test_a_broken_fallback_provider_fails(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path, stt_fallback="openai"))
        assert check(report, "stt fallback").status == "fail"


class TestNotifiersAndRules:
    def test_half_a_telegram_setup_is_a_failure(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path, notify_telegram_token="123456:your-telegram-bot-token"))
        assert check(report, "telegram").status == "fail"
        assert report.exit_code == 2

    def test_a_webhook_url_is_never_printed(self, tmp_path: Path) -> None:
        url = "https://hooks.example.test/services/your-webhook-secret"
        report = run(settings_for(tmp_path, notify_webhook_url=url))
        assert check(report, "webhook").detail.startswith("set")
        assert "your-webhook-secret" not in report.render(color=False)

    def test_a_valid_rules_file_passes_and_counts_rules(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path), rules_file=EXAMPLE_RULES)
        assert check(report, "rules").status == "pass"
        assert "rule(s)" in check(report, "rules").detail

    def test_an_invalid_rules_file_fails_with_the_problem(self, tmp_path: Path) -> None:
        bad = tmp_path / "rules.yaml"
        bad.write_text("version: 1\nrules:\n  - id: x\n    type: nonsense\n", encoding="utf-8")
        report = run(settings_for(tmp_path), rules_file=bad)
        assert check(report, "rules").status == "fail"
        assert report.exit_code == 2

    def test_a_missing_rules_file_fails(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path), rules_file=tmp_path / "gone.yaml")
        assert check(report, "rules").status == "fail"


class TestStreamCheck:
    def test_skipped_without_a_url(self, tmp_path: Path) -> None:
        assert check(run(settings_for(tmp_path)), "stream").status == "info"

    def test_an_existing_local_file_passes(self, tmp_path: Path) -> None:
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"x")
        report = run(settings_for(tmp_path), url=str(clip))
        assert check(report, "stream").status == "pass"
        assert report.exit_code == 0

    def test_a_missing_local_file_is_exit_3(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path), url=str(tmp_path / "gone.wav"))
        assert check(report, "stream").status == "fail"
        assert report.exit_code == 3

    def test_a_resolvable_url_reports_its_title_and_liveness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def resolve(url: str, **_kw: object) -> StreamInfo:
            return StreamInfo(url=url, title="Morning show", is_live=True)

        monkeypatch.setattr(doctor_mod, "resolve_stream", resolve)
        report = run(settings_for(tmp_path), url="https://example.com/live")
        assert check(report, "stream").detail == "Morning show (live=yes)"

    def test_a_resolution_failure_is_exit_3_and_is_redacted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def resolve(url: str, **_kw: object) -> StreamInfo:
            raise StreamResolutionError(
                "failed: https://cdn.example.test/a.m3u8?token=your-token-value"  # gitleaks:allow
            )

        monkeypatch.setattr(doctor_mod, "resolve_stream", resolve)
        report = run(settings_for(tmp_path), url="https://example.com/live")
        assert report.exit_code == 3
        assert "your-token-value" not in report.render(color=False)

    def test_a_stream_failure_does_not_hide_an_environment_failure(self, tmp_path: Path) -> None:
        report = run(settings_for(tmp_path, stt_provider="openai"), url=str(tmp_path / "gone"))
        assert report.exit_code == 2


class TestRendering:
    def test_plain_output_has_no_escape_codes(self, tmp_path: Path) -> None:
        assert "\x1b" not in run(settings_for(tmp_path)).render(color=False)

    def test_forced_colour_paints_the_status(self, tmp_path: Path) -> None:
        assert "\x1b[" in run(settings_for(tmp_path)).render(color=True)


class TestCommand:
    def test_exit_code_and_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LST_STT_PROVIDER", "mock")
        monkeypatch.setenv("LST_OUT_DIR", str(tmp_path / "out"))
        assert main(["doctor", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("lst doctor (livestream-transcriber ")
        assert "PASS  ffmpeg" in out

    def test_failures_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LST_STT_PROVIDER", "openai")
        monkeypatch.setenv("LST_OUT_DIR", str(tmp_path / "out"))
        assert main(["doctor", "--no-color"]) == 2
        assert "FAIL  stt" in capsys.readouterr().out

    def test_show_config_hides_secrets(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LST_STT_PROVIDER", "mock")
        monkeypatch.setenv("LST_OUT_DIR", str(tmp_path / "out"))
        monkeypatch.setenv("LST_OPENAI_API_KEY", "your-openai-key")
        assert main(["doctor", "--no-color", "--show-config"]) == 0
        out = capsys.readouterr().out
        assert "your-openai-key" not in out
        summary = json.loads(out[out.index("{") :])
        assert summary["stt_provider"] == "mock"
