from __future__ import annotations

from pathlib import Path

import pytest

from livestream_transcriber.config import (
    DEFAULT_STT_MODELS,
    STT_PROVIDERS,
    ConfigError,
    Settings,
    stt_key_family,
)

OPENAI_KEY = "your-openai-key"  # no vendor prefix: family "unknown", accepted by any provider
# The prefixes are what the key-family check looks at; they are joined here so that the source
# holds no key-shaped literal.
OPENAI_SK = "sk" + "-your-openai-key"
OPENROUTER_SK = "sk" + "-or-your-openrouter-key"


def test_defaults_are_sane():
    s = Settings()
    assert s.capture_sample_rate == 16000
    assert s.capture_live_chunk_seconds == 2.5
    assert s.capture_file_chunk_seconds == 5.0
    assert s.capture_stream_format == "bestaudio/best"
    assert s.capture_hls_window is False
    assert s.stt_provider == "local"
    assert s.stt_language is None, "language is auto-detected unless set"
    assert s.stt_model is None
    assert s.model_for() == DEFAULT_STT_MODELS["local"] == "small"
    assert s.stt_fallback == "none"
    assert s.resume_auto is True
    assert s.notify_webhook_format == "json"
    assert s.database_path == Path("out") / "lst.db"
    assert s.heartbeat_seconds == 60.0
    assert s.log_level == "INFO"
    s.validate_stt()  # the default needs no key


def test_every_default_model_belongs_to_a_known_provider():
    assert set(DEFAULT_STT_MODELS) <= set(STT_PROVIDERS)


def test_environment_uses_the_lst_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LST_STT_PROVIDER", "MOCK")
    monkeypatch.setenv("LST_CAPTURE_LIVE_CHUNK_SECONDS", "1.5")
    monkeypatch.setenv("LST_RESUME_AUTO", "false")
    monkeypatch.setenv("STT_PROVIDER", "openai")  # no prefix: must be ignored
    s = Settings()
    assert s.stt_provider == "mock"
    assert s.capture_live_chunk_seconds == 1.5
    assert s.resume_auto is False


def test_env_file_is_read_and_process_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "LST_STT_LANGUAGE=de\nLST_CAPTURE_LIVE_CHUNK_SECONDS=3\nLST_LOG_LEVEL=debug\n"
        "UNRELATED_VARIABLE=1\n"
    )
    monkeypatch.setenv("LST_CAPTURE_LIVE_CHUNK_SECONDS", "1.25")
    s = Settings.load(env_file)
    assert s.stt_language == "de"
    assert s.capture_live_chunk_seconds == 1.25  # process env beat the file
    assert s.log_level == "DEBUG"


def test_default_env_file_is_used_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / ".env").write_text("LST_STT_PROVIDER=none\n")
    monkeypatch.chdir(tmp_path)
    assert Settings.load().stt_provider == "none"


def test_missing_explicit_env_file_is_an_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="env file not found"):
        Settings.load(tmp_path / "nope.env")


def test_overrides_beat_everything_and_none_is_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LST_STT_PROVIDER", "mock")
    s = Settings.load(stt_provider="none", stt_language=None, out_dir=Path("elsewhere"))
    assert s.stt_provider == "none"
    assert s.stt_language is None
    assert s.out_dir == Path("elsewhere")
    assert Settings.load(stt_provider=None).stt_provider == "mock"


def test_with_overrides_returns_a_validated_copy():
    base = Settings(openai_api_key=OPENAI_SK)
    derived = base.with_overrides(stt_provider="openai", stt_language="DE")
    assert derived.stt_provider == "openai"
    assert derived.stt_language == "de"
    assert derived.api_key_for("openai") == OPENAI_SK
    assert base.stt_provider == "local"
    with pytest.raises(ConfigError):
        base.with_overrides(stt_workers=0)


@pytest.mark.parametrize("raw", ["", "auto", "AUTO", "none"])
def test_language_auto_and_blank_mean_autodetect(raw: str):
    assert Settings(stt_language=raw).stt_language is None


def test_blank_environment_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LST_STT_PROVIDER", "")
    monkeypatch.setenv("LST_CAPTURE_SAMPLE_RATE", "")
    s = Settings()
    assert s.stt_provider == "local"
    assert s.capture_sample_rate == 16000


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("stt_provider", "whisperx", "unknown provider"),
        ("stt_fallback", "gemini", "unknown provider"),
        ("stt_workers", 0, "greater than or equal to 1"),
        ("capture_live_chunk_seconds", 0, "greater than 0"),
        ("stt_max_drop_ratio", 1.5, "less than or equal to 1"),
        ("stt_overload_window_seconds", 5, "greater than or equal to 10"),
        ("openai_base_url", "ftp://example.com", "http"),
        ("log_level", "loud", "unknown log level"),
        ("notify_webhook_format", "teams", "json"),
    ],
)
def test_invalid_values_are_config_errors_naming_the_variable(
    field: str, value: object, fragment: str
):
    with pytest.raises(ConfigError, match=f"LST_{field.upper()}") as info:
        Settings.load(**{field: value})
    assert fragment in str(info.value)


def test_invalid_environment_value_is_reported_without_the_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LST_STT_WORKERS", "many-secret-words")
    with pytest.raises(ConfigError) as info:
        Settings.load()
    assert "LST_STT_WORKERS" in str(info.value)
    assert "many-secret-words" not in str(info.value)


def test_settings_are_immutable():
    with pytest.raises(ValueError, match="frozen"):
        Settings().stt_workers = 3  # type: ignore[misc]


# ---------------------------------------------------------- provider / keys


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_cloud_provider_without_key_is_a_clear_error(provider: str):
    s = Settings(stt_provider=provider)
    with pytest.raises(ConfigError) as info:
        s.validate_stt()
    assert f"LST_{provider.upper()}_API_KEY" in str(info.value)


def test_cloud_fallback_without_key_is_a_clear_error():
    s = Settings(stt_provider="local", stt_fallback="openai")
    with pytest.raises(ConfigError, match="LST_OPENAI_API_KEY"):
        s.validate_stt()


def test_fallback_must_differ_from_the_provider():
    with pytest.raises(ConfigError, match="must differ"):
        Settings(stt_provider="local", stt_fallback="local").validate_stt()


def test_key_for_the_wrong_vendor_is_rejected_on_the_public_api():
    with pytest.raises(ConfigError, match="OpenRouter key"):
        Settings(stt_provider="openai", openai_api_key=OPENROUTER_SK).validate_stt()
    with pytest.raises(ConfigError, match="OpenAI key"):
        Settings(stt_provider="openrouter", openrouter_api_key=OPENAI_SK).validate_stt()


def test_key_shape_is_not_checked_for_custom_endpoints():
    s = Settings(
        stt_provider="openai",
        openai_api_key=OPENROUTER_SK,
        openai_base_url="http://127.0.0.1:8000/v1",
    )
    s.validate_stt()


def test_matching_keys_validate():
    Settings(stt_provider="openai", openai_api_key=OPENAI_SK).validate_stt()
    Settings(stt_provider="openrouter", openrouter_api_key=OPENROUTER_SK).validate_stt()
    Settings(stt_provider="openai", openai_api_key=OPENAI_KEY).validate_stt()


def test_error_messages_never_contain_the_key():
    s = Settings(stt_provider="openai", openai_api_key=OPENROUTER_SK)
    with pytest.raises(ConfigError) as info:
        s.validate_stt()
    assert OPENROUTER_SK not in str(info.value)


def test_keys_are_read_from_lst_and_vendor_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_SK)
    assert Settings().api_key_for("openai") == OPENAI_SK
    monkeypatch.setenv("LST_OPENAI_API_KEY", OPENAI_SK + "-2")
    assert Settings().api_key_for("openai") == OPENAI_SK + "-2"
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_SK)
    assert Settings().api_key_for("openrouter") == OPENROUTER_SK
    assert Settings().api_key_for("local") is None


def test_base_urls_are_normalised():
    s = Settings(openai_base_url="http://localhost:8080/v1/")
    assert s.base_url_for("openai") == "http://localhost:8080/v1"
    assert s.base_url_for("openrouter") == "https://openrouter.ai/api/v1"
    assert s.base_url_for("local") is None


def test_model_for_applies_the_model_override_to_the_primary_provider_only():
    s = Settings(stt_provider="openai", stt_fallback="local", stt_model="gpt-4o-mini-transcribe")
    assert s.model_for() == "gpt-4o-mini-transcribe"
    assert s.model_for("openai") == "gpt-4o-mini-transcribe"
    assert s.model_for("local") == "small"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_key_family_of_a_missing_key(key: str | None):
    assert stt_key_family(key) is None


def test_key_family_classification():
    assert stt_key_family(OPENROUTER_SK) == "openrouter"
    assert stt_key_family(OPENAI_SK) == "openai"
    assert stt_key_family("something-else") == "unknown"


# ------------------------------------------------------------ redaction, paths


def test_redacted_summary_hides_every_secret():
    s = Settings(
        openai_api_key=OPENAI_SK,
        capture_proxy="http://user:proxy-password@proxy.example.com:3128",
        notify_webhook_url="https://hooks.example.com/services/webhook-secret",
        notify_webhook_token="webhook-bearer-token",
        notify_telegram_token="123456:your-telegram-bot-token",
        notify_telegram_chat_id="12345",
    )
    summary = s.redacted_summary()
    assert summary["openai_api_key"] == "set"
    assert summary["openrouter_api_key"] == "unset"
    assert summary["capture_proxy"] == "set"
    assert summary["notify_webhook_url"] == "set"
    blob = repr(summary)
    for secret in (
        OPENAI_SK,
        "proxy-password",
        "webhook-secret",
        "webhook-bearer-token",
        "your-telegram-bot-token",
    ):
        assert secret not in blob
    assert summary["notify_telegram_chat_id"] == "set"
    assert "12345" not in blob
    assert summary["stt_provider"] == "local"
    assert summary["out_dir"] == "out"
    assert set(summary) == set(Settings.model_fields)


def test_redacted_summary_hides_the_home_directory(tmp_path: Path):
    summary = Settings().redacted_summary()
    assert summary["stt_models_dir"] == "~/.cache/livestream-transcriber/models"
    assert str(Path.home()) not in repr(summary)
    elsewhere = Settings(out_dir=tmp_path / "out").redacted_summary()
    assert elsewhere["out_dir"] == str(tmp_path / "out")


def test_repr_and_dump_do_not_expose_secrets():
    s = Settings(openai_api_key=OPENAI_SK)
    assert OPENAI_SK not in repr(s)
    assert OPENAI_SK not in str(s.model_dump_json())


def test_database_path_defaults_to_the_output_directory(tmp_path: Path):
    assert Settings(out_dir=tmp_path).database_path == tmp_path / "lst.db"
    explicit = tmp_path / "elsewhere" / "x.db"
    assert Settings(out_dir=tmp_path, db_path=explicit).database_path == explicit


def test_ensure_dirs_creates_output_and_database_directories(tmp_path: Path):
    s = Settings(out_dir=tmp_path / "out", db_path=tmp_path / "db" / "x.db")
    s.ensure_dirs()
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "db").is_dir()
