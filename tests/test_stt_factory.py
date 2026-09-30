"""Provider registry, wrapper assembly and loud startup failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from livestream_transcriber.config import STT_PROVIDERS, ConfigError, Settings
from livestream_transcriber.netutil import HttpError
from livestream_transcriber.stt.base import FixtureTranscriber, MockTranscriber, NullTranscriber
from livestream_transcriber.stt.factory import (
    PROVIDER_BUILDERS,
    build_transcriber,
    close_transcriber,
    provider_problem,
    set_fallback_enabled,
    transcriber_health,
)
from livestream_transcriber.stt.providers.faster_whisper import FasterWhisperTranscriber
from livestream_transcriber.stt.providers.openai import OpenAITranscriber
from livestream_transcriber.stt.providers.openrouter import OpenRouterTranscriber
from livestream_transcriber.stt.providers.sherpa_onnx import (
    ONNX_MODELS,
    SherpaOnnxTranscriber,
    release_recognizers,
)
from livestream_transcriber.stt.wrappers import (
    BudgetGuardTranscriber,
    CachedTranscriber,
    FallbackSttTranscriber,
    SttSpend,
)
from tests.support.audio import RATE, tone

OPENAI_KEY = "your-openai-key"
OPENROUTER_KEY = "your-openrouter-key"


def settings(**kwargs) -> Settings:
    kwargs.setdefault("openai_api_key", OPENAI_KEY)
    kwargs.setdefault("openrouter_api_key", OPENROUTER_KEY)
    return Settings(**kwargs)


@pytest.fixture
def fake_local(monkeypatch):
    """Pretend both optional extras are installed."""
    import importlib.util as util

    real = util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in {"faster_whisper", "sherpa_onnx"}:
            return object()
        return real(name, *args, **kwargs)

    monkeypatch.setattr("livestream_transcriber.stt.factory.importlib.util.find_spec", find_spec)


def test_the_registry_covers_every_configurable_provider():
    assert set(PROVIDER_BUILDERS) == set(STT_PROVIDERS)


def test_stand_ins_need_nothing():
    assert isinstance(build_transcriber(Settings(stt_provider="none")), NullTranscriber)
    assert isinstance(build_transcriber(Settings(stt_provider="mock")), MockTranscriber)


def test_mock_transcribes_numbered_text_so_chunks_stay_distinct():
    stt = build_transcriber(Settings(stt_provider="mock"))
    texts = [stt.transcribe(b"x", RATE, start=i, end=i + 1).text for i in range(2)]
    assert texts == ["mock transcript 1", "mock transcript 2"]


def test_mock_with_fixtures_replays_them(tmp_path: Path):
    fixtures = tmp_path / "speech.jsonl"
    fixtures.write_text('{"start": 0, "end": 5, "text": "from the fixture"}\n', encoding="utf-8")
    stt = build_transcriber(Settings(stt_provider="mock"), fixtures=fixtures)
    assert isinstance(stt, FixtureTranscriber)


def test_stand_ins_are_not_wrapped(tmp_path: Path):
    stt = build_transcriber(
        Settings(stt_provider="mock", stt_budget_usd=1.0), cache_dir=tmp_path, fallback="local"
    )
    assert isinstance(stt, MockTranscriber)


# ------------------------------------------------------------------- cloud --


def test_openai_uses_settings_for_key_model_url_and_language():
    cfg = settings(
        stt_provider="openai", stt_model="gpt-4o-mini-transcribe", stt_language="de",
        openai_base_url="http://localhost:8000/v1", stt_word_timestamps=True,
        stt_timeout_seconds=12,
    )  # fmt: skip
    stt = build_transcriber(cfg)
    assert isinstance(stt, OpenAITranscriber)
    assert (stt.model, stt.language, stt.base_url) == (
        "gpt-4o-mini-transcribe", "de", "http://localhost:8000/v1",
    )  # fmt: skip
    assert stt.api_key == OPENAI_KEY and stt.timeout == 12 and stt.word_timestamps
    assert stt.breaker is not None and stt.breaker.failure_threshold == cfg.stt_breaker_failures


def test_openrouter_defaults_and_auto_language():
    stt = build_transcriber(settings(stt_provider="openrouter"))
    assert isinstance(stt, OpenRouterTranscriber)
    assert stt.model == "openai/whisper-large-v3"
    assert stt.language is None
    assert stt.base_url == "https://openrouter.ai/api/v1"


def test_a_missing_key_fails_loudly_with_the_variable_name():
    with pytest.raises(ConfigError, match="LST_OPENAI_API_KEY"):
        build_transcriber(Settings(stt_provider="openai"))
    with pytest.raises(ConfigError, match="LST_OPENROUTER_API_KEY"):
        build_transcriber(Settings(stt_provider="openrouter"))


def test_the_provider_override_builds_another_provider_with_its_own_default_model():
    cfg = settings(stt_provider="openai", stt_model="gpt-4o-transcribe")
    stt = build_transcriber(cfg, provider="openrouter")
    assert isinstance(stt, OpenRouterTranscriber)
    assert stt.model == "openai/whisper-large-v3"  # stt_model belongs to the primary only
    explicit = build_transcriber(cfg, provider="openrouter", model="some/model")
    assert explicit.model == "some/model"


# ------------------------------------------------------------- local + onnx --


def test_local_builds_lazily_from_settings(fake_local, tmp_path):
    cfg = Settings(
        stt_provider="local", stt_language="de", stt_device="cpu", stt_compute_type="int8",
        stt_threads=2, stt_vad_filter=False, stt_word_timestamps=True, stt_models_dir=tmp_path,
        stt_nice=5,
    )  # fmt: skip
    stt = build_transcriber(cfg)
    assert isinstance(stt, FasterWhisperTranscriber)
    assert (stt.model_name, stt.language, stt.threads, stt.nice) == ("small", "de", 2, 5)
    assert (stt.vad_filter, stt.word_timestamps, stt.models_dir) == (False, True, tmp_path)
    assert stt.describe()["loaded"] is False  # nothing is loaded at build time


def test_a_missing_local_extra_fails_loudly(monkeypatch):
    monkeypatch.setattr(
        "livestream_transcriber.stt.factory.importlib.util.find_spec", lambda name: None
    )
    with pytest.raises(ConfigError, match=r"livestream-transcriber\[local\]"):
        build_transcriber(Settings(stt_provider="local"))


def test_onnx_needs_the_extra_and_the_model_files(fake_local, tmp_path, monkeypatch):
    cfg = Settings(stt_provider="onnx", stt_models_dir=tmp_path)
    with pytest.raises(ConfigError, match=r"lst models fetch --model parakeet-tdt-0\.6b-v3"):
        build_transcriber(cfg)
    base = tmp_path / "parakeet-tdt-0.6b-v3"
    base.mkdir()
    for name in ONNX_MODELS["parakeet-tdt-0.6b-v3"].files:
        (base / name).write_bytes(b"x")
    stt = build_transcriber(cfg)
    assert isinstance(stt, SherpaOnnxTranscriber)
    monkeypatch.setattr(
        "livestream_transcriber.stt.factory.importlib.util.find_spec", lambda name: None
    )
    with pytest.raises(ConfigError, match=r"livestream-transcriber\[onnx\]"):
        build_transcriber(cfg)
    release_recognizers()


def test_an_unknown_onnx_model_is_reported(fake_local):
    problem = provider_problem("onnx", Settings(stt_provider="onnx", stt_model="whisper-tiny"))
    assert problem is not None and "unknown onnx model" in problem


def test_provider_problem_is_a_pure_check(fake_local):
    assert provider_problem("none", Settings()) is None
    assert provider_problem("local", Settings()) is None
    assert provider_problem("openai", Settings()) == (
        "an API key is required: set LST_OPENAI_API_KEY"
    )
    assert provider_problem("openai", settings()) is None
    assert "unknown STT provider" in provider_problem("carrier-pigeon", Settings())


# ------------------------------------------------------------------ wrappers --


def test_wrapper_order_is_provider_budget_fallback_cache(fake_local, tmp_path):
    cfg = settings(stt_provider="openai", stt_fallback="local", stt_budget_usd=1.0)
    spend = SttSpend()
    stt = build_transcriber(cfg, cache_dir=tmp_path, budget_spend=spend)
    assert isinstance(stt, CachedTranscriber)
    assert isinstance(stt.inner, FallbackSttTranscriber)
    assert stt.inner.provider_name == "local"
    guard = stt.inner.primary
    assert isinstance(guard, BudgetGuardTranscriber) and guard.spend is spend
    assert isinstance(guard.inner, OpenAITranscriber)
    assert isinstance(stt.inner.fallback, FasterWhisperTranscriber)
    assert (stt.provider, stt.model) == ("openai", "whisper-1")


def test_the_cache_tag_carries_language_and_word_timestamps(tmp_path):
    cfg = settings(stt_provider="openai", stt_language="de", stt_word_timestamps=True)
    assert build_transcriber(cfg, cache_dir=tmp_path).param_tag == "de-words"
    assert (
        build_transcriber(settings(stt_provider="openai"), cache_dir=tmp_path).param_tag == "auto"
    )


def test_no_budget_means_no_guard():
    stt = build_transcriber(settings(stt_provider="openai"))
    assert isinstance(stt, OpenAITranscriber)


def test_a_local_primary_is_never_budgeted(fake_local):
    stt = build_transcriber(Settings(stt_provider="local", stt_budget_usd=1.0))
    assert isinstance(stt, FasterWhisperTranscriber)


def test_the_fallback_provider_gets_its_own_default_model(fake_local):
    cfg = settings(stt_provider="openai", stt_model="gpt-4o-transcribe", stt_fallback="local")
    stt = build_transcriber(cfg)
    assert isinstance(stt, FallbackSttTranscriber)
    assert stt.primary.model == "gpt-4o-transcribe"
    assert stt.fallback.model_name == "small"


def test_the_fallback_can_be_switched_off_per_call():
    cfg = settings(stt_provider="openai", stt_fallback="local")
    assert isinstance(build_transcriber(cfg, fallback="none"), OpenAITranscriber)


def test_a_fallback_with_a_problem_fails_at_startup(monkeypatch):
    monkeypatch.setattr(
        "livestream_transcriber.stt.factory.importlib.util.find_spec", lambda name: None
    )
    with pytest.raises(ConfigError, match="'local' cannot start"):
        build_transcriber(settings(stt_provider="openai", stt_fallback="local"))


def test_end_to_end_outage_falls_back_and_reports_health(fake_local, monkeypatch):
    def down(url, fields, files, *, headers=None, timeout=60.0):
        raise HttpError(503, "upstream down")

    monkeypatch.setattr("livestream_transcriber.stt.providers.openai.post_multipart", down)
    cfg = settings(stt_provider="openai", stt_fallback="local", stt_budget_usd=5.0)
    stt = build_transcriber(cfg)
    assert isinstance(stt, FallbackSttTranscriber)
    stt.fallback = MockTranscriber("local words")  # keep the real model out of the test
    for i in range(6):
        got = stt.transcribe(tone(), RATE, start=float(i), end=float(i) + 1.0)
        assert got is not None and got.text == "local words" and got.degraded
    health = transcriber_health(stt)
    assert health["breakers"]["whisper-1"]["state"] == "open"
    assert health["fallback"]["hits"] == 6
    assert health["budget"]["tripped"] is False
    assert health["providers"][0]["failures"] == 3  # the breaker stopped the rest
    assert set_fallback_enabled(stt, False) is True
    assert transcriber_health(stt)["fallback"]["enabled"] is False
    assert set_fallback_enabled(MockTranscriber(), False) is False


def test_health_of_a_bare_transcriber_is_empty_not_an_error():
    health = transcriber_health(MockTranscriber())
    assert health["breakers"] == {} and health["fallback"] is None
    assert health["cloud_successes"] == 0 and health["cost_usd"] == 0.0


def test_close_reaches_every_node_and_never_raises():
    closed: list[str] = []

    class Closable:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name, self.fail = name, fail

        def transcribe(self, *args, **kwargs):
            return None

        def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("boom")

    chain = FallbackSttTranscriber(Closable("primary", fail=True), Closable("backup"))
    close_transcriber(chain)
    assert sorted(closed) == ["backup", "primary"]
