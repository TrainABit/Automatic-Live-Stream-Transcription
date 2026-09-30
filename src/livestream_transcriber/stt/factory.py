"""Build the configured STT chain and describe or steer a built one.

One registry maps a provider name (``local``, ``onnx``, ``openai``,
``openrouter``, ``mock``, ``none``) to a builder. Building fails loudly with an
actionable message when the provider cannot work (missing optional extra,
missing key, missing model files): a run that silently produced no
transcripts for an hour is worse than one that refuses to start.

The chain is assembled innermost first: provider, budget guard (paid providers
with ``stt_budget_usd``), cloud-to-fallback wrapper (``stt_fallback``), disk
cache (when a cache directory is given).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import CLOUD_STT_PROVIDERS, STT_PROVIDERS, ConfigError, Settings
from ..logging_setup import get_logger
from .base import (
    CircuitBreaker,
    FixtureTranscriber,
    MockTranscriber,
    NullTranscriber,
    Transcriber,
)
from .providers import sherpa_onnx
from .providers._cloud import CloudTranscriber
from .providers.faster_whisper import FasterWhisperTranscriber
from .providers.openai import OpenAITranscriber
from .providers.openrouter import OpenRouterTranscriber
from .wrappers import (
    BudgetGuardTranscriber,
    CachedTranscriber,
    FallbackSttTranscriber,
    SttSpend,
    iter_chain,
)

__all__ = [
    "PROVIDER_BUILDERS",
    "build_transcriber",
    "close_transcriber",
    "provider_problem",
    "set_fallback_enabled",
    "transcriber_health",
]

log = get_logger(__name__)

MOCK_TEXT = "mock transcript {n}"

#: Providers that are stand-ins: no wrappers, no key, no extra.
_STAND_INS = frozenset({"mock", "none"})


def provider_problem(name: str, settings: Settings) -> str | None:
    """Why ``name`` cannot run with ``settings``, or ``None`` when it can.

    Cheap and side-effect free (no model load, no network): ``lst doctor`` uses it
    to report every provider's readiness, and :func:`build_transcriber` uses it
    to refuse to start.
    """
    if name in _STAND_INS:
        return None
    if name == "local":
        if importlib.util.find_spec("faster_whisper") is None:
            return "faster-whisper is not installed: pip install 'livestream-transcriber[local]'"
        return None
    if name == "onnx":
        if importlib.util.find_spec("sherpa_onnx") is None:
            return "sherpa-onnx is not installed: pip install 'livestream-transcriber[onnx]'"
        model = settings.model_for("onnx")
        if model not in sherpa_onnx.ONNX_MODELS:
            return (
                f"unknown onnx model {model!r}; known: {', '.join(sorted(sherpa_onnx.ONNX_MODELS))}"
            )
        if not sherpa_onnx.model_present(model, settings.stt_models_dir):
            return f"onnx model files are missing: run `lst models fetch --model {model}`"
        return None
    if name in CLOUD_STT_PROVIDERS:
        if not settings.api_key_for(name):
            return f"an API key is required: set LST_{name.upper()}_API_KEY"
        return None
    return f"unknown STT provider {name!r}; choose one of {', '.join(STT_PROVIDERS)}"


class _Context:
    """What a provider builder needs, resolved once from the settings."""

    def __init__(
        self,
        settings: Settings,
        provider: str,
        model: str | None,
        fixtures: str | Path | None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.model = model or settings.model_for(provider)
        self.language = settings.stt_language
        self.fixtures = fixtures

    def breaker(self) -> CircuitBreaker:
        s = self.settings
        return CircuitBreaker(
            self.model,
            failure_threshold=s.stt_breaker_failures,
            cooldown_seconds=s.stt_breaker_cooldown_seconds,
        )

    def cloud_kwargs(self) -> dict[str, Any]:
        s = self.settings
        return {
            "model": self.model,
            "base_url": s.base_url_for(self.provider),
            "language": self.language,
            "timeout": s.stt_timeout_seconds,
            "word_timestamps": s.stt_word_timestamps,
            "breaker": self.breaker(),
        }


def _build_local(ctx: _Context) -> Transcriber:
    s = ctx.settings
    return FasterWhisperTranscriber(
        model=ctx.model,
        device=s.stt_device,
        compute_type=s.stt_compute_type,
        threads=s.stt_threads,
        language=ctx.language,
        vad_filter=s.stt_vad_filter,
        word_timestamps=s.stt_word_timestamps,
        models_dir=s.stt_models_dir,
        nice=s.stt_nice,
    )


def _build_onnx(ctx: _Context) -> Transcriber:
    s = ctx.settings
    return sherpa_onnx.SherpaOnnxTranscriber(
        ctx.model,
        model_root=s.stt_models_dir,
        threads=max(1, s.stt_threads),
        language=ctx.language,
        word_timestamps=s.stt_word_timestamps,
        nice=s.stt_nice,
    )


def _build_openai(ctx: _Context) -> Transcriber:
    return OpenAITranscriber(ctx.settings.api_key_for("openai") or "", **ctx.cloud_kwargs())


def _build_openrouter(ctx: _Context) -> Transcriber:
    return OpenRouterTranscriber(ctx.settings.api_key_for("openrouter") or "", **ctx.cloud_kwargs())


def _build_mock(ctx: _Context) -> Transcriber:
    if ctx.fixtures:
        try:
            return FixtureTranscriber.from_jsonl(ctx.fixtures)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot load the mock fixtures: {exc}") from exc
    return MockTranscriber(MOCK_TEXT)


def _build_none(ctx: _Context) -> Transcriber:
    return NullTranscriber()


#: The provider registry: name -> builder. Extend it to add a provider.
PROVIDER_BUILDERS: dict[str, Callable[[_Context], Transcriber]] = {
    "local": _build_local,
    "onnx": _build_onnx,
    "openai": _build_openai,
    "openrouter": _build_openrouter,
    "mock": _build_mock,
    "none": _build_none,
}


def _build_provider(
    settings: Settings, name: str, model: str | None, fixtures: str | Path | None
) -> Transcriber:
    problem = provider_problem(name, settings)
    if problem is not None:
        raise ConfigError(f"STT provider {name!r} cannot start: {problem}")
    try:
        builder = PROVIDER_BUILDERS[name]
    except KeyError:
        raise ConfigError(f"unknown STT provider {name!r}") from None
    return builder(_Context(settings, name, model, fixtures))


def build_transcriber(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    fallback: str | None = None,
    cache_dir: str | Path | None = None,
    fixtures: str | Path | None = None,
    budget_spend: SttSpend | None = None,
    on_budget_exceeded: Callable[[float], None] | None = None,
) -> Transcriber:
    """Build the STT chain for ``settings``.

    ``provider``, ``model`` and ``fallback`` override ``stt_provider``,
    ``stt_model`` and ``stt_fallback`` (the benchmark uses this to build several
    providers from one configuration; pass ``fallback="none"`` to build a bare
    provider). ``fixtures`` is a JSONL file for the ``mock`` provider.
    ``budget_spend`` is the process-wide :class:`SttSpend` to meter paid calls
    against, so a rebuilt session keeps counting. Raises :class:`ConfigError`
    when a provider cannot start.
    """
    name = (provider or settings.stt_provider).strip().lower()
    inner = _build_provider(settings, name, model, fixtures)
    if name in _STAND_INS:
        return inner

    if settings.stt_budget_usd is not None and name in CLOUD_STT_PROVIDERS:
        inner = BudgetGuardTranscriber(
            inner,
            settings.stt_budget_usd,
            on_exceeded=on_budget_exceeded,
            spend=budget_spend,
        )

    fallback_name = (fallback or settings.stt_fallback).strip().lower()
    if fallback_name not in ("none", name):
        backup = _build_provider(settings, fallback_name, None, fixtures)
        inner = FallbackSttTranscriber(inner, backup, provider_name=fallback_name)

    if cache_dir is not None:
        tag = settings.stt_language or "auto"
        if settings.stt_word_timestamps:
            tag += "-words"
        inner = CachedTranscriber(
            inner,
            cache_dir,
            provider=name,
            model=model or settings.model_for(name),
            param_tag=tag,
        )
    return inner


def set_fallback_enabled(transcriber: Any, enabled: bool) -> bool:
    """Switch every fallback wrapper in the chain on or off.

    Returns True when the chain has one. The overload guard uses it to stop a
    fallback that cannot keep up from starving the rest of the pipeline.
    """
    found = False
    for node in iter_chain(transcriber):
        if isinstance(node, FallbackSttTranscriber):
            node.enabled = bool(enabled)
            found = True
    return found


def close_transcriber(transcriber: Any) -> None:
    """Call ``close()`` on every transcriber in the chain that has one. Never raises."""
    for node in iter_chain(transcriber):
        closer = getattr(node, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                log.exception("closing an stt provider failed")


def transcriber_health(transcriber: Any) -> dict[str, Any]:
    """Operational STT state for health reports. Never raises, never blocks.

    ``providers`` lists every provider in the chain with its counters,
    ``breakers`` maps each cloud model to its circuit state, ``fallback`` and
    ``budget`` are ``None`` when the chain has no such wrapper.
    """
    out: dict[str, Any] = {
        "providers": [],
        "breakers": {},
        "cloud_successes": 0,
        "cloud_last_success_age_s": None,
        "cost_usd": 0.0,
        "fallback": None,
        "budget": None,
        "cache": None,
    }
    for node in iter_chain(transcriber):
        if isinstance(node, CloudTranscriber):
            state = node.describe()
            out["providers"].append(state)
            out["cloud_successes"] += int(state["successes"])
            out["cost_usd"] += float(state["cost_usd"])
            age = state["last_success_age_s"]
            best = out["cloud_last_success_age_s"]
            if age is not None and (best is None or age < best):
                out["cloud_last_success_age_s"] = age
            if node.breaker is not None:
                out["breakers"][node.model] = node.breaker.describe()
        elif isinstance(node, FallbackSttTranscriber):
            out["fallback"] = node.describe()
        elif isinstance(node, BudgetGuardTranscriber):
            out["budget"] = node.describe()
        elif isinstance(node, CachedTranscriber):
            out["cache"] = node.describe()
        elif callable(getattr(node, "describe", None)):
            out["providers"].append(node.describe())
    out["cost_usd"] = round(out["cost_usd"], 6)
    return out
