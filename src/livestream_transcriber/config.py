"""Runtime configuration, loaded from the environment and an optional ``.env`` file.

Precedence, highest first: explicit overrides (command line flags), process
environment, ``.env`` file, defaults. Every setting has the prefix ``LST_``:
``stt_provider`` is ``LST_STT_PROVIDER``. The two API keys additionally accept
the vendor's conventional name (``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``) so
an existing shell setup keeps working.

Field names are grouped by prefix (``capture_*``, ``stt_*``, ``resume_*``,
``notify_*``, ``rules_*``) rather than nested models, which keeps the
environment variable names flat and predictable.

Loading never requires credentials: ``lst doctor`` and ``lst rules test`` must
work without a key. Commands that transcribe call
:meth:`Settings.validate_stt` first, so a missing key or a provider/key
mismatch is a startup error instead of an hour of silence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .redact import redact_text

__all__ = [
    "CLOUD_STT_PROVIDERS",
    "DEFAULT_ENV_FILE",
    "DEFAULT_STT_MODELS",
    "STT_PROVIDERS",
    "ConfigError",
    "Settings",
    "stt_key_family",
]

DEFAULT_ENV_FILE = ".env"

#: Provider names accepted by ``stt_provider`` / ``stt_fallback`` (``none`` disables speech).
STT_PROVIDERS = ("local", "onnx", "openai", "openrouter", "mock", "none")
#: Providers that need an API key.
CLOUD_STT_PROVIDERS = frozenset({"openai", "openrouter"})

#: Model used when ``stt_model`` is not set. Every one of them can be overridden.
DEFAULT_STT_MODELS: dict[str, str] = {
    "local": "small",
    "onnx": "parakeet-tdt-0.6b-v3",
    "openai": "whisper-1",
    "openrouter": "openai/whisper-large-v3",
    "mock": "mock",
}

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_LOG_LEVELS = frozenset(logging.getLevelNamesMapping()) - {"NOTSET", "WARN", "FATAL"}


class ConfigError(ValueError):
    """The configuration cannot work. The message never contains a secret value."""


def stt_key_family(api_key: str | None) -> str | None:
    """Classify an API key by prefix: ``"openrouter"``, ``"openai"``, ``"unknown"``.

    Returns ``None`` for a missing or blank key. Never returns the key itself.
    """
    key = (api_key or "").strip()
    if not key:
        return None
    if key.split("-")[:2] == ["sk", "or"]:
        return "openrouter"
    if key.startswith("sk-"):
        return "openai"
    return "unknown"


def _secret(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    text = value.get_secret_value().strip()
    return text or None


class Settings(BaseSettings):
    """All tunables of the transcriber. Immutable once loaded."""

    model_config = SettingsConfigDict(
        env_prefix="LST_",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    # -- capture ---------------------------------------------------------------
    capture_sample_rate: int = Field(16000, ge=8000, le=48000)
    """Mono PCM sample rate handed to the STT providers."""
    capture_live_chunk_seconds: float = Field(2.5, gt=0)
    """Chunk length for live sources: short, to keep end-to-end latency low."""
    capture_file_chunk_seconds: float = Field(5.0, gt=0)
    """Chunk length for files and recordings: longer, since latency is irrelevant."""
    capture_queue_size: int = Field(32, ge=1)
    """Live capture queue depth; when full the oldest chunk is dropped."""
    capture_file_queue_size: int = Field(8, ge=1)
    """File capture queue depth; the producer waits instead of dropping."""
    capture_reconnect_initial_delay: float = Field(1.0, ge=0.1)
    capture_reconnect_max_delay: float = Field(30.0, ge=0.1)
    capture_max_reconnect_attempts: int = Field(0, ge=0)
    """Consecutive reconnect attempts before giving up; 0 means never give up."""
    capture_resolve_cache_seconds: float = Field(20.0, ge=0)
    capture_audio_stall_seconds: float = Field(30.0, ge=0)
    """Restart the capture when no audio arrives for this long; 0 disables."""
    capture_stale_playlist_stalls: int = Field(2, ge=0)
    capture_stream_format: str = "bestaudio/best"
    """yt-dlp format selector for resolved sources."""
    capture_ffmpeg_binary: str = "ffmpeg"
    capture_ffmpeg_loglevel: str = "warning"
    capture_cookies_file: Path | None = None
    """Netscape cookie file passed to yt-dlp (age-gated or bot-checked sources)."""
    capture_proxy: SecretStr | None = None
    """Proxy URL for yt-dlp; may embed credentials, hence a secret."""
    capture_hls_window: bool = False
    """Opt in to reading HLS playlists ourselves for a controlled live window."""
    capture_hls_live_start_index: int = -3
    """Segment to start from, relative to the live edge, when the window is used."""

    # -- speech to text ----------------------------------------------------------
    stt_provider: str = "local"
    stt_model: str | None = None
    """Model name; ``None`` selects the provider default (see ``DEFAULT_STT_MODELS``)."""
    stt_language: str | None = None
    """ISO language code, or ``None`` to auto-detect."""
    stt_fallback: str = "none"
    """Provider used when the primary keeps failing (typically ``local``)."""
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_threads: int = Field(4, ge=0)
    """CPU threads used by local inference; raise it on a multi-core machine."""
    stt_nice: int = Field(0, ge=0, le=19)
    """Niceness applied to local inference threads; 0 leaves priority alone."""
    stt_word_timestamps: bool = False
    stt_vad_filter: bool = True
    stt_models_dir: Path = Path.home() / ".cache" / "livestream-transcriber" / "models"
    """Where downloaded ONNX models live (see ``lst models fetch``)."""
    stt_cache_dir: Path | None = None
    """Cache transcripts of identical audio here, so replays and benchmarks of the same
    recording do not pay for the same chunk twice. Off when unset."""
    stt_workers: int = Field(1, ge=1)
    stt_queue_size: int = Field(8, ge=1)
    """Chunks buffered in RAM ahead of the STT workers."""
    stt_spill_chunks: int = Field(240, ge=0)
    """Chunks that may spill to disk beyond the RAM queue; 0 disables spilling."""
    stt_budget_usd: float | None = Field(None, ge=0)
    """Stop using a cloud provider once its estimated spend reaches this; ``None`` is unlimited."""
    stt_timeout_seconds: float = Field(60.0, gt=0)
    stt_breaker_failures: int = Field(3, ge=1)
    stt_breaker_cooldown_seconds: float = Field(30.0, ge=1)
    stt_max_lag_seconds: float = Field(45.0, ge=0)
    """Overload guard: pause STT when it falls this far behind; 0 disables."""
    stt_max_drop_ratio: float = Field(0.25, ge=0, le=1)
    stt_overload_window_seconds: float = Field(120.0, ge=10)
    stt_recovery_probe_seconds: float = Field(60.0, ge=5)
    stt_outage_seconds: float = Field(300.0, ge=60)
    """Consecutive failure time after which an STT outage is reported."""
    stt_health_lag_seconds: float = Field(30.0, ge=0)
    stt_health_drop_ratio: float = Field(0.05, ge=0, le=1)

    openai_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("LST_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str = Field(
        OPENAI_DEFAULT_BASE_URL,
        validation_alias=AliasChoices("LST_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )
    """Point this at any OpenAI-compatible transcription server."""
    openrouter_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("LST_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = OPENROUTER_DEFAULT_BASE_URL

    # -- resuming after a stream ends -------------------------------------------
    resume_auto: bool = True
    """Keep probing after a stream ends or is offline, and resume when it returns."""
    resume_probe_interval_seconds: float = Field(30.0, ge=5)
    resume_backoff_max_seconds: float = Field(900.0, ge=5)
    resume_online_stable_seconds: float = Field(15.0, ge=0)
    """How long a source must stay up before it counts as live again."""
    resume_end_settle_seconds: float = Field(30.0, ge=0)
    """Grace period after the media ends, to drain in-flight audio and transcripts."""
    resume_uncertain_hold_seconds: float = Field(90.0, ge=10)
    """How long to hold when the source state is ambiguous before deciding."""

    # -- notifications -------------------------------------------------------------
    notify_webhook_url: SecretStr | None = None
    """Webhook URL. Many webhook URLs embed a token, hence a secret."""
    notify_webhook_format: Literal["json", "slack", "discord"] = "json"
    notify_webhook_token: SecretStr | None = None
    """Optional bearer token sent as ``Authorization`` with webhook requests."""
    notify_telegram_token: SecretStr | None = None
    notify_telegram_chat_id: str | None = None
    notify_timeout_seconds: float = Field(10.0, gt=0)

    # -- rules ---------------------------------------------------------------------
    rules_file: Path | None = None
    rules_llm_base_url: str | None = None
    """OpenAI-compatible chat endpoint for ``llm`` rules; unset disables them."""
    rules_llm_model: str | None = None
    rules_llm_api_key: SecretStr | None = None
    rules_llm_timeout_seconds: float = Field(20.0, gt=0)

    # -- paths and process -----------------------------------------------------------
    out_dir: Path = Path("out")
    recordings_dir: Path = Path("recordings")
    db_path: Path | None = None
    """SQLite database; ``None`` means ``<out_dir>/lst.db``."""
    heartbeat_seconds: float = Field(60.0, ge=0)
    """Interval of the periodic health line; 0 disables it."""
    memory_limit_mb: float = Field(0.0, ge=0)
    """Soft RSS limit that triggers back-pressure warnings; 0 disables."""
    log_level: str = "INFO"

    # ---------------------------------------------------------------- validators

    @field_validator("stt_provider", "stt_fallback")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        name = value.strip().lower()
        if name not in STT_PROVIDERS:
            raise ValueError(
                f"unknown provider {value!r}; choose one of {', '.join(STT_PROVIDERS)}"
            )
        return name

    @field_validator("stt_language", "stt_model", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "auto", "none"}:
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("stt_language")
    @classmethod
    def _language_lower(cls, value: str | None) -> str | None:
        return value.lower() if value else None

    @field_validator("log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"unknown log level {value!r}; choose one of {sorted(_LOG_LEVELS)}")
        return level

    @field_validator("openai_base_url", "openrouter_base_url", "rules_llm_base_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = value.strip().rstrip("/")
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return url

    @field_validator("notify_webhook_url")
    @classmethod
    def _webhook_url(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().lower().startswith(
            ("http://", "https://")
        ):
            raise ValueError("must start with http:// or https://")
        return value

    # ------------------------------------------------------------------- loading

    @classmethod
    def load(cls, env_file: str | Path | None = None, **overrides: Any) -> Settings:
        """Build settings from env, ``.env`` and overrides; raise :class:`ConfigError`.

        ``env_file=None`` uses ``./.env`` when it exists. An explicitly named
        file must exist. ``None`` values in ``overrides`` are ignored, so a
        command line parser can pass every flag whether or not it was given.
        """
        if env_file is not None and not Path(env_file).is_file():
            raise ConfigError(f"env file not found: {env_file}")
        source: Path | None = Path(env_file) if env_file is not None else None
        if source is None and Path(DEFAULT_ENV_FILE).is_file():
            source = Path(DEFAULT_ENV_FILE)
        given = {k: v for k, v in overrides.items() if v is not None}
        try:
            return cls(_env_file=source, **given)
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc)) from exc

    def with_overrides(self, **overrides: Any) -> Settings:
        """A validated copy with some fields replaced (``None`` values are ignored)."""
        changes = {k: v for k, v in overrides.items() if v is not None}
        try:
            return type(self).model_validate({**self.model_dump(), **changes})
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc)) from exc

    # ------------------------------------------------------------------- helpers

    @property
    def database_path(self) -> Path:
        """The SQLite path to use: ``db_path`` or ``<out_dir>/lst.db``."""
        return self.db_path if self.db_path is not None else self.out_dir / "lst.db"

    def ensure_dirs(self) -> None:
        """Create the output and database directories (recordings are created on demand)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def api_key_for(self, provider: str) -> str | None:
        """The API key for a cloud provider as plain text, or ``None``."""
        if provider == "openai":
            return _secret(self.openai_api_key)
        if provider == "openrouter":
            return _secret(self.openrouter_api_key)
        return None

    def base_url_for(self, provider: str) -> str | None:
        """The API base URL for a cloud provider, or ``None`` for local providers."""
        if provider == "openai":
            return self.openai_base_url
        if provider == "openrouter":
            return self.openrouter_base_url
        return None

    def model_for(self, provider: str | None = None) -> str:
        """The model to use for ``provider`` (default: the primary provider).

        ``stt_model`` applies to the primary provider only; a fallback provider
        gets its own default, since model names are not portable between them.
        """
        name = provider or self.stt_provider
        if name == self.stt_provider and self.stt_model:
            return self.stt_model
        return DEFAULT_STT_MODELS.get(name, "")

    def validate_stt(self) -> None:
        """Fail fast when the chosen provider (or fallback) cannot work.

        A missing key or a key that belongs to another vendor is a
        configuration error, not "no speech detected". Key values are never
        included in the message.
        """
        if self.stt_fallback == self.stt_provider and self.stt_provider != "none":
            raise ConfigError("LST_STT_FALLBACK must differ from LST_STT_PROVIDER")
        for name in (self.stt_provider, self.stt_fallback):
            self._validate_provider_key(name)

    def _validate_provider_key(self, name: str) -> None:
        if name not in CLOUD_STT_PROVIDERS:
            return
        var = f"LST_{name.upper()}_API_KEY"
        key = self.api_key_for(name)
        family = stt_key_family(key)
        if family is None:
            raise ConfigError(
                f"STT provider {name!r} needs an API key: set {var}. "
                "Speech is not disabled; this is a configuration error. "
                "Use LST_STT_PROVIDER=local for a key-free run."
            )
        if name == "openai" and family == "openrouter" and self._default_url("openai"):
            raise ConfigError(
                "STT provider 'openai' is configured with an OpenRouter key, which OpenAI "
                "rejects (HTTP 401). Use LST_STT_PROVIDER=openrouter or an OpenAI key."
            )
        if name == "openrouter" and family == "openai" and self._default_url("openrouter"):
            raise ConfigError(
                "STT provider 'openrouter' is configured with an OpenAI key, which OpenRouter "
                "rejects (HTTP 401). Use LST_STT_PROVIDER=openai or an OpenRouter key."
            )

    def _default_url(self, provider: str) -> bool:
        """True while the provider still points at its public API (keys are checkable)."""
        expected = OPENAI_DEFAULT_BASE_URL if provider == "openai" else OPENROUTER_DEFAULT_BASE_URL
        return self.base_url_for(provider) == expected

    def redacted_summary(self) -> dict[str, Any]:
        """A log-safe view of every setting: secrets become ``"set"`` / ``"unset"``."""
        summary: dict[str, Any] = {}
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            if "SecretStr" in str(field.annotation):
                is_set = value is not None and bool(value.get_secret_value())
                summary[name] = "set" if is_set else "unset"
            elif name in _MASKED_FIELDS:
                summary[name] = "set" if value else "unset"
            elif isinstance(value, Path):
                summary[name] = _display_path(value)
            elif isinstance(value, str):
                summary[name] = redact_text(value)
            else:
                summary[name] = value
        return summary


#: Settings reported only as set/unset: they identify an account without being secrets.
_MASKED_FIELDS = frozenset({"notify_telegram_chat_id"})


def _display_path(path: Path) -> str:
    """``~/...`` for a path under the home directory, so summaries stored in the
    database or pasted into a bug report do not carry the account name."""
    try:
        return str(Path("~") / path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _format_validation_error(exc: ValidationError) -> str:
    """One line per problem, naming the environment variable, never the value."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        env_name = loc if loc.upper().startswith("LST_") else f"LST_{loc.upper()}"
        lines.append(f"{env_name}: {err['msg']}")
    return "invalid configuration: " + "; ".join(lines)
