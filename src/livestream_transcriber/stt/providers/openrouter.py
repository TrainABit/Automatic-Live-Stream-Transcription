"""OpenRouter speech-to-text.

``POST {base_url}/audio/transcriptions`` with a JSON body, not multipart:
``{"model", "input_audio": {"data": <base64 wav>, "format": "wav"}, "language",
"temperature", ...}``. The answer carries ``text`` and, depending on the model,
``segments``/``words`` and ``usage.cost``. What a model supports varies, so
nothing is assumed beyond the text: timestamps are used when present, and a 400
to ``verbose_json`` downgrades to ``json`` once (see :class:`CloudTranscriber`).

The default model is a Whisper variant; ``LST_STT_MODEL`` selects any other
transcription model OpenRouter lists.
"""

from __future__ import annotations

import base64
from typing import Any

from ...netutil import post_json
from ..base import Transcript, normalise_timed_items
from ._cloud import CloudTranscriber

__all__ = ["OpenRouterTranscriber"]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/whisper-large-v3"


class OpenRouterTranscriber(CloudTranscriber):
    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        language: str | None = None,
        prompt: str | None = None,
        timeout: float = 60.0,
        word_timestamps: bool = False,
        temperature: float | None = 0.0,
        provider_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key,
            model=model,
            base_url=base_url,
            language=language,
            prompt=prompt,
            timeout=timeout,
            word_timestamps=word_timestamps,
            **kwargs,
        )
        self.temperature = temperature
        self.provider_options = provider_options

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/audio/transcriptions"

    def _payload(self, wav: bytes) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
            "response_format": self.response_format,
        }
        if self.language:
            body["language"] = self.language
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.prompt:
            body["prompt"] = self.prompt
        if self.response_format == "verbose_json":
            granularities = ["segment", "word"] if self.word_timestamps else ["segment"]
            body["timestamp_granularities"] = granularities
        if self.provider_options:
            body["provider"] = {"options": self.provider_options}
        return body

    def _send(self, wav: bytes, timeout: float) -> dict[str, Any]:
        return post_json(
            self.endpoint,
            self._payload(wav),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def _reported_cost(self, data: dict[str, Any]) -> float | None:
        usage = data.get("usage")
        if isinstance(usage, dict) and usage.get("cost") is not None:
            try:
                return max(0.0, float(usage["cost"]))
            except (TypeError, ValueError):
                return None
        return None

    def _interpret(
        self, data: dict[str, Any], *, start: float, end: float, cost: float
    ) -> Transcript | None:
        segments = normalise_timed_items(data.get("segments"))
        text = str(data.get("text") or "").strip()
        if not text and segments:
            text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
        if not text:
            return None
        confidence = data.get("confidence")
        return Transcript(
            start=start,
            end=end,
            text=text,
            confidence=float(confidence) if isinstance(confidence, int | float) else None,
            segments=segments,
            words=normalise_timed_items(data.get("words"), text_keys=("word", "text")),
            cost_usd=cost,
            model=self.model,
            provider=self.provider,
            language=data.get("language")
            if isinstance(data.get("language"), str)
            else self.language,
        )
