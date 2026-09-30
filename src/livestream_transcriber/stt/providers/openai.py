"""OpenAI speech-to-text, and any server that speaks the same protocol.

``POST {base_url}/audio/transcriptions`` with a multipart body. The default
base URL is OpenAI's; point it at another OpenAI-compatible server (a local
faster-whisper server, a proxy) to use that instead. ``verbose_json`` is asked
for so segments and, on request, word times come back; models that only
support ``json`` (the ``gpt-4o-*-transcribe`` family) are detected by name, and
any server that answers 400 to ``verbose_json`` is downgraded once at runtime.
"""

from __future__ import annotations

from typing import Any

from ...netutil import post_multipart
from ..base import Transcript, normalise_timed_items
from ._cloud import CloudTranscriber

__all__ = ["OpenAITranscriber"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "whisper-1"


def _supports_verbose_json(model: str) -> bool:
    return not model.lower().startswith("gpt-4o")


class OpenAITranscriber(CloudTranscriber):
    provider = "openai"

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
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault(
            "response_format", "verbose_json" if _supports_verbose_json(model) else "json"
        )
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

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/audio/transcriptions"

    def _fields(self) -> dict[str, str]:
        fields = {"model": self.model, "response_format": self.response_format}
        if self.language:
            fields["language"] = self.language
        if self.prompt:
            fields["prompt"] = self.prompt
        return fields

    def _send(self, wav: bytes, timeout: float) -> dict[str, Any]:
        fields = self._fields()
        if self.response_format == "verbose_json" and self.word_timestamps:
            # Word granularity replaces the segment list in the OpenAI API; the
            # words are what subtitles need, and segments are the default otherwise.
            fields["timestamp_granularities[]"] = "word"
        return post_multipart(
            self.endpoint,
            fields,
            {"file": ("chunk.wav", wav, "audio/wav")},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def _interpret(
        self, data: dict[str, Any], *, start: float, end: float, cost: float
    ) -> Transcript | None:
        text = str(data.get("text") or "").strip()
        if not text:
            return None
        language = data.get("language")
        return Transcript(
            start=start,
            end=end,
            text=text,
            segments=normalise_timed_items(data.get("segments")),
            words=normalise_timed_items(data.get("words"), text_keys=("word", "text")),
            cost_usd=cost,
            model=self.model,
            provider=self.provider,
            language=language if isinstance(language, str) else self.language,
        )
