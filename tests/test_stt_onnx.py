"""The sherpa-onnx provider without the package or any model files."""

from __future__ import annotations

import io
import threading
from types import SimpleNamespace

import pytest

from livestream_transcriber.audio.timing import sessionize_transcript_timestamps
from livestream_transcriber.config import Settings
from livestream_transcriber.stt.base import STT_UNAVAILABLE
from livestream_transcriber.stt.providers import sherpa_onnx
from livestream_transcriber.stt.providers.sherpa_onnx import (
    DEFAULT_MODEL_ROOT,
    ONNX_MODELS,
    SherpaOnnxTranscriber,
    ensure_model,
    model_present,
    release_recognizers,
)
from tests.support.audio import RATE, tone


class FakeRecognizer:
    def __init__(self, text: str, stamps: list[float], tokens: list[str] | None = None) -> None:
        self.text, self.stamps, self.tokens = text, stamps, tokens or []
        self.calls = 0

    def create_stream(self):
        return SimpleNamespace(accept_waveform=lambda rate, audio: None, result=None)

    def decode_stream(self, stream) -> None:
        self.calls += 1
        stream.result = SimpleNamespace(text=self.text, timestamps=self.stamps, tokens=self.tokens)


@pytest.fixture(autouse=True)
def _fresh_recognizers():
    release_recognizers()
    yield
    release_recognizers()


def with_model(tmp_path, name="parakeet-tdt-0.6b-v3"):
    base = tmp_path / name
    base.mkdir(parents=True)
    for f in ONNX_MODELS[name].files:
        (base / f).write_bytes(b"x")
    return tmp_path


def use(monkeypatch, recognizer) -> None:
    monkeypatch.setattr(SherpaOnnxTranscriber, "_build", lambda self: recognizer)


def test_transcribes_with_chunk_relative_segments_that_land_on_the_session(tmp_path, monkeypatch):
    use(monkeypatch, FakeRecognizer(" Good morning everyone. ", [0.8, 1.2, 2.0]))
    stt = SherpaOnnxTranscriber(model_root=with_model(tmp_path))
    got = stt.transcribe(tone(5), RATE, start=100.0, end=105.0)
    assert got is not None and got.text == "Good morning everyone."
    assert (got.provider, got.model, got.degraded) == ("onnx", "onnx/parakeet-tdt-0.6b-v3", False)
    assert got.segments == [
        {"start": 0.8, "end": pytest.approx(2.4), "text": "Good morning everyone."}
    ]
    seg = sessionize_transcript_timestamps(got).segments[0]
    assert seg["start"] == pytest.approx(100.8) and seg["end"] == pytest.approx(102.4)


def test_the_segment_end_never_passes_the_chunk(tmp_path, monkeypatch):
    use(monkeypatch, FakeRecognizer("late words", [1.0, 2.9]))
    got = SherpaOnnxTranscriber(model_root=with_model(tmp_path)).transcribe(
        tone(3), RATE, start=0.0, end=3.0
    )
    assert got is not None and got.segments[0]["end"] == pytest.approx(3.0)


def test_word_times_are_grouped_from_sentencepiece_tokens(tmp_path, monkeypatch):
    tokens = ["▁Good", "▁morn", "ing", "▁all"]
    use(monkeypatch, FakeRecognizer("Good morning all", [0.5, 1.0, 1.3, 1.8], tokens))
    stt = SherpaOnnxTranscriber(model_root=with_model(tmp_path), word_timestamps=True)
    got = stt.transcribe(tone(3), RATE, start=0.0, end=3.0)
    assert got is not None
    assert got.words == [
        {"start": 0.5, "end": 1.0, "text": "Good"},
        {"start": 1.0, "end": 1.8, "text": "morning"},
        {"start": 1.8, "end": pytest.approx(2.2), "text": "all"},
    ]
    plain = SherpaOnnxTranscriber(model_root=tmp_path).transcribe(tone(3), RATE, start=0, end=3)
    assert plain is not None and plain.words is None


def test_empty_text_is_no_speech_and_short_audio_is_skipped(tmp_path, monkeypatch):
    fake = FakeRecognizer("", [])
    use(monkeypatch, fake)
    stt = SherpaOnnxTranscriber(model_root=with_model(tmp_path))
    assert stt.transcribe(tone(3), RATE, start=0, end=3) is None
    assert stt.transcribe(tone(0.3), RATE, start=0, end=0.3) is None
    assert fake.calls == 1


def test_missing_model_files_are_unavailable_and_never_loaded(tmp_path, monkeypatch):
    built: list[int] = []
    monkeypatch.setattr(SherpaOnnxTranscriber, "_build", lambda self: built.append(1))
    stt = SherpaOnnxTranscriber(model_root=tmp_path)
    for _ in range(3):
        got = stt.transcribe(tone(2), RATE, start=0, end=2)
        assert got is not None and got.status == STT_UNAVAILABLE
    assert built == [] and stt.describe()["load_failed"] is True


def test_a_missing_package_is_unavailable(tmp_path, monkeypatch):
    def no_package(self):
        raise ImportError("no sherpa_onnx")

    monkeypatch.setattr(SherpaOnnxTranscriber, "_build", no_package)
    got = SherpaOnnxTranscriber(model_root=with_model(tmp_path)).transcribe(
        tone(2), RATE, start=0, end=2
    )
    assert got is not None and got.unavailable


def test_an_english_only_model_refuses_another_language(tmp_path, monkeypatch):
    root = with_model(tmp_path, "moonshine-base-en")
    use(monkeypatch, FakeRecognizer("hello", [0.1]))
    de = SherpaOnnxTranscriber("moonshine-base-en", model_root=root, language="de")
    assert de.transcribe(tone(2), RATE, start=0, end=2).status == STT_UNAVAILABLE
    en = SherpaOnnxTranscriber("moonshine-base-en", model_root=root, language="en")
    assert en.transcribe(tone(2), RATE, start=0, end=2).text == "hello"
    auto = SherpaOnnxTranscriber("moonshine-base-en", model_root=root)
    assert auto.transcribe(tone(2), RATE, start=0, end=2).text == "hello"


def test_an_unknown_model_is_an_error():
    with pytest.raises(ValueError, match="unknown onnx model"):
        SherpaOnnxTranscriber("whisper-tiny")


def test_instances_share_one_recognizer_and_close_releases_it(tmp_path, monkeypatch):
    built: list[int] = []

    def build(self):
        built.append(1)
        return FakeRecognizer("shared", [0.1])

    monkeypatch.setattr(SherpaOnnxTranscriber, "_build", build)
    root = with_model(tmp_path)
    first, second = SherpaOnnxTranscriber(model_root=root), SherpaOnnxTranscriber(model_root=root)
    assert first.transcribe(tone(2), RATE, start=0, end=2).text == "shared"
    assert second.transcribe(tone(2), RATE, start=0, end=2).text == "shared"
    assert len(built) == 1
    first.close()
    assert sherpa_onnx._SHARED  # the second still holds it
    second.close()
    assert not sherpa_onnx._SHARED
    second.close()  # idempotent
    third = SherpaOnnxTranscriber(model_root=root)
    third.transcribe(tone(2), RATE, start=0, end=2)
    assert len(built) == 2


def test_decodes_are_serialised_per_recognizer(tmp_path, monkeypatch):
    active = 0
    peak = 0
    guard = threading.Lock()

    class Slow(FakeRecognizer):
        def decode_stream(self, stream) -> None:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            threading.Event().wait(0.02)
            with guard:
                active -= 1
            super().decode_stream(stream)

    use(monkeypatch, Slow("text", [0.1]))
    stt = SherpaOnnxTranscriber(model_root=with_model(tmp_path))
    threads = [
        threading.Thread(target=stt.transcribe, args=(tone(2), RATE), kwargs={"start": 0, "end": 2})
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert peak == 1


# ------------------------------------------------------------------ fetching --


def test_ensure_model_downloads_each_missing_file_once(tmp_path, monkeypatch):
    fetched: list[str] = []

    def fake_urlopen(url, timeout=0):
        fetched.append(url)
        return io.BytesIO(b"weights")

    monkeypatch.setattr(sherpa_onnx.urllib.request, "urlopen", fake_urlopen)
    announced: list[str] = []
    base = ensure_model("moonshine-tiny-en", tmp_path, on_file=announced.append)
    files = ONNX_MODELS["moonshine-tiny-en"].files
    assert model_present("moonshine-tiny-en", tmp_path)
    assert len(fetched) == len(files) and announced == list(files)
    assert all("csukuangfj/sherpa-onnx-moonshine-tiny-en-int8/resolve/main/" in u for u in fetched)
    assert not list(base.glob("*.part"))
    ensure_model("moonshine-tiny-en", tmp_path)
    assert len(fetched) == len(files)  # nothing missing, nothing fetched


def test_a_truncated_download_is_not_taken_for_the_model(tmp_path, monkeypatch):
    class Short(io.BytesIO):
        headers = {"Content-Length": "100000"}

    monkeypatch.setattr(
        sherpa_onnx.urllib.request, "urlopen", lambda url, timeout=0: Short(b"x" * 100)
    )
    with pytest.raises(OSError, match="got 100 of 100000 bytes"):
        ensure_model("moonshine-tiny-en", tmp_path)
    assert not model_present("moonshine-tiny-en", tmp_path)
    assert not list((tmp_path / "moonshine-tiny-en").glob("*"))


def test_the_default_model_root_matches_the_setting():
    assert Settings().stt_models_dir == DEFAULT_MODEL_ROOT


def test_nothing_is_downloaded_implicitly(tmp_path, monkeypatch):
    """Loading a provider must never fetch: the conftest network guard would fail it."""
    fetched: list[str] = []
    monkeypatch.setattr(
        sherpa_onnx.urllib.request, "urlopen", lambda *a, **k: fetched.append("hit")
    )
    stt = SherpaOnnxTranscriber(model_root=tmp_path)
    assert stt.transcribe(tone(2), RATE, start=0, end=2).unavailable
    assert fetched == []
