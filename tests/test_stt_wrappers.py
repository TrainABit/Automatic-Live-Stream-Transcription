"""The cache, fallback and budget wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from livestream_transcriber.stt.base import MockTranscriber, Transcript, unavailable
from livestream_transcriber.stt.wrappers import (
    BudgetGuardTranscriber,
    CachedTranscriber,
    FallbackSttTranscriber,
    SttSpend,
    iter_chain,
)
from tests.support.audio import RATE, tone


class Scripted:
    """Returns each scripted item in turn; carries the attributes wrappers read."""

    def __init__(self, *script, model="scripted"):
        self.script = list(script)
        self.model = model
        self.calls = 0

    def transcribe(self, pcm, sample_rate, *, start, end):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return item


class Spending:
    """Adds a fixed cost per call, like a paid provider."""

    model = "paid"

    def __init__(self, per_call: float) -> None:
        self.per_call = per_call
        self.cost_usd = 0.0
        self.calls = 0

    def transcribe(self, pcm, sample_rate, *, start, end):
        self.calls += 1
        self.cost_usd += self.per_call
        return Transcript(start=start, end=end, text=f"call {self.calls}")


# --------------------------------------------------------------------- cache --


def test_cache_hits_do_not_recall_the_inner(tmp_path: Path):
    inner = MockTranscriber("hello")
    cached = CachedTranscriber(inner, tmp_path, provider="openai", model="whisper-1")
    first = cached.transcribe(tone(), RATE, start=5.0, end=10.0)
    second = cached.transcribe(tone(), RATE, start=5.0, end=10.0)
    assert first is not None and second is not None
    assert first.text == second.text == "hello"
    assert len(inner.calls) == 1
    assert (cached.hits, cached.misses) == (1, 1)


def test_cache_key_covers_provider_model_tag_start_and_audio(tmp_path: Path):
    cached = CachedTranscriber(
        MockTranscriber("x"), tmp_path, provider="openrouter", model="openai/whisper-large-v3",
        param_tag="de-words",
    )  # fmt: skip
    path = cached.cache_path(tone(), RATE, 12.345)
    assert path.parent == tmp_path / "openrouter" / "openai_whisper-large-v3"
    assert path.name.startswith("12.35_16000_") or path.name.startswith("12.34_16000_")
    assert path.name.endswith("_de-words.json")
    assert cached.cache_path(tone(1.5), RATE, 12.345) != path
    assert cached.cache_path(tone(), RATE, 13.0) != path


def test_a_different_tag_does_not_share_entries(tmp_path: Path):
    inner = MockTranscriber("hello")
    a = CachedTranscriber(inner, tmp_path, provider="p", model="m", param_tag="de")
    b = CachedTranscriber(inner, tmp_path, provider="p", model="m", param_tag="en")
    a.transcribe(tone(), RATE, start=0.0, end=1.0)
    b.transcribe(tone(), RATE, start=0.0, end=1.0)
    assert len(inner.calls) == 2


def test_cache_stores_the_rich_transcript(tmp_path: Path):
    rich = Transcript(
        start=0, end=1, text="hello", confidence=0.9, provider="openai", model="whisper-1",
        segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        words=[{"start": 0.0, "end": 0.5, "text": "hello"}],
        provider_latency=0.4, cost_usd=0.0001, language="en",
    )  # fmt: skip
    cached = CachedTranscriber(Scripted(rich), tmp_path, provider="openai", model="whisper-1")
    cached.transcribe(tone(), RATE, start=2.0, end=3.0)
    again = cached.transcribe(tone(), RATE, start=2.0, end=3.0)
    assert again is not None
    assert again.segments == rich.segments and again.words == rich.words
    assert (again.confidence, again.cost_usd, again.language) == (0.9, 0.0001, "en")
    assert cached.hits == 1
    (file,) = tmp_path.rglob("*.json")
    assert "sk-" not in file.read_text()


def test_cache_shares_entries_across_wrapper_instances(tmp_path: Path):
    inner = MockTranscriber("hello")
    first = CachedTranscriber(inner, tmp_path, provider="p", model="m")
    second = CachedTranscriber(inner, tmp_path, provider="p", model="m")
    first.transcribe(tone(), RATE, start=5.0, end=10.0)
    assert second.transcribe(tone(), RATE, start=5.0, end=10.0).text == "hello"
    assert len(inner.calls) == 1 and second.hits == 1


def test_failures_no_speech_and_degraded_results_are_not_cached(tmp_path: Path):
    inner = Scripted(
        unavailable(0, 1),
        None,
        Transcript(start=0, end=1, text="from the fallback", degraded=True),
        Transcript(start=0, end=1, text="real"),
    )
    cached = CachedTranscriber(inner, tmp_path, provider="p", model="m")
    assert cached.transcribe(tone(), RATE, start=1.0, end=2.0).unavailable
    assert cached.transcribe(tone(), RATE, start=1.0, end=2.0) is None
    assert cached.transcribe(tone(), RATE, start=1.0, end=2.0).degraded
    assert not list(tmp_path.rglob("*.json"))
    assert cached.transcribe(tone(), RATE, start=1.0, end=2.0).text == "real"
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert inner.calls == 4


def test_cache_only_never_calls_the_inner(tmp_path: Path):
    inner = MockTranscriber("hello")
    CachedTranscriber(inner, tmp_path, provider="p", model="m").transcribe(
        tone(), RATE, start=0.0, end=1.0
    )
    offline = CachedTranscriber(inner, tmp_path, provider="p", model="m", cache_only=True)
    assert offline.transcribe(tone(), RATE, start=0.0, end=1.0).text == "hello"
    assert offline.transcribe(tone(), RATE, start=9.0, end=10.0) is None
    assert len(inner.calls) == 1


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path: Path, caplog):
    inner = MockTranscriber("fresh")
    cached = CachedTranscriber(inner, tmp_path, provider="p", model="m")
    path = cached.cache_path(tone(), RATE, 0.0)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert cached.transcribe(tone(), RATE, start=0.0, end=1.0).text == "fresh"
    assert json.loads(path.read_text())["text"] == "fresh"  # and the entry is repaired


# ------------------------------------------------------------------ fallback --


def test_fallback_takes_over_when_the_primary_is_unavailable():
    primary = Scripted(unavailable(0, 1))
    backup = Scripted(Transcript(start=0, end=1, text="from local", model="faster-whisper/small"))
    wrapper = FallbackSttTranscriber(primary, backup, provider_name="local")
    got = wrapper.transcribe(tone(), RATE, start=10.0, end=11.0)
    assert got is not None and got.text == "from local"
    assert got.degraded is True
    assert (got.provider, got.model) == ("local", "faster-whisper/small")
    assert wrapper.fallback_hits == 1


def test_fallback_also_covers_an_exhausted_quota():
    primary = Scripted(unavailable(0, 1, quota_exceeded=True))
    backup = Scripted(Transcript(start=0, end=1, text="local"))
    got = FallbackSttTranscriber(primary, backup).transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.text == "local" and got.degraded


def test_a_primary_success_passes_through_untouched():
    primary = Scripted(Transcript(start=0, end=1, text="cloud"))
    backup = Scripted(Transcript(start=0, end=1, text="never"))
    got = FallbackSttTranscriber(primary, backup).transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.text == "cloud" and not got.degraded
    assert backup.calls == 0


def test_no_speech_from_the_primary_is_not_a_reason_to_fall_back():
    backup = Scripted(Transcript(start=0, end=1, text="never"))
    assert (
        FallbackSttTranscriber(Scripted(None), backup).transcribe(tone(), RATE, start=0, end=1)
        is None
    )
    assert backup.calls == 0


def test_a_fallback_that_hears_nothing_is_no_speech():
    wrapper = FallbackSttTranscriber(Scripted(unavailable(0, 1)), Scripted(None))
    assert wrapper.transcribe(tone(), RATE, start=0, end=1) is None
    assert wrapper.fallback_misses == 1


def test_when_both_fail_the_result_is_unavailable():
    class Crashing:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("boom")

    for backup in (Crashing(), Scripted(unavailable(0, 1))):
        got = FallbackSttTranscriber(Scripted(unavailable(0, 1)), backup).transcribe(
            tone(), RATE, start=0, end=1
        )
        assert got is not None and got.unavailable


def test_a_disabled_fallback_is_not_called():
    backup = MockTranscriber("local")
    wrapper = FallbackSttTranscriber(Scripted(unavailable(0, 1)), backup)
    wrapper.enabled = False
    got = wrapper.transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.unavailable
    assert backup.calls == []


def test_a_tripped_budget_hands_over_to_the_fallback():
    guard = BudgetGuardTranscriber(Spending(0.06), 0.05)
    wrapper = FallbackSttTranscriber(guard, MockTranscriber("local"))
    first = wrapper.transcribe(tone(), RATE, start=0, end=1)
    assert first is not None and first.text == "call 1" and not first.degraded  # crosses the cap
    second = wrapper.transcribe(tone(), RATE, start=1, end=2)
    assert second is not None and second.text == "local" and second.degraded
    assert wrapper.cost_usd == pytest.approx(0.06)


def test_iter_chain_walks_wrappers_primary_first():
    inner = MockTranscriber("x")
    backup = MockTranscriber("y")
    chain = CachedTranscriber(
        FallbackSttTranscriber(BudgetGuardTranscriber(inner, 1.0), backup),
        "unused",
        provider="p",
    )
    kinds = [type(n).__name__ for n in iter_chain(chain)]
    assert kinds == [
        "CachedTranscriber",
        "FallbackSttTranscriber",
        "BudgetGuardTranscriber",
        "MockTranscriber",
        "MockTranscriber",
    ]


# -------------------------------------------------------------------- budget --


def test_budget_guard_stops_after_the_cap_is_crossed():
    inner = Spending(0.05)
    exceeded: list[float] = []
    guard = BudgetGuardTranscriber(inner, 0.10, on_exceeded=exceeded.append)
    assert guard.transcribe(b"x", RATE, start=0.0, end=1.0) is not None  # 0.05
    assert exceeded == []
    assert guard.transcribe(b"x", RATE, start=1.0, end=2.0) is not None  # 0.10, crossed
    assert exceeded == [pytest.approx(0.10)]
    tripped = guard.transcribe(b"x", RATE, start=2.0, end=3.0)
    # Tripped means "could not listen", never "heard nothing".
    assert tripped is not None and tripped.unavailable and tripped.quota_exceeded
    assert inner.calls == 2 and len(exceeded) == 1


def test_a_failing_budget_callback_does_not_drop_the_transcript():
    def boom(spent):
        raise RuntimeError("notifier down")

    guard = BudgetGuardTranscriber(Spending(0.05), 0.04, on_exceeded=boom)
    got = guard.transcribe(b"x", RATE, start=0.0, end=1.0)
    assert got is not None and got.text == "call 1"


def test_the_spend_of_every_session_counts_toward_one_cap():
    spend = SttSpend()
    alerts: list[float] = []
    first = BudgetGuardTranscriber(Spending(0.06), 0.10, on_exceeded=alerts.append, spend=spend)
    assert first.transcribe(b"x", RATE, start=0.0, end=1.0) is not None  # 0.06
    # The session ends; the next one builds a new transcriber that starts at zero.
    second = BudgetGuardTranscriber(Spending(0.06), 0.10, on_exceeded=alerts.append, spend=spend)
    assert not second.tripped
    assert second.transcribe(b"x", RATE, start=1.0, end=2.0) is not None  # 0.12
    assert second.tripped and alerts == [spend.spent_usd]
    third_inner = Spending(0.06)
    third = BudgetGuardTranscriber(third_inner, 0.10, on_exceeded=alerts.append, spend=spend)
    assert third.tripped
    refused = third.transcribe(b"x", RATE, start=2.0, end=3.0)
    assert refused is not None and refused.unavailable
    assert third_inner.calls == 0 and len(alerts) == 1


def test_without_a_shared_counter_each_guard_counts_its_own():
    alerts: list[float] = []
    for _ in range(2):
        BudgetGuardTranscriber(Spending(0.06), 0.10, on_exceeded=alerts.append).transcribe(
            b"x", RATE, start=0.0, end=1.0
        )
    assert alerts == []


def test_spend_already_on_the_inner_transcriber_is_not_counted_twice():
    spend = SttSpend()
    inner = Spending(0.05)
    inner.cost_usd = 0.05  # spent before the guard wrapped it
    BudgetGuardTranscriber(inner, 1.0, spend=spend).transcribe(b"x", RATE, start=0.0, end=1.0)
    assert spend.spent_usd == pytest.approx(0.05)


def test_spend_ignores_negative_amounts_and_trips_once():
    spend = SttSpend()
    assert spend.add(-1.0) == 0.0
    assert spend.trip() is True and spend.trip() is False
