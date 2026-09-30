# Benchmarks

Two independent parts:

* **Part A** is a benchmark you can reproduce in a few minutes on your own machine, with
  no account and no network access beyond the one-time model download.
* **Part B** is a set of field notes from an earlier, larger measurement of cloud
  speech-to-text providers on German live-stream speech. The raw data is not part of this
  repository, so treat the numbers as context, not as something `lst bench` regenerates.

Neither part is a leaderboard. Both are small, single-machine measurements with the
caveats listed under them.

---

## Part A: Reproducible local benchmark

### How to run it

```bash
pip install 'livestream-transcriber[local]'
scripts/make_tts_clips.sh clips            # ten synthetic clips + reference .txt files
lst bench --clips clips --stt local:tiny,local:base,local:small \
          --out bench.json --markdown bench.md
```

`scripts/make_tts_clips.sh` synthesises five English and five German sentences from a
fixed list with the macOS `say` command (or `espeak-ng` where `say` is missing), converts
them to 16 kHz mono WAV with ffmpeg and writes the sentence next to each clip as its
reference. The sentences avoid numerals on purpose: models write digits and scoring "10"
against "ten" would measure spelling conventions, not recognition.

`lst bench` cuts every clip into 5 s chunks, the pipeline's own unit of work, and
transcribes chunk by chunk. Providers run one after another, never concurrently. Before
the timed pass one *cold pass* transcribes the first chunk, so model loading is reported
as its own number instead of inflating p95 latency and RTF. WER and CER are
micro-averaged (errors summed over all clips, divided by the total reference length) on
text normalised by NFC, case folding and punctuation removal.

The language is detected automatically (`--language` is not set), so the set mixes English
and German clips through one model.

### Test machine

| | |
|---|---|
| CPU | Apple M3 Pro (11 cores) |
| RAM | 18 GB |
| OS | macOS 27.0.1, arm64 |
| Python | 3.11.15 |
| Engine | faster-whisper 1.2.1 (CTranslate2 4.8.2), CPU, `int8`; sherpa-onnx 1.13.8 (Parakeet TDT 0.6B v3) |
| ffmpeg | 8.1 |
| Date | 2026-09-30 |
| Load | Other jobs were running during the measurement (load average about 7 on 11 cores). Latencies are therefore pessimistic and noisy; the WER columns are not affected. |

### Results with one CPU thread (`LST_STT_THREADS=1`)

10 clips, 130 reference words, 1 to 2 chunks per clip.

| Model | WER | CER | p50 latency per 5 s chunk | p95 latency | RTF | Cold start |
|---|---:|---:|---:|---:|---:|---:|
| `tiny` | 6.9 % | 2.4 % | 0.72 s | 0.76 s | 0.19 | 1.64 s |
| `base` | 6.9 % | 2.3 % | 1.51 s | 2.41 s | 0.44 | 2.05 s |
| `small` | 3.1 % | 1.1 % | 5.90 s | 6.22 s | 1.59 | 6.54 s |

RTF is processing time divided by audio time; below 1 keeps up with a live stream. With
one CPU thread, `small` does **not** keep up (RTF 1.59), which is why the default is four
threads (next table). On a machine with fewer cores, lower `LST_STT_THREADS` only together
with a smaller model: the overload guard pauses transcription once the lag grows past
`LST_STT_MAX_LAG_SECONDS`.

### Results with four CPU threads (`LST_STT_THREADS=4`, the default)

| Model | WER | CER | p50 latency per 5 s chunk | p95 latency | RTF | Cold start |
|---|---:|---:|---:|---:|---:|---:|
| `tiny` | 6.9 % | 2.4 % | 0.31 s | 0.33 s | 0.08 | 1.11 s |
| `base` | 6.9 % | 2.3 % | 0.65 s | 1.20 s | 0.19 | 0.88 s |
| `small` | 3.1 % | 1.1 % | 2.15 s | 2.26 s | 0.58 | 2.87 s |

Accuracy is identical (same model, same weights); only speed changes. Cold start is the
first request with the model already on disk. The very first run additionally downloads
the model from Hugging Face (`small` took about 78 s including the download on this
connection).

### Parakeet (sherpa-onnx) against Whisper `small`

Same ten clips, measured back to back in one session (`lst models fetch --model
parakeet-tdt-0.6b-v3` first). Whisper `small` was re-run next to it so both rows share the same
machine load.

```bash
lst bench --clips clips --stt onnx:parakeet-tdt-0.6b-v3,local:small --out bench.json
```

| Threads | Model | WER | CER | p50 latency per 5 s chunk | p95 latency | RTF | Cold start |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Parakeet TDT 0.6B v3 | 4.6 % | 2.9 % | 0.29 s | 0.55 s | 0.08 | 2.74 s |
| 1 | Whisper `small` | 3.1 % | 1.1 % | 6.36 s | 7.18 s | 1.73 | 9.82 s |
| 4 | Parakeet TDT 0.6B v3 | 4.6 % | 2.9 % | 0.13 s | 0.17 s | 0.03 | 1.85 s |
| 4 | Whisper `small` | 3.1 % | 1.1 % | 2.20 s | 2.48 s | 0.60 | 3.53 s |

Parakeet is roughly 17 to 20 times faster and keeps up with a live stream even on a single
thread; Whisper `small` is slightly more accurate here. The extra errors sit in two German
clips (compound words and a name split differently). On ten clean TTS clips a 1.5 point WER
gap is a handful of words, so treat it as "comparable", not as a ranking. Parakeet covers 25
European languages, cannot be forced to a language, and returns chunk-level rather than
word-level timestamps.

### What the errors were

Ten clips are too few for statistics; a look at the actual mistakes is more useful than a
decimal place on the WER.

* The largest single contributor is the **hard 5 s cut**. One German sentence runs slightly
  past 5 s, so the last word (a long compound) is split across two chunks and both halves
  are transcribed as nonsense by `tiny` and `base`; `small` garbles it too, though less. Real
  streams see the same effect at every chunk boundary; the pipeline's speech stream
  stitches overlapping or repeated text, but it cannot repair a word that was cut in half
  by the audio.
* One error in every model is a spelling variant ("start up" in the reference, "startup"
  in the output), which is not a recognition error.
* The other `tiny`/`base` errors are a dropped pronoun and near-miss word forms in German.
* No clip triggered the hallucination check.

### Caveats

* **TTS audio is clean.** One synthetic voice per language, no background noise, no music,
  no overlapping speakers, perfect enunciation. Real speech is harder, so every WER here is
  optimistic. Use the table to compare models with each other on this machine, not to
  predict accuracy on your stream.
* **Ten clips, 130 words.** One word is 0.8 percentage points. The difference between
  `tiny` and `base` is noise; the difference to `small` is about five words.
* **Latency is machine dependent** and was measured under load. Run the commands above on
  your own hardware.
* **Chunking is part of the measurement**, on purpose: it is how a live run experiences the
  models. Whole-clip transcription would score better.
* No cloud provider was run here (no keys). `lst bench --stt local,openai,openrouter` runs
  them side by side and adds a cost column.

---

## Part B: Field notes on cloud speech-to-text for German live-stream speech (2026-09)

These notes summarise an earlier benchmark of six cloud speech-to-text models. Only
provider-level aggregates are reported.

**Dataset.** German live-stream speech, 5 s chunks, 16 kHz mono, about 60 unique controlled
chunks (including 7 non-speech chunks) plus a real-world set (33 non-speech chunks, 20 of
them music), sent through OpenRouter's `/audio/transcriptions` endpoint (one model through
the chat-completions endpoint with audio input). Measured September 2026 from a laptop.

The six models, as labelled in that benchmark: MAI-Transcribe 2, Grok STT 1.0, GPT-4o Mini
Transcribe, Whisper Large V3 Turbo, Deepgram Nova-3 and Gemini 3.8 Flash.

### Latency (clean run, one provider at a time, 78 requests each, 5 s chunks)

| Provider | p50 | p95 | Max | RTF (one worker) |
|---|---:|---:|---:|---:|
| GPT-4o Mini Transcribe | 1.59 s | 3.42 s | 4.67 s | 0.36 |
| Grok STT 1.0 | 1.63 s | 2.38 s | 2.85 s | 0.34 |
| MAI-Transcribe 2 | 1.82 s | 4.37 s | 6.59 s | 0.42 |
| Whisper Large V3 Turbo | 2.35 s | 5.75 s | 11.76 s | 0.54 |
| Deepgram Nova-3 | 5.81 s | 18.37 s | 23.95 s | 1.43 |
| Gemini 3.8 Flash | 11.38 s | 29.42 s | 60.08 s | 2.78 |

An RTF above 1 means one worker cannot keep up with a live stream even in principle:
Deepgram Nova-3 needs at least two workers and Gemini 3.8 Flash at least three.

### Live keep-up (paced 5 s chunks in real time, two workers, drop-when-full queue)

A 30-minute paced run per provider (10 minutes for Gemini 3.8 Flash, for budget). Each
provider ran alone at a different time of the evening.

| Provider | Dropped chunks | Max backlog (chunks) | Max age of the oldest waiting chunk | Longest single request | Keeps up? |
|---|---:|---:|---:|---:|---|
| Grok STT 1.0 | 0 | 4 | 10 s | 16 s | Yes, flattest curve |
| GPT-4o Mini Transcribe | 0 | 7 | 25 s | 23 s | Yes |
| Deepgram Nova-3 | 0 | 8 | 30 s | 39 s | Yes, two slow patches |
| Whisper Large V3 Turbo | 0 | 11 | 45 s | 121 s | Yes; one stalled request cost about four minutes of recovery |
| MAI-Transcribe 2 | 0 | 44 | 210 s | 90 s and 108 s (two runs) | Yes, but up to about 3.5 minutes behind for about 20 minutes |
| Gemini 3.8 Flash | 0 | 30 | 140 s | 33 s | No: backlog grew steadily, end-to-end lag rose from about 25 s to about 150 s |

The lesson that transfers is the **tail, not the median**. With two workers, one hung
request ties up half the capacity, and a slow hour turned a 1.8 s median into a
multi-minute lag for one provider. Median latency also moved 3 to 4 times between the
clean run and the live runs, and changed with the hour of the day inside one evening, so
live latency should not be read as a clean provider ranking. This is the reason the
pipeline has a bounded queue with disk spill, an ordering gate for late results, an
overload guard, and a per-request timeout with a circuit breaker.

### Hallucination on silence and music

Non-speech chunks that came back with any text at all (a count for the controlled set, a share for the real-world set):

| Provider | Controlled (7 non-speech chunks) | Real-world (33 non-speech chunks) | Real-world music (20 chunks) |
|---|---:|---:|---:|
| Deepgram Nova-3 | 0 | 0 % | 0 / 20 |
| Grok STT 1.0 | 0 | 6 % | 0 / 20 |
| GPT-4o Mini Transcribe | 0 | 12 % | 9 / 20 |
| MAI-Transcribe 2 | 0 | 18 % | 8 / 20 |
| Gemini 3.8 Flash | 2 | 24 % | 2 / 20 |
| Whisper Large V3 Turbo | 6 | 100 % | 20 / 20 |

* **Whisper Large V3 Turbo** filled every real-world non-speech chunk, mostly with short
  tags and stock sign-off phrases, and sometimes inserted a stock phrase into real speech.
* **Gemini 3.8 Flash**, a general chat model asked to transcribe, wrote fluent and
  plausible sentences on near-silent audio, and different ones on each run. That is the
  most dangerous failure for anything that consumes transcripts automatically, because the
  output looks right.
* **MAI-Transcribe 2 and GPT-4o Mini Transcribe** transcribed background song vocals and
  produced short foreign-language or nonsense lines on loud instrumental music.
* **Deepgram Nova-3** returned nothing on every non-speech and music chunk.

A few of the non-empty results on quiet audio were probably faint real lyrics rather than
model errors (four unrelated providers agreed on the same words), so the labels, not only
the models, have some noise. This is why the pipeline does not trust silence detection
alone: it applies an energy gate before sending a chunk, and `lst bench` flags generic
boilerplate and repetition loops so you can check a provider on your own quiet audio.

### Cost (measured per request, projected to continuous audio)

| Provider | USD per audio hour | Per 30 days of continuous audio |
|---|---:|---:|
| Whisper Large V3 Turbo | about 0.01 to 0.02 | about 9 to 17 |
| MAI-Transcribe 2 | 0.10 | 72 |
| Grok STT 1.0 | 0.10 | 72 |
| GPT-4o Mini Transcribe | 0.10 | 72 |
| Deepgram Nova-3 | 0.26 | 184 |
| Gemini 3.8 Flash | 1.2 to 1.3 | about 880 to 930 |

Projections assume every 5 s chunk is sent. Silence gating skips almost nothing on real
stream audio, which is rarely silent. `LST_STT_BUDGET_USD` exists because a
continuous-audio bill adds up.

### Caveats

* **Small n.** About 60 unique controlled chunks, 78 requests per provider for latency,
  and tens of non-speech chunks. Differences of a chunk or two are not significant.
* **One language and one domain.** German, spoken live, one kind of content. Results for
  other languages, studio audio or other speaking styles may differ, in either direction.
* **Laptop measurements.** The machine slept during part of the real-world run, and the
  live runs happened at different times of the evening, so latency is reported per run and
  not treated as a ranking. A server-side run would be needed for a reliability figure.
* **Labels are machine-made.** Ground truth came from two machine references plus a signal
  measure; nobody listened to every chunk.
* **Provider pricing and models change.** The prices are what the requests were billed in
  September 2026. Check the current price list before budgeting, and re-run
  `lst bench` on your own audio before choosing a provider.
* **Not reproducible from this repository.** The audio is not published. Part A is.
