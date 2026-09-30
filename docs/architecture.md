# Architecture

This page expands the diagram in the README: how one run is organised, where the queues and
threads are, what each failure does, and where to plug in a new provider, notifier or sink.

## One run, end to end

`lst run`, `lst record` and `lst replay` all build a `SessionPlan` and hand it to a
`SessionRunner` (`pipeline/session.py`). The runner owns what outlives a single connection:
signal handling, the availability state and heartbeat, the SQLite database and the
notification dispatcher (so events that were not delivered when one session ended are
retried in the next).

```
SessionRunner.run()
  loop:
    wait_until_live()          probe candidates (ProbeBackoff), alert on state changes
    connect()                  resolve URL (yt-dlp, or direct for m3u8 / http / file), start ffmpeg
    drive audio into a fresh TranscriptionPipeline until the source ends
    reselect / fail over       when a segment ends empty or a better candidate is live
    end: exit, or (live source, resume enabled) go back to waiting
```

## Data path inside a session

```
on_audio -> AudioJobQueue -> STT workers -> MediaOrderGate -> SpeechStream -> sinks
            (RAM + spill)    (thread pool)  (media order)      (dedup, stitch)   |
                                                                                 v
                                                            rule engine -> dispatcher
```

| Stage | Runs on | Bounded by | When it is full or slow |
|---|---|---|---|
| ffmpeg capture (`stream/source.py`) | asyncio + subprocess | `LST_CAPTURE_QUEUE_SIZE` (live) or `LST_CAPTURE_FILE_QUEUE_SIZE` (files) | Live: drop the oldest chunk, count it. Files and replay: the producer waits. |
| `AudioJobQueue` (`audio/lane.py`) | event loop | `LST_STT_QUEUE_SIZE` in RAM, then `LST_STT_SPILL_CHUNKS` on disk | Live: drop the oldest waiting job, count it. Replay: wait. |
| STT workers (`stt/`) | thread pool, `LST_STT_WORKERS` | request timeout, circuit breaker | Failure becomes an `unavailable` transcript, never `None`. |
| `MediaOrderGate` (`audio/ordering.py`) | event loop | number of outstanding chunks | Holds finished chunks until every earlier one is back. |
| `SpeechStream` (`audio/speech.py`) | event loop | fixed history | Emits an utterance once, or as a revision of the previous one. |
| Rules and notifications | own stages behind small queues | queue size, rate limiter | A slow webhook delays alerts, never transcripts. |

The `OverloadGuard` (`pipeline/overload.py`) sits beside the job queue. It takes lag and drop
ratio as numbers and returns verdicts (`pause`, `probe`, `resume`); it has no asyncio in it
and an injectable clock.

## Three clocks

Every `AudioChunk` carries `media_ts` (seconds since the start of the capture segment, from
counting samples), `ts` (seconds since the session began, continuous across reconnects) and
`wallclock` (when it was read). Ordering and windows use `ts`, text is placed on the timeline
with `media_ts`, and latency is measured against `wallclock`. Rule cooldowns use stream time
so a replay reproduces the live alerts.

## Failure handling at a glance

| Failure | Response |
|---|---|
| ffmpeg exits or the network drops | Supervisor restarts capture after a full-jitter exponential delay; `LST_CAPTURE_MAX_RECONNECT_ATTEMPTS=0` never gives up. |
| Process alive, no audio | Stall watchdog (`LST_CAPTURE_AUDIO_STALL_SECONDS`) restarts capture; a frozen HLS playlist is held and re-checked. |
| Stream offline or ended | Auto-resume probes with back-off and starts a new session when it is stable (`LST_RESUME_*`). |
| Bot check or probe error | Backed off exponentially; alerts once per change, not once per probe. |
| Cloud provider failing | Circuit breaker opens; with `LST_STT_FALLBACK=local` transcription continues on the local model. |
| STT slower than real time | Overload guard pauses STT, probes for recovery, doubles the probe interval after a relapse. |
| Spend limit reached | Budget guard stops paid calls (`LST_STT_BUDGET_USD`). |
| Notifier down | Event is already in SQLite; only the failed targets are retried, also after a restart. |
| Corrupt SQLite file | Moved aside with a `.corrupt-*` suffix and recreated. |
| Invalid configuration | Startup error (exit 2) naming the variable, never the value. |

## Storage

SQLite (WAL, one shared connection guarded by a lock): `sessions`, `segments`, `transcripts`,
`events` and `deliveries`. An event is committed before any notifier runs;
`Database.pending_notifications()` is what the dispatcher retries. Files written next to it:
`transcript.jsonl` (one utterance per line), `transcript.srt` and `transcript.vtt`
(incremental writers; a cue per provider segment when the provider reports times, else one
per chunk). A run never overwrites an earlier one: when `transcript.*` already exists in the
output directory the next files are `transcript-2.*`, and so on.

Recordings (`lst record`, `lst run --record DIR`) are `manifest.json`, `audio.jsonl` and one
WAV per capture segment.

## Extending

* **A new speech-to-text provider:** implement the `Transcriber` protocol from `stt/base.py`,
  a synchronous `transcribe(pcm, sample_rate, *, start, end) -> Transcript | None` that
  returns `None` for silence and an `unavailable(...)` transcript for a failure. Register a
  builder in `stt/factory.py` and add the name to `STT_PROVIDERS` in `config.py`. Wrappers
  (budget, fallback) apply to it automatically.
* **A new notifier:** implement `Notifier` from `notify/base.py` (`name` and
  `send(event) -> bool`; raise `NotifyError` to keep the event pending) and add it to
  `notify/factory.py`.
* **A new output:** implement the `TranscriptSink` protocol from `outputs/base.py` and add
  it to the composite in `pipeline/session.py`.
* **A new source type:** subclass `StreamSource` from `stream/base.py`
  (`connect`, `get_audio`, `close`). `ReplayStreamSource` is the smallest example.
