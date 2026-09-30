#!/usr/bin/env bash
# Generate a small, reproducible benchmark set: ten short clips (five English, five
# German) with a reference transcript next to each one.
#
#   scripts/make_tts_clips.sh [OUTPUT_DIR]        # default: clips/
#
# Speech comes from the macOS `say` command, or from `espeak-ng` where `say` is not
# available. Audio is converted to 16 kHz mono WAV with ffmpeg. The sentences are fixed,
# so two runs on the same machine give the same references.
#
# Synthetic speech is clean, so error rates measured on it are optimistic; see
# docs/benchmarks.md.
set -euo pipefail

out="${1:-clips}"
en_voice="${LST_TTS_EN_VOICE:-Samantha}"   # macOS voice names; ignored by espeak-ng
de_voice="${LST_TTS_DE_VOICE:-Anna}"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required" >&2; exit 1; }
if command -v say >/dev/null 2>&1; then
  engine=say
elif command -v espeak-ng >/dev/null 2>&1; then
  engine=espeak-ng
else
  echo "need macOS 'say' or 'espeak-ng' to synthesise speech" >&2
  exit 1
fi

mkdir -p "$out"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# name|language|sentence (no numerals: models write digits, which would be scored as errors)
clips=(
  "en_01|en|Welcome to the stream. Today we are going to talk about the new release."
  "en_02|en|The weather forecast says it will rain in the afternoon, so bring an umbrella."
  "en_03|en|Please remember to subscribe and leave a comment if you have any questions."
  "en_04|en|The new version adds support for offline playback and a faster start up."
  "en_05|en|We will take a short break now and continue again in a few minutes."
  "de_01|de|Willkommen zum Stream. Heute sprechen wir über die neue Version."
  "de_02|de|Der Wetterbericht sagt, dass es am Nachmittag regnen wird, also nehmt einen Regenschirm mit."
  "de_03|de|Vergesst nicht, den Kanal zu abonnieren und einen Kommentar zu schreiben, wenn ihr Fragen habt."
  "de_04|de|Die neue Version bringt Unterstützung für die Wiedergabe ohne Internetverbindung."
  "de_05|de|Wir machen jetzt eine kurze Pause und sind in einigen Minuten wieder da."
)

for entry in "${clips[@]}"; do
  IFS='|' read -r name lang text <<<"$entry"
  raw="$tmp/$name"
  if [ "$engine" = say ]; then
    voice="$en_voice"; [ "$lang" = de ] && voice="$de_voice"
    say -v "$voice" -o "$raw.aiff" -- "$text"
    src="$raw.aiff"
  else
    espeak-ng -v "$lang" -w "$raw.wav" -- "$text"
    src="$raw.wav"
  fi
  ffmpeg -nostdin -loglevel error -y -i "$src" -ar 16000 -ac 1 -c:a pcm_s16le "$out/$name.wav"
  printf '%s\n' "$text" >"$out/$name.txt"
  echo "wrote $out/$name.wav"
done
echo "done: ${#clips[@]} clips in $out (engine: $engine)"
