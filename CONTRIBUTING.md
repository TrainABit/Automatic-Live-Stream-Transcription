# Contributing

Thanks for taking a look. The project is small and the bar is simple: tests pass offline,
lint and types are clean, and a change explains *why* in its commit message.

## Setup

```bash
git clone https://github.com/TrainABit/Automatic-Live-Stream-Transcription.git
cd Automatic-Live-Stream-Transcription
uv venv --python 3.11
uv pip install -e '.[dev]'          # add ,local to run the faster-whisper tests for real
source .venv/bin/activate
```

You also need `ffmpeg` on the `PATH`; a few tests decode a synthetic clip with it.

## Checks

Run all four before you open a pull request. CI runs the same commands on Python 3.11
and 3.12.

```bash
pytest                              # offline; network access is blocked by a fixture
ruff check .
ruff format --check .
mypy src
```

* `LST_RUN_SLOW=1 pytest` also runs the tests that start a real ffmpeg process.
* Tests that need the public internet carry the `network` marker and are opt-in
  (`LST_RUN_NETWORK=1`). Nothing in the default run may touch the network.
* Test audio is generated (ffmpeg `lavfi` sources or NumPy-written WAVs). Do not commit
  recordings, models or databases.
* After changing `src/livestream_transcriber/config.py`, run
  `python scripts/gen_config_docs.py` to refresh `docs/configuration.md` and `.env.example`.

## Conventions

* Python 3.11+, type hints everywhere (`mypy --strict`-ish), small modules, dataclasses
  over ad-hoc dicts. Docstrings explain *why* a thing exists, not what the code already says.
* Standard library HTTP only (`netutil.py`); tests guard and patch it.
* Anything that can carry a secret goes through `redact.py` before it is logged.
* New settings are `LST_`-prefixed fields on `Settings`, validated at startup, with a
  docstring under the field (the docs are generated from it).
* Commits are conventional and scoped: `feat(stream): ...`, `fix(rules): ...`,
  `docs: ...`, `test(pipeline): ...`, `chore(ci): ...`. Scope is the package directory.

## Reporting problems

Open an issue with the output of `lst doctor` (it hides secrets) and the command you ran.
Please redact stream URLs that carry tokens before pasting them.
