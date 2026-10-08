# Python 3.12 migration verification — 8 October 2026

The project now requires Python `==3.12.*`. `.python-version` selects 3.12, and the shared runtime platform check rejects other Python minor versions for CLI and backend use. Existing Apple Silicon/macOS requirements remain in place.

The project virtual environment was rebuilt with native arm64 CPython 3.12.13 using `uv sync --locked --python 3.12 --extra voice --group dev`. MLX 0.32.3, MLX-LM 0.32.0 and the pinned optional speech dependencies resolved and installed without changing their selected versions.

Validation performed:

- `.venv/bin/python --version`: Python 3.12.13.
- Installed Mica metadata: `Requires-Python: ==3.12.*`.
- `.venv/bin/python -m pytest -q`: **111 passed in 27.22s**. Tests ran with Metal and localhost access, including four explicit checks accepting 3.12 and rejecting 3.10, 3.11 and 3.13.
- `uv lock --check --offline`: passed; 128 packages resolved from the existing cache.
- Project metadata, lockfile and interpreter pin consistency: passed.
- `git diff --check`: passed.

This migration did not rerun real-model benchmarks or interactive microphone/Whisper/Kokoro playback. Earlier hardware reports remain historical Python 3.11 results. The current voice extra does not install the separate `en_core_web_sm` English phonemizer model; that asset is still required before English offline speech, as documented in the README.
