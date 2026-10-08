# Voice compatibility review — 9 October 2026

The implementation and optional speech dependencies are compatible with native arm64 Python 3.12.13 on macOS 15.8.1 / Apple M3. The project remains restricted to Python >=3.12,<3.13.

The installed distribution metadata reports:

| Package | Version | Requires Python |
| --- | --- | --- |
| MLX | 0.32.3 | >=3.10 |
| MLX-LM | 0.32.0 | >=3.11 |
| mlx-whisper | 0.4.3 | >=3.8 |
| mlx-audio | 0.4.3 | >=3.10 |
| Misaki | 0.9.4 | >=3.8,<3.13 |
| sounddevice | 0.5.3 | >=3.7 |
| spaCy | 3.8.16 | >=3.9,<3.15 |

No upgrades to the pinned MLX or speech-library versions were required. The reviewed lockfile had omitted the optional speech dependency graph; it now resolves 129 packages for Python 3.12, including the complete voice extra. Transitive package versions and native Python 3.12 wheels are fixed by uv.lock.

The English Kokoro setup was incomplete: the worker requires `en_core_web_sm`, but installing the previous voice extra did not provide it. The extra now includes the official 3.8.0 wheel from the spaCy model release, constraining spaCy to its compatible 3.8 series. Misaki's packaged dictionaries and espeakng-loader's native library/data also worked offline. README setup instructions and the missing-asset message now match the dependency graph.

Implementation review confirmed that both I/O modes share the existing Engine/Conversation generation loop, optional speech imports remain lazy, Whisper receives mono float32 PCM at 16 kHz, Kokoro receives an absolute voice-file path, and speech inference runs in a reusable local spawn worker with socket connections denied. Cancellation terminates the worker and playback aborts queued device audio. A startup-failure cleanup gap was fixed: both pipe ends and the unstarted process are now closed if spawning fails. A fake-process regression test covers this path.

Verification performed:

- `uv sync --locked --python 3.12 --extra voice --group dev`: passed.
- `uv run --locked --extra voice --python 3.12 pytest -q`: **115 passed in 22.32s**, including existing core/API and voice adapter tests; execution had Metal/localhost access.
- `uv lock --check --offline`: passed, 129 packages.
- `scripts/voice_dependency_smoke.py`: real imports/API signatures, MLX device access, Whisper tokenizer, PortAudio library and English Kokoro phonemizer passed with socket connections blocked before speech imports. PortAudio reported V19.7.0-devel.
- `git diff --check`: passed.

These checks did not download Whisper/Kokoro weights, run real speech inference, open a microphone/playback device, or exercise interactive recording controls. End-to-end voice inference and device cancellation still require the README's manual Apple Silicon smoke test. Japanese/Chinese Kokoro phonemizers remain outside this extra. Prior verification reports describe their original checks and environments; this report supersedes their note that the English spaCy asset must be installed separately.

## Project asset staging

The project now stages model snapshots under `assets/models/<owner>/<repository>/` and packaged phonemizer/tokenizer resources under `assets/runtime/`. `MICA_ASSETS_DIR` overrides the root; assets are Git-ignored. Snapshot symlinks are dereferenced, and loaders prefer staged assets before looking in the global Hub cache.

The default Llama 3B (1.70 GiB), Whisper small (0.45 GiB), and Kokoro (0.36 GiB) repositories were copied/downloaded successfully. Runtime copies include en_core_web_sm, Misaki, espeakng-loader and mlx-whisper. No symlinks remain in the staged tree. Online `voice-download` completed real local synthesis without opening an audio device. With `HF_HOME` pointing to an empty temporary cache, `voice-download --offline` also loaded the speech models and synthesized successfully, with the speech worker denying socket connections.

The complete suite passed **123 tests**, including eight asset-storage tests for cache-independent copies, staged offline lookup, packaged resources and path validation. Targeted checks also cover LLM provenance preservation and retry behavior for interrupted staging. Interactive microphone capture, transcription of live speech, and audible playback have not been performed.
