# Mica

**Mica — Local conversations, powered by MLX.**

A local conversation tool with a CLI and a [backend API](src/mica/api/README.md) for native arm64 Python 3.12 on macOS Apple Silicon. Inference uses MLX-LM exclusively. Use it directly from Python or opt into its localhost HTTP adapter. No server runs by default, and there are no background daemons, telemetry, conversation uploads, conversion, or alternative inference backends.

## Use as a backend

```python
from mica.api import Mica

with Mica("mlx-community/Llama-3.2-3B-Instruct-4bit") as backend:
    result = backend.generate("Explain why the sky is blue.")
    text = result.text
```

For HTTP access, run `uv run python -m mica.api --model mlx-community/Llama-3.2-3B-Instruct-4bit`, then POST JSON to `http://127.0.0.1:8000/v1/generate`. Both interfaces default to offline model access and reuse the model already downloaded by Mica. See the [API guide](src/mica/api/README.md) for streaming, chat sessions, cancellation and backend integration.

## Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then run from this directory in a native terminal:

```sh
uv python install 3.12
uv sync --locked --python 3.12 --group dev
uv run mica doctor
```

`uv sync` creates the project `.venv`. `uv.lock` pins the complete dependency graph and distribution hashes. The tested inference releases are MLX 0.32.3 and MLX-LM 0.32.0; Python is 3.12.13. For a runtime-only installation use `uv sync --locked --no-dev`. No dependency installation uses pip.

Activation is optional:

```sh
source .venv/bin/activate
mica --help
```

Use `uv run --locked ...` without activation. Once installed, use `.venv/bin/mica` or the activated entry point to avoid uv's dependency checks when working without an internet connection. If using uv itself offline, use `uv run --offline --locked mica ...` in addition to the application's `--offline` flag.

Intel, Linux, Rosetta, macOS before 14, and Python versions other than 3.12 are rejected. Metal must be available in the executing process. macOS sandbox/headless environments may prevent GPU access even on supported hardware; `doctor` explains import failures.

## Select and download a model

```sh
uv run mica models
uv run mica download --model mlx-community/Llama-3.2-3B-Instruct-4bit
uv run mica chat --offline --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

The picker offers verified [Llama 3.2 3B](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) as a speed-oriented starting choice and [Qwen 2.5 7B](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) as a quality-oriented alternative. These are choices to compare, not promises about speed or answer quality. Actual repository revisions and validation evidence are recorded in `verification/presets.json`. Weight sizes are approximately 1.68 GiB and 3.99 GiB, respectively; tokenizer and cache storage are additional.

```sh
uv run mica download --model mlx-community/Qwen2.5-7B-Instruct-4bit
uv run mica chat --offline --model mlx-community/Qwen2.5-7B-Instruct-4bit
uv run mica chat --offline --model /absolute/path/to/native-mlx-model
uv run mica chat
```

The final command uses the last successfully selected model, or opens a picker on first use. Repository IDs beyond these presets are accepted after the same validation. Local directories work entirely offline. Models needing custom tokenizer/architecture code are rejected. `models` validates configuration, supported architecture, template presence, safetensors header packing and shard indexes; unavailable/incompatible presets are omitted with a reason. Listing may retrieve small configuration/index files and tensor header ranges, but never saves full weights. Only selecting/downloading a model retrieves its weights. Repositories are downloaded at the validated commit SHA. Existing complete Hugging Face snapshots are reused with no network checks. Gated models require license acceptance and `uv run hf auth login` or `HF_TOKEN`.

## Conversation controls

```sh
uv run mica chat --offline \
  --system 'Answer clearly and briefly.' \
  --temperature 0.7 --top-p 0.9 --seed 42 \
  --context-budget 4096 --max-tokens 512 --memory-gb 9
```

Enter sends. Esc then Enter inserts a newline. Terminal history supports recalling multiline input. Ctrl+C cancels generation and returns to the prompt, or clears current input. EOF exits. Sampling settings and the last model persist after successful loading.

| Command | Effect |
| --- | --- |
| `/help` | Show controls |
| `/model [ID or directory]` | Pick/switch model; clear conversation and release previous weights/caches |
| `/new` | Clear conversation and cache |
| `/save [path]` | Save JSON session; defaults to `sessions/last.json` under the configuration directory |
| `/load [path]` | Restore model identity, settings, and messages; rebuild cache |
| `/system [instructions]` | Replace system instructions; omit instructions to clear them; invalidate cache |
| `/stats` | Show most recent generation measurements |
| `/exit` | Exit |

Quote file paths containing spaces in `/save` and `/load`. Session files preserve incomplete assistant output with `incomplete: true`. Before the next successful input, that interrupted user/assistant pair is removed from active context. Save it first if you want to retain the partial answer. Generation errors use the same incomplete-turn policy. Exiting does not silently autosave a conversation.

The official tokenizer chat template formats role boundaries and system instructions. The total context budget defaults to 4,096 tokens, capped at the model's declared context. The output limit defaults to 512 and is reserved before input. Oldest complete user/assistant pairs are removed until the formatted prompt fits. A single input that cannot fit with system instructions and template overhead is rejected without changing history. Unsupported system roles produce an actionable template error; instructions are never silently folded into a user message.

Prompt caches reuse only exact matching token prefixes. MLX-LM's trim API removes generated suffixes before reuse. Non-trimmable caches are rebuilt. Trimming history, changing system instructions, loading a session, switching models, cancellation, and generation errors invalidate caches. Cache size is bounded by the conversation's context/output budget rather than a rotating cache that could discard still-active context.

## Accepted artifacts and memory

Acceptance requires native MLX affine quantization metadata (`bits: 4`, supported group size), packed uint32 matrix weights, floating scales/biases, and tensor shapes consistent with eight 4-bit weights per uint32. All matrix weight tensors must satisfy this validation; ordinary one-dimensional normalization weights and biases may be unquantized. Mixed-bit artifacts are rejected. GPTQ, AWQ, GGUF, Q4_K_M, and Q4_0 are rejected before weight loading. Files are never converted or requantized. MLX-LM constructs its quantized modules when loading an already quantized artifact; this does not convert source weights. Transformers is installed transitively only for MLX-LM tokenizer utilities; the application never loads or runs a Transformers model.

The default MLX memory guideline is 9 GiB, further capped at 60% of device memory. MLX's allocation cache is limited to 256 MiB; prompt prefill uses 256-token chunks. This guideline is not a hard process-memory cap or a guarantee against swap. Weight-size checks reserve estimated headroom before loading, while KV cache, activations, Python, and other applications add memory use. A fanless laptop can slow under sustained workloads. Start with the 3B model and modest context, close memory-heavy apps, and use measured results. The application changes supported process-local MLX controls only; no system-wide wired-memory setting is altered.

Downloads display estimated weight size and available disk space, with an additional 512 MiB margin required. Hugging Face's normal cache is reused. Interrupted downloads can be retried with the same command. Full-precision sources and incompatible artifacts are never fetched for conversion.

Configuration, downloaded-model revision records, and terminal history live in `~/Library/Application Support/Mica`. Existing installations reuse `~/Library/Application Support/localchat` when the new directory is absent, preserving settings, sessions, history, and downloaded-model revisions. Set `MICA_HOME` to change this directory; the previous `LOCALCHAT_HOME` variable is also supported. JSON files are written atomically with owner-only permissions. Conversations and history stay local. Hugging Face metadata/download requests are made only when needed online. `--offline` uses local files/cache and sets Hub/tokenizer offline modes before importing them; it prevents metadata checks and downloads. The revision registry ensures pinned downloads remain usable even without a cached `main` reference. If changing `MICA_HOME`, use a model's local snapshot directory or download/register it under the new configuration directory.

## Verify and benchmark

```sh
uv run pytest -q
uv run python scripts/verify_presets.py
uv run python scripts/hardware_smoke.py
uv run mica benchmark --offline \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --runs 3 --max-tokens 128 --output verification/benchmark-3b.json
```

The hardware smoke script expects the 3B model to have been downloaded. It blocks socket connections, runs a real streaming multi-turn conversation, checks prompt reuse, raises SIGINT, saves/reloads the incomplete session, regenerates, and verifies old weights are collected before reloading. The 7B model is not automatically downloaded by any verification script.

Benchmark settings and prompt are saved alongside measurements. Cold load means first load in the benchmark process; download/validation time is excluded and OS disk caches are not flushed. The first generation is reported separately from subsequent warm runs. Each benchmark prompt starts with an empty KV cache, measuring warm weights rather than a cached-prefix hit. TTFT measures elapsed time to the first MLX-LM token response. Prompt-processing speed and generation tokens/second are MLX-LM's reported values; prompt speed covers only the uncached suffix when caches are reused. Token counts include MLX-LM's final stop response. Peak MLX allocation bytes are reset per generation; process peak RSS is a lifetime high-water mark on macOS, not a per-turn increment. Load peak allocation is reported separately. Read `verification/RESULTS.md` for actual results and outstanding checks.

Implementation follows the [official MLX-LM API and source](https://github.com/ml-explore/mlx-lm) and [MLX memory-control documentation](https://ml-explore.github.io/mlx/build/html/python/memory_management.html). The installed 0.32.0 source was inspected for loader behavior, architecture dispatch, streaming metrics, and cache trimming; `uv.lock` fixes the tested APIs.

## Local voice mode

Text remains the default. Install the optional speech extra, prepare assets while online, then start voice mode:

```sh
uv sync --locked --python 3.12 --extra voice --group dev
uv run --locked --extra voice mica voice-download
uv run --locked --extra voice mica chat --io-mode voice --offline \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

At `Voice>` press Enter to start recording, then Enter again to stop and submit. Recording also stops automatically at the configured duration limit. Typed slash commands remain available before recording starts. `/mode` shows the mode; `/mode text` and `/mode voice` switch within the same conversation without reloading the language model or invalidating its KV cache. In voice mode typed messages require switching to text. Transcribed slash commands are ordinary user content. `/load` restores voice settings as part of the session and recreates speech resources if voice is active; speech setup failure returns to text mode.

The terminal displays Recording, Transcribing, Generating, and Speaking states. It displays exactly one `You (voice)>` transcript before using the existing generation loop. Silence/empty transcripts do not create turns. Assistant text streams as usual while completed sentences are synthesized and played in order. A bounded queue holds up to four sentences; backpressure slows text generation if playback falls behind. The final fragment is spoken when generation finishes. Fenced code is buffered until its closing fence and omitted from speech. Recording remains half-duplex: it never overlaps generation or playback. Kokoro inference runs in the existing local speech process while the CLI thread continues MLX-LM generation; the two models can contend for GPU memory and throughput. Ctrl+C cancels recording/transcription without creating a turn; during generation it stops active/queued speech and retains the existing incomplete-turn semantics (already spoken sentences cannot be recalled); during speech it stops speech and keeps the completed text. Speech errors also keep completed text. `/mode text` releases speech resources; failed voice setup leaves text mode usable. Models remain cached in one local speech worker between successful voice turns; cancellation/errors terminate that worker and the next turn reloads it. No server is started.

Configure speech using the `voice` object in Mica's `config.json` (merge this object into the existing JSON):

```json
{
  "voice": {
    "whisper_model": "mlx-community/whisper-small-mlx",
    "language": "auto",
    "kokoro_model": "mlx-community/Kokoro-82M-bf16",
    "voice": "af_heart",
    "speed": 1.0,
    "microphone": null,
    "output_device": null,
    "max_duration": 60.0
  }
}
```

Both models accept a repository ID or absolute local directory. Whisper `language` accepts `auto` or an ISO code recognized by the installed mlx-whisper tokenizer (for example `en`, `hi`, `fr`). English-only models reject other explicit languages. Kokoro's language is derived independently from the voice prefix: `a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p` Brazilian Portuguese. Use a matching voice safetensors file actually present in the selected model's `voices/` directory. The default repository includes `af_heart`; other repositories may supply additional languages. Japanese and Chinese Kokoro phonemizers are outside this initial extra. Automatic Whisper language detection does not select a different Kokoro voice; choose a voice suitable for the assistant's output language. Voice files/tensor dimensions, model architecture and tokenizer language support are checked on setup. Speed must be finite and between 0.5 and 2.0; duration between 1 and 300 seconds. Devices accept null (system default), a device name, or a nonnegative numeric index. List devices with `uv run --locked --extra voice python -m sounddevice`. Grant your terminal microphone access in macOS System Settings → Privacy & Security → Microphone. Devices must support mono float32 input at 16 kHz and output at 24 kHz.

The locked speech APIs are mlx-whisper 0.4.3, mlx-audio 0.4.3, Misaki 0.9.4 and sounddevice 0.5.3. The dependency lock targets Python 3.12 alongside MLX 0.32.3 and MLX-LM 0.32.0. Misaki 0.9.4 requires Python >=3.8,<3.13, so Python 3.12 is supported and the project excludes 3.13. See [Whisper's source](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and [Kokoro's pipeline](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/kokoro/pipeline.py). Misaki's English phonemizer uses `en_core_web_sm` 3.8.0, installed from a pinned wheel by the voice extra before offline English speech use. Its dictionaries are packaged with Misaki. `espeakng-loader` bundles the native eSpeak NG library and language data used by the English fallback and other supported languages; no Homebrew phonemizer installation is required. The extra has substantial transitive dependencies (including spaCy and Torch for phonemizer dependencies); inference uses the local MLX models. Text-only installs need none of these speech packages.

`voice-download` explicitly fetches complete speech snapshots and runs a short synthesis without opening any audio devices, verifying the configured voice and phonemizer. Re-run after changing repositories/voice. Local directories must already contain complete MLX assets: Whisper config/weights, Kokoro config/weights/voice safetensors. Speech inference uses absolute local asset paths, offline library flags and denied socket connections. A missing spaCy model is rejected before Misaki can invoke its downloader. `--offline` never checks remote metadata or fetches speech assets; missing assets name the setup command. Download/install every dependency and language model before disconnecting networking. For a fully cached offline launch use:

```sh
.venv/bin/mica voice-download --offline
.venv/bin/mica chat --offline --io-mode voice --model YOUR_CACHED_LLM
```

Or use `uv run --offline --locked --extra voice ...` along with Mica's `--offline`. Speech adds memory beyond the language engine's guideline: use the small Whisper default, the 3B LLM and modest context initially. The microphone buffer is bounded (at most 300 seconds of mono 16 kHz PCM). Captured audio is held only in memory, never saved by default. Synthesis uses sentence chunks with a bounded queue and playback backpressure, and device streams are closed on success, error and cancellation. Session JSON stores transcripts and raw assistant text. Markdown/code filtering applies only to the spoken copy: fenced code is announced as omitted, links are read by their labels, and common formatting markers are removed.

### Manual Apple Silicon voice smoke test

These steps require interactive audio hardware and have **not been performed** as part of automated verification:

1. Install the extra, download the LLM, run `mica voice-download`, and start text chat. Send a typed turn; switch with `/mode voice` and confirm the previous conversation is remembered without another LLM load.
2. Press Enter, speak, and press Enter again. Confirm microphone permission, a single displayed transcript, streamed response, and audible Kokoro playback. Verify no capture occurs during playback. Wait through the duration cap on another recording and confirm automatic submission.
3. Record silence and confirm no turn is added. Say a slash command such as “slash new”; confirm it is user content and does not clear history.
4. Press Ctrl+C during recording, transcription, language generation, synthesis, and playback. Confirm return to the prompt, no delayed playback, and the expected history policy. Save/load sessions to inspect incomplete flags. Switch modes repeatedly and confirm devices remain available to other apps.
5. Test an unavailable device, revoked microphone permission, unknown voice, and missing cached model/voice. Confirm actionable messages and successful text chat after failed voice setup. Confirm a TTS failure retains a successful text turn.
6. Disable Wi-Fi/Ethernet after setup. Run the two offline commands above, record/transcribe/generate/play multiple turns, save/load, and switch modes. Confirm no downloads or network prompts. Restore networking when finished.

Automated fake-adapter coverage in `tests/test_voice.py` covers shared conversation routing, mode switching, cancellation at each stage, empty transcription, TTS failure, cache-only asset resolution, resource cleanup and text-only dependency isolation. Run `uv run --locked --extra voice pytest -q` for the complete suite.

To verify the installed speech APIs, Whisper tokenizer, PortAudio library and English phonemizer without downloading weights or opening audio devices:

```sh
uv run --locked --extra voice python scripts/voice_dependency_smoke.py
```

This check blocks socket connections before importing speech libraries. On Python 3.12.13, the real dependency smoke check and all 115 automated tests passed; live microphone recording and Whisper/Kokoro inference/playback remain manual checks. See [voice compatibility results](verification/voice-python312.md).

## Project asset folder

Download all assets needed for the default text/voice setup with:

```sh
uv sync --locked --python 3.12 --extra voice
uv run --locked --extra voice mica download --model mlx-community/Llama-3.2-3B-Instruct-4bit
uv run --locked --extra voice mica voice-download
```

The download commands populate `assets/` under the current working directory:

- `models/<owner>/<repository>/`: language/speech model weights, configs, tokenizers and Kokoro voice files. Complete snapshot files are copied with Hub symlinks dereferenced.
- `runtime/en_core_web_sm/`: English spaCy model.
- `runtime/misaki/`: phonemizer code and dictionaries.
- `runtime/espeakng_loader/`: bundled eSpeak native library and language data.
- `runtime/mlx_whisper/`: Whisper code and packaged tokenizer/filter resources.

These runtime copies are selected by the speech worker before importing speech packages. Python 3.12 and the installed voice dependencies are still required; `assets/` is not a replacement for the virtual environment. The folder is Git-ignored because it contains large binaries. Set `MICA_ASSETS_DIR=/absolute/path/to/assets` to choose a fixed location, especially when running from a different directory. Both model loaders prefer this folder before the global Hub cache. Existing global caches remain available for migration; download commands copy their contents into the asset folder. Explicit local model directory arguments remain supported.

Once setup completes, run from the project directory (or set `MICA_ASSETS_DIR`):

```sh
.venv/bin/mica voice-download --offline
.venv/bin/mica chat --offline --io-mode voice --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Use `mica download --model OTHER_LLM` for additional language models. `voice-download` stages the Whisper/Kokoro repositories configured in `config.json`; it does not automatically download every alternative model. Asset provenance files record repository/revision or packaged resource names. Raw recordings and saved conversations are not stored in this folder.

Sentence speech now overlaps LLM generation. Fake-adapter tests cover first-sentence playback before generation ends, token boundaries, fenced code, ordering, bounded queues, cancellation, and speech failures. A real offline Llama/Kokoro overlap check was also performed without opening audio devices; details are in [streaming speech verification](verification/streaming-speech.md).
