# Mica

**Mica — Local conversations, powered by MLX.**

A local conversation tool with a CLI and a [backend API](src/mica/api/README.md) for native arm64 Python 3.11 on macOS Apple Silicon. Inference uses MLX-LM exclusively. Use it directly from Python or opt into its localhost HTTP adapter. No server runs by default, and there are no background daemons, telemetry, conversation uploads, conversion, or alternative inference backends.

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
uv python install 3.11
uv sync --locked --python 3.11 --group dev
uv run mica doctor
```

`uv sync` creates the project `.venv`. `uv.lock` pins the complete dependency graph and distribution hashes. The tested inference releases are MLX 0.32.3 and MLX-LM 0.32.0; Python is 3.11.15. For a runtime-only installation use `uv sync --locked --no-dev`. No dependency installation uses pip.

Activation is optional:

```sh
source .venv/bin/activate
mica --help
```

Use `uv run --locked ...` without activation. Once installed, use `.venv/bin/mica` or the activated entry point to avoid uv's dependency checks when working without an internet connection. If using uv itself offline, use `uv run --offline --locked mica ...` in addition to the application's `--offline` flag.

Intel, Linux, Rosetta, macOS before 14, and Python versions other than 3.11 are rejected. Metal must be available in the executing process. macOS sandbox/headless environments may prevent GPU access even on supported hardware; `doctor` explains import failures.

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
