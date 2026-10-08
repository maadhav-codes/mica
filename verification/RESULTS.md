# Actual verification results — 8 October 2026

Host: Apple M3, 16 GiB unified memory, macOS 15.8.1, native arm64 Python 3.11.15. Installed using uv 0.10.9 and project-local `.venv`; exact packages and hashes are in `uv.lock`.

Tested direct runtime versions: MLX 0.32.3, MLX-LM 0.32.0, huggingface-hub 1.33.0, typer 0.27.3, rich 15.0.0, prompt-toolkit 3.0.53. pytest 9.1.1 is a development dependency.

## Model validation

Both exact preset repositories were accessible. Configuration, built-in architecture dispatch, declared context, official template presence, weight files/indexes and native affine 4-bit packed tensor headers passed validation against the installed MLX-LM release. See [presets.json](presets.json) for exact commit SHAs.

| Preset | Architecture | Packed layers | Weight bytes | Context |
| --- | --- | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct-4bit | llama | 197 | 1,807,496,278 | 131,072 |
| Qwen2.5-7B-Instruct-4bit | qwen2 | 198 | 4,284,346,255 | 32,768 |

Only the 3B weights were downloaded. Its tokenizer/template and weights were loaded by MLX-LM, and its templates were exercised with system, user, and assistant messages. The 7B tokenizer configuration/template was inspected, but 7B weights were not loaded or benchmarked.

## Automated and real hardware checks

`uv run pytest -q` passed all tests; the captured final output is in [pytest.txt](pytest.txt). Coverage includes format/name/magic rejection, actual 4-bit packed shapes, truncation and shard indexes, settings persistence and invalid settings, platform/Rosetta rejection, formatted-token budgeting, whole-turn trimming, oversize-input rejection without state mutation, model context bounds, session validation, incomplete-turn handling, offline resolution, pinned revision lookup, prompt-cache reuse/cancellation and interactive cache invalidation after system/history/session/model changes.

[hardware-smoke.json](hardware-smoke.json) records an actual streaming multi-turn smoke test with socket connections blocked. The model answered “Hello Mira.” then recalled “You told me your name is Mira.” The second turn reused 52 prompt tokens and processed 21 new tokens. SIGINT interrupted generation; the cache cleared, incomplete output survived JSON save/load, and the interrupted pair was discarded before successful regeneration. The old model's weak reference became empty after reload, demonstrating weight release. Generation after reload also passed offline.

A separate real pseudo-terminal check verified Esc+Enter multiline input, streamed output, `/save`, `/load`, Ctrl+C during a 512-token-limit generation, return to the prompt, incomplete session storage, successful next-turn recovery, and Ctrl+D exit with status 0. The first stored user message contained a newline. The cancelled response was explicitly stored with `incomplete: true`.

Metal access failed inside the execution sandbox and succeeded in the approved native process. Hardware tests and benchmarks ran with GPU access outside that sandbox.

## 3B benchmark

Command:

```sh
uv run mica benchmark --offline \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --runs 3 --max-tokens 128 --output verification/benchmark-3b.json
```

Cold model load: **1.705 seconds**, with **1.807 GB** peak MLX allocation during loading. This is a first load in a fresh benchmark process, with OS filesystem caches retained; validation and download time are excluded.

The fixed prompt used 56 formatted tokens; every run had an empty prompt cache. Temperature 0, top-p 1, seed 42, context budget 4096, output limit 128. Each run stopped after 110 library-reported generation tokens.

| Phase | TTFT seconds | Prompt tokens/s | Generation tokens/s | Peak MLX GB | Process peak RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| First generation | 0.289 | 225.56 | 19.07 | 2.002 | 2.294 |
| Warm 1 | 0.308 | 182.79 | 19.12 | 1.996 | 2.294 |
| Warm 2 | 0.302 | 186.35 | 18.87 | 1.996 | 2.294 |
| Warm 3 | 0.310 | 181.20 | 18.30 | 1.996 | 2.294 |

GB in this table means decimal 10^9 bytes. TTFT is wall time to the first token response. Prompt and generation rates are MLX-LM's metrics, including its final stop token in the generation count. MLX peak allocation was reset per generation; process RSS is the process's lifetime high-water mark, not a per-turn measurement. These are distinct memory measurements and need not match. Raw values, device information, prompt, settings and revision are in [benchmark-3b.json](benchmark-3b.json).

## Checks not run

No 7B streaming/benchmark run or switch between distinct model repositories was performed; the real release-and-reload check reused the 3B artifacts, while automated CLI checks exercise distinct model identities. No extended thermal/battery or near-memory-limit stress test was performed. These short runs do not establish sustained fanless-laptop throughput. Real gated authentication failures, a disk-full download and a physical memory exhaustion event were not induced.

To exercise the quality alternative and switch between distinct models:

```sh
uv run mica download --model mlx-community/Qwen2.5-7B-Instruct-4bit
uv run mica benchmark --offline --model mlx-community/Qwen2.5-7B-Instruct-4bit --output verification/benchmark-7b.json
uv run mica chat --offline --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Inside chat, send a message, use `/model mlx-community/Qwen2.5-7B-Instruct-4bit`, send another message, then switch back. No 7B performance figures are claimed.
