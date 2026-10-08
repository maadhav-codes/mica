# MLX HTTP thread-affinity fix — 8 October 2026

The reported failure was reproduced in a fresh process: load the model on the main thread, then generate from another thread. MLX-LM failed at `generate_step` while evaluating the prompt cache with `RuntimeError: There is no Stream(gpu, 0) in current thread.`

Mica now owns a single persistent inference worker. It resolves/loads the model, evaluates every generation step, invalidates caches and closes streams/models on that same worker. HTTP request threads handle input/output and await that worker. Existing single-generation locking and cooperative cancellation remain in place. No inference backend, dependency or quantization format changed.

The original API smoke generated text through Python before sending HTTP requests, so it did not cover cold HTTP startup. The new `scripts/api_cold_start_smoke.py` starts a fresh `python -m mica.api` subprocess and makes the exact reported request as its first generation:

```json
{"prompt":"Explain MLX briefly.","options":{"max_tokens":128,"temperature":0.3}}
```

Real Apple M3 checks passed for that first request (HTTP 200 with text), NDJSON streaming, multi-turn session memory and prompt-cache reuse across fresh HTTP request threads, cancellation/incomplete-turn recovery and SIGINT shutdown with exit code 0. Raw results are in [api-cold-start.json](api-cold-start.json). These are functional checks rather than answer-quality or throughput claims.

New automated regressions verify that load, generation, stream cleanup, invalidation, model switching and release all use one worker thread, even when callers use several different threads. They also verify that a rejected model selection preserves the current session.

The full suite passed: **80 tests passed**.

Repeat using:

```sh
uv run pytest -q
uv run python scripts/api_cold_start_smoke.py
```

Restart an already running API process after updating the source; a live process retains its previously imported code.
