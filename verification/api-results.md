# Backend API verification — 8 October 2026

The existing 52 checks and 26 new API checks passed: **78 tests passed**. The new tests cover stateless and isolated-session generation, settings overrides, partial streams, cancellation, session save/load/reset, model changes, one-model ownership, concurrent-operation rejection, JSON and NDJSON HTTP responses, streaming cancellation, HTTP errors, optional authentication, loopback binding, request size/JSON validation, Host/Origin handling, and bounded session storage.

Real native MLX inference was exercised on the Apple M3 with the already downloaded `mlx-community/Llama-3.2-3B-Instruct-4bit` artifact. No additional model was downloaded. Remote socket connections were blocked; only loopback sockets were allowed for HTTP verification.

- Python multi-turn generation returned “Hello Mira!” then “Your name is Mira.”
- The second turn reused 47 prompt tokens and processed 19 new tokens.
- Closing a partial stream saved an incomplete assistant response; save/load and next-turn recovery passed.
- The local HTTP adapter returned “Hello!” from a real generation request.
- Its streaming endpoint returned text deltas for “Ready.” followed by a `done` event with real inference measurements.

Raw smoke results are in [api-smoke.json](api-smoke.json). These short output checks are functional tests, not a sustained throughput benchmark.

Repeat with:

```sh
uv run pytest -q
uv run python scripts/api_smoke.py
```

The HTTP adapter is local-only and opt-in. No external deployment or browser frontend integration was tested. Concurrency/error cases use a deterministic fake engine; the real smoke separately verifies native MLX operation through both public interfaces.
