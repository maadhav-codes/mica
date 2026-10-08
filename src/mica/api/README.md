# Mica backend API

Use Mica directly from Python, or run its optional localhost HTTP adapter. Both interfaces use the same validated native MLX 4-bit model, official chat template, context budgeting, session storage and inference engine as the CLI. No web framework or additional dependency is installed. Importing `mica.api` never loads weights, downloads a model or starts a server.

## Python integration

Install the project with `uv sync --locked`. In your own uv project, add this checkout with `uv add /absolute/path/to/mica`. Run on native arm64 Python 3.11 on macOS 14+ with Metal access. Download a selected model once, then use the API offline:

```python
from mica.api import Mica, Settings

with Mica(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    offline=True,
    settings=Settings(temperature=0.7, max_tokens=512),
) as backend:
    result = backend.generate("Explain why the sky is blue.")
    print(result.text)
    print(result.stats)
```

`generate()` is stateless unless you pass a session. It returns a `GenerationResult` containing `text`, `model`, `session_id`, `stats` and `incomplete`. Use `result.to_dict()` in your application's JSON response. The API produces no terminal output and does not change CLI settings/history.

```python
from contextlib import closing
from mica.api import Mica

with Mica("mlx-community/Llama-3.2-3B-Instruct-4bit") as backend:
    session = backend.new_session(system="Answer briefly.")
    backend.generate("My name is Mira.", session=session)
    with closing(backend.stream("What is my name?", session=session)) as stream:
        for delta in stream:
            send_to_your_frontend(delta)  # Replace with your own callback.
    backend.save_session(session, "conversation.json")
```

Streams yield strings. Always consume the iterator fully or close it; `contextlib.closing` guarantees cleanup on early return, exceptions or disconnection. Partial assistant output is stored with `incomplete: true`, and its user/assistant pair is removed before the next successful turn. Closing a stream clears its KV cache. `backend.cancel()` can be called from another thread and returns whether generation was active; cancellation is observed at a token boundary, so long prompt prefill is not interrupted immediately.

Each session has isolated messages and system instructions. Switching sessions invalidates the shared prompt cache. `session.messages` and `session.settings` return copies. Only the owning backend can use a session.

Additional operations:

```python
session = backend.new_session()
result = backend.generate(
    "Hello", session=session,
    options={"temperature": 0.3, "top_p": 0.9, "seed": 7, "max_tokens": 128},
)
backend.set_system(session, "Use plain language.")
backend.reset(session)
backend.save_session(session, "conversation.json")
restored = backend.load_session("conversation.json")
backend.switch_model("/path/to/another/native-mlx-model")
```

Per-request `options` may contain only `temperature`, `top_p`, `seed`, `context_budget`, and `max_tokens`. They do not change session defaults. Sessions from a previous model cannot be used after switching; create a new session or load a saved session with the selected model identity. A failed model load can leave the instance without a loaded model; close it and create another.

Keep **one Mica instance per process** and reuse it. A second live instance raises `MicaBusyError`. Simultaneous generation/session mutations also raise `MicaBusyError`; finish or close the active stream before another operation. This protects one loaded model and its cache. Mica owns one persistent inference worker: model loading, generation, cache invalidation, cancellation cleanup and model release run on that same thread, preserving MLX GPU stream ownership even when HTTP requests arrive on different threads. In an async web application, call the synchronous API from a worker thread and serialize inference; do not block the event loop or create a Mica instance for each request. Avoid multiple worker processes, each of which would load its own model.

Offline is the default. `offline=False` explicitly permits a missing selected model to be validated/downloaded. Prefer provisioning model artifacts once before starting your backend. Hub/tokenizer offline environment flags are process-wide and read during library import; use a fresh online process for downloads after enabling offline mode. All inference remains local in either mode.

## Optional HTTP adapter

The adapter starts only when explicitly requested. It binds to **127.0.0.1** and defaults to offline model access:

```sh
uv run python -m mica.api \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --port 8000
```

Add `--online` only to allow model metadata checks/downloads. Set `MICA_API_TOKEN` to require `Authorization: Bearer <token>` on every request. No tokens or request contents are logged. This adapter is intended for a local backend; it rejects browser-origin requests and provides no CORS access. Have your own backend call it and relay results to a frontend.

Start it programmatically instead of using any terminal command:

```python
from mica.api import Mica
from mica.api.server import serve

with Mica("mlx-community/Llama-3.2-3B-Instruct-4bit") as backend:
    serve(backend, port=8000)
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service/model status |
| POST | `/v1/generate` | Stateless generation, or a session reply; optional streaming |
| POST | `/v1/sessions` | Create an in-memory session with optional system instructions |
| GET | `/v1/sessions/{id}` | Read session settings and messages |
| DELETE | `/v1/sessions/{id}` | Delete an unused session |
| POST | `/v1/cancel` | Cooperatively cancel the active generation |

### Generate text

```sh
curl http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain MLX briefly.","options":{"max_tokens":128,"temperature":0.3}}'
```

Response:

```json
{"text":"...","model":"mlx-community/Llama-3.2-3B-Instruct-4bit","session_id":null,"stats":{},"incomplete":false}
```

`stats` contains actual inference measurements. There are no fabricated timing defaults.

### Stream text

```sh
curl -N http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Tell me a short story.","stream":true}'
```

Streaming uses newline-delimited JSON (`application/x-ndjson`), not an OpenAI-compatible protocol:

```json
{"type":"delta","text":"Once"}
{"type":"delta","text":" upon a time..."}
{"type":"done","model":"...","session_id":null,"stats":{},"incomplete":false}
```

Validation errors before streaming starts use ordinary HTTP errors. Errors after HTTP 200 produce a final `{"type":"error","error":"..."}` event; treat them as failed/incomplete generations. Disconnecting closes the iterator and invalidates its cache once a socket write detects the disconnect. Use `/v1/cancel` when you need an explicit cancellation signal.

### Multi-turn conversation

Create a session:

```sh
curl http://127.0.0.1:8000/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"system":"Answer clearly and briefly."}'
```

Use the returned `id` as `session_id` in subsequent `/v1/generate` requests. Sessions are isolated and remain in memory until deleted or shutdown. There is a default limit of 32 sessions; delete unused ones. Durable session save/load is available through the Python API, so HTTP callers never supply server file paths.

Requests require JSON objects, accept at most 512 KiB, reject nonfinite numbers/unknown fields, and do not accept chunked request bodies. Invalid input returns 400, unknown sessions/routes 404, session limit 429, and a busy inference engine 503. Generation is serialized rather than queued; retry 503 after the active request completes. A health response means the model is loaded, not that no request is currently generating.

## Verification

```sh
uv run pytest -q
uv run python scripts/api_smoke.py
uv run python scripts/api_cold_start_smoke.py
```

The smoke script requires the downloaded 3B preset and uses only local HTTP and local model artifacts. It does not download another model. The API tests use a fake engine for predictable error/concurrency checks; the smoke script verifies real MLX generation separately.

The cold-start script launches a fresh API subprocess and sends an HTTP generation request before any inference warmup. It then checks streaming, chat history/cache reuse across requests, cancellation/recovery and clean shutdown. It uses an available loopback port so it can run alongside your normal server.

See [actual API verification results](../../../verification/api-results.md).
