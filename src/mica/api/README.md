# Mica backend API

Use Mica directly from Python, or run its optional localhost HTTP adapter. Both interfaces use the same validated native MLX 4-bit model, official chat template, context budgeting, session storage and inference engine as the CLI. No web framework or additional dependency is installed. Importing `mica.api` never loads weights, downloads a model or starts a server.

## Python integration

Install the project with `uv sync --locked`. In your own uv project, add this checkout with `uv add /absolute/path/to/mica`. Run on native arm64 Python 3.12 on macOS 14+ with Metal access. Download a selected model once, then use the API offline:

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

Add `--online` only to allow model metadata checks/downloads. Set `MICA_API_TOKEN` to require `Authorization: Bearer <token>` on every request. No tokens or request contents are logged. Browser origins are rejected by default. Use an explicitly allowed loopback origin for a local frontend (see audio endpoints below), or have your own backend relay requests.

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
| POST | `/v1/cancel` | Cancel active text/audio work |

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

Text/session requests require JSON objects, accept at most 512 KiB, reject nonfinite numbers/unknown fields, and do not accept chunked request bodies. Invalid input returns 400, unknown sessions/routes 404, session limit 429, and a busy inference engine 503. Generation is serialized rather than queued; retry 503 after the active request completes. A health response means the model is loaded, not that no request is currently generating.

## Verification

```sh
uv run pytest -q
uv run python scripts/api_smoke.py
uv run python scripts/api_cold_start_smoke.py
```

The smoke script requires the downloaded 3B preset and uses only local HTTP and local model artifacts. It does not download another model. The API tests use a fake engine for predictable error/concurrency checks; the smoke script verifies real MLX generation separately.

The cold-start script launches a fresh API subprocess and sends an HTTP generation request before any inference warmup. It then checks streaming, chat history/cache reuse across requests, cancellation/recovery and clean shutdown. It uses an available loopback port so it can run alongside your normal server.

See [actual API verification results](../../../verification/api-results.md).

## Local audio endpoints

Install the voice extra and prepare the speech assets before starting the server:

```sh
uv sync --locked --python 3.12 --extra voice
uv run --locked --extra voice mica voice-download
uv run --locked --extra voice python -m mica.api \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit --voice \
  --allow-origin http://localhost:5173
```

`--voice` enables audio endpoints. Models load lazily on the first audio request and remain cached in the existing local speech worker; text-only servers do not import speech dependencies. The default is offline and uses staged `assets/` files. `--online` explicitly allows missing models to be downloaded. CLI startup reads the `voice` settings object from Mica's `config.json`; programmatic startup accepts `audio=True, audio_settings={...}` on `MicaHTTPServer` or `serve`. For example `audio_settings={"voice":"af_heart", "language":"auto", "speed":1.0}`. All inference is local; no microphone, speaker or temporary audio files are used by the server. The frontend owns capture/playback.

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| POST | `/v1/audio/transcribe` | Binary PCM16 WAV, `Content-Type: audio/wav` | JSON `{"text":"..."}` |
| POST | `/v1/audio/synthesize` | JSON `{"text":"Hello"}` | Mono PCM16 `audio/wav` |
| POST | `/v1/audio/synthesize` | JSON `{"text":"Hello", "stream":true}` | NDJSON audio chunks, then `done` |
| POST | `/v1/audio/cancel` | JSON `{}` | JSON `{"cancel_requested":true|false}` |
| POST | `/v1/voice/chat` | JSON `{"text":"What the user spoke"}` | Assistant text and Base64 WAV in one JSON response |

`/health` includes `audio_enabled`. Disabled audio endpoints return 503 with setup instructions. Transcription and synthesis do not modify conversation history. `/v1/voice/chat` runs the complete text-to-answer-to-speech flow on the backend and supports conversation sessions. Speech requests are serialized independently of text generation, allowing local LLM and speech inference to overlap while sharing GPU resources. Concurrent audio requests receive 503 rather than being queued.

### Voice chat: user text → Mica answer → audio

Send the user's words (from frontend speech recognition or typed input). The backend generates an answer with Mica MLX, synthesizes that answer, and returns both in one response:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/voice/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain MLX in one sentence.","options":{"max_tokens":80}}'
```

```json
{"text":"MLX is ...","model":"...","session_id":null,"stats":{},"incomplete":false,"audio":{"format":"wav","mime_type":"audio/wav","encoding":"base64","data":"BASE64_WAV_BYTES"}}
```

Optional `session_id` and `options` work as on `/v1/generate`. Use `/v1/sessions` to create a conversation first. The frontend displays `text`, decodes `audio.data` as a complete WAV, and plays it. It does not call `/v1/generate` or `/v1/audio/synthesize` separately. If the frontend sends raw recorded audio, first obtain text through `/v1/audio/transcribe`, then send that text to `/v1/voice/chat`.

```javascript
// Call from a user interaction so browser audio playback is permitted.
async function voiceReply(userText, sessionId) {
  const context = new AudioContext();
  await context.resume();
  try {
    const response = await fetch('http://127.0.0.1:8000/v1/voice/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'}, // Add Bearer token if configured.
      body: JSON.stringify({text: userText, ...(sessionId ? {session_id: sessionId} : {}),
                            options: {max_tokens: 128}}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    const bytes = Uint8Array.from(atob(result.audio.data), c => c.charCodeAt(0));
    const source = context.createBufferSource();
    source.buffer = await context.decodeAudioData(bytes.buffer);
    source.connect(context.destination);
    source.onended = () => context.close();
    source.start();
    return result.text;
  } catch (error) {
    await context.close();
    throw error;
  }
}
```

This route returns after generation and synthesis finish; it does not accept `stream`. `/v1/cancel` cancels whichever stage is active; cancelled generation is not synthesized and returns 409. A synthesis failure returns an HTTP error; a successful generated turn remains in session history even if speech fails. Existing synthesis text/audio limits apply to the generated answer, so use short answers and bounded `max_tokens`. The independent streaming endpoints remain available for frontends that need incremental output.

### Transcribe WAV

```sh
curl http://127.0.0.1:8000/v1/audio/transcribe \
  -H 'Content-Type: audio/wav' --data-binary @recording.wav
```

Accepts uncompressed PCM16 WAV, one or two channels, at 8–48 kHz; stereo is downmixed and resampled to 16 kHz. The configured maximum recording duration applies (default 60 seconds), with a 20 MiB upload cap. Silence returns empty text. Browser MediaRecorder WebM/Opus, multipart uploads and arbitrary server file paths are not accepted: encode PCM16 WAV in the frontend before upload. Unsupported content types return 415, invalid/truncated audio 400, and oversized bodies 413. Uploads require Content-Length and cannot use chunked transfer encoding.

### Synthesize WAV

```sh
curl http://127.0.0.1:8000/v1/audio/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Mica."}' --output response.wav
```

Text must be nonempty and at most 10,000 characters. Common Markdown is prepared for speech and fenced code is omitted, as in CLI voice mode. Responses are limited to 20 MiB of PCM or 300 seconds. Voice, speed, models and language are configured on the server; arbitrary per-request model paths are not accepted. The default Kokoro output is 24 kHz. Static WAV responses are sent after synthesis completes.

### Stream synthesized audio

```sh
curl -N http://127.0.0.1:8000/v1/audio/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello. This is another sentence.","stream":true}'
```

The response is `application/x-ndjson`, using the existing adapter's streaming transport rather than WebSockets:

```json
{"type":"audio","format":"pcm_s16le","sample_rate":24000,"channels":1,"audio":"BASE64_PCM_BYTES"}
{"type":"audio","format":"pcm_s16le","sample_rate":24000,"channels":1,"audio":"BASE64_PCM_BYTES"}
{"type":"done"}
```

Base64-decode `audio` as signed 16-bit little-endian mono samples, normalize by 32768 for Web Audio, and schedule chunks consecutively at `sample_rate`. Events contain at most 32 KiB of raw PCM. JSON validation/model errors before the first chunk use HTTP errors. After HTTP 200, failures end with `{"type":"error","error":"..."}` instead of `done`; discard any pending frontend playback when cancellation is requested. A detected streaming disconnect closes the generator and releases the worker. Disconnects before any socket write cannot be detected immediately; use the explicit cancellation endpoint for those operations.

```sh
curl http://127.0.0.1:8000/v1/audio/cancel \
  -H 'Content-Type: application/json' -d '{}'
```

Nonstreaming cancelled requests return 409. Streaming cancellation produces a final error event if the connection remains open. `/v1/cancel` now requests cancellation of both active text and audio work. Server shutdown cancels and releases speech resources. Audio errors do not invalidate successful text generation or session history.

### Local browser/native clients

Browser origins remain rejected unless explicitly listed with repeatable `--allow-origin` flags (or `allowed_origins=("http://localhost:5173",)` programmatically). Only exact HTTP/HTTPS loopback origins are accepted; no wildcard or remote-site origins. Preflight OPTIONS supports Content-Type and Authorization. Actual requests still enforce the existing local Host and Bearer token checks. Native clients or a frontend's backend can omit Origin.

```javascript
const response = await fetch('http://127.0.0.1:8000/v1/audio/synthesize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'}, // Add Authorization when configured.
  body: JSON.stringify({text: 'Hello from Mica.'}),
});
if (!response.ok) throw new Error(await response.text());
const context = new AudioContext();
const buffer = await context.decodeAudioData(await response.arrayBuffer());
const source = context.createBufferSource();
source.buffer = buffer;
source.connect(context.destination);
source.start();
```

Configure an allowed origin matching your frontend's actual scheme/host/port. Have the frontend own audio permission prompts and cancellation of its playback queue. No new HTTP framework or WebSocket dependency is required. Automated tests use fake speech adapters to cover WAV conversion, binary/streaming responses, request validation, busy/cancel/recovery behavior, origin/auth rules and cleanup; actual models require prepared assets.

Repeat the real offline audio roundtrip (Kokoro synthesis → WAV HTTP upload → Whisper transcription, plus streamed synthesis) with `uv run --locked --extra voice python scripts/audio_api_smoke.py`. It uses staged defaults and opens no microphone or speaker devices.
