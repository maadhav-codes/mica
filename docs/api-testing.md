# Test Mica's text and audio API flows

Run these steps on your Apple Silicon Mac from the project root. Python 3.12 is required. Two terminals are used: one runs the API server, the other acts as the frontend/client. All inference stays local. For voice chat, `/v1/voice/chat` accepts the user's text and returns the generated answer and audio together. The separate endpoints below can also be tested individually.

## Combined voice chat test

Complete setup and health checks in steps 1–3 below, then run:

```sh
mica_api http://127.0.0.1:8000/v1/voice/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain MLX in one short sentence.","options":{"max_tokens":80}}' \
  --output .mica-test/api/voice-chat.json

.venv/bin/python - <<'PY'
import base64
import json
from pathlib import Path
result = json.loads(Path('.mica-test/api/voice-chat.json').read_text())
assert not result['incomplete'] and result['text'].strip()
assert result['audio']['format'] == 'wav' and result['audio']['encoding'] == 'base64'
print(result['text'])
Path('.mica-test/api/voice-chat.wav').write_bytes(base64.b64decode(result['audio']['data'], validate=True))
PY

afplay .mica-test/api/voice-chat.wav
```

Expected: the printed answer and played speech convey the same generated reply. The server performs both MLX generation and speech synthesis. The input text is not simply read aloud.

To test spoken input, your frontend speech recognition supplies `text` to this same request. If using Mica for recognition, record and transcribe using steps 7–8 below, then submit the returned transcript:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
text = json.loads(Path('.mica-test/api/transcript.json').read_text())['text']
assert text.strip(), 'Record a nonempty question first'
Path('.mica-test/api/voice-chat-request.json').write_text(json.dumps({'text': text, 'options': {'max_tokens': 80}}))
PY

mica_api http://127.0.0.1:8000/v1/voice/chat \
  -H 'Content-Type: application/json' \
  --data-binary @.mica-test/api/voice-chat-request.json \
  --output .mica-test/api/voice-chat.json
```

Repeat the decode/play commands above to hear the answer. No separate generation or synthesis requests are needed. Optional `session_id` preserves conversation history. This combined response is sent after synthesis completes; use the individual streaming endpoints for incremental output.

## 1. Install and prepare assets

In Terminal 1:

```sh
cd /Users/mayurdayal/Developer/engineer/mlx-chat
uv sync --locked --python 3.12 --extra voice --group dev
uv run --locked --extra voice mica download --model mlx-community/Llama-3.2-3B-Instruct-4bit
uv run --locked --extra voice mica voice-download
```

These setup commands can use the network. Models and speech resources are placed in `assets/`. Repeated downloads reuse cached files. `voice-download` should end with `Speech models, configured voice and phonemizer validated for offline use.`

## 2. Start the API

Still in Terminal 1:

```sh
.venv/bin/python -m mica.api \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --voice --port 8000 \
  --allow-origin http://localhost:5173
```

Leave it running. It should print `Mica API listening at http://127.0.0.1:8000`.

The server defaults to offline model access; no `--online` is needed after setup. `--allow-origin` permits a browser frontend served from exactly `http://localhost:5173`; curl/native clients do not require it. If using a different frontend port, change the allowed origin to match. If port 8000 is already used, select another port and update all client URLs.

Optional authentication: set `MICA_API_TOKEN` to the same value in both terminals **before** starting the server/client. Do not publish the token. The helper below includes it automatically; when unset, the server requires no token.

## 3. Prepare the client and check health

In Terminal 2:

```sh
cd /Users/mayurdayal/Developer/engineer/mlx-chat
mkdir -p .mica-test/api

mica_api() {
  curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer ${MICA_API_TOKEN:-}" "$@"
}

mica_api http://127.0.0.1:8000/health | .venv/bin/python -m json.tool
```

Expected: `service` is `mica`, `status` is `ready`, and `audio_enabled` is `true`. The first audio request may take longer while speech models load; later requests reuse the worker. Run each next step only after the prior command succeeds.

## Scenario A: typed text → text reply + spoken reply

### 4. Ask the language model a question

In Terminal 2:

```sh
mica_api http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain why the sky is blue in two short sentences.","options":{"max_tokens":128,"temperature":0.3}}' \
  --output .mica-test/api/text-response.json
```

### 5. Display the answer and prepare its speech request

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path('.mica-test/api/text-response.json').read_text())
assert not result['incomplete'], 'Generation was interrupted; do not submit partial text.'
assert result['text'].strip(), 'The response is empty.'
print('Assistant:', result['text'])
Path('.mica-test/api/text-speech-request.json').write_text(
    json.dumps({'text': result['text']})
)
PY
```

Expected: the assistant's text is printed, and the same raw answer is submitted to synthesis. Kokoro uses a separate speech-friendly representation, so Markdown may be simplified and code omitted.

### 6. Synthesize the answer and play it

```sh
mica_api http://127.0.0.1:8000/v1/audio/synthesize \
  -H 'Content-Type: application/json' \
  --data-binary @.mica-test/api/text-speech-request.json \
  --output .mica-test/api/text-response.wav

afplay .mica-test/api/text-response.wav
```

Pass condition: you received the text answer and heard its spoken version. No microphone or transcription request was needed. The API server does not play audio itself; `afplay` represents frontend playback.

## Scenario B: recorded audio → transcript → text reply + spoken reply

### 7. Record a short question as PCM16 WAV

Run this in Terminal 2, then speak immediately after the prompt, for example: “Explain what MLX does in two short sentences.” Recording stops after six seconds.

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import sounddevice as sd
import wave
rate = 16000
print('Recording for 6 seconds. Speak your question now.', flush=True)
try:
    audio = sd.rec(6 * rate, samplerate=rate, channels=1, dtype='int16')
    sd.wait()
finally:
    sd.stop()
path = Path('.mica-test/api/question.wav')
with wave.open(str(path), 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    wav.writeframes(audio.astype('<i2').tobytes())
print('Saved:', path)
PY
```

Allow microphone permission for your terminal in macOS System Settings → Privacy & Security → Microphone if prompted. List devices with `.venv/bin/python -m sounddevice`; to select one, add `device=DEVICE_INDEX` to `sd.rec(...)`.

Alternatively, place an existing recording at `.mica-test/api/question.wav`. Supported uploads are **PCM16 WAV**, mono/stereo, **8–48 kHz**, no longer than the configured maximum duration (default **60 seconds**), and at most **20 MiB**. Browser MediaRecorder WebM/Opus cannot be uploaded directly; the frontend must encode WAV first.

For a device-free test, generate synthetic input instead of recording:

```sh
mica_api http://127.0.0.1:8000/v1/audio/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain what MLX does in two short sentences."}' \
  --output .mica-test/api/question.wav
```

This tests transcription/inference but does not verify microphone capture.

### 8. Transcribe the question

```sh
mica_api http://127.0.0.1:8000/v1/audio/transcribe \
  -H 'Content-Type: audio/wav' \
  --data-binary @.mica-test/api/question.wav \
  --output .mica-test/api/transcript.json
```

### 9. Display the transcript and prepare the chat request

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
transcript = json.loads(Path('.mica-test/api/transcript.json').read_text())['text']
print('You (voice):', transcript)
assert transcript.strip(), 'No speech detected. Record again before continuing.'
Path('.mica-test/api/voice-chat-request.json').write_text(json.dumps({
    'prompt': transcript,
    'options': {'max_tokens': 128, 'temperature': 0.3},
}))
PY
```

Expected: recognizable words matching your question. Minor spelling differences are possible. Transcription alone does not send the question to the language model or create conversation history.

### 10. Ask the language model using the transcript

```sh
mica_api http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  --data-binary @.mica-test/api/voice-chat-request.json \
  --output .mica-test/api/voice-response.json
```

### 11. Display the answer and prepare synthesis

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path('.mica-test/api/voice-response.json').read_text())
assert not result['incomplete'], 'Generation was interrupted.'
assert result['text'].strip(), 'The response is empty.'
print('Assistant:', result['text'])
Path('.mica-test/api/voice-speech-request.json').write_text(
    json.dumps({'text': result['text']})
)
PY
```

### 12. Synthesize and play the spoken answer

```sh
mica_api http://127.0.0.1:8000/v1/audio/synthesize \
  -H 'Content-Type: application/json' \
  --data-binary @.mica-test/api/voice-speech-request.json \
  --output .mica-test/api/voice-response.wav

afplay .mica-test/api/voice-response.wav
```

Pass condition: your recording becomes a displayed transcript, that transcript receives a text answer, and you hear the answer. This section tests three individual API requests. For the combined backend flow, replace steps 9–12 with the `/v1/voice/chat` test at the beginning of this guide.

## 13. Check streaming and cancellation

Stream text directly:

```sh
mica_api -N http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Tell me a short story.","stream":true,"options":{"max_tokens":128}}'
```

Expected: `delta` events followed by `done`, or an explicit `error` event.

Stream audio to a file:

```sh
mica_api -N http://127.0.0.1:8000/v1/audio/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello. This is streamed audio from Mica.","stream":true}' \
  --output .mica-test/api/audio-events.jsonl
```

Verify and convert its PCM chunks into a playable WAV:

```sh
.venv/bin/python - <<'PY'
import base64
import json
from pathlib import Path
import wave
events = [json.loads(line) for line in Path('.mica-test/api/audio-events.jsonl').read_text().splitlines()]
assert events and events[-1]['type'] == 'done', 'Streaming failed or was cancelled.'
assert not any(event['type'] == 'error' for event in events)
chunks = [event for event in events if event['type'] == 'audio']
assert chunks, 'No audio chunks received.'
rate = chunks[0]['sample_rate']
with wave.open('.mica-test/api/streamed-response.wav', 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    for chunk in chunks:
        assert chunk['sample_rate'] == rate and chunk['channels'] == 1
        assert chunk['format'] == 'pcm_s16le'
        wav.writeframesraw(base64.b64decode(chunk['audio'], validate=True))
print('Verified audio chunks:', len(chunks))
PY

afplay .mica-test/api/streamed-response.wav
```

This verifies the streaming transport, but saves all chunks before playback. A real frontend should schedule chunks as they arrive. To overlap an LLM reply with speech, the frontend must collect completed sentences from text deltas and submit them for synthesis in order, with only one active speech request at a time.

While a longer audio request is running, use another terminal to cancel it:

```sh
curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer ${MICA_API_TOKEN:-}" \
  -H 'Content-Type: application/json' \
  -d '{}' http://127.0.0.1:8000/v1/audio/cancel
```

Expected: `cancel_requested: true` when speech was active, otherwise `false`. Cancelled nonstreaming requests return HTTP 409; streaming requests end with an error event if still connected. Use `/v1/cancel` with the same body to cancel text and audio work together. Frontend playback already buffered on the client must also be stopped by the frontend; server cancellation cannot stop `afplay` or browser audio already received.

## 14. Test conversation continuity if needed

Both scenarios above are stateless. To retain context:

```sh
mica_api http://127.0.0.1:8000/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"system":"Answer briefly."}'
```

Copy the returned `id` into `session_id` on each `/v1/generate` request, whether its prompt came from typing or transcription. Only generation calls change that session; speech endpoints do not.

## 15. Run automated verification and stop the server

```sh
uv run --locked --extra voice pytest -q
uv run --locked --extra voice python scripts/audio_api_smoke.py
```

The smoke script starts its own server on a temporary port, tests a real offline Kokoro→Whisper HTTP roundtrip and streamed synthesis, then shuts down. It requires prepared default assets and opens no audio devices. It can run alongside the server above.

When finished, press Ctrl+C in Terminal 1 to stop your API server. The test files in `.mica-test/api/` contain your recording/transcript/answers; keep or remove them as appropriate. The server itself does not persist raw recordings.

Troubleshooting:

- **503, audio disabled:** restart the server with `--voice`.
- **503, speech busy:** wait for the current audio request to finish or cancel it, then retry.
- **401:** use the same `MICA_API_TOKEN` in client and server terminals.
- **403:** use the exact allowed browser origin and call `127.0.0.1`, not another hostname.
- **415:** send PCM16 WAV with `Content-Type: audio/wav` for transcription.
- **400, missing assets:** run the setup commands online from the project root. If launching elsewhere, set `MICA_ASSETS_DIR` to the absolute assets directory.
- **Connection refused:** confirm Terminal 1 is still running and the port matches.

The automated HTTP roundtrip has been verified using synthetic audio. Live microphone capture, speaker playback and your browser frontend still require the manual steps above.
