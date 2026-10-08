# Local HTTP audio endpoints — 9 October 2026

MicaHTTPServer now exposes opt-in `/v1/audio/transcribe` and `/v1/audio/synthesize` endpoints through the existing local speech worker. Transcription accepts bounded PCM16 mono/stereo WAV at 8–48 kHz and resamples to Whisper's mono 16 kHz input. Synthesis returns PCM16 WAV or bounded base64 PCM chunks over NDJSON. Audio requests do not mutate text sessions and do not use server-side microphone/speaker devices or temporary files.

Speech dependencies/models remain lazy and optional. Startup accepts `--voice`, reads configured voice settings, defaults to offline staged assets and retains one reusable speech worker. Each speech operation acquires a nonblocking lease; overlap receives 503. Text inference uses its separate existing worker and can overlap speech. `/v1/audio/cancel` cancels speech; `/v1/cancel` cancels text and speech. Cancellation polling kills pending speech work, and closing an incomplete streaming generator discards the worker. Successful idle shutdown now requests a normal child exit so speech-library semaphore finalizers run; failed/cancelled operations still terminate the worker.

Browser access is opt-in through exact loopback HTTP/HTTPS origins. Preflight supports Content-Type/Authorization, while actual requests preserve Host and Bearer checks. Default servers continue rejecting browser origins. No WebSocket/web-framework dependency was added.

Validation on Python 3.12.13 / Apple M3:

- Full automated suite: **151 passed in 18.41s**. Fake speech/HTTP checks cover resampling, silence, WAV/NDJSON formats, body/text limits, malformed uploads, cancellation/busy/recovery, failure events, explicit origins/Bearer auth, disabled endpoints, early iterator closure and graceful worker cleanup.
- `scripts/audio_api_smoke.py` performed real offline HTTP Kokoro synthesis, uploaded the generated WAV to real mlx-whisper, and exercised streaming synthesis with an empty external Hugging Face cache and prepared project assets.
- Synthesized WAV: **181,244 bytes**, **3.775 seconds**. Recognized transcript: “Hello from Micah. This speech is generated locally.” The input text used “Mica”; Whisper's spelling difference is recorded rather than treated as an exact transcript match.
- Streaming synthesis produced **3 audio events** followed by `done`.
- A repeated real roundtrip exited without the initial leaked-semaphore warning after graceful shutdown was added. Torch emitted its existing third-party deprecation warning.
- `git diff --check` passed. CLI help exposes `--voice` and repeatable `--allow-origin`.

Real microphone/speaker use, browser playback, and real-inference cancellation were not exercised by this check. Transport tests use fake speech adapters for deterministic cancellation/error coverage. WAV PCM16 is the supported upload format; MediaRecorder WebM/Opus, multipart uploads, arbitrary file paths and WebSockets are not provided.

## Combined voice chat

`POST /v1/voice/chat` accepts user text with optional session/options, runs the existing MLX backend and synthesizes the assistant answer. It returns generation metadata and a Base64 WAV in one JSON response after both stages complete. It requires `--voice`; partial/cancelled generation is not spoken. No dependencies were added.

- Python **3.12.13** full suite: **160 passed in 23.49s**, including combined response decoding, session/options forwarding, validation before inference, disabled audio, cancelled generation and synthesis failure.
- Real offline `scripts/audio_api_smoke.py` verified the combined route with staged Llama/Kokoro models: user text “Say hello in one short sentence.” produced assistant text “Hello.” and a valid **64,844-byte** mono PCM16 WAV.
- The same smoke run retained the Whisper roundtrip and three streaming audio chunks from the earlier checks. It opened no microphone or speaker devices.
- Native MLX/loopback tests were run with sandbox approval after restricted execution failed. Browser playback remains a manual check.
