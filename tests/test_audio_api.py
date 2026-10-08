import base64
from io import BytesIO
import json
from threading import Event, Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

import numpy as np
import pytest

from mica.api.audio import AudioService, MAX_AUDIO_BYTES, decode_wav
from mica.api.server import MicaHTTPServer
from mica.api.client import GenerationResult
from mica.config import ChatError


def wav_bytes(rate=16000, channels=1, seconds=0.05, silent=False):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(np.full((int(rate * seconds), channels), 0 if silent else 1200, dtype="<i2").tobytes())
    return buffer.getvalue()


class FakeSpeech:
    def __init__(self):
        self.audio = None
        self.closed = 0
        self.spoken = []
        self.entered = Event()
        self.service = None
        self.hold = False
        self.fail = False
    def transcribe(self, audio):
        self.audio = audio
        self.entered.set()
        if self.hold:
            assert self.service.cancelled.wait(3)
        return " local transcript "
    def synthesize(self, text):
        self.spoken.append(text)
        yield np.array([0.0, 0.5, -0.5], dtype=np.float32), 24000
        if self.fail:
            raise RuntimeError("synthesis failed")
        yield np.array([0.25], dtype=np.float32), 24000
    def close(self):
        self.closed += 1


@pytest.fixture
def audio_http():
    fake = FakeSpeech()
    service = AudioService(factory=lambda: fake)
    fake.service = service
    backend = SimpleNamespace(_offline=True, model="fake", cancel=lambda: False)
    server = MicaHTTPServer(backend, port=0, allowed_origins=("http://localhost:5173",))
    server.audio = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server, fake
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


def request(base, path, body, content_type="application/json", headers=None, method="POST"):
    raw = json.dumps(body).encode() if isinstance(body, dict) else body
    return urlopen(Request(base+path, data=raw, method=method,
        headers={"Content-Type": content_type, **(headers or {})}), timeout=5)


def test_http_transcribe_downmixes_resamples_and_leaves_sessions_untouched(audio_http):
    base, server, fake = audio_http
    with request(base, "/v1/audio/transcribe", wav_bytes(48000, 2), "audio/wav") as response:
        assert json.load(response) == {"text": "local transcript"}
    assert fake.audio.dtype == np.float32 and fake.audio.shape == (800,)
    assert not server.sessions


def test_silent_audio_never_loads_speech(audio_http):
    base, server, fake = audio_http
    with request(base, "/v1/audio/transcribe", wav_bytes(silent=True), "audio/wav") as response:
        assert json.load(response) == {"text": ""}
    assert server.audio.speech is None and fake.audio is None


def test_wav_and_streaming_synthesis(audio_http):
    base, server, fake = audio_http
    with request(base, "/v1/audio/synthesize", {"text": "**Hello**"}) as response:
        assert response.headers.get_content_type() == "audio/wav"
        with wave.open(BytesIO(response.read()), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 24000, 4)
    with request(base, "/v1/audio/synthesize", {"text": "Hello", "stream": True}) as response:
        events = [json.loads(line) for line in response]
    assert events[-1] == {"type": "done"}
    assert len(events) == 3
    assert events[0]["format"] == "pcm_s16le" and events[0]["sample_rate"] == 24000
    assert len(base64.b64decode(events[0]["audio"])) == 6
    assert fake.spoken == ["Hello", "Hello"] and not server.sessions


@pytest.mark.parametrize("path,body,ctype,status", [
    ("transcribe", b"bad WAV", "audio/wav", 400),
    ("transcribe", b"bad", "audio/webm", 415),
    ("synthesize", {"text": ""}, "application/json", 400),
    ("synthesize", {"text": "x"*10001}, "application/json", 400),
    ("synthesize", {"text": "hello", "voice": "unknown"}, "application/json", 400),
    ("synthesize", {"text": "hello", "stream": "yes"}, "application/json", 400),
])
def test_validation(audio_http, path, body, ctype, status):
    base, server, _ = audio_http
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/audio/"+path, body, ctype)
    assert exc.value.code == status and json.load(exc.value)["error"]
    assert server.audio.speech is None


def test_audio_body_limit(audio_http):
    base, _, _ = audio_http
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/audio/transcribe", b"short", "audio/wav", {"Content-Length": str(MAX_AUDIO_BYTES+1)})
    assert exc.value.code == 413


def test_cancel_busy_and_recovery(audio_http):
    base, server, fake = audio_http
    fake.hold = True
    result = []
    def transcribe():
        try:
            request(base, "/v1/audio/transcribe", wav_bytes(), "audio/wav")
        except HTTPError as e:
            result.append(e.code)
    thread = Thread(target=transcribe)
    thread.start()
    assert fake.entered.wait(2)
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/audio/synthesize", {"text": "busy"})
    assert exc.value.code == 503
    with request(base, "/v1/audio/cancel", {}) as response:
        assert json.load(response)["cancel_requested"]
    thread.join(3)
    assert not thread.is_alive() and result == [409] and fake.closed
    fake.hold = False
    with request(base, "/v1/audio/transcribe", wav_bytes(), "audio/wav") as response:
        assert json.load(response)["text"] == "local transcript"
    assert not server.audio.active.is_set()


def test_stream_failure_is_final_error_and_discards_worker(audio_http):
    base, server, fake = audio_http
    fake.fail = True
    with request(base, "/v1/audio/synthesize", {"text": "hi", "stream": True}) as response:
        events = [json.loads(line) for line in response]
    assert events[0]["type"] == "audio" and events[-1]["type"] == "error"
    assert not any(e["type"] == "done" for e in events)
    assert server.audio.speech is None and fake.closed


def test_allowed_browser_origin_preflight_and_token(audio_http):
    base, server, _ = audio_http
    server.token = "secret"
    origin = {"Origin": "http://localhost:5173"}
    with request(base, "/v1/audio/synthesize", b"", headers={**origin,
        "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,authorization"}, method="OPTIONS") as response:
        assert response.status == 204
        assert response.headers["Access-Control-Allow-Origin"] == origin["Origin"]
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/audio/synthesize", {"text": "hello"}, headers=origin)
    assert exc.value.code == 401
    with request(base, "/v1/audio/synthesize", {"text": "hello"}, headers={**origin, "Authorization": "Bearer secret"}) as response:
        assert response.headers["Access-Control-Allow-Origin"] == origin["Origin"]
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/audio/synthesize", {"text": "hello"}, headers={"Origin": "https://evil.example", "Authorization": "Bearer secret"})
    assert exc.value.code == 403


def test_disabled_audio_is_lazy(audio_http):
    base, server, _ = audio_http
    server.audio.close()
    server.audio = None
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/audio/synthesize", {"text": "hi"})
    assert exc.value.code == 503 and "--voice" in json.load(exc.value)["error"]


def test_invalid_and_long_wav():
    with pytest.raises(ChatError, match="at most"):
        decode_wav(wav_bytes(seconds=0.1), 0.01)
    with pytest.raises(ChatError):
        decode_wav(wav_bytes()[:-10], 60)


def test_closing_synthesis_stream_cleans_pending_work():
    fake = FakeSpeech()
    service = AudioService(factory=lambda: fake)
    stream = service.synthesize("hello")
    next(stream)
    stream.close()
    assert fake.closed and service.speech is None and not service.active.is_set()


def test_reject_nonlocal_browser_origins():
    backend = SimpleNamespace(_offline=True)
    with pytest.raises(ValueError, match="loopback"):
        MicaHTTPServer(backend, port=0, allowed_origins=("https://example.com",))


def test_idle_service_shutdown_uses_graceful_worker_cleanup():
    events = []
    fake = FakeSpeech()
    fake.shutdown = lambda: events.append("graceful")
    service = AudioService(factory=lambda: fake)
    assert service.wav("hello")
    service.close()
    assert events == ["graceful"] and service.closed and service.speech is None


def test_voice_chat_generates_answer_and_returns_text_and_wav(audio_http):
    base, server, fake = audio_http
    calls = []
    session = SimpleNamespace(id="conversation")
    server.sessions[session.id] = session
    def generate(text, **kwargs):
        calls.append((text, kwargs))
        return GenerationResult("**Assistant answer.**", "fake", session.id, {"tokens": 3})
    server.backend.generate = generate
    with request(base, "/v1/voice/chat", {"text": "What I spoke", "session_id": session.id,
                 "options": {"max_tokens": 64}}) as response:
        payload = json.load(response)
    assert calls == [("What I spoke", {"session": session, "options": {"max_tokens": 64}})]
    assert payload["text"] == "**Assistant answer.**" and not payload["incomplete"]
    assert payload["session_id"] == session.id and payload["stats"] == {"tokens": 3}
    assert fake.spoken == ["Assistant answer."]
    assert payload["audio"]["mime_type"] == "audio/wav"
    assert payload["audio"]["encoding"] == "base64" and payload["audio"]["format"] == "wav"
    with wave.open(BytesIO(base64.b64decode(payload["audio"]["data"])), "rb") as wav:
        assert wav.getnframes() == 4 and wav.getframerate() == 24000


@pytest.mark.parametrize("body,status", [({}, 400), ({"text": " "}, 400),
    ({"text": 42}, 400), ({"text": "hi", "stream": True}, 400),
    ({"text": "hi", "session_id": "missing"}, 404)])
def test_voice_chat_validates_before_generation(audio_http, body, status):
    base, server, fake = audio_http
    server.backend.generate = lambda *args, **kwargs: pytest.fail("Unexpected generation")
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/voice/chat", body)
    assert exc.value.code == status and not fake.spoken


def test_voice_chat_cancelled_generation_is_not_spoken(audio_http):
    base, server, fake = audio_http
    server.backend.generate = lambda *args, **kwargs: GenerationResult("partial", "fake", None, {}, True)
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/voice/chat", {"text": "hello"})
    assert exc.value.code == 409 and not fake.spoken


def test_voice_chat_synthesis_failure_returns_no_success(audio_http):
    base, server, fake = audio_http
    server.backend.generate = lambda *args, **kwargs: GenerationResult("answer", "fake", None, {})
    fake.fail = True
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/voice/chat", {"text": "hello"})
    assert exc.value.code == 500 and "synthesis failed" in json.load(exc.value)["error"]


def test_voice_chat_requires_audio_enabled_before_generation(audio_http):
    base, server, fake = audio_http
    server.audio.close()
    server.audio = None
    server.backend.generate = lambda *args, **kwargs: pytest.fail("Unexpected generation")
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/voice/chat", {"text": "hi"})
    assert exc.value.code == 503
