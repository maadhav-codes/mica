"""Optional local JSON/NDJSON/WAV HTTP transport using the standard library."""
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import base64
from threading import Lock
from urllib.parse import urlsplit

from .client import Mica, MicaBusyError
from ..config import ChatError

MAX_BODY_BYTES = 512 * 1024


class RequestError(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


class MicaHTTPServer(ThreadingHTTPServer):
    """Loopback service; text and optional speech each serialize their own work."""
    daemon_threads = False

    def __init__(self, backend: Mica, *, host="127.0.0.1", port=8000, token=None, max_sessions=32,
                 audio=False, audio_settings=None, allowed_origins=()):
        if host != "127.0.0.1":
            raise ValueError("Mica's built-in HTTP adapter binds only to 127.0.0.1.")
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        if token is not None and not isinstance(token, str):
            raise ValueError("token must be a string or None.")
        self.backend = backend
        self.token = token
        self.max_sessions = max_sessions
        self.sessions = {}
        self.sessions_lock = Lock()
        self.allowed_origins = set(allowed_origins)
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
                raise ValueError("Allow only exact loopback browser origins, such as http://localhost:5173.")
        self.audio = None
        if audio:
            from .audio import AudioService
            self.audio = AudioService(audio_settings, offline=backend._offline)
        super().__init__((host, port), Handler)

    def server_close(self):
        self.backend.cancel()
        try:
            if self.audio is not None:
                self.audio.close()
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = "Mica/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *args):
        # Never write conversations or request contents to terminal logs.
        pass

    def _authorize(self, preflight=False):
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
            raise RequestError("Invalid local Host header.", 403)
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.allowed_origins:
            raise RequestError("Browser origin is not allowed. Configure an exact loopback origin or call from your backend.", 403)
        if self.server.token and not preflight:
            import hmac
            actual = self.headers.get("Authorization", "")
            if not hmac.compare_digest(actual.encode("utf-8"), f"Bearer {self.server.token}".encode("utf-8")):
                raise RequestError("A valid Bearer token is required.", 401)

    def end_headers(self):
        origin = self.headers.get("Origin")
        if origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def _raw_body(self, limit):
        if self.headers.get("Transfer-Encoding"):
            raise RequestError("Chunked request bodies are unsupported.")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as e:
            raise RequestError("A valid Content-Length is required.") from e
        if not 0 < length <= limit:
            raise RequestError(f"Request body must be between 1 byte and {limit} bytes.", 413)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("Incomplete request body.")
        return raw

    def _json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise RequestError("Chunked request bodies are unsupported.")
        if self.headers.get_content_type() != "application/json":
            raise RequestError("Use Content-Type: application/json.", 415)
        raw = self._raw_body(MAX_BODY_BYTES)
        try:
            def reject_constant(value):
                raise ValueError(f"Nonfinite JSON value: {value}")
            data = json.loads(raw, parse_constant=reject_constant)
        except (ValueError, UnicodeError) as e:
            raise RequestError("Invalid JSON body.") from e
        if not isinstance(data, dict):
            raise RequestError("JSON body must be an object.")
        return data

    def _fields(self, body, allowed):
        if set(body) - allowed:
            raise RequestError("Unknown request fields: " + ", ".join(sorted(set(body) - allowed)))

    def _session(self, identity):
        if not isinstance(identity, str):
            raise RequestError("session_id must be a string.")
        with self.server.sessions_lock:
            session = self.server.sessions.get(identity)
        if session is None:
            raise RequestError("Session not found.", 404)
        return session

    def _request(self, method):
        try:
            self._authorize()
            path = urlsplit(self.path).path
            if method == "GET" and path == "/health":
                self._json(200, {"service": "mica", "model": self.server.backend.model, "status": "ready",
                                 "audio_enabled": self.server.audio is not None})
            elif method == "GET" and path.startswith("/v1/sessions/"):
                session = self._session(path.removeprefix("/v1/sessions/"))
                self._json(200, self.server.backend.session_info(session))
            elif method == "DELETE" and path.startswith("/v1/sessions/"):
                session = self._session(path.removeprefix("/v1/sessions/"))
                self.server.backend.reset(session)
                with self.server.sessions_lock:
                    self.server.sessions.pop(session.id, None)
                self._json(200, {"deleted": session.id})
            elif method == "POST":
                if path == "/v1/audio/transcribe":
                    self._transcribe()
                    return
                body = self._body()
                if path == "/v1/sessions":
                    self._fields(body, {"system"})
                    if "system" in body and not isinstance(body["system"], str):
                        raise RequestError("system must be a string.")
                    with self.server.sessions_lock:
                        if len(self.server.sessions) >= self.server.max_sessions:
                            raise RequestError("Session limit reached. Delete unused sessions.", 429)
                        session = self.server.backend.new_session(system=body.get("system"))
                        self.server.sessions[session.id] = session
                    self._json(201, {"id": session.id, "model": self.server.backend.model})
                elif path == "/v1/cancel":
                    self._fields(body, set())
                    text_cancelled = self.server.backend.cancel()
                    audio_cancelled = self.server.audio.cancel() if self.server.audio is not None else False
                    self._json(200, {"cancel_requested": text_cancelled or audio_cancelled})
                elif path == "/v1/audio/cancel":
                    self._fields(body, set())
                    self._json(200, {"cancel_requested": self._audio().cancel()})
                elif path == "/v1/audio/synthesize":
                    self._synthesize(body)
                elif path == "/v1/voice/chat":
                    self._voice_chat(body)
                elif path == "/v1/generate":
                    self._generate(body)
                else:
                    raise RequestError("Endpoint not found.", 404)
            else:
                raise RequestError("Endpoint not found.", 404)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception as e:
            if isinstance(e, RequestError):
                status, message = e.status, str(e)
            elif isinstance(e, MicaBusyError):
                status, message = 503, str(e)
            elif isinstance(e, ChatError):
                from .audio import AudioCancelledError
                status, message = (409 if isinstance(e, AudioCancelledError) else 400), str(e)
            else:
                status, message = 500, f"Local inference failed: {e}"
            self._json(status, {"error": message})

    def _audio(self):
        if self.server.audio is None:
            raise RequestError("Audio endpoints are disabled. Install the voice extra, prepare assets and start with --voice.", 503)
        return self.server.audio

    def _transcribe(self):
        audio = self._audio()
        if self.headers.get_content_type() not in ("audio/wav", "audio/x-wav"):
            raise RequestError("Use Content-Type: audio/wav with PCM16 WAV data.", 415)
        from .audio import MAX_AUDIO_BYTES
        text = audio.transcribe(self._raw_body(MAX_AUDIO_BYTES))
        self._json(200, {"text": text})

    def _voice_chat(self, body):
        self._fields(body, {"text", "session_id", "options"})
        audio = self._audio()
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError("text must be a nonempty string.")
        session = self._session(body["session_id"]) if "session_id" in body else None
        result = self.server.backend.generate(text, session=session, options=body.get("options"))
        if result.incomplete:
            raise RequestError("Voice reply generation cancelled; no audio was synthesized.", 409)
        # Synthesize the assistant's answer, never the user's transcript.
        wav = audio.wav(result.text)
        self._json(200, {**result.to_dict(), "audio": {
            "format": "wav", "mime_type": "audio/wav", "encoding": "base64",
            "data": base64.b64encode(wav).decode("ascii"),
        }})

    def _synthesize(self, body):
        self._fields(body, {"text", "stream"})
        if type(body.get("stream", False)) is not bool:
            raise RequestError("stream must be a boolean.")
        audio = self._audio()
        if not body.get("stream"):
            data = audio.wav(body.get("text"))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        with closing(audio.synthesize(body.get("text"))) as stream:
            first = next(stream)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            def event(data):
                self.wfile.write(json.dumps(data).encode() + b"\n")
                self.wfile.flush()
            def chunk(raw, rate):
                event({"type": "audio", "format": "pcm_s16le", "sample_rate": rate,
                       "channels": 1, "audio": base64.b64encode(raw).decode("ascii")})
            try:
                chunk(*first)
                for raw, rate in stream:
                    chunk(raw, rate)
                event({"type": "done"})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            except Exception as e:
                event({"type": "error", "error": str(e)})

    def _generate(self, body):
        self._fields(body, {"prompt", "session_id", "options", "stream"})
        if type(body.get("stream", False)) is not bool:
            raise RequestError("stream must be a boolean.")
        session = self._session(body["session_id"]) if "session_id" in body else None
        kwargs = {"session": session, "options": body.get("options")}
        if not body.get("stream"):
            result = self.server.backend.generate(body.get("prompt"), **kwargs)
            self._json(200, result.to_dict())
            return
        receipt = {}
        with closing(self.server.backend._stream(body.get("prompt"), receipt=receipt, **kwargs)) as stream:
            # Validate and start inference before sending HTTP 200.
            first = next(stream)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            def event(data):
                self.wfile.write(json.dumps(data, ensure_ascii=False, allow_nan=False).encode() + b"\n")
                self.wfile.flush()
            try:
                event({"type": "delta", "text": first})
                for text in stream:
                    event({"type": "delta", "text": text})
                event({"type": "done", **receipt})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return  # closing(stream) preserves incomplete output and clears KV cache
            except Exception as e:
                event({"type": "error", "error": str(e)})

    def do_GET(self):
        self._request("GET")

    def do_POST(self):
        self._request("POST")

    def do_DELETE(self):
        self._request("DELETE")

    def do_OPTIONS(self):
        try:
            self._authorize(preflight=True)
            if self.headers.get("Origin") not in self.server.allowed_origins:
                raise RequestError("An allowed browser origin is required.", 403)
            if self.headers.get("Access-Control-Request-Method") not in ("GET", "POST", "DELETE"):
                raise RequestError("Unsupported preflight method.")
            requested = {s.strip().lower() for s in self.headers.get("Access-Control-Request-Headers", "").split(",") if s.strip()}
            if requested - {"content-type", "authorization"}:
                raise RequestError("Unsupported preflight headers.")
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except RequestError as e:
            self._json(e.status, {"error": str(e)})


def serve(backend: Mica, *, port=8000, token=None, max_sessions=32, audio=False,
          audio_settings=None, allowed_origins=()):
    """Block until shutdown; caller owns backend lifetime. No server starts on import."""
    with MicaHTTPServer(backend, port=port, token=token, max_sessions=max_sessions,
                        audio=audio, audio_settings=audio_settings, allowed_origins=allowed_origins) as server:
        server.serve_forever(poll_interval=0.5)
