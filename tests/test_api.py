from contextlib import closing
from copy import deepcopy
import json
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mica.api import ChatError, Mica, MicaBusyError, Session, Settings
from mica.api.server import MicaHTTPServer


@pytest.fixture
def backend(monkeypatch):
    import mica.api.client as api
    from mica.models import reject_name

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = deepcopy(messages)
            return list(range(2 + sum(3 + len(m["content"]) for m in messages)))

    class FakeEngine:
        def __init__(self):
            self.model = self.spec = None
            self.tokenizer = Tokenizer()
            self.stats = {}
            self.prompts = []
            self.invalidations = 0
            self.fail = False
            self.gate = None
            self.thread_calls = []
        def load(self, spec, settings):
            self.thread_calls.append(("load", get_ident()))
            self.model, self.spec = object(), spec
            self.invalidate()
        def invalidate(self):
            self.thread_calls.append(("invalidate", get_ident()))
            self.invalidations += 1
        def close(self):
            self.thread_calls.append(("close", get_ident()))
            self.model = self.spec = None
        def stream(self, tokens, settings):
            self.thread_calls.append(("generate", get_ident()))
            self.prompts.append(deepcopy(self.tokenizer.messages))
            complete = False
            try:
                yield "Hello"
                if self.gate:
                    assert self.gate.wait(5), "HTTP test did not release generation"
                if self.fail:
                    raise RuntimeError("allocation failed")
                yield " world"
                complete = True
            finally:
                self.thread_calls.append(("stream_cleanup", get_ident()))
                self.stats = {"finish_reason": "stop" if complete else "cancelled", "generation_tokens": 2, "seed": settings.seed}

    def resolve(model, **kwargs):
        reject_name(model)
        assert kwargs["offline"] is True
        return SimpleNamespace(identity=model, context=4096)
    monkeypatch.setattr(api, "Engine", FakeEngine)
    monkeypatch.setattr(api, "resolve", resolve)
    monkeypatch.setattr(api, "check_platform", lambda: None)
    monkeypatch.setattr(api, "offline_environment", lambda: None)
    with Mica("org/model") as instance:
        yield instance


def test_generate_is_stateless_and_returns_metrics(backend):
    result = backend.generate("one", options={"seed": 7})
    assert result.text == "Hello world" and result.session_id is None
    assert result.stats["seed"] == 7 and not result.incomplete
    assert result.to_dict()["model"] == "org/model"
    backend.generate("two")
    assert [m["content"] for m in backend._engine.prompts[-1] if m["role"] == "user"] == ["two"]


def test_model_and_stream_operations_keep_thread_affinity(backend):
    main_thread = get_ident()
    outputs = []
    errors = []
    def request():
        try:
            outputs.append(backend.generate("Explain MLX briefly.", options={"temperature": 0.3}).text)
        except Exception as e:
            errors.append(e)
    # The first request and each following request come from fresh HTTP-like threads.
    for _ in range(3):
        thread = Thread(target=request)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    with closing(backend.stream("cancelled")) as stream:
        next(stream)
    backend.switch_model("org/other")
    backend.close()
    assert not errors and outputs == ["Hello world"] * 3
    calls = backend._engine.thread_calls
    assert {name for name, _ in calls} >= {"load", "generate", "stream_cleanup", "invalidate", "close"}
    assert len({identity for _, identity in calls}) == 1
    assert calls[0][1] != main_thread


def test_invalid_model_selection_preserves_session(backend):
    session = backend.new_session()
    with pytest.raises(ChatError, match="GGUF"):
        backend.switch_model("invalid.gguf")
    assert backend.generate("still valid", session=session).text


def test_isolated_sessions_and_settings_copies(backend):
    first = backend.new_session(system="Be brief")
    second = backend.new_session()
    backend.generate("My name is Mira", session=first)
    backend.generate("What is my name?", session=first)
    assert len(first.messages) == 4
    backend.generate("Other conversation", session=second)
    assert len(second.messages) == 2
    assert "Mira" not in str(backend._engine.prompts[-1])
    copy = first.messages
    copy.clear()
    assert len(first.messages) == 4
    first.settings.system = "external mutation"
    assert first.settings.system == "Be brief"


def test_closed_stream_records_incomplete_and_recovers(backend):
    session = backend.new_session()
    with closing(backend.stream("first", session=session)) as stream:
        assert next(stream) == "Hello"
        with pytest.raises(MicaBusyError):
            backend.generate("concurrent")
    assert session.messages[-1] == {"role": "assistant", "content": "Hello", "incomplete": True}
    result = backend.generate("second", session=session)
    assert result.text == "Hello world"
    assert len(session.messages) == 2 and session.messages[0]["content"] == "second"


def test_cancel_busy_operation_and_preserve_partial(backend):
    session = backend.new_session()
    assert not backend.cancel()
    with closing(backend.stream("question", session=session)) as stream:
        next(stream)
        with pytest.raises(MicaBusyError):
            backend.close()
        assert backend.cancel()
        assert list(stream) == []
    assert session.messages[-1]["incomplete"]
    assert backend.stats["finish_reason"] == "cancelled"
    assert not backend.cancel()


@pytest.mark.parametrize("prompt,options", [("", None), (None, None), ("q", {"temperature": -1}), ("q", {"backend": "torch"}), ("q", {"max_tokens": 5000}), ("q", [])])
def test_invalid_requests_leave_history_unchanged(backend, prompt, options):
    session = backend.new_session()
    with pytest.raises(ChatError):
        backend.generate(prompt, session=session, options=options)
    assert session.messages == []
    assert backend.generate("valid", session=session).text


def test_inference_error_is_incomplete_not_a_success(backend):
    session = backend.new_session()
    backend._engine.fail = True
    with pytest.raises(RuntimeError, match="allocation"):
        backend.generate("first", session=session)
    assert session.messages[-1]["incomplete"]
    backend._engine.fail = False
    backend.generate("second", session=session)
    assert session.messages[0]["content"] == "second"


def test_sessions_save_load_reset_system_and_model_switch(backend, tmp_path):
    session = backend.new_session()
    backend.generate("saved", session=session)
    path = tmp_path / "session.json"
    backend.save_session(session, path)
    restored = backend.load_session(path)
    assert restored.id != session.id and restored.messages == session.messages
    backend.set_system(restored, "Changed")
    assert restored.settings.system == "Changed"
    backend.reset(restored)
    assert restored.messages == []
    backend.switch_model("org/other")
    with pytest.raises(ChatError, match="previous model"):
        backend.generate("q", session=session)
    with pytest.raises(ChatError, match="different model"):
        backend.load_session(path)
    with pytest.raises(ChatError, match="GGUF"):
        backend.switch_model("model.gguf")
    assert backend.model == "org/other"  # validation failure keeps the current model


def test_single_model_lease_and_closed_lifecycle(backend):
    with pytest.raises(MicaBusyError, match="already owns"):
        Mica("org/second")
    backend.close()
    backend.close()
    with pytest.raises(ChatError, match="closed"):
        backend.generate("q")
    with pytest.raises(ChatError, match="GGUF"):
        Mica("model.gguf")
    with Mica("org/replacement") as replacement:
        assert replacement.generate("q").text


@pytest.fixture
def http_api(backend):
    server = MicaHTTPServer(backend, port=0, max_sessions=2)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def request(base, path, body=None, method=None, headers=None):
    req = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                  method=method, headers={"Content-Type": "application/json", **(headers or {})})
    return urlopen(req, timeout=5)


def test_http_generation_sessions_and_ndjson(http_api):
    base, _ = http_api
    with request(base, "/health") as response:
        assert json.load(response)["service"] == "mica"
    with request(base, "/v1/sessions", {"system": "Be concise"}) as response:
        identity = json.load(response)["id"]
        assert response.status == 201
    with request(base, "/v1/generate", {"prompt": "Hi", "session_id": identity}) as response:
        result = json.load(response)
        assert result["text"] == "Hello world" and result["session_id"] == identity
    with request(base, "/v1/generate", {"prompt": "Next", "session_id": identity, "stream": True}) as response:
        assert response.headers.get_content_type() == "application/x-ndjson"
        events = [json.loads(line) for line in response]
        assert "".join(e["text"] for e in events if e["type"] == "delta") == "Hello world"
        assert events[-1]["type"] == "done" and not events[-1]["incomplete"]
    with request(base, "/v1/sessions/" + identity) as response:
        assert len(json.load(response)["messages"]) == 4
    with request(base, "/v1/sessions/" + identity, method="DELETE") as response:
        assert json.load(response)["deleted"] == identity


@pytest.mark.parametrize("body,status", [({"prompt": "", "stream": True}, 400), ({"prompt": "q", "session_id": "absent"}, 404), ({"prompt": "q", "stream": "yes"}, 400), ({"prompt": "q", "unknown": 1}, 400), ({"prompt": "q", "options": {"seed": -1}}, 400)])
def test_http_actionable_validation(http_api, body, status):
    base, _ = http_api
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/generate", body)
    assert exc.value.code == status and json.load(exc.value)["error"]


def test_http_loopback_auth_origin_and_session_limit(http_api):
    base, server = http_api
    server.token = "test-secret"
    with pytest.raises(HTTPError) as exc:
        request(base, "/health")
    assert exc.value.code == 401
    headers = {"Authorization": "Bearer test-secret"}
    with request(base, "/health", headers=headers) as response:
        assert response.status == 200
    with pytest.raises(HTTPError) as exc:
        request(base, "/health", headers={**headers, "Origin": "https://example.com"})
    assert exc.value.code == 403
    for _ in range(2):
        request(base, "/v1/sessions", {}, headers=headers).close()
    with pytest.raises(HTTPError) as exc:
        request(base, "/v1/sessions", {}, headers=headers)
    assert exc.value.code == 429
    with pytest.raises(ValueError, match="127.0.0.1"):
        MicaHTTPServer(server.backend, host="0.0.0.0")


@pytest.mark.parametrize("data,headers,status", [
    (b'{"prompt":"q","options":{"temperature":NaN}}', {}, 400),
    (b"not json", {}, 400),
    (b'{}', {"Content-Type": "text/plain"}, 415),
    (b'{}', {"Content-Length": "524289"}, 413),
    (b'{}', {"Host": "untrusted.example"}, 403),
])
def test_http_body_and_host_validation(http_api, data, headers, status):
    base, _ = http_api
    req = Request(base + "/v1/generate", data=data, headers={"Content-Type": "application/json", **headers})
    with pytest.raises(HTTPError) as exc:
        urlopen(req, timeout=5)
    assert exc.value.code == status


def test_http_cancel_stream_and_busy_status(http_api, backend):
    base, _ = http_api
    backend._engine.gate = Event()
    try:
        with request(base, "/v1/generate", {"prompt": "q", "stream": True}) as response:
            assert json.loads(response.readline())["type"] == "delta"
            with pytest.raises(HTTPError) as exc:
                request(base, "/v1/generate", {"prompt": "concurrent"})
            assert exc.value.code == 503
            with request(base, "/v1/cancel", {}) as cancel:
                assert json.load(cancel)["cancel_requested"]
            backend._engine.gate.set()
            assert json.loads(response.readline())["incomplete"] is True
    finally:
        backend._engine.gate.set()
