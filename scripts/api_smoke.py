"""Real native MLX smoke checks for the public Python and local HTTP APIs."""
from contextlib import closing
import json
from pathlib import Path
import socket
from threading import Thread
from urllib.request import Request, urlopen

from mica.api import Mica, Settings
from mica.api.server import MicaHTTPServer
from mica.config import atomic_json


def main():
    # Permit localhost API sockets; block remote network connections.
    connect = socket.socket.connect
    def local_only(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"Unexpected non-local connection: {address}")
        return connect(sock, address)
    socket.socket.connect = local_only
    results = {}
    try:
        with Mica("mlx-community/Llama-3.2-3B-Instruct-4bit", settings=Settings(temperature=0, max_tokens=32)) as backend:
            session = backend.new_session(system="Answer briefly.")
            first = backend.generate("My name is Mira. Say hello.", session=session)
            second = backend.generate("What is my name?", session=session)
            assert first.text.strip() and "mira" in second.text.lower()
            assert second.stats["cached_prompt_tokens"] > 0
            results["python_multiturn"] = [first.to_dict(), second.to_dict()]
            with closing(backend.stream("Write a long story.", session=session)) as stream:
                next(stream)
            assert session.messages[-1]["incomplete"] is True
            backend.save_session(session, ".mica-test/api-partial.json")
            restored = backend.load_session(".mica-test/api-partial.json")
            assert restored.messages[-1]["incomplete"]
            assert backend.generate("Say ready.", session=restored).text.strip()
            results["stream_close_session_recovery"] = "passed"
            server = MicaHTTPServer(backend, port=0)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                def request(body):
                    return urlopen(Request(base + "/v1/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=60)
                with request({"prompt": "Say hello.", "options": {"max_tokens": 16}}) as response:
                    record = json.load(response)
                    assert record["text"].strip()
                    results["http_generation"] = record
                with request({"prompt": "Say ready.", "stream": True, "options": {"max_tokens": 16}}) as response:
                    events = [json.loads(line) for line in response]
                    assert any(e.get("text") for e in events)
                    assert events[-1]["type"] == "done" and not events[-1]["incomplete"]
                    results["http_streaming"] = events
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        atomic_json(Path("verification/api-smoke.json"), results)
        print(json.dumps(results, indent=2))
    finally:
        socket.socket.connect = connect


if __name__ == "__main__":
    main()
