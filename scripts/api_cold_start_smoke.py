"""Exercise a fresh API server without any earlier Python generation/warmup."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from mica.config import atomic_json


def main():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "mica.api", "--model", "mlx-community/Llama-3.2-3B-Instruct-4bit", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    records = {"startup": "fresh subprocess, no inference warmup", "port": port}
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("MICA_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    def request(path, body=None):
        req = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                      headers=headers)
        return urlopen(req, timeout=60)
    try:
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                raise AssertionError("Server exited during startup: " + process.stdout.read())
            try:
                with urlopen(Request(base + "/health", headers=headers), timeout=1) as response:
                    assert json.load(response)["status"] == "ready"
                break
            except (URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise AssertionError("API startup exceeded 60 seconds")
                time.sleep(0.1)
        # Exact request reported by the user: first generation runs in HTTP.
        body = {"prompt": "Explain MLX briefly.", "options": {"max_tokens": 128, "temperature": 0.3}}
        with request("/v1/generate", body) as response:
            record = json.load(response)
            assert response.status == 200 and record["text"].strip() and not record["incomplete"]
            records["first_http_request"] = record
        with request("/v1/generate", {"prompt": "Say ready.", "stream": True, "options": {"max_tokens": 16}}) as response:
            events = [json.loads(line) for line in response]
            assert events[-1]["type"] == "done" and not events[-1]["incomplete"]
            records["streaming"] = events
        with request("/v1/sessions", {"system": "Answer briefly."}) as response:
            identity = json.load(response)["id"]
        turns = []
        for prompt in ("My name is Mira. Say hello.", "What is my name?"):
            with request("/v1/generate", {"session_id": identity, "prompt": prompt, "options": {"max_tokens": 32, "temperature": 0}}) as response:
                turns.append(json.load(response))
        assert "mira" in turns[-1]["text"].lower()
        assert turns[-1]["stats"]["cached_prompt_tokens"] > 0
        records["multi_turn_across_http_threads"] = turns
        body = {"session_id": identity, "prompt": "Write a very long detailed story with twenty chapters about a voyage through space.", "stream": True, "options": {"max_tokens": 512}}
        with request("/v1/generate", body) as response:
            assert json.loads(response.readline())["type"] == "delta"
            with request("/v1/cancel", {}) as cancelled:
                assert json.load(cancelled)["cancel_requested"]
            remaining = [json.loads(line) for line in response]
            assert remaining[-1]["type"] == "done" and remaining[-1]["incomplete"]
        with request("/v1/sessions/" + identity) as response:
            assert json.load(response)["messages"][-1]["incomplete"] is True
        with request("/v1/generate", {"session_id": identity, "prompt": "What is my name?", "options": {"max_tokens": 24, "temperature": 0}}) as response:
            result = json.load(response)
            assert "mira" in result["text"].lower() and not result["incomplete"]
        records["cancellation_and_recovery"] = "passed"
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        records["server_exit_code"] = process.returncode
        records["server_output"] = process.stdout.read()
        process.stdout.close()
    assert process.returncode == 0, records["server_output"]
    atomic_json(Path("verification/api-cold-start.json"), records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
