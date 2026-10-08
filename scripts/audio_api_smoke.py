"""Offline HTTP speech roundtrip with staged models; no audio devices are opened.

Run: uv run --locked --extra voice python scripts/audio_api_smoke.py
"""
from io import BytesIO
import base64
import json
from threading import Thread
from urllib.request import Request, urlopen
import wave

from mica.api import Mica
from mica.api.server import MicaHTTPServer


def main():
    with Mica("mlx-community/Llama-3.2-3B-Instruct-4bit", offline=True) as backend:
        server = MicaHTTPServer(backend, port=0, audio=True)
        thread = Thread(target=server.serve_forever)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            request = Request(base+"/v1/audio/synthesize", data=json.dumps({"text": "Hello from Mica. This speech is generated locally."}).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=90) as response:
                assert response.headers.get_content_type() == "audio/wav"
                data = response.read()
            with wave.open(BytesIO(data), "rb") as wav:
                assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
                duration = wav.getnframes() / wav.getframerate()
            request = Request(base+"/v1/audio/transcribe", data=data, headers={"Content-Type": "audio/wav"})
            with urlopen(request, timeout=90) as response:
                transcript = json.load(response)["text"]
            assert transcript.strip()
            request = Request(base+"/v1/audio/synthesize", data=json.dumps({"text": "Streaming audio works.", "stream": True}).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=90) as response:
                events = [json.loads(line) for line in response]
            assert events[0]["type"] == "audio" and events[-1]["type"] == "done"
            request = Request(base+"/v1/voice/chat", data=json.dumps({"text": "Say hello in one short sentence.", "options": {"max_tokens": 40, "temperature": 0}}).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=120) as response:
                reply = json.load(response)
            assert reply["text"].strip() and not reply["incomplete"]
            reply_wav = base64.b64decode(reply["audio"]["data"], validate=True)
            with wave.open(BytesIO(reply_wav), "rb") as wav:
                assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getnframes() > 0
            print(json.dumps({"wav_bytes": len(data), "duration_seconds": duration, "transcript": transcript, "streamed_chunks": len(events)-1,
                              "voice_reply": reply["text"], "voice_reply_wav_bytes": len(reply_wav)}))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)
            assert not thread.is_alive()


if __name__ == "__main__":
    main()
