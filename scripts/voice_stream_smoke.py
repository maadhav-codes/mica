"""Measure offline LLM/Kokoro overlap without opening microphone/playback devices.

Requires staged default models: uv run --locked --extra voice python scripts/voice_stream_smoke.py
"""
import time
from mica.config import Settings, offline_environment
from mica.models import resolve
from mica.inference import Engine
from mica.conversation import Conversation
from mica.voice import VoiceSettings, LocalSpeech, resolve_assets, VoiceIO

def main():
    offline_environment()
    settings = Settings(temperature=0, max_tokens=180)
    voice_settings = VoiceSettings()
    paths = resolve_assets(voice_settings, True)
    speech = LocalSpeech(voice_settings, paths)
    started = time.monotonic()
    played = []
    class Audio:
        def play(self, audio, rate):
            played.append((time.monotonic() - started, len(audio)))
        def close(self):
            pass
    audio = Audio()
    voice = VoiceIO(voice_settings, audio, speech, audio)
    engine = Engine()
    pipeline = None
    try:
        spec = resolve('mlx-community/Llama-3.2-3B-Instruct-4bit', offline=True)
        engine.load(spec, settings)
        conversation = Conversation(spec.identity, settings)
        tokens, _ = conversation.prepare('Start your answer with exactly "Hello." Then write ten short sentences explaining why the sky is blue.', engine.tokenizer, spec.context)
        started = time.monotonic()
        pipeline = voice.start_stream()
        output = []
        for piece in engine.stream(tokens, settings):
            output.append(piece)
            pipeline.feed(piece)
        generated = time.monotonic() - started
        pipeline.finish()
        print({'first_speech_audio_seconds': played[0][0], 'generation_finished_seconds': generated, 'audio_chunks': len(played), 'overlap': played[0][0] < generated})
        assert played[0][0] < generated
    finally:
        if pipeline is not None:
            pipeline.cancel()
        voice.close()
        engine.close()

if __name__ == '__main__':
    main()
