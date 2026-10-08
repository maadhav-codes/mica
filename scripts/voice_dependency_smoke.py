"""Verify optional speech APIs and packaged phonemizer offline, without model weights/audio devices.

Run: uv run --locked --extra voice python scripts/voice_dependency_smoke.py
"""
import sys, socket, inspect, importlib.metadata as metadata
assert sys.version_info[:2] == (3, 12)
def denied(*args, **kwargs):
    raise AssertionError('Unexpected network access during speech compatibility check')
socket.socket.connect = denied
socket.socket.connect_ex = denied
from mica.config import offline_environment
offline_environment()
import mlx.core as mx
import mlx_whisper
from mlx_whisper.tokenizer import get_tokenizer
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline
import sounddevice
import spacy
assert spacy.util.is_package('en_core_web_sm')
spacy.cli.download = denied
pipeline = KokoroPipeline(lang_code='a', model=None, repo_id='local')
phonemes, _ = pipeline.g2p('Hello from Mica on Python three point twelve.')
assert phonemes and pipeline.g2p.fallback is not None
assert get_tokenizer(True, language='en', task='transcribe').encode('Hello')
assert 'path_or_hf_repo' in inspect.signature(mlx_whisper.transcribe).parameters
assert any(p.kind == p.VAR_KEYWORD for p in inspect.signature(load_model).parameters.values())
assert 'repo_id' in inspect.signature(KokoroPipeline).parameters
assert 'speed' in inspect.signature(KokoroPipeline.__call__).parameters
print('Python:', sys.version.split()[0], 'MLX device:', mx.device_info()['device_name'])
for package in ('mlx','mlx-lm','mlx-whisper','mlx-audio','misaki','sounddevice','spacy','en-core-web-sm','espeakng-loader'):
    print(package, metadata.version(package), 'Requires-Python:', metadata.metadata(package).get('Requires-Python'))
print('PortAudio:', sounddevice.get_portaudio_version())
print('Offline phonemizer:', phonemes)
print('Speech API signatures and Whisper tokenizer: passed')
