import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mica.config import ChatError, Settings
from mica.conversation import Conversation
from mica.voice import VoiceIO, VoiceSettings, asset_path, resolve_assets, speech_text, LocalSpeech


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m["content"]) + 3 for m in messages)))


class Adapters:
    def __init__(self, transcript="hello", fail=None):
        self.transcript, self.fail = transcript, fail
        self.closed, self.played, self.spoken = 0, [], []

    def record(self, settings, wait):
        if self.fail == "record":
            raise KeyboardInterrupt
        return [1]

    def transcribe(self, audio):
        if self.fail == "transcribe":
            raise KeyboardInterrupt
        return self.transcript

    def synthesize(self, text):
        self.spoken.append(text)
        if self.fail == "synthesize":
            raise KeyboardInterrupt
        if self.fail == "tts":
            raise ChatError("bad TTS")
        yield [1], 24000
        yield [2], 24000

    def play(self, audio, rate):
        if self.fail == "play":
            raise KeyboardInterrupt
        self.played.append(audio)

    def close(self):
        self.closed += 1


def run_chat(monkeypatch, tmp_path, inputs, adapter=None, generation_fail=None, enable_fail=False, args=(), pieces=None, between_pieces=None):
    import mica.cli as cli
    import mica.voice as voice
    import prompt_toolkit
    monkeypatch.setenv("MICA_HOME", str(tmp_path))
    prompts = iter(inputs)
    loaded, contexts, invalidations = [], [], []
    class Engine:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.stats = {}
        def load(self, spec, settings):
            self.spec, self.model, self.load_seconds = spec, spec.identity, 0
            loaded.append(spec.identity)
        def invalidate(self):
            invalidations.append(True)
        def stream(self, tokens, settings):
            contexts.append(tokens)
            for piece in pieces or ["**answer**"]:
                yield piece
                if between_pieces:
                    between_pieces()
            if generation_fail:
                raise generation_fail
        def close(self):
            pass
    calls = []
    def create(settings, offline):
        calls.append(True)
        if enable_fail:
            raise ChatError("missing voice dependencies")
        return VoiceIO(settings, adapter, adapter, adapter)
    monkeypatch.setattr(cli, "setup", lambda offline: None)
    monkeypatch.setattr(cli, "Engine", Engine)
    monkeypatch.setattr(cli, "resolve", lambda identity, *a, **kw: SimpleNamespace(identity=identity, context=4096))
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **kw: SimpleNamespace(prompt=lambda *a: next(prompts)))
    monkeypatch.setattr(voice, "create_voice", create)
    result = CliRunner().invoke(cli.app, ["chat", "--model", "org/model", *args])
    assert result.exit_code == 0, (result.stdout, result.exception)
    return result.stdout, loaded, contexts, calls, invalidations


def test_text_does_not_import_speech_dependencies():
    # Separate interpreter proves dependency absence, even after other tests import adapters.
    code = '''
import sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'mlx_whisper', 'mlx_audio', 'sounddevice', 'misaki', 'spacy', 'numpy'}:
            raise AssertionError('speech dependency imported: ' + fullname)
sys.meta_path.insert(0, Deny())
from mica.cli import app
from mica.config import Settings
Settings().validate()
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_default_text_and_switch_preserve_conversation(monkeypatch, tmp_path):
    adapter = Adapters("/new")  # Spoken slash commands are content, never commands.
    saved = tmp_path / "turns.json"
    out, loaded, contexts, calls, invalidations = run_chat(monkeypatch, tmp_path,
        ["first", "/mode", "/mode voice", "", "/mode text", "third", f"/save {saved}", "/exit"], adapter)
    state = Conversation.load(saved)
    assert [m["content"] for m in state.messages[::2]] == ["first", "/new", "third"]
    assert loaded == ["org/model"] and len(contexts) == 3
    assert not invalidations and len(calls) == 1
    assert out.count("You (voice)> /new") == 1
    assert "I/O mode: text" in out and "Unknown command" not in out
    assert adapter.spoken == ["answer"] and adapter.played == [[1], [2]]
    assert adapter.closed
    assert state.messages[1]["content"] == "**answer**"


def test_text_default_never_creates_voice(monkeypatch, tmp_path):
    out, _, contexts, calls, _ = run_chat(monkeypatch, tmp_path, ["hello", "/exit"])
    assert len(contexts) == 1 and not calls and "You (voice)>" not in out


@pytest.mark.parametrize("transcript", ["", "   "])
def test_empty_transcript_does_not_change_history(monkeypatch, tmp_path, transcript):
    saved = tmp_path / "empty.json"
    out, _, contexts, _, _ = run_chat(monkeypatch, tmp_path,
        ["", f"/save {saved}", "/exit"], Adapters(transcript), args=["--io-mode", "voice"])
    assert not contexts and not Conversation.load(saved).messages
    assert "No speech detected" in out and "You (voice)>" not in out


@pytest.mark.parametrize("stage", ["record", "transcribe"])
def test_input_cancellation_discards_pending_turn(monkeypatch, tmp_path, stage):
    saved = tmp_path / "cancelled.json"
    adapter = Adapters(fail=stage)
    out, _, contexts, _, _ = run_chat(monkeypatch, tmp_path,
        ["", f"/save {saved}", "/exit"], adapter, args=["--io-mode", "voice"])
    assert not contexts and not Conversation.load(saved).messages
    assert adapter.closed and not adapter.played
    assert "Voice input cancelled" in out


@pytest.mark.parametrize("stage", ["synthesize", "play", "tts"])
def test_speech_cancellation_or_failure_preserves_completed_text(monkeypatch, tmp_path, stage):
    saved = tmp_path / "completed.json"
    adapter = Adapters(fail=stage)
    out, _, contexts, _, _ = run_chat(monkeypatch, tmp_path,
        ["", f"/save {saved}", "/mode text", "/exit"], adapter, args=["--io-mode", "voice"])
    messages = Conversation.load(saved).messages
    assert len(contexts) == 1 and len(messages) == 2
    assert messages[-1] == {"role": "assistant", "content": "**answer**", "incomplete": False}
    assert "text retained" in out.lower() and not adapter.played and adapter.closed


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("generation failed")])
def test_generation_cancellation_never_speaks(monkeypatch, tmp_path, failure):
    adapter = Adapters()
    saved = tmp_path / "incomplete.json"
    run_chat(monkeypatch, tmp_path, ["", f"/save {saved}", "/exit"], adapter,
        generation_fail=failure, args=["--io-mode", "voice"])
    assert Conversation.load(saved).messages[-1]["incomplete"]
    assert not adapter.spoken and not adapter.played


def test_failed_voice_enable_keeps_text_usable(monkeypatch, tmp_path):
    out, loaded, contexts, _, _ = run_chat(monkeypatch, tmp_path,
        ["/mode voice", "hello", "/exit"], enable_fail=True)
    assert len(contexts) == 1 and loaded == ["org/model"]
    assert "missing voice dependencies" in out


def test_offline_only_uses_cache(monkeypatch):
    import huggingface_hub
    calls = []
    def snapshot(identity, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"local_files_only": True}
        raise FileNotFoundError("missing")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(ChatError, match="voice-download"):
        asset_path("org/model", True, prepare=True)
    assert len(calls) == 1


def test_asset_voice_and_model_compatibility(tmp_path):
    whisper, kokoro = tmp_path / "whisper", tmp_path / "kokoro"
    whisper.mkdir(); kokoro.mkdir(); (kokoro / "voices").mkdir()
    (whisper / "config.json").write_text(json.dumps({"n_mels": 80, "n_vocab": 51864}))
    (whisper / "weights.npz").touch()
    (kokoro / "config.json").write_text(json.dumps({"istftnet": {}, "plbert": {}, "vocab": {}, "style_dim": 128}))
    (kokoro / "model.safetensors").touch()
    settings = VoiceSettings(whisper_model=str(whisper), kokoro_model=str(kokoro))
    with pytest.raises(ChatError, match="voice asset missing"):
        resolve_assets(settings, True)
    (kokoro / "voices" / "af_heart.safetensors").touch()
    assert resolve_assets(settings, True)[2].endswith("af_heart.safetensors")
    settings.language = "fr"
    with pytest.raises(ChatError, match="English-only"):
        resolve_assets(settings, True)


@pytest.mark.parametrize("change", [{"speed": float("nan")}, {"speed": True}, {"max_duration": 301}, {"microphone": -1}, {"voice": "bogus"}, {"voice": "jf_alpha"}])
def test_invalid_voice_settings(change):
    with pytest.raises(ChatError):
        VoiceSettings(**change).validate()


def test_speech_representation():
    raw = "# Title\n**Hello** [friend](https://example.com)\n```python\nprint('secret')\n```"
    assert speech_text(raw) == "Title\nHello friend\n Code omitted."
    assert "secret" in raw


def test_worker_close_kills_and_clears_stale_connections():
    events = []
    process = SimpleNamespace(pid=123, terminate=lambda: events.append("terminate"), join=lambda **kw: events.append("join"),
        is_alive=lambda: False, close=lambda: events.append("process-close"))
    speech = LocalSpeech.__new__(LocalSpeech)
    speech.process = process
    speech.connection = SimpleNamespace(close=lambda: events.append("pipe-close"))
    speech.close(); speech.close()
    assert events == ["terminate", "join", "process-close", "pipe-close"]
    assert speech.process is None and speech.connection is None


def test_playback_cancellation_aborts_device_queue(monkeypatch):
    import numpy as np
    from mica.voice import SoundDeviceAudio
    events = []
    class Stream:
        def start(self): events.append("start")
        def write(self, data):
            events.append("write")
            raise KeyboardInterrupt
        def stop(self): events.append("drain")
        def abort(self): events.append("abort")
        def close(self): events.append("close")
    audio = SoundDeviceAudio.__new__(SoundDeviceAudio)
    audio.sd = SimpleNamespace(OutputStream=lambda **kw: Stream())
    audio.np, audio.settings = np, VoiceSettings()
    with pytest.raises(KeyboardInterrupt):
        audio.play(np.ones(5000), 24000)
    assert events == ["start", "write", "abort", "close"]


def test_capture_duration_bound_and_cleanup():
    import numpy as np
    from mica.voice import SoundDeviceAudio
    closed = []
    class Stop(Exception): pass
    class Stream:
        def __init__(self, callback): self.callback = callback
        def __enter__(self):
            try:
                # More input frames than the configured one-second buffer.
                self.callback(np.ones((20000, 1), dtype="float32"), 20000, None, None)
            except Stop:
                pass
            return self
        def __exit__(self, *args): closed.append(True)
    audio = SoundDeviceAudio.__new__(SoundDeviceAudio)
    audio.sd = SimpleNamespace(InputStream=lambda **kw: Stream(kw["callback"]), CallbackStop=Stop)
    audio.np, audio.settings = np, VoiceSettings(max_duration=1)
    def wait(full): assert full.is_set()
    pcm = audio.record(audio.settings, wait)
    assert pcm.shape == (16000,) and pcm.dtype == np.float32 and closed == [True]


def test_worker_crash_is_actionable():
    speech = LocalSpeech.__new__(LocalSpeech)
    speech.connection = SimpleNamespace(poll=lambda timeout: False)
    speech.process = SimpleNamespace(is_alive=lambda: False)
    with pytest.raises(ChatError, match="memory"):
        speech._receive()


def test_voice_settings_persist_in_existing_session_format(tmp_path, monkeypatch):
    monkeypatch.setenv("MICA_HOME", str(tmp_path))
    settings = Settings(voice={"language": "en", "voice": "bf_emma", "speed": 1.2, "microphone": 2})
    settings.save()
    assert Settings.load().voice == settings.voice
    path = tmp_path / "session.json"
    Conversation("model", settings).save(path)
    assert Conversation.load(path).settings.voice == settings.voice


def test_failed_worker_start_releases_both_pipe_ends(monkeypatch):
    import mica.voice as voice
    events = []
    def fail_start():
        raise OSError("Cannot spawn worker")
    process = SimpleNamespace(pid=None, start=fail_start, close=lambda: events.append("process"))
    parent = SimpleNamespace(close=lambda: events.append("parent"))
    child = SimpleNamespace(close=lambda: events.append("child"))
    context = SimpleNamespace(Pipe=lambda: (parent, child), Process=lambda **kw: process)
    monkeypatch.setattr(voice, "get_context", lambda method: context)
    with pytest.raises(OSError, match="Cannot spawn"):
        LocalSpeech(VoiceSettings(), ("whisper", "kokoro", "voice"))
    assert events == ["child", "process", "parent"]


def test_sentence_chunker_handles_token_splits_and_fenced_code():
    from mica.voice import SentenceChunks
    chunks = SentenceChunks()
    assert chunks.feed("Hello wor") == []
    assert chunks.feed("ld. Next?") == ["Hello world.", "Next?"]
    assert chunks.feed("\n```python\nprint('secret.')\n") == []
    assert chunks.feed("```\nLast words") == ["Code omitted."]
    assert chunks.finish() == ["Last words"]


def test_streaming_speech_starts_before_generation_finishes_and_keeps_order():
    import threading
    from mica.voice import StreamingSpeech
    started = threading.Event()
    adapter = Adapters()
    original = adapter.synthesize
    def synthesize(text):
        started.set()
        yield from original(text)
    adapter.synthesize = synthesize
    voice = VoiceIO(VoiceSettings(), adapter, adapter, adapter)
    pipeline = StreamingSpeech(voice, capacity=2)
    try:
        pipeline.feed("First sentence.")
        assert started.wait(2), "Speech waited for generation completion"
        pipeline.feed(" Second sentence! Remaining tail")
        pipeline.finish()
        assert adapter.spoken == ["First sentence.", "Second sentence!", "Remaining tail"]
        assert adapter.played == [[1], [2]] * 3
    finally:
        pipeline.cancel()
    assert not pipeline.thread.is_alive()


def test_streaming_cancel_discards_queued_sentences_and_stale_audio():
    import threading
    from mica.voice import StreamingSpeech
    started, released = threading.Event(), threading.Event()
    adapter = Adapters()
    def synthesize(text):
        adapter.spoken.append(text)
        started.set()
        assert released.wait(2)
        yield [99], 24000
    adapter.synthesize = synthesize
    adapter.close = released.set
    voice = VoiceIO(VoiceSettings(), adapter, adapter, adapter)
    pipeline = StreamingSpeech(voice, capacity=2)
    pipeline.feed("First.")
    assert started.wait(2)
    pipeline.feed(" Second. Third.")
    assert pipeline.queue.qsize() <= 2
    pipeline.cancel()
    assert not pipeline.thread.is_alive()
    assert adapter.spoken == ["First."] and not adapter.played
    assert pipeline.queue.empty()


def test_streaming_failure_discards_audio_without_interrupting_text_producer():
    import threading
    from mica.voice import StreamingSpeech
    failed = threading.Event()
    adapter = Adapters()
    def synthesize(text):
        failed.set()
        raise ChatError("TTS failure")
        yield
    adapter.synthesize = synthesize
    voice = VoiceIO(VoiceSettings(), adapter, adapter, adapter)
    pipeline = StreamingSpeech(voice)
    pipeline.feed("First.")
    assert failed.wait(2)
    pipeline.feed(" Text continues. Final tail")  # No error propagated into LLM generation.
    with pytest.raises(ChatError, match="TTS failure"):
        pipeline.finish()
    assert not adapter.played and not pipeline.thread.is_alive()


def test_cli_speaks_while_generator_is_still_active(monkeypatch, tmp_path):
    import threading
    started = threading.Event()
    adapter = Adapters()
    original = adapter.play
    def play(audio, rate):
        original(audio, rate)
        started.set()
    adapter.play = play
    def between():
        assert started.wait(2), "Generation must not finish before first speech"
    saved = tmp_path / "streamed.json"
    run_chat(monkeypatch, tmp_path, ["", f"/save {saved}", "/exit"], adapter,
        args=["--io-mode", "voice"], pieces=["First. ", "Second!", " Tail"], between_pieces=between)
    assert adapter.spoken == ["First.", "Second!", "Tail"]
    assert Conversation.load(saved).messages[-1]["content"] == "First. Second! Tail"


def test_cli_interrupt_after_first_sentence_stops_future_speech(monkeypatch, tmp_path):
    import threading
    played = threading.Event()
    adapter = Adapters()
    adapter.play = lambda audio, rate: played.set()
    def between():
        assert played.wait(2)
        raise KeyboardInterrupt
    saved = tmp_path / "partial.json"
    run_chat(monkeypatch, tmp_path, ["", f"/save {saved}", "/mode text", "typed", "/exit"], adapter,
        args=["--io-mode", "voice"], pieces=["First. ", "Never spoken."], between_pieces=between)
    assert adapter.spoken == ["First."]
    assert Conversation.load(saved).messages[-1]["incomplete"] is True


def test_idle_worker_shutdown_exits_without_termination():
    events = []
    alive = iter([True, False])
    speech = LocalSpeech.__new__(LocalSpeech)
    speech.process = SimpleNamespace(pid=123, is_alive=lambda: next(alive),
        join=lambda **kw: events.append("join"), close=lambda: events.append("process-close"),
        terminate=lambda: pytest.fail("Idle worker should exit normally"))
    speech.connection = SimpleNamespace(send=lambda message: events.append(message), close=lambda: events.append("pipe-close"))
    speech.shutdown()
    assert events == [("close", None), "join", "process-close", "pipe-close"]
