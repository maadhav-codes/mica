"""Optional, half-duplex speech adapters. No speech imports on the text path."""
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import json
import re
import threading
from typing import Protocol

from .config import ChatError, offline_environment


@dataclass
class VoiceSettings:
    whisper_model: str = "mlx-community/whisper-small-mlx"
    language: str = "auto"
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    voice: str = "af_heart"
    speed: float = 1.0
    microphone: str | int | None = None
    output_device: str | int | None = None
    max_duration: float = 60.0

    def validate(self):
        import math
        for name in ("whisper_model", "kokoro_model", "language", "voice"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ChatError(f"voice.{name} must be a nonempty string.")
        for name, low, high in (("speed", 0.5, 2.0), ("max_duration", 1, 300)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ChatError(f"voice.{name} must be between {low} and {high}.")
        for name in ("microphone", "output_device"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (str, int) or value == "" or isinstance(value, int) and value < 0):
                raise ChatError(f"voice.{name} must be a device name or nonnegative index.")
        if not re.fullmatch(r"[abefhip][fm]_[a-z0-9]+", self.voice):
            raise ChatError("Use a Kokoro voice from the selected model (a/b English, e Spanish, f French, h Hindi, i Italian, p Portuguese). Japanese/Chinese phonemizers are not included in this extra.")
        return self


def speech_text(text):
    """A separate spoken representation; never changes terminal/history text."""
    text = re.sub(r"```.*?(?:```|$)|~~~.*?(?:~~~|$)", " Code omitted. ", text, flags=re.S)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s*|>\s*|[-*+]\s+)", "", text)
    return re.sub(r"[`*_]", "", text).strip()


class Capture(Protocol):
    def record(self, settings, wait_for_stop): ...
    def close(self): ...


class Speech(Protocol):
    def transcribe(self, audio): ...
    def synthesize(self, text): ...
    def close(self): ...


class Playback(Protocol):
    def play(self, audio, rate): ...
    def close(self): ...


class SoundDeviceAudio:
    def __init__(self, settings):
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as e:
            raise ChatError("Install speech dependencies: uv sync --locked --extra voice") from e
        self.sd, self.np, self.settings = sd, np, settings
        try:
            sd.check_input_settings(device=settings.microphone, channels=1, dtype="float32", samplerate=16000)
            sd.check_output_settings(device=settings.output_device, channels=1, dtype="float32", samplerate=24000)
        except Exception as e:
            raise ChatError(f"Audio device unavailable: {e}. Check macOS microphone permission for your terminal and device settings; list devices with uv run --extra voice python -m sounddevice.") from e

    def record(self, settings, wait_for_stop):
        # Whisper accepts mono float32 PCM at 16 kHz, avoiding ffmpeg/resampling.
        buffer = self.np.empty(int(16000 * settings.max_duration), dtype="float32")
        count = 0
        full = threading.Event()
        error = []

        def callback(data, frames, timing, status):
            nonlocal count
            if status:
                error.append(str(status))
                full.set()
                raise self.sd.CallbackStop
            n = min(frames, len(buffer) - count)
            buffer[count:count+n] = data[:n, 0]
            count += n
            if count == len(buffer):
                full.set()
                raise self.sd.CallbackStop

        try:
            with self.sd.InputStream(device=settings.microphone, channels=1, samplerate=16000, dtype="float32", callback=callback):
                wait_for_stop(full)
            if error:
                raise ChatError(f"Recording failed ({error[0]}). Check microphone/device load and retry.")
            audio = buffer[:count].copy()
            if not count or float(self.np.max(self.np.abs(audio))) < 0.001:
                return None
            return audio
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise ChatError(f"Microphone capture failed: {e}. Allow microphone access for your terminal in macOS Settings.") from e

    def play(self, audio, rate):
        # Own stream, no global queue or background playback surviving cancellation.
        if getattr(self, "_playback_cancelled", False):
            return
        stream = self.sd.OutputStream(device=self.settings.output_device, channels=1, samplerate=rate, dtype="float32")
        self._playback_stream = stream
        try:
            stream.start()
            audio = self.np.asarray(audio, dtype="float32").reshape(-1, 1)
            for start in range(0, len(audio), 2048):
                if getattr(self, "_playback_cancelled", False):
                    return
                stream.write(audio[start:start+2048])
            stream.stop()  # Drain only on successful playback.
        finally:
            try:
                stream.abort()  # Cancellation/errors discard queued device audio immediately.
            finally:
                stream.close()
                self._playback_stream = None

    def begin_speech(self):
        self._playback_cancelled = False

    def close(self):
        self._playback_cancelled = True
        stream = getattr(self, "_playback_stream", None)
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass  # Playback's finally block owns closing the stream.


def asset_path(identity, offline, prepare=False):
    path = Path(identity).expanduser()
    if path.is_dir():
        return path.resolve()
    from .assets import model_path, stage_model
    staged = model_path(identity)
    if staged.is_dir() and (not prepare or offline):
        return staged
    from huggingface_hub import snapshot_download
    if prepare and not offline:
        source = Path(snapshot_download(identity))
        return stage_model(identity, source, source.name)
    try:
        # Always try cache first; no metadata request when cached.
        source = Path(snapshot_download(identity, local_files_only=True))
        return stage_model(identity, source, source.name)
    except Exception as e:
        if offline:
            raise ChatError(f"Missing cached speech model {identity}. Run mica voice-download while online, or configure a complete local directory.") from e
    source = Path(snapshot_download(identity))
    return stage_model(identity, source, source.name)


def resolve_assets(settings, offline, prepare=False):
    settings.validate()
    whisper = asset_path(settings.whisper_model, offline, prepare)
    kokoro = asset_path(settings.kokoro_model, offline, prepare)
    try:
        w = json.loads((whisper / "config.json").read_text())
        k = json.loads((kokoro / "config.json").read_text())
        if "n_mels" not in w or not any((whisper / name).is_file() for name in ("weights.npz", "weights.safetensors")):
            raise ValueError("Whisper needs MLX config.json and weights.npz/weights.safetensors")
        required = {"istftnet", "plbert", "vocab", "style_dim"}
        if k.get("model_type") not in (None, "kokoro") or not required <= k.keys() or not list(kokoro.glob("*.safetensors")):
            raise ValueError("Kokoro needs its architecture config and safetensors weights")
        voice = kokoro / "voices" / f"{settings.voice}.safetensors"
        if not voice.is_file():
            raise ValueError(f"voice asset missing: {voice}")
        if settings.language != "auto" and w.get("n_vocab", 0) == 51864 and settings.language != "en":
            raise ValueError("English-only Whisper model requires language=en or auto")
    except (OSError, ValueError) as e:
        raise ChatError(f"Invalid/incomplete speech assets: {e}. Run mica voice-download online or select compatible local MLX models.") from e
    return str(whisper), str(kokoro), str(voice)


def _worker(connection, settings, paths):
    """Spawned process: cached models; no sockets, even inside third-party loaders."""
    offline_environment()
    from .assets import asset_root
    import sys
    runtime = asset_root() / "runtime"
    if runtime.is_dir():
        sys.path.insert(0, str(runtime))
    import socket
    def denied(*args, **kwargs):
        raise ChatError("Speech asset attempted network access. Run mica voice-download online to prepare all assets.")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    try:
        import numpy as np
        import mlx_whisper
        from mlx_whisper.tokenizer import LANGUAGES
        from mlx_audio.tts.utils import load_model
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline
        if settings.language != "auto" and settings.language not in LANGUAGES:
            raise ChatError("Invalid Whisper language; use auto or an ISO language code from mlx_whisper.tokenizer.LANGUAGES.")
        # Misaki otherwise calls spacy.cli.download (a pip subprocess) when absent.
        import spacy
        if settings.voice[0] in "ab" and not spacy.util.is_package("en_core_web_sm"):
            raise ChatError("Missing English phonemizer asset en_core_web_sm. Run uv sync --locked --python 3.12 --extra voice online.")
        spacy.cli.download = denied
        from misaki.espeak import EspeakFallback
        EspeakFallback(british=settings.voice[0] == "b")  # fail explicitly, never silently skip fallback
        import importlib
        import mlx.core as mx
        holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
        whisper = holder.get_model(paths[0], mx.float16)
        if settings.language != "auto" and settings.language != "en" and not whisper.is_multilingual:
            raise ChatError("This Whisper model is English-only; select a multilingual model or language=en.")
        model = load_model(Path(paths[1]), model_type="kokoro")
        pipeline = KokoroPipeline(lang_code=settings.voice[0], model=model, repo_id=paths[1])
        pack = pipeline.load_voice(paths[2])
        if len(pack.shape) != 3 or pack.shape[0] < 510 or pack.shape[1] != 1 or pack.shape[2] != 2 * model.config.style_dim:
            raise ChatError("Voice tensor is incompatible with the selected Kokoro model.")
        from mlx_whisper.tokenizer import get_tokenizer
        get_tokenizer(whisper.is_multilingual, num_languages=whisper.num_languages,
                      language=None if settings.language == "auto" else settings.language, task="transcribe")
        connection.send(("ready", None))
        while True:
            operation, value = connection.recv()
            if operation == "close":
                return
            elif operation == "transcribe":
                result = mlx_whisper.transcribe(value, path_or_hf_repo=paths[0], language=None if settings.language == "auto" else settings.language, verbose=None)
                del value  # Do not retain raw recordings while the language model generates.
                connection.send(("result", result["text"].strip()))
            else:
                segments = [value[i:i+400] for i in range(0, len(value), 400)]
                for result in pipeline(segments, voice=paths[2], speed=settings.speed):
                    if result.audio is not None:
                        connection.send(("audio", (np.asarray(result.audio, dtype=np.float32), model.sample_rate)))
                        # Backpressure: at most one generated chunk waiting for playback.
                        connection.recv()
                connection.send(("done", None))
    except BaseException as e:
        try:
            connection.send(("error", f"{type(e).__name__}: {e}"))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class LocalSpeech:
    def __init__(self, settings, paths, cancel_event=None):
        self.settings, self.paths = settings, paths
        self.cancel_event = cancel_event
        self.process = self.connection = None
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        def check_cancel():
            event = getattr(self, "cancel_event", None)
            if event is not None and event.is_set():
                raise ChatError("Speech operation cancelled.")
        check_cancel()
        while not self.connection.poll(0.05):
            check_cancel()
            if not self.process.is_alive():
                raise ChatError("Speech worker exited. Reduce model size/recording duration and check memory.")
        try:
            kind, value = self.connection.recv()
        except (EOFError, OSError) as e:
            raise ChatError("Speech worker disconnected. Check available memory and retry, or switch to text mode.") from e
        if kind == "error":
            raise ChatError(f"Local speech failed: {value}. Check voice assets, phonemizer setup and available memory; text chat remains usable.")
        return kind, value

    def _start(self):
        ctx = get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, self.settings, self.paths), daemon=True)
        try:
            try:
                self.process.start()
            finally:
                child.close()
            self._receive()
        except BaseException:
            self.close()
            raise

    def transcribe(self, audio):
        if self.process is None:
            self._start()
        self.connection.send(("transcribe", audio))
        return self._receive()[1]

    def synthesize(self, text):
        if self.process is None:
            self._start()
        self.connection.send(("synthesize", text))
        while True:
            kind, value = self._receive()
            if kind == "done":
                return
            yield value
            self.connection.send(("next", None))

    def shutdown(self):
        """Release an idle worker normally so third-party finalizers can run."""
        self.close(graceful=True)

    def close(self, graceful=False):
        if self.process is not None:
            if self.process.pid is not None:
                if graceful and self.process.is_alive() and self.connection is not None:
                    try:
                        self.connection.send(("close", None))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    self.process.join(timeout=2)
                else:
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class VoiceIO:
    def __init__(self, settings, capture, speech, playback):
        self.settings, self.capture, self.speech, self.playback = settings, capture, speech, playback

    def listen(self, wait_for_stop, state):
        try:
            state("Recording — Enter stops; Ctrl+C cancels.")
            audio = self.capture.record(self.settings, wait_for_stop)
            if audio is None:
                return ""
            state("Transcribing…")
            return self.speech.transcribe(audio).strip()
        except BaseException:
            self.close()
            raise

    def speak(self, text, state):
        text = speech_text(text)
        if not text:
            return
        chunks = None
        try:
            if hasattr(self.playback, "begin_speech"):
                self.playback.begin_speech()
            state("Speaking — synthesizing and playing; Ctrl+C stops.")
            chunks = self.speech.synthesize(text)
            for audio, rate in chunks:
                self.playback.play(audio, rate)
        except BaseException:
            self.close()
            raise
        finally:
            if chunks is not None and hasattr(chunks, "close"):
                chunks.close()

    def start_stream(self):
        return StreamingSpeech(self)

    def close(self):
        try:
            self.capture.close()
        finally:
            try:
                self.playback.close()
            finally:
                self.speech.close()


def create_voice(settings, offline):
    audio = SoundDeviceAudio(settings)
    try:
        paths = resolve_assets(settings, offline)
        return VoiceIO(settings, audio, LocalSpeech(settings, paths), audio)
    except BaseException:
        audio.close()
        raise


class SentenceChunks:
    """Incremental sentence boundaries, retaining fenced code until it closes."""
    def __init__(self):
        self.pending = ""

    def feed(self, piece):
        self.pending += piece
        chunks = []
        start = 0
        fence = None
        for match in re.finditer(r"```|~~~|[.!?](?=\s|$)|\n", self.pending):
            marker = match.group()
            if marker in ("```", "~~~"):
                if fence is None:
                    fence = marker
                elif fence == marker:
                    fence = None
            elif fence is None:
                text = speech_text(self.pending[start:match.end()])
                if text:
                    chunks.append(text)
                start = match.end()
        self.pending = self.pending[start:]
        return chunks

    def finish(self):
        text = speech_text(self.pending)
        self.pending = ""
        return [text] if text else []


class StreamingSpeech:
    """Bounded sentence producer/consumer; MLX speech stays in its own process."""
    def __init__(self, voice, capacity=4):
        import queue
        self.voice = voice
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.cleanup_lock = threading.Lock()
        self.error = None
        self.chunks = SentenceChunks()
        if hasattr(voice.playback, "begin_speech"):
            voice.playback.begin_speech()
        self.thread = threading.Thread(target=self._run, name="mica-speech")
        self.thread.start()

    def _run(self):
        import queue
        try:
            while not self.stop.is_set():
                try:
                    text = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if text is None:
                    return
                audio_chunks = self.voice.speech.synthesize(text)
                try:
                    for audio, rate in audio_chunks:
                        if self.stop.is_set():
                            return
                        self.voice.playback.play(audio, rate)
                finally:
                    if hasattr(audio_chunks, "close"):
                        audio_chunks.close()
        except BaseException as e:
            if not self.stop.is_set():
                self.error = e
                self.stop.set()
                self._close()

    def _close(self):
        with self.cleanup_lock:
            self.voice.close()

    def _submit(self, text):
        import queue
        while not self.stop.is_set():
            try:
                self.queue.put(text, timeout=0.05)
                return
            except queue.Full:
                pass  # Bounded backpressure and interruptible waits.

    def feed(self, piece):
        for text in self.chunks.feed(piece):
            self._submit(text)

    def finish(self):
        for text in self.chunks.finish():
            self._submit(text)
        self._submit(None)
        while self.thread.is_alive():
            self.thread.join(timeout=0.05)
        if self.error is not None:
            raise self.error

    def cancel(self):
        self.stop.set()
        self.chunks.pending = ""
        # Abort device buffers and kill uninterruptible inference before joining.
        self._close()
        while self.thread.is_alive():
            self.thread.join(timeout=0.05)
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
