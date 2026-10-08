"""Bounded local speech operations; no microphone, speaker or temporary files."""
from contextlib import contextmanager, closing
from io import BytesIO
from threading import Event, Lock
import math
import wave

from ..config import ChatError
from .client import MicaBusyError

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_SPEECH_SECONDS = 300
MAX_TEXT_CHARS = 10000


class AudioCancelledError(ChatError):
    pass


def decode_wav(data, max_duration):
    import numpy as np
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            if width != 2 or channels not in (1, 2) or not 8000 <= rate <= 48000:
                raise ChatError("Upload PCM16 WAV with one/two channels at 8–48 kHz.")
            if not frames or frames / rate > max_duration:
                raise ChatError(f"Recording must be nonempty and at most {max_duration:g} seconds.")
            raw = wav.readframes(frames)
            if len(raw) != frames * channels * width:
                raise ChatError("Truncated WAV audio.")
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
        if rate != 16000:
            from scipy.signal import resample_poly
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        return audio
    except (wave.Error, EOFError, ValueError) as e:
        raise ChatError(f"Invalid WAV: {e}") from e


def pcm16(audio):
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all():
        raise ChatError("Speech synthesis returned invalid audio samples.")
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


class AudioService:
    def __init__(self, settings=None, offline=True, factory=None):
        from ..voice import VoiceSettings
        self.settings = VoiceSettings(**(settings or {})).validate()
        self.offline, self.factory = offline, factory
        self.speech = None
        self.lock, self.active, self.cancelled = Lock(), Event(), Event()
        self.closed = False

    def _check(self):
        if self.cancelled.is_set():
            raise AudioCancelledError("Speech operation cancelled.")

    def _discard(self, graceful=False):
        if self.speech is not None:
            if graceful and hasattr(self.speech, "shutdown"):
                self.speech.shutdown()
            else:
                self.speech.close()
            self.speech = None

    @contextmanager
    def _operation(self):
        if not self.lock.acquire(blocking=False):
            raise MicaBusyError("Speech is busy. Retry after the active audio request finishes.")
        try:
            if self.closed:
                raise ChatError("Audio service is closed.")
            self.cancelled.clear()
            self.active.set()
            if self.speech is None:
                if self.factory is not None:
                    self.speech = self.factory()
                else:
                    from ..voice import LocalSpeech, resolve_assets
                    paths = resolve_assets(self.settings, self.offline)
                    self._check()
                    self.speech = LocalSpeech(self.settings, paths, cancel_event=self.cancelled)
            self._check()
            yield self.speech
            self._check()
        except BaseException as e:
            self._discard()
            if self.cancelled.is_set() and isinstance(e, Exception):
                raise AudioCancelledError("Speech operation cancelled.") from e
            raise
        finally:
            self.active.clear()
            self.lock.release()

    def transcribe(self, data):
        audio = decode_wav(data, self.settings.max_duration)
        import numpy as np
        if float(np.max(np.abs(audio))) < 0.001:
            return ""
        with self._operation() as speech:
            return speech.transcribe(audio).strip()

    def synthesize(self, text):
        from ..voice import speech_text
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise ChatError(f"text must be nonempty and at most {MAX_TEXT_CHARS} characters.")
        spoken = speech_text(text)
        if not spoken:
            raise ChatError("Text contains no speakable content.")
        with self._operation() as speech:
            stream = speech.synthesize(spoken)
            total, sample_rate = 0, None
            try:
                for audio, rate in stream:
                    self._check()
                    if type(rate) is not int or not 8000 <= rate <= 48000 or sample_rate not in (None, rate):
                        raise ChatError("Synthesis returned inconsistent/unsupported sample rates.")
                    sample_rate = rate
                    raw = pcm16(audio)
                    total += len(raw)
                    if total > MAX_AUDIO_BYTES or total / (2 * rate) > MAX_SPEECH_SECONDS:
                        raise ChatError("Synthesized audio exceeds the output limit; shorten text.")
                    # Bound individual HTTP events even when Kokoro returns long chunks.
                    for start in range(0, len(raw), 32768):
                        self._check()
                        yield raw[start:start+32768], rate
                if sample_rate is None:
                    raise ChatError("Synthesis produced no audio.")
            finally:
                stream.close()

    def wav(self, text):
        buffer = BytesIO()
        with closing(self.synthesize(text)) as stream:
            first, rate = next(stream)
            with wave.open(buffer, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.writeframesraw(first)
                for raw, _ in stream:
                    output.writeframesraw(raw)
        return buffer.getvalue()

    def cancel(self):
        if not self.active.is_set():
            return False
        self.cancelled.set()
        return True

    def close(self):
        self.cancel()
        with self.lock:
            self.closed = True
            self._discard(graceful=True)
