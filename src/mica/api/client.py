from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event, Lock
from typing import Iterator
from uuid import uuid4

from ..config import ChatError, Settings, check_platform, offline_environment
from ..conversation import Conversation
from ..inference import Engine
from ..models import resolve

_MODEL_SLOT = Lock()


class MicaBusyError(ChatError):
    """Another operation/stream owns the single inference engine."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    session_id: str | None
    stats: dict
    incomplete: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Session:
    """An isolated conversation; mutate it through its owning Mica instance."""

    def __init__(self, state: Conversation, owner: object):
        self.id = str(uuid4())
        self._state = state
        self._owner = owner

    @property
    def messages(self) -> list[dict]:
        return deepcopy(self._state.messages)

    @property
    def settings(self) -> Settings:
        return replace(self._state.settings)


class Mica:
    """One native MLX model, with stateless generation and isolated chat sessions.

    Offline by default. No terminal I/O or CLI configuration changes. Consume a
    stream fully or close it (use contextlib.closing) before another operation.
    Concurrent operations fail with MicaBusyError instead of duplicating models.
    """

    def __init__(self, model: str, *, offline: bool = True, settings: Settings | None = None):
        if not isinstance(model, str) or not model.strip():
            raise ChatError("Provide a repository ID or local model directory.")
        if type(offline) is not bool:
            raise ChatError("offline must be a boolean.")
        self._settings = replace(settings or Settings()).validate()
        self._offline = offline
        self._lock = Lock()
        self._cancel = Event()
        self._generating = Event()
        self._closed = False
        self._owner = object()
        self._cache_session = None
        if not _MODEL_SLOT.acquire(blocking=False):
            raise MicaBusyError("A Mica instance already owns this process's model. Reuse it or close it first.")
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mica-inference")
        try:
            if offline:
                offline_environment()
            check_platform()
            self._engine = Engine()
            self._call(self._load_model, model)
        except BaseException:
            try:
                try:
                    if hasattr(self, "_engine"):
                        # A caller interrupt can leave the queued load still running.
                        self._call(lambda: self._engine.close() if self._engine.model is not None else None)
                finally:
                    self._worker.shutdown(wait=True)
            finally:
                _MODEL_SLOT.release()
            raise

    def _call(self, function, *args):
        # MLX graphs and streams stay on the same thread from load to cleanup.
        return self._worker.submit(function, *args).result()

    def _load_model(self, model, switching=False):
        spec = resolve(model, offline=self._offline, announce=lambda _: None)
        try:
            self._engine.load(spec, self._settings)
        finally:
            if switching:
                self._owner = object()
                self._cache_session = None

    def _invalidate(self):
        self._call(self._engine.invalidate)

    def _worker_stream(self, tokens, settings):
        stream = self._engine.stream(tokens, settings)
        try:
            while True:
                try:
                    piece = self._call(next, stream)
                except StopIteration:
                    return
                yield piece
        finally:
            self._call(stream.close)

    @property
    def model(self) -> str:
        return self._engine.spec.identity if self._engine.spec else ""

    @property
    def stats(self) -> dict:
        return deepcopy(self._engine.stats)

    @contextmanager
    def _operation(self):
        if not self._lock.acquire(blocking=False):
            raise MicaBusyError("Mica is busy. Finish/close the active stream or cancel it before another operation.")
        try:
            if self._closed:
                raise ChatError("Mica is closed. Create a new instance.")
            if self._engine.model is None:
                raise ChatError("No model is loaded. Close this instance and create another.")
            yield
        finally:
            self._lock.release()

    def _session(self, session):
        if not isinstance(session, Session) or session._owner is not self._owner:
            raise ChatError("Session belongs to another Mica instance or a previous model. Create/load a session for this model.")
        return session._state

    def new_session(self, *, system: str | None = None) -> Session:
        with self._operation():
            settings = replace(self._settings)
            if system is not None:
                settings.system = system
            settings.validate()
            return Session(Conversation(self.model, settings), self._owner)

    def stream(self, prompt: str, *, session: Session | None = None, options: dict | None = None) -> Iterator[str]:
        """Yield text deltas. Closing/cancelling saves a partial turn as incomplete."""
        return self._stream(prompt, session=session, options=options)

    def _stream(self, prompt, *, session=None, options=None, receipt=None):
        with self._operation():
            if not isinstance(prompt, str) or not prompt.strip():
                raise ChatError("prompt must be a nonempty string.")
            state = self._session(session) if session is not None else Conversation(self.model, replace(self._settings))
            settings = replace(state.settings)
            if options is not None:
                allowed = {"temperature", "top_p", "seed", "context_budget", "max_tokens"}
                if not isinstance(options, dict) or set(options) - allowed:
                    raise ChatError(f"options must contain only: {', '.join(sorted(allowed))}.")
                for key, value in options.items():
                    setattr(settings, key, value)
            settings.validate()
            # Budget using per-request settings without persisting these overrides.
            previous_settings = state.settings
            state.settings = settings
            try:
                tokens, trimmed = state.prepare(prompt, self._engine.tokenizer, self._engine.spec.context)
            finally:
                state.settings = previous_settings
            key = session.id if session else None
            if trimmed or session is None or key != self._cache_session:
                self._invalidate()
            self._cancel.clear()
            self._generating.set()
            output = []
            complete = False
            stream = self._worker_stream(tokens, settings)
            try:
                for piece in stream:
                    if self._cancel.is_set():
                        break
                    output.append(piece)
                    yield piece
                else:
                    complete = True
            finally:
                try:
                    stream.close()
                finally:
                    state.finish("".join(output), incomplete=not complete)
                    if not complete or session is None:
                        self._invalidate()
                    self._cache_session = key if complete and session is not None else None
                    self._generating.clear()
                    if receipt is not None:
                        receipt.update(model=self.model, session_id=key, stats=self.stats, incomplete=not complete)

    def generate(self, prompt: str, *, session: Session | None = None, options: dict | None = None) -> GenerationResult:
        """Return text and measurements. Cancellation returns an incomplete result."""
        receipt = {}
        stream = self._stream(prompt, session=session, options=options, receipt=receipt)
        try:
            text = "".join(stream)
        finally:
            stream.close()
        return GenerationResult(text=text, **receipt)

    def session_info(self, session: Session) -> dict:
        with self._operation():
            state = self._session(session)
            return {"id": session.id, "model": state.model, "settings": asdict(state.settings), "messages": session.messages}

    def cancel(self) -> bool:
        """Request cooperative cancellation from another thread; observed at a token boundary."""
        if not self._generating.is_set():
            return False
        self._cancel.set()
        return True

    def reset(self, session: Session):
        with self._operation():
            self._session(session).messages.clear()
            self._invalidate()
            self._cache_session = None

    def set_system(self, session: Session, instructions: str):
        with self._operation():
            state = self._session(session)
            settings = replace(state.settings, system=instructions).validate()
            state.settings = settings
            self._invalidate()
            self._cache_session = None

    def save_session(self, session: Session, path: str | Path):
        with self._operation():
            self._session(session).save(Path(path).expanduser())

    def load_session(self, path: str | Path) -> Session:
        with self._operation():
            state = Conversation.load(Path(path).expanduser())
            if state.model != self.model:
                raise ChatError("Saved session uses a different model. Open Mica with that model before loading it.")
            self._invalidate()
            self._cache_session = None
            return Session(state, self._owner)

    def switch_model(self, model: str):
        with self._operation():
            self._call(self._load_model, model, True)

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise MicaBusyError("Cannot close during generation. Cancel, then finish/close the active stream.")
        try:
            if not self._closed:
                try:
                    self._call(self._engine.close)
                finally:
                    try:
                        self._worker.shutdown(wait=True)
                    finally:
                        self._closed = True
                        _MODEL_SLOT.release()
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
