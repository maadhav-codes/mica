from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


class ChatError(Exception):
    """An actionable error safe to display to the user."""


def check_platform():
    if platform.system() != "Darwin":
        raise ChatError("Requires macOS Apple Silicon with native arm64 Python 3.12.")
    translated = subprocess.run(["sysctl", "-in", "sysctl.proc_translated"], capture_output=True, text=True)
    if translated.stdout.strip() == "1" or platform.machine() != "arm64":
        raise ChatError("Intel/Rosetta execution is unsupported. Open a native terminal and install arm64 Python 3.12 with uv.")
    mac_version = platform.mac_ver()[0]
    if not mac_version or int(mac_version.split(".")[0]) < 14:
        raise ChatError("The locked MLX releases require macOS 14 or later. Upgrade macOS before installing.")
    if sys.version_info[:2] != (3, 12):
        raise ChatError("Use native Python 3.12: uv sync --python 3.12")


def home() -> Path:
    override = os.environ.get("MICA_HOME") or os.environ.get("LOCALCHAT_HOME")
    if override:
        return Path(override).expanduser()
    current = Path("~/Library/Application Support/Mica").expanduser()
    legacy = Path("~/Library/Application Support/localchat").expanduser()
    # Reuse prior settings, sessions, history, and pinned model revisions in place.
    return legacy if not current.exists() and legacy.exists() else current


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".mica-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass
class Settings:
    system: str = "You are a helpful assistant."
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    context_budget: int = 4096
    max_tokens: int = 512
    memory_gb: float = 9.0
    last_model: str | None = None
    voice: dict = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.system, str) or (self.last_model is not None and not isinstance(self.last_model, str)):
            raise ChatError("System instructions and model identity must be strings.")
        for name in ("seed", "context_budget", "max_tokens"):
            if type(getattr(self, name)) is not int:
                raise ChatError(f"{name} must be an integer.")
        for name in ("temperature", "top_p", "memory_gb"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ChatError(f"{name} must be a finite number.")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ChatError("Temperature must be nonnegative and top-p must be in (0, 1].")
        if not 0 <= self.seed < 2**32:
            raise ChatError("Seed must be between 0 and 2^32-1.")
        if not 0 < self.max_tokens < self.context_budget <= 32768:
            raise ChatError("Require 0 < max_tokens < context_budget <= 32768.")
        if not 1 <= self.memory_gb <= 12:
            raise ChatError("MLX memory limit must be between 1 and 12 GiB; leave room for macOS.")
        from .voice import VoiceSettings
        if not isinstance(self.voice, dict):
            raise ChatError("voice settings must be an object.")
        try:
            VoiceSettings(**self.voice).validate()
        except TypeError as e:
            raise ChatError(f"Invalid voice settings: {e}") from e
        return self

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(**data).validate()
        except (TypeError, ValueError) as e:
            raise ChatError(f"Invalid settings: {e}") from e

    @classmethod
    def load(cls):
        path = home() / "config.json"
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError) as e:
            raise ChatError(f"Cannot read {path}: {e}") from e

    def save(self):
        self.validate()
        atomic_json(home() / "config.json", asdict(self))


def offline_environment():
    # Set before importing Hub, Transformers tokenizer utilities, or MLX-LM.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
