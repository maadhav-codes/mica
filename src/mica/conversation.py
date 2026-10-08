from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from .config import ChatError, Settings, atomic_json


@dataclass
class Conversation:
    model: str
    settings: Settings
    messages: list[dict] = field(default_factory=list)

    def finish(self, text: str, incomplete: bool = False):
        self.messages.append({"role": "assistant", "content": text, "incomplete": incomplete})

    def discard_incomplete(self) -> bool:
        # Remove both halves of an interrupted turn before the next generation.
        if self.messages and self.messages[-1].get("incomplete"):
            del self.messages[-2:]
            return True
        return False

    def prepare(self, text, tokenizer, model_context):
        budget = min(self.settings.context_budget, model_context)
        allowance = budget - self.settings.max_tokens
        if allowance <= 0:
            raise ChatError("Output limit leaves no input space in this model's context.")
        system = [{"role": "system", "content": self.settings.system}] if self.settings.system else []
        user = {"role": "user", "content": text}

        def encode(messages):
            try:
                return list(tokenizer.apply_chat_template(system + messages, tokenize=True, add_generation_prompt=True))
            except Exception as e:
                raise ChatError(f"Official chat template cannot format these roles/instructions: {e}") from e

        # Work on a copy: rejection must not mutate conversation or cache state.
        history = self.messages[:]
        incomplete = bool(history and history[-1].get("incomplete"))
        if incomplete:
            history = history[:-2]
        clean = [{"role": m["role"], "content": m["content"]} for m in history]
        tokens = encode(clean + [user])
        trimmed = incomplete
        while len(tokens) > allowance and history:
            history = history[2:]
            clean = clean[2:]
            trimmed = True
            tokens = encode(clean + [user])
        if len(tokens) > allowance:
            raise ChatError(f"Individual input plus system/template is {len(tokens)} tokens; only {allowance} fit. Shorten input/system or raise context budget. Input was not truncated.")
        self.messages = history + [user]
        return tokens, trimmed

    def save(self, path: Path):
        atomic_json(path, {"version": 1, "model": self.model, "settings": asdict(self.settings), "messages": self.messages})

    @classmethod
    def load(cls, path: Path):
        try:
            data = json.loads(path.read_text())
            if data["version"] != 1 or not isinstance(data["model"], str) or not data["model"]:
                raise ValueError("invalid session version/model")
            settings = Settings.from_dict(data["settings"])
            messages = data["messages"]
            if not isinstance(messages, list) or len(messages) % 2:
                raise ValueError("session must contain complete user/assistant pairs")
            for i, m in enumerate(messages):
                if not isinstance(m, dict) or m.get("role") != ("user" if i % 2 == 0 else "assistant") or not isinstance(m.get("content"), str):
                    raise ValueError("invalid message roles/content")
                if "incomplete" in m and type(m["incomplete"]) is not bool:
                    raise ValueError("incomplete flag must be boolean")
                if m.get("incomplete") and (i != len(messages) - 1 or i % 2 == 0):
                    raise ValueError("only the final assistant response may be incomplete")
            return cls(data["model"], settings, messages)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ChatError(f"Cannot load session {path}: {e}") from e
