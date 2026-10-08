"""Run only after explicitly downloading the 3B preset. No network is allowed."""
import gc
import json
from pathlib import Path
import signal
import socket
import weakref

from mica.config import Settings, atomic_json, offline_environment

offline_environment()


def forbid_network(*args, **kwargs):
    raise AssertionError("Offline smoke test attempted a network connection")


socket.socket.connect = forbid_network
socket.create_connection = forbid_network

from mica.conversation import Conversation
from mica.inference import Engine
from mica.models import resolve


def main():
    settings = Settings(max_tokens=48, temperature=0, top_p=1, system="Answer briefly and accurately.")
    spec = resolve("mlx-community/Llama-3.2-3B-Instruct-4bit", offline=True)
    engine = Engine()
    result = {"model": spec.identity, "revision": spec.revision, "network_connections_allowed": False, "turns": []}
    try:
        engine.load(spec, settings)
        state = Conversation(spec.identity, settings)
        for text in ("My name is Mira. Say hello using my name.", "What name did I tell you?"):
            tokens, trimmed = state.prepare(text, engine.tokenizer, spec.context)
            if trimmed:
                engine.invalidate()
            chunks = list(engine.stream(tokens, settings))
            answer = "".join(chunks)
            assert answer.strip(), "No streaming text produced"
            state.finish(answer)
            result["turns"].append({"input": text, "output": answer, "chunks": len(chunks), **engine.stats})
        assert engine.stats["cached_prompt_tokens"] > 0
        assert "mira" in result["turns"][1]["output"].lower()
        tokens, _ = state.prepare("List many interesting facts about space.", engine.tokenizer, spec.context)
        stream = engine.stream(tokens, settings)
        partial = ""
        try:
            partial = next(stream)
            signal.raise_signal(signal.SIGINT)
        except KeyboardInterrupt:
            pass
        finally:
            stream.close()
        state.finish(partial, incomplete=True)
        assert engine.cache is None
        session_path = Path(".mica-test/interrupted.json")
        state.save(session_path)
        restored = Conversation.load(session_path)
        assert restored.messages[-1]["incomplete"] is True
        tokens, trimmed = restored.prepare("Reply with the word ready.", engine.tokenizer, spec.context)
        assert trimmed
        assert all(not m.get("incomplete") for m in restored.messages)
        engine.invalidate()
        answer = "".join(engine.stream(tokens, settings))
        assert answer.strip()
        restored.finish(answer)
        result["cancellation_and_recovery"] = "passed: SIGINT, incomplete session roundtrip, discarded pair, regenerated"
        settings.system = "Answer in one sentence."
        engine.invalidate()
        assert not engine.cached_tokens
        old_model = weakref.ref(engine.model)
        engine.load(spec, settings)
        gc.collect()
        assert old_model() is None, "Previous model still referenced after switch"
        assert engine.cache is None
        result["model_release_and_reload"] = "passed: previous model collected, caches cleared, same 3B artifacts reloaded"
        result["different_repository_switch"] = "not run: 7B weights were not downloaded"
        tokens, _ = Conversation(spec.identity, settings).prepare("Say hello.", engine.tokenizer, spec.context)
        assert "".join(engine.stream(tokens, settings)).strip()
        result["offline_generation_after_reload"] = "passed"
        atomic_json(Path("verification/hardware-smoke.json"), result)
        print(json.dumps(result, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
