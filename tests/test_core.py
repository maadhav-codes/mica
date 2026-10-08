from dataclasses import asdict
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mica.config import ChatError, Settings, check_platform
from mica.conversation import Conversation
from mica.inference import Engine, common_prefix
from mica.models import GGUF_ERROR, decode_header, reject_name, validate_config, validate_index, validate_local, validate_tensors


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # Role and framing tokens are included; tests don't count plain text alone.
        self.messages = messages
        return list(range(2 + sum(3 + len(m["content"]) for m in messages)))


@pytest.mark.parametrize("name", ["thing.gguf", "repo/GGUF", "Q4_K_M", "Q4_0", "thing-GPTQ", "AWQ-model"])
def test_forbidden_names(name):
    with pytest.raises(ChatError) as exc:
        reject_name(name)
    if "gguf" in name.lower():
        assert str(exc.value) == GGUF_ERROR


def test_gguf_magic_rejected_even_with_misleading_filename(tmp_path):
    from mica.models import resolve
    path = tmp_path / "weights.bin"
    path.write_bytes(b"GGUF" + b"\x00" * 32)
    with pytest.raises(ChatError) as exc:
        resolve(str(path), offline=True)
    assert str(exc.value) == GGUF_ERROR


@pytest.mark.parametrize("change", [{"temperature": -1}, {"temperature": float("nan")}, {"top_p": 0}, {"seed": -1}, {"seed": True}, {"context_budget": 512}, {"max_tokens": 0}, {"memory_gb": 16}, {"system": []}])
def test_invalid_settings(change):
    with pytest.raises(ChatError):
        Settings(**change).validate()


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MICA_HOME", str(tmp_path))
    s = Settings(last_model="org/model", temperature=0.3, seed=7)
    s.save()
    assert Settings.load() == s
    assert (tmp_path / "config.json").stat().st_mode & 0o777 == 0o600
    (tmp_path / "config.json").write_text('{"max_tokens": "512"}')
    with pytest.raises(ChatError):
        Settings.load()


def test_platform_rejections(monkeypatch):
    import mica.config as config
    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    with pytest.raises(ChatError, match="macOS"):
        check_platform()
    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="1"))
    with pytest.raises(ChatError, match="Rosetta"):
        check_platform()
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="0"))
    monkeypatch.setattr(config.platform, "mac_ver", lambda: ("13.6", (), ""))
    with pytest.raises(ChatError, match="macOS 14"):
        check_platform()


@pytest.mark.parametrize("python_version", [(3, 11, 15), (3, 12, 13), (3, 13, 0)])
def test_platform_requires_python_312(monkeypatch, python_version):
    import mica.config as config
    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config.platform, "mac_ver", lambda: ("15.8.1", (), ""))
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="0"))
    monkeypatch.setattr(config.sys, "version_info", python_version)
    if python_version[:2] == (3, 12):
        check_platform()
    else:
        with pytest.raises(ChatError, match="Python 3.12"):
            check_platform()


def test_budget_counts_template_and_trims_pairs():
    s = Settings(system="S", context_budget=40, max_tokens=10)
    c = Conversation("m", s, [{"role": "user", "content": "123456"}, {"role": "assistant", "content": "abcdef"}])
    tok = Tokenizer()
    tokens, trimmed = c.prepare("hello world", tok, 100)
    assert trimmed
    assert len(tokens) == 20  # two start tokens + system framing + user framing
    assert tok.messages == [{"role": "system", "content": "S"}, {"role": "user", "content": "hello world"}]
    assert c.messages == [{"role": "user", "content": "hello world"}]


def test_rejected_input_preserves_state_and_reserves_output():
    c = Conversation("m", Settings(system="S", context_budget=40, max_tokens=10))
    c.messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    before = c.messages[:]
    with pytest.raises(ChatError, match="not truncated"):
        c.prepare("X" * 30, Tokenizer(), 100)
    assert c.messages == before
    with pytest.raises(ChatError, match="no input space"):
        c.prepare("u", Tokenizer(), 8)


def test_model_context_bounds_budget():
    c = Conversation("m", Settings(system="", context_budget=100, max_tokens=10))
    with pytest.raises(ChatError):
        c.prepare("12345678901", Tokenizer(), 20)


def test_interrupted_session_persistence(tmp_path):
    c = Conversation("org/model", Settings())
    c.messages.append({"role": "user", "content": "question"})
    c.finish("partial", incomplete=True)
    path = tmp_path / "session.json"
    c.save(path)
    restored = Conversation.load(path)
    assert restored.messages[-1]["incomplete"] is True
    assert restored.settings == c.settings
    assert restored.model == c.model
    assert restored.discard_incomplete()
    assert restored.messages == []


def test_prepare_removes_interrupted_pair():
    c = Conversation("m", Settings())
    c.messages = [{"role": "user", "content": "bad"}, {"role": "assistant", "content": "partial", "incomplete": True}]
    _, trimmed = c.prepare("new", Tokenizer(), 4096)
    assert trimmed
    assert c.messages == [{"role": "user", "content": "new"}]


@pytest.mark.parametrize("messages", [[{"role": "user", "content": "u"}], [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}], [{"role": "user", "content": "u", "incomplete": True}, {"role": "assistant", "content": "a"}], [{"role": "user", "content": "u"}, {"role": "assistant", "content": 3}]])
def test_reject_malformed_session(tmp_path, messages):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "model": "m", "settings": asdict(Settings()), "messages": messages}))
    with pytest.raises(ChatError):
        Conversation.load(path)


def native_config():
    return {"model_type": "llama", "hidden_size": 128, "num_hidden_layers": 1, "intermediate_size": 256, "num_attention_heads": 4, "rms_norm_eps": 1e-5, "vocab_size": 256, "max_position_embeddings": 4096, "quantization": {"bits": 4, "group_size": 64}}


def native_tensors():
    return {"model.layers.0.q_proj.weight": {"dtype": "U32", "shape": [128, 16]}, "model.layers.0.q_proj.scales": {"dtype": "F16", "shape": [128, 2]}, "model.layers.0.q_proj.biases": {"dtype": "F16", "shape": [128, 2]}, "model.norm.weight": {"dtype": "F16", "shape": [128]}}


def test_native_tensor_evidence():
    assert validate_tensors(native_tensors(), native_config()) == 1
    tensors = native_tensors()
    tensors["model.layers.0.q_proj.weight"]["shape"] = [128, 32]  # 8-bit packing masquerading as 4-bit
    with pytest.raises(ChatError, match="inconsistent"):
        validate_tensors(tensors, native_config())


@pytest.mark.parametrize("mutation", ["float", "no_scales", "gptq", "empty", "bad_biases"])
def test_reject_non_native_tensors(mutation):
    t = native_tensors()
    if mutation == "float":
        t["model.layers.0.q_proj.weight"]["dtype"] = "F16"
    elif mutation == "no_scales":
        del t["model.layers.0.q_proj.scales"]
    elif mutation == "gptq":
        t["model.qweight"] = {"dtype": "I32", "shape": [1, 1]}
    elif mutation == "empty":
        t = {}
    else:
        t["model.layers.0.q_proj.biases"]["shape"] = [128, 4]
    with pytest.raises(ChatError):
        validate_tensors(t, native_config())


@pytest.mark.parametrize("change", [{"quantization": {"bits": 8, "group_size": 64}}, {"quantization": None}, {"quantization_config": {"quant_method": "awq"}}, {"model_type": "not_a_real_model"}, {"model_file": "custom.py"}, {"quantization": {"bits": 4, "group_size": 64, "model.layer": {"bits": 8}}}])
def test_reject_incompatible_config(change):
    config = native_config() | change
    with pytest.raises(ChatError):
        validate_config(config, {"chat_template": "template"}, ["model.safetensors", "tokenizer.json"])


def test_valid_architecture_and_required_chat_template():
    assert validate_config(native_config(), {"chat_template": "template"}, ["model.safetensors", "tokenizer.json"])[0] == 4096
    with pytest.raises(ChatError, match="chat template"):
        validate_config(native_config(), {}, ["model.safetensors", "tokenizer.json"])


@pytest.mark.parametrize("raw", [b"GGUF1234", struct.pack("<Q", 2**63), b"", struct.pack("<Q", 10) + b"{}"])
def test_bad_headers(raw):
    with pytest.raises(ChatError):
        decode_header(io.BytesIO(raw))


def test_local_validation_detects_truncated_download(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(native_config()))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "template"}))
    (tmp_path / "tokenizer.json").write_text("{}")
    tensors = native_tensors()
    for t in tensors.values():
        t["data_offsets"] = [0, 10000]
    raw = json.dumps(tensors).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(ChatError, match="truncated"):
        validate_local(tmp_path)


def test_missing_or_inconsistent_shard_index():
    with pytest.raises(ChatError, match="missing shards"):
        validate_index({"weight_map": {"layer.weight": "model-00002.safetensors"}}, {"layer.weight": {}}, ["model-00001.safetensors"])
    with pytest.raises(ChatError, match="does not match"):
        validate_index({"weight_map": {"other.weight": "model.safetensors"}}, {"layer.weight": {}}, ["model.safetensors"])


def test_prefix_and_invalidation():
    assert common_prefix([1, 2, 3], [1, 2, 8]) == 2
    assert common_prefix([1, 2, 3], [1, 2, 3]) == 2  # always evaluate at least one token
    e = Engine()
    e.cache = object()
    e.cached_tokens = [1, 2]
    e.invalidate()
    assert e.cache is None and e.cached_tokens == []


def test_cli_validation_and_help():
    from mica.cli import app
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    rejected = runner.invoke(app, ["download", "--model", "x.gguf"])
    assert rejected.exit_code == 1
    assert GGUF_ERROR in rejected.stdout


def test_offline_blocks_remote_inspection(monkeypatch):
    import huggingface_hub
    import mica.models as models
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no cache")))
    monkeypatch.setattr(models, "inspect_remote", lambda *a: pytest.fail("Network metadata called in offline mode"))
    with pytest.raises(ChatError, match="Offline"):
        models.resolve("org/model", offline=True)


def test_offline_resolves_pinned_download_without_main_ref(tmp_path, monkeypatch):
    import huggingface_hub
    import mica.models as models
    monkeypatch.setenv("MICA_HOME", str(tmp_path))
    (tmp_path / "models.json").write_text(json.dumps({"org/model": "commit-sha"}))
    calls = []
    def cached_snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        assert kwargs["revision"] == "commit-sha"
        assert kwargs["local_files_only"] is True
        return "/cache/snapshots/commit-sha"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_snapshot)
    monkeypatch.setattr(models, "validate_local", lambda path, identity, revision: SimpleNamespace(identity=identity, revision=revision))
    monkeypatch.setattr(models, "inspect_remote", lambda *a: pytest.fail("Offline metadata lookup"))
    assert models.resolve("org/model", offline=True).revision == "commit-sha"
    assert len(calls) == 1


def test_cache_reuse_and_cancellation(monkeypatch):
    import mlx_lm
    import mlx_lm.models.cache as cache_api

    class Cache:
        offset = 0
        def is_trimmable(self):
            return True
        def trim(self, n):
            self.offset -= n
            return n

    seen = []
    def fake_stream(model, tokenizer, tokens, **kwargs):
        seen.append(tokens)
        cache = kwargs["prompt_cache"][0]
        cache.offset += len(tokens)
        yield SimpleNamespace(text="a", prompt_tps=5, generation_tokens=1, generation_tps=10, finish_reason=None)
        cache.offset += 1
        yield SimpleNamespace(text="b", prompt_tps=5, generation_tokens=2, generation_tps=10, finish_reason="length")

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream)
    monkeypatch.setattr(cache_api, "make_prompt_cache", lambda model: [Cache()])
    engine = Engine()
    assert "".join(engine.stream([1, 2, 3], Settings())) == "ab"
    assert engine.cached_tokens == [1, 2, 3]
    assert engine.cache[0].offset == 3
    list(engine.stream([1, 2, 3, 4, 5], Settings()))
    assert seen[-1] == [4, 5]
    assert engine.stats["cached_prompt_tokens"] == 3
    stream = engine.stream([1, 2, 3, 4, 5, 6], Settings())
    next(stream)
    stream.close()
    assert engine.cache is None
    assert engine.stats["finish_reason"] == "cancelled"


def test_interactive_state_changes_do_not_reuse_stale_cache(tmp_path, monkeypatch):
    import mica.cli as cli
    import prompt_toolkit
    monkeypatch.setenv("MICA_HOME", str(tmp_path / "home"))
    cached_at_generation = []
    loaded = []
    save_path = tmp_path / "session.json"
    inputs = iter(["one", "two", "/system Changed", "three", "/new", "four", f"/save {save_path}", f"/load {save_path}", "five", "/model org/next", "six", "/exit"])

    class FakeEngine:
        def __init__(self):
            self.cache = None
            self.tokenizer = Tokenizer()
            self.stats = {}
        def load(self, spec, settings):
            self.spec = spec
            self.model = spec.identity
            self.cache = None
            self.load_seconds = 0.1
            loaded.append(spec.identity)
        def invalidate(self):
            self.cache = None
        def stream(self, tokens, settings):
            cached_at_generation.append(self.cache is not None)
            self.cache = object()
            yield "answer"
        def close(self):
            self.model = self.cache = None

    monkeypatch.setattr(cli, "Engine", FakeEngine)
    monkeypatch.setattr(cli, "setup", lambda offline: None)
    monkeypatch.setattr(cli, "resolve", lambda identity, *a, **kw: SimpleNamespace(identity=identity, context=4096))
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **kw: SimpleNamespace(prompt=lambda *a: next(inputs)))
    result = CliRunner().invoke(cli.app, ["chat", "--model", "org/first", "--offline"])
    assert result.exit_code == 0, result.stdout
    assert cached_at_generation == [False, True, False, False, False, False]
    assert loaded == ["org/first", "org/first", "org/next"]
    saved = Conversation.load(save_path)
    assert saved.settings.system == "Changed"
    assert saved.messages[0]["content"] == "four"
    assert saved.model == "org/first"
    assert Settings.load().last_model == "org/next"


@pytest.mark.parametrize("python_version", [(3, 10), (3, 11), (3, 12), (3, 13)])
def test_platform_requires_python_312_only(monkeypatch, python_version):
    import mica.config as config
    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config.platform, "mac_ver", lambda: ("15.8.1", (), ""))
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="0"))
    monkeypatch.setattr(config.sys, "version_info", python_version)
    if python_version == (3, 12):
        check_platform()
    else:
        with pytest.raises(ChatError, match="Python 3.12"):
            check_platform()
