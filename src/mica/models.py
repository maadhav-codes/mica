"""Validate metadata and safetensors headers without loading tensor payloads."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
from urllib.request import Request, urlopen

from .config import ChatError, atomic_json, home

GGUF_ERROR = "GGUF models are not supported. This tool requires native MLX 4-bit model artifacts, such as compatible mlx-community repositories."
PRESETS = (
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Speed-oriented 3B"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", "Quality-oriented 7B"),
)
PATTERNS = ["config.json", "model*.safetensors", "model.safetensors.index.json", "tokenizer*", "special_tokens_map.json", "chat_template*"]


def reject_name(name):
    lower = str(name).lower()
    if "gguf" in lower:
        raise ChatError(GGUF_ERROR)
    if any(s in lower for s in ("q4_k_m", "q4_0", "gptq", "awq")):
        raise ChatError("Only native MLX 4-bit artifacts are accepted; GGUF/GPTQ/AWQ and llama.cpp quantizations are unsupported.")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ChatError(f"Missing or invalid {path.name}: {e}") from e


def validate_config(config, tokenizer, names):
    for name in names:
        reject_name(name)
    if config.get("model_file") or config.get("auto_map"):
        raise ChatError("Custom architecture code is unsupported; use a built-in MLX-LM architecture.")
    q = config.get("quantization")
    if not isinstance(q, dict) or q.get("bits") != 4 or type(q.get("group_size")) is not int or q["group_size"] not in (32, 64, 128):
        raise ChatError("Requires native MLX quantization metadata with bits=4 and group_size=32, 64, or 128. No conversion is performed.")
    for meta in (q, config.get("quantization_config", {})):
        if not isinstance(meta, dict) or meta.get("quant_method") or meta.get("mode", "affine") != "affine":
            raise ChatError("Only native MLX affine 4-bit quantization is accepted; GPTQ/AWQ/other formats are rejected.")
        if "bits" in meta and meta["bits"] != 4:
            raise ChatError("Mixed precision quantization is unsupported; every quantized layer must be 4-bit.")
    for name, meta in q.items():
        if name in ("bits", "group_size", "mode"):
            continue
        if not isinstance(meta, dict) or meta.get("bits") != 4 or meta.get("mode", "affine") != "affine":
            raise ChatError(f"Layer {name} is not native MLX affine 4-bit.")
        if meta.get("group_size") not in (32, 64, 128):
            raise ChatError(f"Invalid group size for {name}.")
    if not isinstance(config.get("model_type"), str) or not config["model_type"].replace("_", "").isalnum():
        raise ChatError("Missing or invalid model architecture.")
    from mlx_lm.utils import _get_classes
    try:
        _, args = _get_classes(config)
        args.from_dict(config)
    except (ValueError, TypeError, KeyError, ImportError) as e:
        raise ChatError(f"Unsupported MLX-LM architecture/configuration: {e}") from e
    context = config.get("max_position_embeddings", config.get("max_seq_len"))
    if type(context) is not int or context <= 0:
        raise ChatError("Model must declare its supported context length.")
    if not tokenizer.get("chat_template") and not any(n.startswith("chat_template") for n in names):
        raise ChatError("Missing official tokenizer chat template; generic templates are not substituted.")
    if tokenizer.get("auto_map"):
        raise ChatError("Tokenizer remote code is disabled; use a self-contained tokenizer.")
    if "tokenizer.json" not in names and "tokenizer.model" not in names:
        raise ChatError("Missing tokenizer.json or tokenizer.model.")
    weights = [n for n in names if Path(n).name.startswith("model") and n.endswith(".safetensors") and "/" not in n]
    if not weights:
        raise ChatError("Missing native model*.safetensors weights.")
    return context, sorted(weights)


def decode_header(reader):
    raw = reader.read(8)
    if raw[:4] == b"GGUF":
        raise ChatError(GGUF_ERROR)
    if len(raw) != 8:
        raise ChatError("Interrupted or invalid safetensors file.")
    size = struct.unpack("<Q", raw)[0]
    if not 2 <= size <= 16 * 1024**2:
        raise ChatError("Invalid safetensors header size.")
    try:
        raw_header = reader.read(size)
        if len(raw_header) != size:
            raise ValueError("truncated header")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError("not a map")
        return header
    except (ValueError, UnicodeError) as e:
        raise ChatError(f"Invalid safetensors header: {e}") from e


def validate_tensors(tensors, config):
    q = config["quantization"]
    count = 0
    for name, t in tensors.items():
        if name == "__metadata__":
            continue
        if not isinstance(t, dict) or not isinstance(t.get("shape"), list):
            raise ChatError(f"Invalid tensor metadata for {name}.")
        if name.endswith((".qweight", ".qzeros", ".g_idx")):
            raise ChatError("GPTQ/AWQ packed tensors are unsupported.")
        if name.endswith(".weight") and len(t["shape"]) >= 2:
            prefix = name[:-7]
            scales, biases = tensors.get(prefix + ".scales"), tensors.get(prefix + ".biases")
            meta = q.get(prefix, q)
            group = meta.get("group_size", q["group_size"])
            if t.get("dtype") != "U32" or not scales or not biases:
                raise ChatError(f"{name} is not native MLX packed uint32 4-bit with scales/biases.")
            shape = t["shape"]
            if scales.get("dtype") not in ("F16", "BF16", "F32") or scales.get("shape") != biases.get("shape") or biases.get("dtype") != scales.get("dtype"):
                raise ChatError(f"Invalid native affine scales/biases for {name}.")
            s = scales["shape"]
            if len(s) != len(shape) or shape[:-1] != s[:-1] or shape[-1] * 8 != s[-1] * group:
                raise ChatError(f"Packed shape for {name} is inconsistent with native 4-bit group size {group}.")
            count += 1
        elif t.get("dtype") == "U32":
            raise ChatError(f"Unexpected packed tensor {name}.")
    if not count:
        raise ChatError("No native MLX quantized weight layers found.")
    return count


def validate_index(index, tensors, shard_names):
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or set(mapping) != set(tensors):
        raise ChatError("Weight index does not match tensor headers; missing or unexpected tensors/shards.")
    if not set(mapping.values()).issubset(set(shard_names)):
        raise ChatError("Weight index refers to missing shards.")


@dataclass
class ModelSpec:
    identity: str
    path: Path
    revision: str | None
    config: dict
    context: int
    weights: list[str]
    size: int
    quantized_layers: int


def validate_local(path: Path, identity=None, revision=None):
    if not path.is_dir():
        raise ChatError(f"Model directory does not exist: {path}")
    names = [p.name for p in path.iterdir()]
    config = read_json(path / "config.json")
    tokenizer = read_json(path / "tokenizer_config.json")
    context, weights = validate_config(config, tokenizer, names)
    tensors = {}
    size = 0
    for name in weights:
        with (path / name).open("rb") as f:
            header = decode_header(f)
            start = f.tell()
        actual = (path / name).stat().st_size
        for key, t in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ChatError(f"Duplicate tensor {key} across shards.")
            offsets = t.get("data_offsets", [])
            if len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= actual - start:
                raise ChatError(f"Interrupted/truncated weight file {name}; rerun download.")
            tensors[key] = t
        size += actual
    layers = validate_tensors(tensors, config)
    if (path / "model.safetensors.index.json").exists():
        validate_index(read_json(path / "model.safetensors.index.json"), tensors, weights)
    return ModelSpec(identity or str(path.resolve()), path, revision, config, context, weights, size, layers)


def remote_header(repo, revision, name):
    from huggingface_hub import get_token, hf_hub_url
    headers = {"Range": "bytes=0-16777223"}
    if token := get_token():
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(hf_hub_url(repo, name, revision=revision), headers=headers), timeout=60) as response:
        return decode_header(response)


def inspect_remote(repo):
    reject_name(repo)
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(repo, files_metadata=True)
    names = [f.rfilename for f in info.siblings]
    revision = info.sha
    path = Path(hf_hub_download(repo, "config.json", revision=revision)).parent
    hf_hub_download(repo, "tokenizer_config.json", revision=revision)
    config, tokenizer = read_json(path / "config.json"), read_json(path / "tokenizer_config.json")
    context, weights = validate_config(config, tokenizer, names)
    tensors = {}
    for name in weights:
        header = remote_header(repo, revision, name)
        for key, t in header.items():
            if key != "__metadata__":
                if key in tensors:
                    raise ChatError(f"Duplicate tensor {key} across shards.")
                tensors[key] = t
    layers = validate_tensors(tensors, config)
    if "model.safetensors.index.json" in names:
        index_path = Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision))
        validate_index(read_json(index_path), tensors, weights)
    size = sum(f.size or 0 for f in info.siblings if f.rfilename in weights)
    if size <= 0:
        raise ChatError("Repository does not expose weight sizes; cannot estimate download space.")
    return ModelSpec(repo, path, revision, config, context, weights, size, layers)


def resolve(model: str, offline=False, announce=print):
    reject_name(model)
    path = Path(model).expanduser()
    if path.is_file():
        with path.open("rb") as f:
            if f.read(4) == b"GGUF":
                raise ChatError(GGUF_ERROR)
        raise ChatError("Select a model directory containing config, tokenizer, and native MLX safetensors; individual weight files are unsupported.")
    if path.exists() or model.startswith(("/", ".", "~")):
        return validate_local(path)
    from .assets import model_path
    staged = model_path(model)
    if staged.is_dir():
        try:
            provenance = staged / "mica-asset.json"
            revision = read_json(provenance).get("revision") if provenance.exists() else None
            return validate_local(staged, model, revision)
        except (ChatError, OSError, ValueError):
            if offline:
                raise
            # A cancelled staging copy can be repaired by rerunning download.
    from huggingface_hub import snapshot_download
    registry_path = home() / "models.json"
    registry = read_json(registry_path) if registry_path.exists() else {}
    if not isinstance(registry, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in registry.items()):
        raise ChatError(f"Invalid model revision registry: {registry_path}")
    # Use a complete cached snapshot without any metadata request.
    try:
        cached = Path(snapshot_download(model, revision=registry.get(model, "main"), local_files_only=True, allow_patterns=PATTERNS))
        return validate_local(cached, model, cached.name)
    except Exception as e:
        if offline:
            raise ChatError(f"Offline model unavailable or invalid: {e}. Download it online first.") from e
    spec = inspect_remote(model)
    spec.path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(spec.path).free
    announce(f"Estimated weights: {spec.size / 1024**3:.2f} GiB; free disk: {free / 1024**3:.2f} GiB. Tokenizer/cache overhead is additional.")
    if free < spec.size + 512 * 1024**2:
        raise ChatError("Insufficient disk space; free at least the estimated weights plus 512 MiB and retry.")
    cached = Path(snapshot_download(model, revision=spec.revision, allow_patterns=PATTERNS))
    validated = validate_local(cached, model, spec.revision)
    registry[model] = spec.revision
    atomic_json(registry_path, registry)
    return validated
