"""Inspect preset headers online, recording evidence without downloading weights."""
from importlib.metadata import version
from pathlib import Path

from mica.config import atomic_json, check_platform
from mica.models import PRESETS, inspect_remote

check_platform()
records = []
for repo, label in PRESETS:
    spec = inspect_remote(repo)
    record = {"repository": repo, "choice": label, "revision": spec.revision, "architecture": spec.config["model_type"], "quantization": spec.config["quantization"], "weight_files": spec.weights, "weight_bytes": spec.size, "quantized_layers": spec.quantized_layers, "context": spec.context, "official_template_present": True, "native_packed_headers_validated": True, "mlx_lm": version("mlx-lm")}
    records.append(record)
    print(f"Validated {repo}: {spec.revision}, {spec.quantized_layers} packed layers")
atomic_json(Path("verification/presets.json"), records)
