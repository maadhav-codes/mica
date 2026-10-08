"""Project-local, self-contained model snapshots and packaged speech resources."""
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .config import ChatError, atomic_json


def asset_root():
    return Path(os.environ.get("MICA_ASSETS_DIR", "assets")).expanduser().resolve()


def model_path(identity):
    # Reject path traversal; repository IDs retain their owner/name hierarchy.
    parts = identity.split("/")
    if len(parts) != 2 or any(not part or part in (".", "..") or "\\" in part for part in parts):
        raise ChatError(f"Invalid repository ID: {identity}")
    return asset_root() / "models" / parts[0] / parts[1]


def copy_asset(source, destination):
    """Replace even read-only cache copies without leaving partial weight files."""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".mica-asset-")
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.chmod(temporary, os.stat(temporary).st_mode | stat.S_IWUSR)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


def stage_model(identity, source, revision=None):
    target = model_path(identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    if source != target.resolve():
        # Dereference Hub symlinks so assets work without the global Hub cache.
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False,
                        copy_function=copy_asset, ignore=shutil.ignore_patterns(".cache"))
    atomic_json(target / "mica-asset.json", {"repository": identity, "revision": revision})
    return target


def stage_speech_resources():
    runtime = asset_root() / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    packages = ("en_core_web_sm", "misaki", "espeakng_loader", "mlx_whisper")
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ChatError(f"Missing speech resource {name}. Run uv sync --locked --python 3.12 --extra voice online.")
        source = Path(next(iter(spec.submodule_search_locations)))
        shutil.copytree(source, runtime / name, dirs_exist_ok=True,
                        copy_function=copy_asset,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    atomic_json(runtime / "mica-resources.json", {"packages": list(packages), "python": "3.12"})
    return runtime
