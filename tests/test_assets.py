from pathlib import Path
from types import SimpleNamespace

import pytest

from mica.assets import asset_root, model_path, stage_model, stage_speech_resources
from mica.config import ChatError


def test_stage_snapshot_is_independent_of_source_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    source = tmp_path / "snapshot"
    source.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    (source / "weights.safetensors").symlink_to(blob)
    target = stage_model("org/model", source, "revision")
    assert not (target / "weights.safetensors").is_symlink()
    blob.unlink()
    assert (target / "weights.safetensors").read_bytes() == b"weights"
    assert (target / "mica-asset.json").is_file()


def test_offline_speech_prefers_project_assets(tmp_path, monkeypatch):
    import huggingface_hub
    from mica.voice import asset_path
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    target = model_path("org/model")
    target.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("Hub lookup"))
    assert asset_path("org/model", True, prepare=True) == target


def test_language_model_prefers_project_assets(tmp_path, monkeypatch):
    import mica.models as models
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    target = model_path("org/model")
    target.mkdir(parents=True)
    monkeypatch.setattr(models, "validate_local", lambda path, identity, revision=None: (path, identity))
    monkeypatch.setattr(models, "inspect_remote", lambda *a: pytest.fail("Network lookup"))
    assert models.resolve("org/model", offline=True) == (target, "org/model")


def test_stage_packaged_speech_assets(tmp_path, monkeypatch):
    import mica.assets as assets
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    package = tmp_path / "package"
    package.mkdir()
    (package / "dictionary.json").write_text("{}")
    monkeypatch.setattr(assets.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(package)]))
    runtime = stage_speech_resources()
    for name in ("en_core_web_sm", "misaki", "espeakng_loader", "mlx_whisper"):
        assert (runtime / name / "dictionary.json").is_file()
    assert (runtime / "mica-resources.json").is_file()


@pytest.mark.parametrize("identity", ["../model", "org/..", "org/a/b", "org\\evil/model"])
def test_repository_path_cannot_escape_assets(identity):
    with pytest.raises(ChatError):
        model_path(identity)


def test_staged_language_model_retains_revision(tmp_path, monkeypatch):
    import mica.models as models
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    source = tmp_path / "source"
    source.mkdir()
    target = stage_model("org/model", source, "pinned-sha")
    monkeypatch.setattr(models, "validate_local", lambda path, identity, revision: (path, revision))
    assert models.resolve("org/model", offline=True) == (target, "pinned-sha")


def test_online_resolver_can_recover_interrupted_staging(tmp_path, monkeypatch):
    import huggingface_hub
    import mica.models as models
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("MICA_HOME", str(tmp_path / "settings"))
    target = model_path("org/model")
    target.mkdir(parents=True)
    def validate(path, identity, revision=None):
        if path == target:
            raise ChatError("Interrupted staging")
        return "cached-model"
    monkeypatch.setattr(models, "validate_local", validate)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(tmp_path / "cached"))
    assert models.resolve("org/model") == "cached-model"
    with pytest.raises(ChatError, match="Interrupted"):
        models.resolve("org/model", offline=True)


def test_restage_read_only_cache_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MICA_ASSETS_DIR", str(tmp_path / "assets"))
    source = tmp_path / "source"
    source.mkdir()
    weights = source / "weights.npz"
    weights.write_bytes(b"weights")
    weights.chmod(0o444)
    target = stage_model("org/model", source, "revision")
    destination = target / "weights.npz"
    assert destination.stat().st_mode & 0o200
    # Reproduce files staged by the prior copytree implementation.
    destination.chmod(0o444)
    stage_model("org/model", source, "revision")
    assert destination.read_bytes() == b"weights"
    assert destination.stat().st_mode & 0o200
    assert not weights.stat().st_mode & 0o200
    assert not list(target.glob(".mica-asset-*"))
