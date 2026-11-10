"""Resolve catalog entries to local model files."""
from __future__ import annotations

from pathlib import Path

import pytest

from hnt.catalog import CatalogEntry, resolve_model


def _entry(tmp_path: Path, sha: str | None = None) -> CatalogEntry:
    data = {
        "id": "test_model_2x",
        "name": "TestModel 2x",
        "family": "HEART",
        "arch": "heart",
        "scale": 2,
        "params": 1,
        "license": "MIT",
        "formats": ["safetensors"],
        "download_url": "https://huggingface.co/Phips/HEART/resolve/main/models/heart_2x.safetensors",
        "vram_hint": "low",
        "description": "test",
    }
    if sha is not None:
        data["sha256"] = sha
    return CatalogEntry.from_dict(data)


def test_resolve_model_downloads_into_cache_dir(monkeypatch, tmp_path: Path):
    fake = tmp_path / "m.safetensors"
    fake.write_bytes(b"weights")
    calls = {}

    def fake_download(repo_id, filename, local_dir):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["local_dir"] = local_dir
        return fake

    monkeypatch.setattr("hnt.catalog.hf_hub_download", fake_download)
    cache = tmp_path / "models"
    p = resolve_model(_entry(tmp_path), cache)
    assert p == fake
    assert calls["repo_id"] == "Phips/HEART"
    assert calls["filename"] == "models/heart_2x.safetensors"
    assert cache.is_dir()


def test_resolve_model_verifies_sha256(monkeypatch, tmp_path: Path):
    import hashlib

    fake = tmp_path / "m.safetensors"
    fake.write_bytes(b"weights")
    monkeypatch.setattr("hnt.catalog.hf_hub_download", lambda *a, **k: fake)
    sha = hashlib.sha256(b"weights").hexdigest()
    p = resolve_model(_entry(tmp_path, sha), tmp_path)
    assert p == fake

    monkeypatch.setattr("hnt.catalog.hf_hub_download", lambda *a, **k: fake)
    with pytest.raises(ValueError):
        resolve_model(_entry(tmp_path, "0" * 64), tmp_path)
