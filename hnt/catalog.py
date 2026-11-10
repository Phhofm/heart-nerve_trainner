"""BODY Studio model catalog: bundled JSON, optional remote update, integrity."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = "https://raw.githubusercontent.com/Phhofm/sisr-body-studio/main/models.json"
BUNDLED = Path(__file__).with_name("models.json")
REQUIRED = {
    "id", "name", "family", "arch", "scale", "params", "license",
    "formats", "download_url", "vram_hint", "description",
}


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    family: str
    arch: str
    scale: int
    params: int
    license: str
    formats: tuple[str, ...]
    download_url: str
    vram_hint: str
    description: str
    sha256: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "CatalogEntry":
        missing = REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        if data["scale"] not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        if data["params"] <= 0:
            raise ValueError("params must be positive")
        if not str(data["download_url"]).startswith("https://"):
            raise ValueError("download_url must be https")
        if not data["formats"]:
            raise ValueError("at least one format required")
        if data["license"] != "MIT":
            raise ValueError("only MIT models are allowed")
        return cls(
            id=data["id"], name=data["name"], family=data["family"],
            arch=data["arch"], scale=int(data["scale"]), params=int(data["params"]),
            license=data["license"], formats=tuple(data["formats"]),
            download_url=data["download_url"], vram_hint=data["vram_hint"],
            description=data["description"],
            sha256=data.get("sha256"),
        )


def _fetch_json(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=10) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload["models"] if isinstance(payload, dict) else payload


def load_catalog(url: str | None = None) -> list[CatalogEntry]:
    """Remote catalog if reachable, else the bundled copy. Invalid remote entries fall back."""
    source = url or DEFAULT_URL
    try:
        raw = _fetch_json(source)
        return [CatalogEntry.from_dict(d) for d in raw]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raw = json.loads(BUNDLED.read_text(encoding="utf-8"))
        return [CatalogEntry.from_dict(d) for d in raw]


def verify_file(path: str | Path, sha256: str | None) -> bool | None:
    """True/False for a known hash; None when no hash is recorded."""
    if sha256 is None:
        return None
    p = Path(path)
    if not p.is_file():
        return False
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == sha256.lower()

from huggingface_hub import hf_hub_download


def resolve_model(entry: CatalogEntry, cache_dir: str | Path) -> Path:
    """Download a catalog model into ``cache_dir`` and verify its hash when present."""
    url = entry.download_url
    if not url.startswith("https://huggingface.co/") or "/resolve/main/" not in url:
        raise ValueError(f"unsupported download_url: {url}")
    repo_id, filename = url[len("https://huggingface.co/"):].split("/resolve/main/", 1)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(cache_dir)))
    if entry.sha256 and verify_file(path, entry.sha256) is not True:
        raise ValueError(f"checksum mismatch for {entry.id}")
    return path
