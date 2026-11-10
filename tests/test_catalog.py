"""BODY Studio model catalog: bundled JSON, network fallback, integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hnt.catalog import CatalogEntry, load_catalog, verify_file


EXPECTED_FAMILIES = {"PULSE_fast", "PULSE", "NERVE", "HEART"}


def test_bundled_catalog_exists_and_is_valid():
    entries = load_catalog()
    assert entries, "catalog must not be empty"
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate model ids"
    assert {e.family for e in entries} == EXPECTED_FAMILIES
    for e in entries:
        assert e.scale in (2, 4)
        assert e.params > 0
        assert e.download_url.startswith("https://")
        assert e.license == "MIT"
        assert e.formats, "at least one inference format"
        assert "safetensors" in e.formats
        assert e.vram_hint and e.description


def test_catalog_order_is_smallest_to_largest():
    entries = load_catalog()
    families = [e.family for e in entries]
    first = families.index("PULSE_fast")
    second = families.index("PULSE", first + 1)
    third = families.index("NERVE", second + 1)
    fourth = families.index("HEART", third + 1)
    assert first < second < third < fourth


def test_network_failure_falls_back_to_bundled(monkeypatch):
    def boom(url: str) -> dict:
        raise OSError("offline")

    monkeypatch.setattr("hnt.catalog._fetch_json", boom)
    entries = load_catalog()
    assert {e.family for e in entries} == EXPECTED_FAMILIES


def test_network_update_replaces_bundled(monkeypatch, tmp_path: Path):
    payload = [
        {
            "id": "test_model_2x",
            "name": "TestModel",
            "family": "HEART",
            "arch": "heart",
            "scale": 2,
            "params": 1,
            "license": "MIT",
            "formats": ["safetensors"],
            "download_url": "https://example.com/test.safetensors",
            "vram_hint": "low",
            "description": "test",
        }
    ]
    monkeypatch.setattr("hnt.catalog._fetch_json", lambda url: payload)
    entries = load_catalog()
    assert len(entries) == 1
    assert entries[0].id == "test_model_2x"


def test_invalid_remote_entry_is_rejected(monkeypatch):
    def bad(url: str) -> dict:
        return [{"id": "broken", "scale": 999, "download_url": "ftp://x", "formats": []}]

    monkeypatch.setattr("hnt.catalog._fetch_json", bad)
    entries = load_catalog()
    assert {e.family for e in entries} == EXPECTED_FAMILIES  # fell back to bundled


def test_verify_file_matches_and_rejects(tmp_path: Path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"weights")
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    assert verify_file(p, good) is True
    assert verify_file(p, "0" * 64) is False
    assert verify_file(p, None) is None  # unknown hash: not verified, not failed
    assert verify_file(tmp_path / "missing.bin", good) is False


def test_catalog_entry_from_dict_requires_all_fields():
    data = {
        "id": "pulse_fast_2x",
        "name": "PULSE_fast",
        "family": "PULSE_fast",
        "arch": "pulse_fast",
        "scale": 2,
        "params": 162192,
        "license": "MIT",
        "formats": ["safetensors", "onnx_fp16"],
        "download_url": "https://example.com/x.safetensors",
        "vram_hint": "very low",
        "description": "x",
    }
    e = CatalogEntry.from_dict(data)
    assert e.id == "pulse_fast_2x"

    with pytest.raises(ValueError):
        CatalogEntry.from_dict({**data, "download_url": "ftp://x"})


def test_cli_catalog_prints_entries():
    import io
    import sys
    from hnt.cli import build_parser

    p = build_parser()
    args = p.parse_args(["catalog"])
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = args.fn(args)
    finally:
        sys.stdout = old
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert len(data) >= 8
    assert {m["family"] for m in data} == EXPECTED_FAMILIES
