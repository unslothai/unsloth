# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export a cached model to a plain folder and import it back (#8798)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hub.services.models import portable


def _cache_with(hub: Path, repo_id: str, files: dict[str, bytes], commit: str = "abc123") -> Path:
    """A real cache layout: blobs plus a snapshot of symlinks, and refs/main."""
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    (repo / "blobs").mkdir(parents = True)
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents = True)
    for name, data in files.items():
        blob = repo / "blobs" / f"sha-{name.replace('/', '_')}"
        blob.write_bytes(data)
        target = snapshot / name
        target.parent.mkdir(parents = True, exist_ok = True)
        os.symlink(blob, target)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit, encoding = "utf-8")
    return snapshot


@pytest.fixture
def hub(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(portable, "_hub_cache", lambda: cache)
    return cache


def test_export_dereferences_the_snapshot_and_writes_a_manifest(hub, tmp_path):
    _cache_with(hub, "org/model", {"config.json": b"{}", "model.safetensors": b"ST", "sub/tok.json": b"[]"})
    out = portable.export_cached_model("org/model", None, str(tmp_path / "out"))
    folder = Path(out["path"])
    assert folder == tmp_path / "out" / "org--model"
    assert out["files"] == 3 and out["size_bytes"] == 6
    assert (folder / "model.safetensors").read_bytes() == b"ST"
    assert not (folder / "model.safetensors").is_symlink(), "plain files, or the copy is useless off this machine"
    manifest = json.loads((folder / portable.MANIFEST_NAME).read_text())
    assert manifest["repo_id"] == "org/model" and manifest["revision"] == "abc123"
    assert sorted(manifest["files"]) == ["config.json", "model.safetensors", "sub/tok.json"]


def test_a_gguf_variant_exports_only_its_own_file_plus_sidecars(hub, tmp_path):
    _cache_with(hub, "org/m-GGUF", {"m-Q4_K_M.gguf": b"4", "m-Q8_0.gguf": b"8", "README.md": b"r"})
    out = portable.export_cached_model("org/m-GGUF", "Q4_K_M", str(tmp_path / "out"))
    names = sorted(p.name for p in Path(out["path"]).iterdir() if p.name != portable.MANIFEST_NAME)
    assert names == ["README.md", "m-Q4_K_M.gguf"]


def test_export_refuses_a_destination_inside_the_cache(hub, tmp_path):
    _cache_with(hub, "org/model", {"config.json": b"{}"})
    with pytest.raises(portable.PortableModelError):
        portable.export_cached_model("org/model", None, str(hub / "somewhere"))


def test_export_of_an_unknown_model_is_not_found(hub, tmp_path):
    _cache_with(hub, "org/model", {"config.json": b"{}"})
    with pytest.raises(FileNotFoundError):
        portable.export_cached_model("org/other", None, str(tmp_path / "out"))


def test_import_round_trips_into_the_cache_layout(hub, tmp_path):
    _cache_with(hub, "org/model", {"config.json": b"{}", "w.safetensors": b"W"})
    out = portable.export_cached_model("org/model", None, str(tmp_path / "out"))
    fresh = tmp_path / "hub2"
    fresh.mkdir()
    portable._hub_cache = lambda: fresh  # a second machine
    result = portable.import_model_folder(out["path"])
    snapshot = Path(result["path"])
    assert result["status"] == "imported" and result["files"] == 2
    assert snapshot == fresh / "models--org--model" / "snapshots" / "abc123"
    assert (snapshot / "w.safetensors").read_bytes() == b"W"
    assert any((fresh / "models--org--model" / "blobs").iterdir()), "content lives in blobs/, as the Hub writes it"
    assert (fresh / "models--org--model" / "refs" / "main").read_text() == "abc123"
    # The inventory scan sees it as a downloaded model.
    from hub.services.models.local_inventory import _discover_hf_cache

    assert [m for _, m, _ in _discover_hf_cache(fresh)] == ["org/model"]
    # Importing again is a no-op, not a second copy.
    assert portable.import_model_folder(out["path"])["status"] == "already_present"


def test_import_refuses_a_folder_without_a_manifest(hub, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "model.gguf").write_bytes(b"g")
    with pytest.raises(portable.PortableModelError):
        portable.import_model_folder(str(plain))


@pytest.mark.parametrize("name", ["../escape.bin", "/abs.bin", "a/../../b.bin"])
def test_import_refuses_manifest_entries_that_leave_the_folder(hub, tmp_path, name):
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / portable.MANIFEST_NAME).write_text(
        json.dumps({"format": "unsloth-export", "version": 1, "repo_id": "org/model", "files": [name]})
    )
    with pytest.raises(portable.PortableModelError):
        portable.import_model_folder(str(folder))
