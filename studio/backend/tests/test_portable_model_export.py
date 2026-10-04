# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export a cached model to a plain folder and import it back (#8798)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hub.services.models import portable


def _cache_with(
    hub: Path,
    repo_id: str,
    files: dict[str, bytes],
    commit: str = "abc123",
) -> Path:
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
    _cache_with(
        hub, "org/model", {"config.json": b"{}", "model.safetensors": b"ST", "sub/tok.json": b"[]"}
    )
    out = portable.export_cached_model("org/model", None, str(tmp_path / "out"))
    folder = Path(out["path"])
    assert folder == tmp_path / "out" / "org--model"
    assert out["files"] == 3 and out["size_bytes"] == 6
    assert (folder / "model.safetensors").read_bytes() == b"ST"
    assert not (
        folder / "model.safetensors"
    ).is_symlink(), "plain files, or the copy is useless off this machine"
    manifest = json.loads((folder / portable.MANIFEST_NAME).read_text())
    assert manifest["repo_id"] == "org/model" and manifest["revision"] == "abc123"
    assert sorted(manifest["files"]) == ["config.json", "model.safetensors", "sub/tok.json"]


def test_a_gguf_variant_exports_only_its_own_file_plus_sidecars(hub, tmp_path):
    _cache_with(
        hub,
        "org/m-GGUF",
        {
            "m-Q2_K.gguf": b"2",
            "m-Q2_K_L.gguf": b"L",
            "UD-Q2_K_XL/m-UD-Q2_K_XL.gguf": b"X",
            "mmproj-F16.gguf": b"p",
            "README.md": b"r",
        },
    )
    out = portable.export_cached_model("org/m-GGUF", "Q2_K", str(tmp_path / "out"))
    manifest = json.loads((Path(out["path"]) / portable.MANIFEST_NAME).read_text())
    assert sorted(manifest["files"]) == ["README.md", "m-Q2_K.gguf", "mmproj-F16.gguf"]


def test_a_second_variant_imports_into_a_snapshot_holding_another(hub, tmp_path, monkeypatch):
    _cache_with(hub, "org/m-GGUF", {"m-Q2_K.gguf": b"2", "m-Q8_0.gguf": b"8", "README.md": b"r"})
    out = portable.export_cached_model("org/m-GGUF", "Q8_0", str(tmp_path / "out"))
    other = tmp_path / "hub2"
    other.mkdir()
    snapshot = _cache_with(other, "org/m-GGUF", {"m-Q2_K.gguf": b"2", "README.md": b"r"})
    monkeypatch.setattr(portable, "_hub_cache", lambda: other)
    assert portable.import_model_folder(out["path"])["status"] == "imported"
    assert (snapshot / "m-Q8_0.gguf").read_bytes() == b"8"
    assert (snapshot / "README.md").is_symlink(), "the cached file is kept, not replaced"


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
    assert any(
        (fresh / "models--org--model" / "blobs").iterdir()
    ), "content lives in blobs/, as the Hub writes it"
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
        json.dumps(
            {"format": "unsloth-export", "version": 1, "repo_id": "org/model", "files": [name]}
        )
    )
    with pytest.raises(portable.PortableModelError):
        portable.import_model_folder(str(folder))


def test_an_api_key_caller_gets_the_folder_redacted_and_a_session_keeps_it(hub, tmp_path):
    """Both routes answer the folder they touched, so they sit behind the inventory path
    boundary like the rest of the hub routes."""
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hub.routes import inventory as inventory_routes

    _cache_with(hub, "org/model", {"config.json": b"{}"})

    def client(via_api_key: bool) -> TestClient:
        app = FastAPI()
        app.include_router(inventory_routes.router, prefix = "/api/hub")
        app.dependency_overrides[get_current_subject] = lambda: "alice"
        app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
        return TestClient(app, raise_server_exceptions = False)

    session = client(False).post(
        "/api/hub/export-model",
        json = {"repo_id": "org/model", "destination": str(tmp_path / "session")},
    )
    assert session.status_code == 200
    assert session.json()["path"] == str(tmp_path / "session" / "org--model")

    keyed = client(True).post(
        "/api/hub/export-model",
        json = {"repo_id": "org/model", "destination": str(tmp_path / "keyed")},
    )
    assert keyed.status_code == 200
    assert keyed.json()["path"] == "" and keyed.json()["files"] == 1

    # The refusal names the offending folder; an API-key caller does not get it back.
    refused = client(True).post(
        "/api/hub/export-model",
        json = {"repo_id": "org/model", "destination": str(hub / "inside")},
    )
    assert refused.status_code == 400
    assert str(hub) not in refused.json()["detail"]

    imported = client(True).post(
        "/api/hub/import-model", json = {"source": str(tmp_path / "session" / "org--model")}
    )
    assert imported.status_code == 200
    assert imported.json()["path"] == ""


def test_a_managed_account_is_refused(hub, tmp_path):
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hub.routes import inventory as inventory_routes
    from utils.account_context import AccountContext, bind_account, reset_account

    _cache_with(hub, "org/model", {"config.json": b"{}"})
    app = FastAPI()
    app.include_router(inventory_routes.router, prefix = "/api/hub")
    app.dependency_overrides[get_current_subject] = lambda: "bob"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False

    async def as_managed(scope, receive, send):
        token = bind_account(AccountContext("bob-id", "bob", "user"))
        try:
            await app(scope, receive, send)
        finally:
            reset_account(token)

    client = TestClient(as_managed, raise_server_exceptions = False)
    exported = client.post(
        "/api/hub/export-model",
        json = {"repo_id": "org/model", "destination": str(tmp_path / "out")},
    )
    assert exported.status_code == 403 and not (tmp_path / "out").exists()
    assert client.post("/api/hub/import-model", json = {"source": str(tmp_path)}).status_code == 403
