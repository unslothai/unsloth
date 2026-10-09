# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""User-chosen display names for custom scan folders, shown in place of "Custom folder"."""

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from hub.services.models import local_inventory
from hub.storage import scan_folders
from routes import models as models_route
from storage import studio_db


@pytest.fixture(autouse = True)
def _registrable_tmp(monkeypatch):
    # macOS denies /private/var, where pytest keeps tmp_path.
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [])


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"GGUF" + b"\0" * 28)
    return path


@pytest.mark.parametrize("storage", [studio_db, scan_folders])
def test_add_stores_the_name_and_a_readd_without_one_keeps_it(tmp_path, storage):
    row, inserted = storage.add_scan_folder_with_status(str(tmp_path), label = "  Jan   app ")
    assert inserted and row["label"] == "Jan app"
    row, changed = storage.add_scan_folder_with_status(str(tmp_path))
    assert not changed and row["label"] == "Jan app"
    row, changed = storage.add_scan_folder_with_status(str(tmp_path), label = "GPT4All")
    assert changed and row["label"] == "GPT4All"
    row, changed = storage.add_scan_folder_with_status(str(tmp_path), label = " ")
    assert changed and row["label"] is None
    assert storage.list_scan_folders()[0]["label"] is None


def test_rename_sets_and_clears_the_name(tmp_path):
    row, _ = studio_db.add_scan_folder_with_status(str(tmp_path))
    assert row["label"] is None
    assert studio_db.update_scan_folder_label(row["id"], "Jan")["label"] == "Jan"
    assert scan_folders.list_scan_folders()[0]["label"] == "Jan"
    assert studio_db.update_scan_folder_label(row["id"], "")["label"] is None
    assert studio_db.update_scan_folder_label(row["id"] + 1, "Jan") is None


def test_an_overlong_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match = "at most"):
        studio_db.add_scan_folder_with_status(str(tmp_path), label = "x" * 41)
    assert studio_db.list_scan_folders() == []


def test_rename_endpoint_reports_missing_and_invalid(tmp_path):
    row = local_inventory.add_scan_folder_response(str(tmp_path), None, "Jan")
    assert row["label"] == "Jan"
    renamed = local_inventory.update_scan_folder_response(row["id"], "KoboldCpp")
    assert renamed["label"] == "KoboldCpp" and renamed["status"] == "ok"
    with pytest.raises(HTTPException) as missing:
        local_inventory.update_scan_folder_response(row["id"] + 1, "Jan")
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as invalid:
        local_inventory.update_scan_folder_response(row["id"], "x" * 41)
    assert invalid.value.status_code == 400


def _folders(tmp_path: Path, label):
    root = tmp_path / "jan-models"
    _gguf(root / "tiny-Q4_K_M.gguf")
    return [{"id": 1, "path": str(root), "recursive": False, "label": label}]


@pytest.mark.parametrize("label", ["Jan", None])
def test_hub_listing_stamps_the_folder_name_on_its_rows(tmp_path, label):
    empty = tmp_path / "_empty"
    empty.mkdir()
    sources = local_inventory._LocalInventorySources(empty, empty, empty, (), (), (), ())
    models = asyncio.run(
        local_inventory._scan_local_models_response(str(empty), _folders(tmp_path, label), sources)
    ).models
    custom = [m for m in models if m.source == "custom"]
    assert custom and {m.source_label for m in custom} == {label}


@pytest.mark.parametrize("label", ["Jan", None])
def test_compat_listing_stamps_the_folder_name_on_its_rows(tmp_path, label):
    empty = tmp_path / "_empty"
    empty.mkdir()
    sources = models_route._CompatLocalInventorySources(empty, empty, empty, (), ())
    models = models_route.collect_local_models(
        empty, custom_folders = _folders(tmp_path, label), sources = sources
    )
    custom = [m for m in models if m.source == "custom"]
    assert custom and {m.source_label for m in custom} == {label}


def test_hub_routes_add_rename_and_list_the_name(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import authenticated_via_api_key, get_current_subject
    from hub.routes import inventory as inventory_routes

    app = FastAPI()
    app.include_router(inventory_routes.router, prefix = "/api/hub")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: True
    client = TestClient(app, raise_server_exceptions = False)

    added = client.post("/api/hub/scan-folders", json = {"path": str(tmp_path), "label": "Jan"})
    assert added.status_code == 201, added.text
    folder = added.json()
    assert folder["label"] == "Jan"
    # An API-key caller still never sees the host path, name or not.
    assert str(tmp_path) not in added.text

    renamed = client.patch(f"/api/hub/scan-folders/{folder['id']}", json = {"label": "GPT4All"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["label"] == "GPT4All"
    assert str(tmp_path) not in renamed.text
    listed = client.get("/api/hub/scan-folders").json()["folders"]
    assert [f["label"] for f in listed] == ["GPT4All"]

    assert client.patch("/api/hub/scan-folders/999999", json = {"label": "x"}).status_code == 404
    # The request schema caps the name at the stored limit, so an over-long one never reaches the db.
    too_long = client.patch(f"/api/hub/scan-folders/{folder['id']}", json = {"label": "x" * 41})
    assert too_long.status_code == 422
    too_long_add = client.post("/api/hub/scan-folders", json = {"path": str(tmp_path), "label": "x" * 41})
    assert too_long_add.status_code == 422
