# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
from types import SimpleNamespace

import pytest

from hub.services.models import cache_inventory, deletion, gguf_variants
from hub.utils import inventory_scan
from utils import hf_cache_settings


@pytest.fixture(params = [False, True])
def cache_locations(monkeypatch, tmp_path, request):
    active_custom = request.param
    store = {}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", store.update)
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "unused")
    default_root = hf_cache_settings.get_hf_cache_paths().hub_cache
    custom_home = tmp_path / "custom"
    repo_id = "Org/Model-GGUF"
    expected = {}

    def write_quant(root, quant):
        repo = root / "models--Org--Model-GGUF"
        snapshot = repo / "snapshots" / ("d" * 40)
        snapshot.mkdir(parents = True)
        filename = f"Model-{quant}.gguf"
        (snapshot / filename).write_bytes(b"\0" * 256)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("d" * 40)
        expected[quant] = (repo, snapshot / filename)

    write_quant(default_root, "Q6_K")
    hf_cache_settings.set_hf_cache_home(str(custom_home))
    write_quant(custom_home / "hub", "Q8_0")
    hf_cache_settings.set_hf_cache_home(None)
    if active_custom:
        hf_cache_settings.set_hf_cache_home(str(custom_home))

    assert custom_home / "hub" in hf_cache_settings.known_hf_hub_caches()
    return repo_id, expected


@pytest.fixture
def cache_client():
    from auth.authentication import allow_ambient_hf_token, authenticated_via_api_key
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hub.routes import inventory
    from routes import models

    app = FastAPI()
    app.include_router(inventory.router, prefix = "/api/hub")
    app.include_router(models.router, prefix = "/api/models")
    app.dependency_overrides[inventory.get_current_subject] = lambda: "test"
    app.dependency_overrides[inventory.get_request_hf_token] = lambda: None
    # A browser session, not an sk-unsloth call: the route reads this before its cache walk.
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True
    with TestClient(app) as client:
        yield client
