# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
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
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hub.routes import inventory
    from routes import models

    app = FastAPI()
    app.include_router(inventory.router, prefix = "/api/hub")
    app.include_router(models.router, prefix = "/api/models")
    app.dependency_overrides[inventory.get_current_subject] = lambda: "test"
    app.dependency_overrides[inventory.get_request_hf_token] = lambda: None
    with TestClient(app) as client:
        yield client


def test_cached_gguf_keeps_quants_in_previous_download_folders(cache_locations, cache_client):
    repo_id, expected = cache_locations
    scans = inventory_scan.all_hf_cache_scans()
    assert sum(repo.repo_id == repo_id for scan in scans for repo in scan.repos) == 2
    rows = [row for row in cache_inventory._scan_cached_gguf() if row["repo_id"] == repo_id]
    repeated = cache_inventory._scan_cached_gguf(cache_scans = scans + scans)
    assert [row for row in repeated if row["repo_id"] == repo_id] == rows
    response = cache_client.get("/api/hub/cached-gguf")
    assert response.status_code == 200
    rows = [row for row in response.json()["cached"] if row["repo_id"] == repo_id]
    found = {}
    for row in rows:
        response = asyncio.run(
            gguf_variants.get_gguf_variants_response(
                repo_id, prefer_local_cache = True, local_path = row["load_id"]
            )
        )
        for variant in response.variants:
            if variant.downloaded:
                found[variant.quant] = row

    assert set(found) == set(expected), "A quant in the inactive cache disappeared"
    assert len({row["inventory_id"] for row in rows}) == 2
    for quant, (repo, file_path) in expected.items():
        assert found[quant]["cache_path"] == str(repo)
        assert found[quant]["load_id"] == str(file_path.parent)
        assert file_path.is_file()
        assert found[quant]["active_cache"] == (
            repo.parent == hf_cache_settings.get_hf_cache_paths().hub_cache
        )
    # Use the exact management path returned by the inventory, even when that
    # folder is inactive. Deleting Q8 must leave the other cache's Q6 intact.
    deletion._delete_cached_model_blocking(repo_id, "Q8_0", None, found["Q8_0"]["cache_path"])
    assert not expected["Q8_0"][1].exists()
    assert expected["Q6_K"][1].is_file()


@pytest.mark.parametrize("action", ["impact", "reveal", "copy"])
def test_cache_actions_target_selected_copy(cache_locations, cache_client, monkeypatch, action):
    repo_id, expected = cache_locations
    default_q8 = expected["Q6_K"][1].with_name("Model-Q8_0.gguf")
    default_q8.write_bytes(b"\0" * 128)
    expected["Q8_0"][1].write_bytes(b"\0" * 512)
    inventory_scan.invalidate_hf_cache_scans()
    selected_path = str(expected["Q6_K"][0])
    payload = {"repo_id": repo_id, "variant": "Q8_0", "cache_path": selected_path}
    if action == "impact":
        response = cache_client.post("/api/hub/delete-impact", json = payload)
        assert response.status_code == 200, response.text
        assert response.json()["reclaimed_bytes"] == 128
    elif action == "reveal":
        revealed = []
        monkeypatch.setattr("utils.paths.path_utils.reveal_in_file_manager", revealed.append)
        response = cache_client.post("/api/models/reveal-cached-model", json = payload)
        assert response.status_code == 200, response.text
        assert revealed == [default_q8]
    else:
        response = cache_client.get("/api/models/cached-model-path", params = payload)
        assert response.status_code == 200, response.text
        assert response.json()["path"] == str(default_q8)
    deletion._delete_cached_model_blocking(repo_id, "Q8_0", None, selected_path)
    assert not default_q8.exists()
    assert expected["Q8_0"][1].is_file()


@pytest.mark.parametrize("logical_id", [False, True])
def test_delete_other_copy_of_loaded_quant(cache_locations, cache_client, monkeypatch, logical_id):
    repo_id, expected = cache_locations
    default_repo, q6 = expected["Q6_K"]
    custom_repo, loaded_file = expected["Q8_0"]
    default_q8 = q6.with_name("Model-Q8_0.gguf")
    default_q8.write_bytes(b"\0" * 128)
    inventory_scan.invalidate_hf_cache_scans()
    backend = SimpleNamespace(
        is_active = True,
        is_loaded = True,
        model_identifier = repo_id if logical_id else str(loaded_file.parent),
        hf_variant = "Q8_0",
        gguf_path = str(loaded_file),
    )
    monkeypatch.setattr("routes.inference.get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(deletion, "_inference_backend_blocks_delete", lambda *args: False)
    monkeypatch.setattr(deletion, "_diffusion_blocks_delete", lambda *args: None)
    monkeypatch.setattr(deletion, "_video_blocks_delete", lambda *args: None)
    payload = {"repo_id": repo_id, "variant": "Q8_0", "cache_path": str(default_repo)}
    response = cache_client.request("DELETE", "/api/hub/delete-cached", json = payload)
    assert response.status_code == 200, response.text
    assert not default_q8.exists()
    assert loaded_file.is_file()
    payload["cache_path"] = str(custom_repo)
    response = cache_client.request("DELETE", "/api/hub/delete-cached", json = payload)
    assert response.status_code == 400, response.text
    assert "Unload" in response.json()["detail"]
    assert loaded_file.is_file()


def test_unknown_loaded_copy_keeps_delete_guard(cache_locations, monkeypatch):
    repo_id, expected = cache_locations
    backend = SimpleNamespace(
        is_active = True,
        is_loaded = True,
        model_identifier = repo_id,
        hf_variant = "Q8_0",
        gguf_path = None,
    )
    monkeypatch.setattr("routes.inference.get_llama_cpp_backend", lambda: backend)
    assert deletion._llama_cpp_blocks_delete(repo_id, "Q8_0", str(expected["Q6_K"][0]))


@pytest.mark.parametrize("media", ["images", "video"])
def test_delete_other_copy_of_loaded_media(cache_locations, cache_client, monkeypatch, media):
    repo_id, expected = cache_locations
    unused_repo, unused_file = expected["Q6_K"]
    loaded_repo, loaded_file = expected["Q8_0"]
    backend = SimpleNamespace(
        status = lambda: {"loaded": True, "repo_id": str(loaded_file.parent)},
        loaded_repo_ids = lambda: [str(loaded_file.parent)],
        loading_repo_ids = lambda: [],
    )
    monkeypatch.setattr(deletion, "_llama_cpp_blocks_delete", lambda *args: False)
    monkeypatch.setattr(deletion, "_inference_backend_blocks_delete", lambda *args: False)
    if media == "images":
        monkeypatch.setattr(
            "core.inference.diffusion_engine_router.get_active_diffusion_engine", lambda: backend
        )
        monkeypatch.setattr(deletion, "_video_blocks_delete", lambda *args: None)
    else:
        monkeypatch.setattr("core.inference.video.get_video_backend", lambda: backend)
        monkeypatch.setattr(deletion, "_diffusion_blocks_delete", lambda *args: None)
    response = cache_client.request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": repo_id, "cache_path": str(unused_repo)},
    )
    assert response.status_code == 200, response.text
    assert not unused_file.exists()
    assert loaded_file.is_file()
    response = cache_client.request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": repo_id, "cache_path": str(loaded_repo)},
    )
    assert response.status_code == 400
    assert loaded_file.is_file()


def test_companion_duplicate_survives_selected_delete(cache_locations, cache_client, monkeypatch):
    from hub.services.models import companion_cleanup
    from hub.utils import companion_assets

    repo_id, expected = cache_locations
    unused_repo, unused_file = expected["Q6_K"]
    surviving_repo, surviving_file = expected["Q8_0"]
    # Both copies hold the same required asset; a different quant is not a substitute.
    duplicate = surviving_file.with_name(unused_file.name)
    duplicate.write_bytes(unused_file.read_bytes())
    inventory_scan.invalidate_hf_cache_scans()
    monkeypatch.setattr(companion_assets, "is_companion_base", lambda _repo: True)
    monkeypatch.setattr(
        companion_assets,
        "required_companion_bases",
        lambda *a, **k: {repo_id.lower(): {"Org/Image-GGUF"}},
    )
    monkeypatch.setattr(companion_assets, "known_companion_base_ids", lambda: {repo_id.lower()})
    monkeypatch.setattr(deletion, "_llama_cpp_blocks_delete", lambda *args: False)
    monkeypatch.setattr(deletion, "_inference_backend_blocks_delete", lambda *args: False)
    monkeypatch.setattr(deletion, "_diffusion_blocks_delete", lambda *args: None)
    monkeypatch.setattr(deletion, "_video_blocks_delete", lambda *args: None)
    duplicate.write_bytes(b"")
    inventory_scan.invalidate_hf_cache_scans()
    assert companion_cleanup._delete_impact_blocking(repo_id, None, str(unused_repo))[
        "blocked_by"
    ] == ["Org/Image-GGUF"]
    duplicate.write_bytes(unused_file.read_bytes())
    inventory_scan.invalidate_hf_cache_scans()
    impact = companion_cleanup._delete_impact_blocking(repo_id, None, str(unused_repo))
    assert impact["blocked_by"] == []
    response = cache_client.request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": repo_id, "cache_path": str(unused_repo)},
    )
    assert response.status_code == 200, response.text
    assert duplicate.is_file()
    assert companion_cleanup._delete_impact_blocking(repo_id, None, str(surviving_repo))[
        "blocked_by"
    ] == ["Org/Image-GGUF"]
    response = cache_client.request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": repo_id, "cache_path": str(surviving_repo)},
    )
    assert response.status_code == 400
    assert duplicate.is_file()
