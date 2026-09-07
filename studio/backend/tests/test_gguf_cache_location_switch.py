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


@pytest.mark.parametrize("endpoint", ["/api/models/gguf-variants", "/api/hub/gguf-variants"])
def test_folder_switch_preserves_logical_row_and_both_quants(
    cache_locations, cache_client, endpoint
):
    from core.inference.llama_cpp import cached_gguf_for_load

    repo_id, expected = cache_locations
    rows = [r for r in cache_inventory._scan_cached_gguf() if r["repo_id"] == repo_id]
    assert len(rows) == 1
    assert rows[0]["load_id"] == repo_id
    response = cache_client.get(
        endpoint,
        params = {
            "repo_id": repo_id,
            "prefer_local_cache": True,
            "offline": True,
            "include_cache_locations": True,
            "local_path": rows[0]["load_id"],
        },
    )
    assert response.status_code == 200, response.text
    variants = {v["quant"]: v for v in response.json()["variants"] if v["downloaded"]}
    assert set(variants) == set(expected)
    for quant, (repo, path) in expected.items():
        assert variants[quant]["cache_path"] == str(repo)
        assert cached_gguf_for_load(repo_id, quant) == str(path)


def test_logical_quant_actions_choose_the_same_cache(cache_locations, cache_client):
    repo_id, expected = cache_locations
    inactive_quant = next(
        q
        for q, (repo, _) in expected.items()
        if repo.parent != hf_cache_settings.get_hf_cache_paths().hub_cache
    )
    repo, path = expected[inactive_quant]
    payload = {"repo_id": repo_id, "variant": inactive_quant}
    copied = cache_client.get("/api/models/cached-model-path", params = payload)
    assert copied.status_code == 200, copied.text
    assert copied.json()["path"] == str(path)
    preview = cache_client.post("/api/hub/delete-impact", json = payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["reclaimed_bytes"] == 256
    assert preview.json()["cache_path"] == str(repo)
    deletion._delete_cached_model_blocking(repo_id, inactive_quant, None)
    assert not path.exists()
    assert all(p.exists() for q, (_, p) in expected.items() if q != inactive_quant)


def test_complete_inactive_copy_beats_torn_active_copy(cache_locations, cache_client):
    from core.inference.llama_cpp import cached_gguf_for_load

    repo_id, expected = cache_locations
    inactive_quant = next(
        q
        for q, (repo, _) in expected.items()
        if repo.parent != hf_cache_settings.get_hf_cache_paths().hub_cache
    )
    active_repo = next(
        repo
        for repo, _ in expected.values()
        if repo.parent == hf_cache_settings.get_hf_cache_paths().hub_cache
    )
    active_snap = active_repo / "snapshots" / ("d" * 40)
    (active_snap / f"Model-{inactive_quant}-00001-of-00002.gguf").write_bytes(b"0" * 256)
    inventory_scan.invalidate_hf_cache_scans()
    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id, prefer_local_cache = True, include_cache_locations = True
        )
    )
    found = next(v for v in response.variants if v.quant == inactive_quant)
    assert found.downloaded and not found.partial
    assert cached_gguf_for_load(repo_id, inactive_quant) == str(expected[inactive_quant][1])


def test_duplicate_quant_prefers_active_cache_and_deletes_only_that_copy(
    cache_locations, cache_client
):
    from core.inference.llama_cpp import cached_gguf_for_load

    repo_id, expected = cache_locations
    for repo, path in expected.values():
        (path.parent / "Model-Q4_K_M.gguf").write_bytes(b"0" * 256)
    inventory_scan.invalidate_hf_cache_scans()
    active_repo = next(
        repo
        for repo, _ in expected.values()
        if repo.parent == hf_cache_settings.get_hf_cache_paths().hub_cache
    )
    active_path = active_repo / "snapshots" / ("d" * 40) / "Model-Q4_K_M.gguf"
    assert cached_gguf_for_load(repo_id, "Q4_K_M") == str(active_path)
    deletion._delete_cached_model_blocking(repo_id, "Q4_K_M", None)
    assert not active_path.exists()
    surviving = [
        path.parent / "Model-Q4_K_M.gguf" for repo, path in expected.values() if repo != active_repo
    ]
    assert all(path.exists() for path in surviving)
    assert cached_gguf_for_load(repo_id, "Q4_K_M") == str(surviving[0])


def test_explicit_snapshot_remains_scoped(cache_locations):
    repo_id, expected = cache_locations
    for quant, (repo, path) in expected.items():
        response = asyncio.run(
            gguf_variants.get_gguf_variants_response(
                repo_id,
                prefer_local_cache = True,
                local_path = str(path.parent),
                include_cache_locations = True,
            )
        )
        assert {v.quant for v in response.variants if v.downloaded} == {quant}


def test_download_worker_reuses_inactive_quant_without_network(cache_locations, monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    repo_id, expected = cache_locations
    quant = next(
        q
        for q, (repo, _) in expected.items()
        if repo.parent != hf_cache_settings.get_hf_cache_paths().hub_cache
    )

    def offline(*args, **kwargs):
        raise ConnectionError("offline test")

    def no_download(*args, **kwargs):
        pytest.fail("The downloaded variant must be reused without moving or fetching weights")

    monkeypatch.setattr("huggingface_hub.list_repo_files", offline)
    monkeypatch.setattr("huggingface_hub.get_paths_info", offline)
    monkeypatch.setattr("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", no_download)
    resolved = LlamaCppBackend()._download_gguf(hf_repo = repo_id, hf_variant = quant)
    assert resolved == str(expected[quant][1])


def test_listing_requires_opt_in(cache_locations, cache_client):
    repo_id, expected = cache_locations
    response = cache_client.get(
        "/api/hub/gguf-variants",
        params = {
            "repo_id": repo_id,
            "prefer_local_cache": True,
            "offline": True,
        },
    )
    assert response.status_code == 200, response.text
    active = {
        quant
        for quant, (repo, _) in expected.items()
        if repo.parent == hf_cache_settings.get_hf_cache_paths().hub_cache
    }
    assert {v["quant"] for v in response.json()["variants"] if v["downloaded"]} == active


def test_media_sources_do_not_enter_chat_resolution(cache_locations, monkeypatch):
    from hub.utils.gguf_sources import cached_gguf_action_path, cached_gguf_sources

    repo_id, expected = cache_locations
    monkeypatch.setattr(
        "hub.services.models.catalog_classification._gguf_path_task", lambda *args: "text-to-image"
    )
    assert cached_gguf_sources(repo_id) == {}
    assert cached_gguf_action_path(repo_id, "Q6_K") is None
    explicit = str(expected["Q6_K"][0])
    assert cached_gguf_action_path(repo_id, "Q6_K", explicit) == explicit


@pytest.mark.parametrize("missing_context", [False, True])
def test_cached_quant_keeps_its_own_context(cache_locations, cache_client, missing_context):
    import struct

    repo_id, expected = cache_locations
    contexts = {"Q6_K": 131072, "Q8_0": None if missing_context else 32768}

    def gguf_string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    for quant, (_, path) in expected.items():
        context = contexts[quant]
        header = b"GGUF" + struct.pack("<IQQ", 3, 0, 1 if context is None else 2)
        header += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("llama")
        if context is not None:
            header += gguf_string("llama.context_length") + struct.pack("<II", 4, context)
        path.write_bytes(header.ljust(256, b"\0"))
    inventory_scan.invalidate_hf_cache_scans()
    response = cache_client.get(
        "/api/models/gguf-variants",
        params = {
            "repo_id": repo_id,
            "prefer_local_cache": True,
            "offline": True,
            "include_cache_locations": True,
        },
    )
    assert response.status_code == 200, response.text
    listing = response.json()
    # This is the native limit the picker supplies when the user selects/configures a quant.
    actual = {
        v["quant"]: v.get("context_length", listing["context_length"])
        for v in listing["variants"]
        if v["downloaded"]
    }
    assert actual == contexts


@pytest.mark.parametrize("undersized", [False, True])
@pytest.mark.parametrize("fail_repeat_listing", [False, True])
def test_inactive_source_uses_scoped_online_status(
    cache_locations, monkeypatch, undersized, fail_repeat_listing
):
    from hub.utils.download_manifest import ExpectedFile
    from hub.utils.gguf import GgufVariantInfo
    from hub.utils.gguf_plan import plan_from_expected_files

    repo_id, expected = cache_locations
    quant, (repo, path) = next(
        (q, source)
        for q, source in expected.items()
        if source[0].parent != hf_cache_settings.get_hf_cache_paths().hub_cache
    )
    # A normal HF snapshot symlink identifies the old blob independently of its byte size.
    payload = path.read_bytes()
    path.unlink()
    blob = repo / "blobs" / ("a" * 64)
    blob.parent.mkdir(exist_ok = True)
    blob.write_bytes(payload[:32] if undersized else payload)
    path.symlink_to(blob)
    variant = GgufVariantInfo(filename = path.name, quant = quant, size_bytes = 256)
    requirement = plan_from_expected_files(quant, [ExpectedFile(path.name, 256, "b" * 64)])
    listing_calls = []

    def list_online(*args, **kwargs):
        listing_calls.append(repo_id)
        if fail_repeat_listing and len(listing_calls) > 1:
            raise ConnectionError("a repeated Hub request failed")
        return [variant], False, []

    monkeypatch.setattr(gguf_variants, "list_gguf_variants", list_online)
    monkeypatch.setattr(gguf_variants, "_variant_requirement_cache_get", lambda *a: requirement)
    inventory_scan.invalidate_hf_cache_scans()
    scoped = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            local_path = str(path.parent),
        )
    )
    listing_calls.clear()
    merged = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            include_cache_locations = True,
        )
    )
    assert listing_calls == [repo_id]
    direct = next(v for v in scoped.variants if v.quant == quant)
    actual = next(v for v in merged.variants if v.quant == quant)
    assert direct.downloaded is not undersized
    assert direct.update_available is not undersized
    assert (actual.downloaded, actual.partial, actual.update_available) == (
        direct.downloaded,
        direct.partial,
        direct.update_available,
    )


def test_memory_estimate_uses_the_copy_selected_for_load(cache_locations):
    from core.inference.llama_cpp import cached_gguf_for_load
    from routes.models import _resolve_quant_gguf

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    for repo, path in expected.values():
        (path.parent / "Model-Q4_K_M.gguf").write_bytes(
            b"0" * (128 if repo.parent == active else 512)
        )
    inventory_scan.invalidate_hf_cache_scans()
    loaded = cached_gguf_for_load(repo_id, "Q4_K_M")
    estimated, size = _resolve_quant_gguf(repo_id, "Q4_K_M", False)
    assert estimated == loaded
    assert size == 128
