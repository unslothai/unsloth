# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from hub.services.models import cache_inventory, deletion, gguf_variants
from hub.utils import inventory_scan
from .gguf_cache_location_fixtures import cache_client, cache_locations
from utils import hf_cache_settings


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


def test_loader_skips_a_cancelled_copy_for_a_healthy_duplicate(cache_locations):
    """The listing prefers the healthy duplicate, so the loader must not open the cancelled copy."""
    from core.inference.llama_cpp import cached_gguf_for_load
    from hub.utils import download_manifest

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant = "Q4_K_M"
    cancelled = healthy = None
    cancelled_repo = None
    for repo, path in expected.values():
        (path.parent / f"Model-{quant}.gguf").write_bytes(b"0" * 256)
        if repo.parent == active:
            cancelled, cancelled_repo = path.parent, repo
        else:
            healthy = path.parent
    assert cancelled is not None and healthy is not None
    assert download_manifest.write_cancel_marker(
        "model", repo_id, quant, hub_cache = cancelled_repo.parent
    )
    inventory_scan.invalidate_hf_cache_scans()
    assert cached_gguf_for_load(repo_id, quant) == str(healthy / f"Model-{quant}.gguf")


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
