# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from hub.services.models import gguf_variants
from hub.utils import inventory_scan
from .gguf_cache_location_fixtures import cache_client, cache_locations
from utils import hf_cache_settings


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


@pytest.mark.parametrize("endpoint", ["/api/models/gguf-variants", "/api/hub/gguf-variants"])
@pytest.mark.parametrize("failure", ["network", 503, 401, 403, 404, 429])
def test_inactive_only_chat_cache_handles_hub_failures(
    cache_locations, cache_client, monkeypatch, endpoint, failure
):
    import httpx

    repo_id, expected = cache_locations
    active_root = hf_cache_settings.get_hf_cache_paths().hub_cache
    active_path = next(path for repo, path in expected.values() if repo.parent == active_root)
    active_path.unlink()
    quant, (repo, path) = next(
        (quant, pair) for quant, pair in expected.items() if pair[0].parent != active_root
    )
    inventory_scan.invalidate_hf_cache_scans()
    error = (
        ConnectionError("Hub temporarily unavailable")
        if failure == "network"
        else (
            httpx.HTTPStatusError(
                "Hub response failed",
                request = httpx.Request("GET", "https://huggingface.co/api/models/Org/Model-GGUF"),
                response = httpx.Response(failure),
            )
        )
    )

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(gguf_variants, "list_gguf_variants", unavailable)
    transient = failure == "network" or failure == 503
    if transient:
        scoped = cache_client.get(
            endpoint,
            params = {"repo_id": repo_id, "local_path": str(path.parent)},
        )
        assert scoped.status_code == 200, scoped.text
        assert {v["quant"] for v in scoped.json()["variants"] if v["downloaded"]} == {quant}

    response = cache_client.get(
        endpoint,
        params = {
            "repo_id": repo_id,
            "local_path": repo_id,
            "include_cache_locations": True,
        },
    )
    if not transient:
        assert response.status_code == failure, response.text
        return
    assert response.status_code == 200, response.text
    variants = [v for v in response.json()["variants"] if v["downloaded"]]
    assert [(v["quant"], v["cache_path"]) for v in variants] == [(quant, str(repo))]
    assert not variants[0]["partial"]


@pytest.mark.parametrize("task", ["text-to-image", "text-to-video", "text-to-speech"])
def test_media_loader_does_not_reuse_inactive_cache(cache_locations, monkeypatch, task):
    from core.inference.llama_cpp import cached_gguf_for_load

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    monkeypatch.setattr(
        "hub.services.models.catalog_classification._gguf_path_task", lambda *args: task
    )
    for quant, (repo, path) in expected.items():
        actual = cached_gguf_for_load(repo_id, quant)
        assert actual == (str(path) if repo.parent == active else None)
