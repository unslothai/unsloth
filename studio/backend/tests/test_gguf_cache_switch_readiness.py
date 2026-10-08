# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from hub.services.models import gguf_variants
from hub.utils import inventory_scan
from .gguf_cache_location_fixtures import cache_client, cache_locations
from utils import hf_cache_settings


@pytest.mark.parametrize("entrypoint", ["source", "listing", "loader", "worker"])
def test_interrupted_projector_prefers_ready_duplicate(cache_locations, monkeypatch, entrypoint):
    from core.inference.llama_cpp import cached_gguf_for_load
    from hub.utils import download_manifest
    from hub.utils.gguf_sources import cached_gguf_sources

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant = "Q4_K_M"
    filename = f"Model-{quant}.gguf"
    healthy = None
    for repo, path in expected.values():
        snapshot = path.parent
        (snapshot / filename).write_bytes(b"0" * 256)
        if repo.parent != active:
            (snapshot / "mmproj-F16.gguf").write_bytes(b"0" * 128)
            healthy = snapshot
        assert download_manifest.write_manifest(
            "model",
            repo_id,
            quant,
            [
                download_manifest.ExpectedFile(filename, 256),
                download_manifest.ExpectedFile("mmproj-F16.gguf", 128),
            ],
            hub_cache = repo.parent,
            commit_hash = snapshot.name,
        )
    assert healthy is not None
    inventory_scan.invalidate_hf_cache_scans()
    if entrypoint == "source":
        assert cached_gguf_sources(repo_id)[quant.lower()].snapshot == healthy
    elif entrypoint == "loader":
        assert cached_gguf_for_load(repo_id, quant) == str(healthy / filename)
    elif entrypoint == "worker":
        from core.inference.llama_cpp import LlamaCppBackend

        def offline(*args, **kwargs):
            raise ConnectionError("offline test")

        monkeypatch.setattr("huggingface_hub.list_repo_files", offline)
        monkeypatch.setattr("huggingface_hub.get_paths_info", offline)
        monkeypatch.setattr(
            "core.inference.llama_cpp.hf_hub_download_with_xet_fallback",
            lambda *args, **kwargs: pytest.fail("must reuse the complete remembered copy"),
        )
        actual = LlamaCppBackend()._download_gguf(hf_repo = repo_id, hf_variant = quant)
        assert actual == str(healthy / filename)
        assert (healthy / "mmproj-F16.gguf").is_file()
    else:
        response = asyncio.run(
            gguf_variants.get_gguf_variants_response(
                repo_id,
                prefer_local_cache = True,
                offline = True,
                include_cache_locations = True,
            )
        )
        variant = next(v for v in response.variants if v.quant == quant)
        assert variant.downloaded and not variant.partial
        assert variant.cache_path == str(healthy.parent.parent)


@pytest.mark.parametrize("token", [False, "hf_fixture_token"])
def test_pin_listing_needs_explicit_token_when_ambient_is_denied(
    cache_locations, monkeypatch, token
):
    from hub.utils.gguf import GgufVariantInfo

    repo_id, expected = cache_locations
    quant, (repo, path) = next(
        (q, value)
        for q, value in expected.items()
        if value[0].parent != hf_cache_settings.get_hf_cache_paths().hub_cache
    )
    variant = GgufVariantInfo(filename = path.name, quant = quant, size_bytes = 256)
    # Main decides this by probing the Hub, and the fixture repo is not really there. The repo is
    # private to an ambient caller and reachable with an explicit token, which is the difference
    # this test measures, so answer the probe instead of the network.
    monkeypatch.setattr(
        "hub.utils.hf_tokens._explicit_token_reaches_repo",
        lambda repo, token, *a, **k: token is not None,
    )
    monkeypatch.setattr(gguf_variants, "list_gguf_variants", lambda *a, **k: ([variant], False, []))
    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            hf_token = token,
            prefer_local_cache = True,
            include_cache_locations = True,
        )
    )
    actual = next(v for v in response.variants if v.quant == quant)
    assert actual.downloaded is bool(token)


@pytest.mark.parametrize("entrypoint", ["listing", "source"])
def test_incomplete_remembered_copy_is_not_advertised_complete(
    cache_locations, monkeypatch, entrypoint
):
    """An offline/local answer has no Hub check to fall back on, so the remembered copy is
    judged by its own manifest: a quant whose companion never finished is not a complete
    copy the picker may offer to load, even though every main shard is on disk."""
    from hub.utils import download_manifest
    from hub.utils.gguf_sources import cached_gguf_source_partial, cached_gguf_sources

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant = "Q4_K_M"
    filename = f"Model-{quant}.gguf"
    # The remembered copy holds every main shard but is missing its projector, which is
    # exactly the torn state a completed-looking snapshot can hide.
    remembered = None
    for repo, path in expected.values():
        if repo.parent == active:
            continue
        snapshot = path.parent
        (snapshot / filename).write_bytes(b"0" * 256)
        remembered = snapshot
        assert download_manifest.write_manifest(
            "model",
            repo_id,
            quant,
            [
                download_manifest.ExpectedFile(filename, 256),
                download_manifest.ExpectedFile("mmproj-F16.gguf", 128),
            ],
            hub_cache = repo.parent,
            commit_hash = snapshot.name,
        )
    assert remembered is not None
    inventory_scan.invalidate_hf_cache_scans()
    assert not (remembered / "mmproj-F16.gguf").is_file()
    assert cached_gguf_source_partial(repo_id, quant, remembered)

    if entrypoint == "source":
        assert cached_gguf_sources(repo_id)[quant.lower()].snapshot == remembered
        return

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            prefer_local_cache = True,
            offline = True,
            include_cache_locations = True,
        )
    )
    variant = next(v for v in response.variants if v.quant == quant)
    assert not variant.downloaded
    assert variant.partial
    assert variant.cache_path == str(remembered.parent.parent)


def test_a_healthy_duplicate_is_preferred_over_a_cancelled_copy(cache_locations):
    """A copy its own snapshot state marks partial -- here a cancel marker -- must not shadow a
    healthy duplicate of the same quant: the merge would report the quant as unusable and offer
    a retry for a copy that is already loadable from the other remembered folder."""
    from hub.utils import download_manifest
    from hub.utils.gguf_sources import cached_gguf_source_partial, cached_gguf_sources

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant = "Q4_K_M"
    filename = f"Model-{quant}.gguf"
    # The same quant is present in two folders, which is exactly what duplicate ranking decides.
    cancelled = healthy = None
    for repo, path in expected.values():
        snapshot = path.parent
        (snapshot / filename).write_bytes(b"0" * 256)
        if repo.parent == active:
            cancelled = snapshot
        else:
            healthy = snapshot
    assert cancelled is not None and healthy is not None
    # Manifest-verified by construction (every file it declares exists), yet partial by its own
    # marker -- the state a manifest-only comparison cannot see.
    assert download_manifest.write_cancel_marker(
        "model", repo_id, quant, hub_cache = cancelled.parent.parent.parent
    )
    inventory_scan.invalidate_hf_cache_scans()
    assert cached_gguf_source_partial(repo_id, quant, cancelled)
    assert not cached_gguf_source_partial(repo_id, quant, healthy)
    assert cached_gguf_sources(repo_id)[quant.lower()].snapshot == healthy

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            prefer_local_cache = True,
            offline = True,
            include_cache_locations = True,
        )
    )
    variant = next(v for v in response.variants if v.quant == quant)
    assert variant.downloaded
    assert not variant.partial
    assert variant.cache_path == str(healthy.parent.parent)


def test_an_authorized_anonymous_caller_still_sees_remembered_caches(cache_locations, monkeypatch):
    """hub_cached_read_refused authorizes a public repo for the sentinel; a second anonymous
    test here re-refused it and dropped a quant that exists only in a remembered folder."""
    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant, (repo, _path) = next(
        (q, value) for q, value in expected.items() if value[0].parent != active
    )
    monkeypatch.setattr("hub.utils.hf_tokens._explicit_token_reaches_repo", lambda *a, **k: True)
    monkeypatch.setattr(gguf_variants, "list_gguf_variants", lambda *a, **k: ([], False, []))
    inventory_scan.invalidate_hf_cache_scans()
    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            hf_token = False,
            prefer_local_cache = True,
            offline = True,
            include_cache_locations = True,
        )
    )
    found = next(v for v in response.variants if v.quant == quant)
    assert found.cache_path == str(repo)
