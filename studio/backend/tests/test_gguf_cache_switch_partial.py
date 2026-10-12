# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from hub.services.models import deletion, gguf_variants
from hub.utils import inventory_scan
from .gguf_cache_location_fixtures import cache_client, cache_locations
from utils import hf_cache_settings


def test_an_interrupted_split_quant_is_still_offered_from_its_folder(cache_locations):
    """list_local_gguf_variants sees the quant while complete_snapshot_variants excludes it, so
    the merge must keep it as a partial source with its folder rather than drop the row."""
    from hub.utils.gguf_sources import cached_gguf_source_partial, cached_gguf_sources

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    remembered = next(repo for repo, _ in expected.values() if repo.parent != active)
    quant = "Q4_K_M"
    snap = remembered / "snapshots" / ("d" * 40)
    (snap / f"Model-{quant}-00001-of-00002.gguf").write_bytes(b"0" * 256)
    inventory_scan.invalidate_hf_cache_scans()

    # Listed by the scan but excluded as incomplete, so it is the fallback that carries it.
    source = cached_gguf_sources(repo_id)[quant.lower()]
    assert source.snapshot == snap

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
    assert variant.cache_path == str(remembered)


def test_merging_a_root_row_recomputes_the_default(cache_locations, monkeypatch):
    """The default is chosen among ROOT rows, so a default picked from the active cache alone
    must be recomputed once a remembered folder contributes a root checkpoint."""
    from hub.utils.gguf import GgufVariantInfo

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    remembered = next(repo for repo, _ in expected.values() if repo.parent != active)
    quant = "Q6_K"
    (remembered / "snapshots" / ("d" * 40) / f"Model-{quant}.gguf").write_bytes(b"0" * 256)
    distilled = GgufVariantInfo(
        filename = "distilled/Model-Q6_K.gguf", quant = "distilled/Model-Q6_K", size_bytes = 256
    )
    monkeypatch.setattr(
        gguf_variants, "list_gguf_variants", lambda *a, **k: ([distilled], False, [])
    )
    inventory_scan.invalidate_hf_cache_scans()
    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id, prefer_local_cache = True, offline = True, include_cache_locations = True
        )
    )
    assert response.default_variant == quant, response.default_variant


def test_duplicate_ranking_honors_current_companion_readiness(cache_locations):
    """No local rule can see a companion the CURRENT revision newly requires, so a duplicate
    the snapshot itself judges partial must not outrank a usable copy of the same quant."""
    from hub.utils.gguf_sources import cached_gguf_sources

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant = "Q4_K_M"
    for repo, path in expected.values():
        (path.parent / f"Model-{quant}.gguf").write_bytes(b"0" * 256)
    inventory_scan.invalidate_hf_cache_scans()
    active_snap = next(path.parent for repo, path in expected.values() if repo.parent == active)
    remembered_snap = next(path.parent for repo, path in expected.values() if repo.parent != active)
    # The active copy satisfies every local rule but its scoped Hub answer does not.
    readiness = {active_snap: False, remembered_snap: True}
    chosen = cached_gguf_sources(
        repo_id, scoped_ready = lambda snapshot, _quant: readiness.get(snapshot)
    )[quant.lower()].snapshot
    assert chosen == remembered_snap

    # The reverse verdict keeps the active copy: ranking is the comparison, not a bias.
    readiness = {active_snap: True, remembered_snap: False}
    kept = cached_gguf_sources(
        repo_id, scoped_ready = lambda snapshot, _quant: readiness.get(snapshot)
    )[quant.lower()].snapshot
    assert kept == active_snap


def test_the_merged_default_prefers_a_ready_row(cache_locations):
    """An offline merge that offers a complete Q8_0 must not recommend a torn remembered
    quant just because that quant ranks higher."""
    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    active_repo = next(repo for repo, _ in expected.values() if repo.parent == active)
    (active_repo / "snapshots" / ("d" * 40) / "Model-Q8_0.gguf").write_bytes(b"0" * 256)
    remembered = next(repo for repo, _ in expected.values() if repo.parent != active)
    snapped = remembered / "snapshots" / ("d" * 40)
    (snapped / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"0" * 256)
    inventory_scan.invalidate_hf_cache_scans()

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id, prefer_local_cache = True, offline = True, include_cache_locations = True
        )
    )
    ready = {v.quant for v in response.variants if v.downloaded and not v.partial}
    assert "Q8_0" in ready
    assert response.default_variant in ready


def test_a_remembered_partial_keeps_its_resume_metadata(cache_locations, monkeypatch):
    """A remembered partial's resume affordance has to describe the root a resume will really
    continue in. A download lands in the ACTIVE cache, so judging the row against the folder it
    happens to sit in would promise a byte-for-byte restart the continuation cannot honour."""
    from hub.utils import download_manifest

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant, (repo, _path) = next(
        (q, value) for q, value in expected.items() if value[0].parent != active
    )
    snap = repo / "snapshots" / ("d" * 40)
    (snap / f"Model-{quant}-00001-of-00002.gguf").write_bytes(b"0" * 256)
    assert download_manifest.write_cancel_marker(
        "model", repo_id, quant, "http", hub_cache = repo.parent
    )
    inventory_scan.invalidate_hf_cache_scans()
    roots = []

    def _resumable(
        _repo_id,
        _quant,
        repo_cache_dir = None,
    ):
        roots.append(repo_cache_dir)
        # Only the remembered folder holds a resumable partial; the active root has none.
        return repo_cache_dir is not None and repo_cache_dir.parent == repo.parent

    monkeypatch.setattr(gguf_variants, "_partial_resumable_for_variant", _resumable)
    monkeypatch.setattr(gguf_variants, "variant_remaining_bytes_from_state", lambda *a, **k: 4096)

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id, prefer_local_cache = True, offline = True, include_cache_locations = True
        )
    )
    variant = next(v for v in response.variants if v.quant == quant)
    assert variant.partial and not variant.downloaded
    # The row still names the copy it was listed from, which is what delete and load resolve.
    assert variant.cache_path == str(repo)
    # The active root holds no marker or manifest for this quant, so there is no transport to
    # name and nothing a resume could reuse: the neutral label, not a promise of "http".
    assert variant.partial_transport is None
    # Judged against the ACTIVE root's repo dir: this partial's own folder is where the resume
    # will NOT go.
    assert roots and all(root.parent == active for root in roots), roots
    assert variant.partial_resumable is False
    assert variant.download_remaining_bytes == 4096

    # The other way round: a partial in the ACTIVE cache is judged by the copy a resume continues.
    quant, (repo, _path) = next(
        (q, value) for q, value in expected.items() if value[0].parent == active
    )
    snap = repo / "snapshots" / ("d" * 40)
    (snap / f"Model-{quant}-00001-of-00002.gguf").write_bytes(b"0" * 256)
    assert download_manifest.write_cancel_marker(
        "model", repo_id, quant, "http", hub_cache = repo.parent
    )
    inventory_scan.invalidate_hf_cache_scans()
    roots.clear()
    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id, prefer_local_cache = True, offline = True, include_cache_locations = True
        )
    )
    variant = next(v for v in response.variants if v.quant == quant)
    assert variant.partial and not variant.downloaded
    assert roots and all(root == repo for root in roots), roots
    assert variant.partial_transport == "http"


def test_a_cached_only_quant_is_judged_by_its_own_partial_state(cache_locations, monkeypatch):
    """A quant the current revision no longer lists has no scoped answer to consult, so the
    merge has to fall back to the copy's own per-snapshot verdict instead of trusting the
    synthesized row and advertising a cancelled download as loadable."""
    from hub.utils import gguf_sources
    from hub.utils.gguf import GgufVariantInfo

    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant, (repo, path) = next(
        (q, value) for q, value in expected.items() if value[0].parent != active
    )
    other = next(q for q, value in expected.items() if value[0].parent == active)
    listed = GgufVariantInfo(filename = f"Model-{other}.gguf", quant = other, size_bytes = 256)
    monkeypatch.setattr(gguf_variants, "list_gguf_variants", lambda *a, **k: ([listed], False, []))
    source = gguf_sources.CachedGgufSource(
        GgufVariantInfo(filename = path.name, quant = quant, size_bytes = 256),
        path.parent,
        False,
    )
    # The current revision cannot describe this quant, so its scoped answer returns no row.
    monkeypatch.setattr(
        gguf_sources, "cached_gguf_sources", lambda *a, **k: {quant.lower(): source}
    )
    monkeypatch.setattr(gguf_sources, "cached_gguf_source_partial", lambda *a, **k: True)
    inventory_scan.invalidate_hf_cache_scans()

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(repo_id, include_cache_locations = True)
    )
    variant = next(v for v in response.variants if v.quant == quant)
    assert not variant.downloaded
    assert variant.partial


def test_the_delete_preview_follows_the_copy_the_row_carries(cache_locations, cache_client):
    """The listing hands each row the copy it resolved, and the confirm dialog must preview
    that same copy: without the reference the preview measures the active duplicate and
    describes a delete the user did not ask for."""
    repo_id, expected = cache_locations
    active = hf_cache_settings.get_hf_cache_paths().hub_cache
    quant = "Q4_K_M"
    for repo, path in expected.values():
        (path.parent / f"Model-{quant}.gguf").write_bytes(b"0" * 256)
    inventory_scan.invalidate_hf_cache_scans()
    remembered = next(repo for repo, _ in expected.values() if repo.parent != active)
    active_repo = next(repo for repo, _ in expected.values() if repo.parent == active)

    unscoped = cache_client.post(
        "/api/hub/delete-impact", json = {"repo_id": repo_id, "variant": quant}
    )
    assert unscoped.status_code == 200, unscoped.text
    assert unscoped.json()["cache_path"] == str(active_repo)

    scoped = cache_client.post(
        "/api/hub/delete-impact",
        json = {"repo_id": repo_id, "variant": quant, "cache_path": str(remembered)},
    )
    assert scoped.status_code == 200, scoped.text
    assert scoped.json()["cache_path"] == str(remembered)
    assert scoped.json()["reclaimed_bytes"] == 256


@pytest.mark.parametrize("invalid_kind", ["unremembered", "expired_reference"])
def test_delete_preview_rejects_an_invalid_explicit_cache_copy(
    cache_locations, cache_client, tmp_path, invalid_kind
):
    """A stale picker target cannot promise a harmless delete that confirmation rejects."""
    from fastapi import HTTPException

    repo_id, expected = cache_locations
    cache_path = (
        str(tmp_path / "unremembered" / "models--Org--Model-GGUF")
        if invalid_kind == "unremembered"
        else "ref:" + "0" * 32
    )
    payload = {"repo_id": repo_id, "variant": "Q6_K", "cache_path": cache_path}
    preview = cache_client.post("/api/hub/delete-impact", json = payload)
    assert preview.status_code == 400, preview.text
    assert preview.json()["detail"] == "Invalid cache_path"
    with pytest.raises(HTTPException) as rejected:
        deletion._delete_cached_model_blocking(repo_id, "Q6_K", None, cache_path)
    assert rejected.value.status_code == 400
    assert rejected.value.detail == "Invalid cache_path"
    assert all(path.exists() for _, path in expected.values())
