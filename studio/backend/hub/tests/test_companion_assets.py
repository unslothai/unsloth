# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reuse, preflight and deletion of shared image-model companion assets (issue #8116).

The shape under test is the reporter's: ``unsloth/FLUX.2-klein-4B-GGUF`` holding two quants,
and one cached copy of ``black-forest-labs/FLUX.2-klein-4B`` carrying the 8.23 GB of text
encoder, VAE, tokenizer and configs that both quants load through.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.services.models import cache_inventory, companion_cleanup, deletion
from hub.utils import companion_assets

GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
# Real byte counts, measured from a cache holding exactly this pair.
Q2_K_BYTES = 1_827_807_808
Q4_K_M_BYTES = 2_604_311_104
COMPANION_BYTES = 8_229_021_460


@pytest.fixture(autouse = True)
def _isolated_state_root(monkeypatch, tmp_path):
    """Companion links live under the app cache root; keep each test to its own."""
    monkeypatch.setattr("utils.paths.storage_roots.cache_root", lambda: tmp_path / "app-cache")


def _file(name: str, size: int, *, snapshot: str = "/cache/snap"):
    return SimpleNamespace(
        file_name = name.rsplit("/", 1)[-1],
        file_path = f"{snapshot}/{name}",
        blob_path = f"/cache/blobs/{name.replace('/', '_')}",
        size_on_disk = size,
    )


def _repo(repo_id: str, files, *, cache: str = "/cache"):
    snapshot = f"{cache}/models--{repo_id.replace('/', '--')}/snapshots/rev1"
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = f"{cache}/models--{repo_id.replace('/', '--')}",
        revisions = [
            SimpleNamespace(
                commit_hash = "rev1",
                snapshot_path = snapshot,
                files = [_file(name, size, snapshot = snapshot) for name, size in files],
            )
        ],
    )


def _gguf_repo(*quants):
    return _repo(GGUF_REPO, [(f"flux-2-klein-4b-{q}.gguf", size) for q, size in quants])


def _base_repo(repo_id: str = BASE_REPO):
    return _repo(
        repo_id,
        [
            ("model_index.json", 460),
            ("text_encoder/model-00001-of-00002.safetensors", COMPANION_BYTES - 1_000_000),
            ("vae/diffusion_pytorch_model.safetensors", 900_000),
            ("tokenizer/tokenizer.json", 99_540),
        ],
    )


def _install(monkeypatch, *repos):
    scans = [SimpleNamespace(repos = list(repos))]
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: scans)
    monkeypatch.setattr(companion_cleanup.cache_inventory, "all_hf_cache_scans", lambda: scans)
    return scans


# --------------------------------------------------------------------------------------------
# 1. Compatible quants reuse ONE cached copy of the companion assets.
# --------------------------------------------------------------------------------------------


def test_two_quants_of_one_family_resolve_to_a_single_companion_base():
    """Both quants derive the same base id, so the cache holds one copy, not two."""
    scans = [SimpleNamespace(repos = [_gguf_repo(("Q2_K", Q2_K_BYTES), ("Q4_K_M", Q4_K_M_BYTES))])]
    required = companion_assets.required_companion_bases(scans)
    assert BASE_REPO.lower() in required
    assert required[BASE_REPO.lower()] == {GGUF_REPO}


# --------------------------------------------------------------------------------------------
# 2. Deletion shows what is reclaimed and what remains.
# --------------------------------------------------------------------------------------------


def test_deleting_one_of_two_quants_retains_the_companions(monkeypatch):
    _install(monkeypatch, _gguf_repo(("Q2_K", Q2_K_BYTES), ("Q4_K_M", Q4_K_M_BYTES)), _base_repo())
    impact = asyncio.run(companion_cleanup.delete_impact_response(GGUF_REPO, "Q2_K"))
    assert impact["reclaimed_bytes"] == Q2_K_BYTES
    retained = impact["retained_companions"]
    assert [r["repo_id"] for r in retained] == [BASE_REPO]
    assert retained[0]["size_bytes"] == COMPANION_BYTES
    # The surviving quant is named as the reason, so the dialog can say who is holding it.
    assert retained[0]["needed_by"] == [GGUF_REPO]
    assert impact["freeable_companions"] == []


def test_deleting_the_last_quant_reports_the_companions_as_freeable(monkeypatch):
    _install(monkeypatch, _gguf_repo(("Q4_K_M", Q4_K_M_BYTES)), _base_repo())
    impact = asyncio.run(companion_cleanup.delete_impact_response(GGUF_REPO, "Q4_K_M"))
    assert impact["reclaimed_bytes"] == Q4_K_M_BYTES
    assert impact["retained_companions"] == []
    freeable = impact["freeable_companions"]
    assert [f["repo_id"] for f in freeable] == [BASE_REPO]
    assert freeable[0]["size_bytes"] == COMPANION_BYTES
    assert freeable[0]["needed_by"] == []


def test_whole_repo_delete_reclaims_every_quant(monkeypatch):
    _install(monkeypatch, _gguf_repo(("Q2_K", Q2_K_BYTES), ("Q4_K_M", Q4_K_M_BYTES)), _base_repo())
    impact = asyncio.run(companion_cleanup.delete_impact_response(GGUF_REPO))
    assert impact["reclaimed_bytes"] == Q2_K_BYTES + Q4_K_M_BYTES
    assert [f["repo_id"] for f in impact["freeable_companions"]] == [BASE_REPO]


# --------------------------------------------------------------------------------------------
# 3. Shared assets are not removed while another installed model needs them. ADVERSARIAL.
# --------------------------------------------------------------------------------------------


def test_shared_base_cannot_be_deleted_while_a_quant_is_installed(monkeypatch):
    """The whole point of the issue: this delete succeeded before the guard, and silently
    left both installed quants unloadable."""
    _install(monkeypatch, _gguf_repo(("Q2_K", Q2_K_BYTES), ("Q4_K_M", Q4_K_M_BYTES)), _base_repo())
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deletion.delete_cached_model_response(BASE_REPO))
    assert excinfo.value.status_code == 400
    assert GGUF_REPO in excinfo.value.detail
    assert "Delete those models first" in excinfo.value.detail


def test_shared_base_delete_is_still_blocked_by_a_single_remaining_quant(monkeypatch):
    _install(monkeypatch, _gguf_repo(("Q4_K_M", Q4_K_M_BYTES)), _base_repo())
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deletion.delete_cached_model_response(BASE_REPO))
    assert excinfo.value.status_code == 400


def test_shared_base_delete_is_blocked_through_the_ungated_mirror_identity(monkeypatch):
    """The same logical base has two ids. Deleting either must be refused while a quant needs it."""
    gguf = _repo("unsloth/FLUX.1-schnell-GGUF", [("flux1-schnell-Q4_K_M.gguf", Q4_K_M_BYTES)])
    _install(
        monkeypatch,
        gguf,
        _base_repo("unsloth/FLUX.1-schnell"),
        _base_repo("black-forest-labs/FLUX.1-schnell"),
    )
    for identity in ("unsloth/FLUX.1-schnell", "black-forest-labs/FLUX.1-schnell"):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(deletion.delete_cached_model_response(identity))
        assert excinfo.value.status_code == 400, identity


def test_a_base_reached_only_through_a_card_tag_is_still_protected(monkeypatch):
    """A GGUF pick caches no model card, so family detection alone would call this base an
    orphan. The link the resolver recorded at load time is what keeps it protected."""
    gguf = _repo("unsloth/FLUX.2-klein-9B-GGUF", [("flux-2-klein-9b-Q4_K_M.gguf", Q4_K_M_BYTES)])
    other_base = "black-forest-labs/FLUX.2-klein-9B"
    _install(monkeypatch, gguf, _base_repo(other_base))
    # Without the link, the family default points at klein-4B and klein-9B looks reclaimable.
    assert companion_cleanup.companion_dependents(other_base) == []
    companion_assets.record_companion_link("unsloth/FLUX.2-klein-9B-GGUF", other_base)
    assert companion_cleanup.companion_dependents(other_base) == ["unsloth/FLUX.2-klein-9B-GGUF"]
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deletion.delete_cached_model_response(other_base))
    assert excinfo.value.status_code == 400


def test_a_link_for_an_uninstalled_checkpoint_does_not_protect_anything(monkeypatch):
    """Links are filtered by what is installed, so a stale one cannot strand assets forever."""
    companion_assets.record_companion_link(GGUF_REPO, BASE_REPO)
    _install(monkeypatch, _base_repo())
    assert companion_cleanup.companion_dependents(BASE_REPO) == []


def test_delete_of_a_companion_base_fails_closed_when_the_cache_cannot_be_read(monkeypatch):
    """Unable to enumerate dependants is not the same as having none."""
    def _boom():
        raise OSError("cache unreadable")

    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", _boom)
    monkeypatch.setattr(companion_cleanup.cache_inventory, "all_hf_cache_scans", _boom)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deletion.delete_cached_model_response(BASE_REPO))
    assert excinfo.value.status_code == 503


def test_the_guard_leaves_ordinary_repos_alone(monkeypatch):
    """Only a known companion base takes the extra check, so a chat GGUF is untouched by it."""
    assert not companion_assets.is_companion_base("unsloth/Qwen3-8B-GGUF")
    assert companion_assets.is_companion_base(BASE_REPO)


# --------------------------------------------------------------------------------------------
# 4. Orphaned companion assets can be found and removed without hand-editing the HF cache.
# --------------------------------------------------------------------------------------------


def test_orphan_listing_is_empty_while_a_quant_is_installed(monkeypatch):
    _install(monkeypatch, _gguf_repo(("Q4_K_M", Q4_K_M_BYTES)), _base_repo())
    result = asyncio.run(companion_cleanup.orphan_companions_response())
    assert result["companions"] == []
    assert result["total_bytes"] == 0


def test_orphan_listing_reports_the_stranded_companions_once_the_quants_are_gone(monkeypatch):
    _install(monkeypatch, _base_repo())
    result = asyncio.run(companion_cleanup.orphan_companions_response())
    assert [c["repo_id"] for c in result["companions"]] == [BASE_REPO]
    assert result["total_bytes"] == COMPANION_BYTES
    # The repo dir, not the cache root: the delete route resolves the owning cache from it.
    assert (
        result["companions"][0]["cache_path"]
        == "/cache/models--black-forest-labs--FLUX.2-klein-4B"
    )


def test_orphan_listing_never_offers_a_repo_that_holds_a_checkpoint(monkeypatch):
    """A base the user picked as a full pipeline, or one that also holds a GGUF, is a model."""
    base = _base_repo()
    base.revisions[0].files.append(
        _file("flux-2-klein-4b-Q4_K_M.gguf", Q4_K_M_BYTES, snapshot = base.revisions[0].snapshot_path)
    )
    _install(monkeypatch, base)
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_orphan_listing_never_offers_an_unrecognised_repo(monkeypatch):
    """Only curated family bases are candidates, so a mis-recorded link cannot promote a repo."""
    stranger = _repo("someone/private-weights", [("model.safetensors", 1_000)])
    companion_assets.record_companion_link(GGUF_REPO, "someone/private-weights")
    _install(monkeypatch, stranger)
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_a_repo_is_never_recorded_as_its_own_companion(monkeypatch):
    """A full pipeline is its own base. Recording that would make it a dependent of itself and
    block its own deletion forever, with no other model to delete to unblock it."""
    assert companion_assets.record_companion_link(BASE_REPO, BASE_REPO) is False
    assert companion_assets.record_companion_link(BASE_REPO, BASE_REPO.upper()) is False
    assert companion_assets.read_companion_links() == {}
    _install(monkeypatch, _base_repo())
    assert companion_cleanup.companion_dependents(BASE_REPO) == []
    assert [
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    ] == [BASE_REPO]


def test_orphan_listing_never_offers_a_base_installed_as_a_full_pipeline(monkeypatch):
    """Several curated bases run on their own. A companion fetch skips the denoiser folder and
    a pipeline pick takes it, so the denoiser's presence says which one put it on disk."""
    base = _base_repo()
    base.revisions[0].files.append(
        _file(
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            7_000_000_000,
            snapshot = base.revisions[0].snapshot_path,
        )
    )
    _install(monkeypatch, base)
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_recording_a_link_keeps_the_ones_already_there():
    """Every write is a read-modify-write of one shared file, so a second checkpoint's link must
    not erase the first: the erased one would look orphaned and be offered for removal."""
    assert companion_assets.record_companion_link(GGUF_REPO, BASE_REPO) is True
    assert (
        companion_assets.record_companion_link(
            "unsloth/FLUX.2-klein-9B-GGUF", "black-forest-labs/FLUX.2-klein-9B"
        )
        is True
    )
    links = companion_assets.read_companion_links()
    assert links[GGUF_REPO.lower()] == [BASE_REPO]
    assert links["unsloth/flux.2-klein-9b-gguf"] == ["black-forest-labs/FLUX.2-klein-9B"]
    # Re-recording the same pair is a no-op rather than a duplicate.
    assert companion_assets.record_companion_link(GGUF_REPO, BASE_REPO) is False
    assert companion_assets.read_companion_links()[GGUF_REPO.lower()] == [BASE_REPO]
