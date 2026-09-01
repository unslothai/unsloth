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


def _file(
    name: str,
    size: int,
    *,
    snapshot: str = "/cache/snap",
):
    return SimpleNamespace(
        file_name = name.rsplit("/", 1)[-1],
        file_path = f"{snapshot}/{name}",
        blob_path = f"/cache/blobs/{name.replace('/', '_')}",
        size_on_disk = size,
    )


def _repo(
    repo_id: str,
    files,
    *,
    cache: str = "/cache",
):
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




def test_two_quants_of_one_family_resolve_to_a_single_companion_base():
    """Both quants derive the same base id, so the cache holds one copy, not two."""
    scans = [SimpleNamespace(repos = [_gguf_repo(("Q2_K", Q2_K_BYTES), ("Q4_K_M", Q4_K_M_BYTES))])]
    required = companion_assets.required_companion_bases(scans)
    assert BASE_REPO.lower() in required
    assert required[BASE_REPO.lower()] == {GGUF_REPO}




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
    """A GGUF pick caches no model card, so family detection alone answers the family DEFAULT
    base. The checkpoint's own identity names the variant it needs, and the link the resolver
    recorded at load time confirms it."""
    gguf = _repo("unsloth/FLUX.2-klein-9B-GGUF", [("flux-2-klein-9b-Q4_K_M.gguf", Q4_K_M_BYTES)])
    other_base = "black-forest-labs/FLUX.2-klein-9B"
    _install(monkeypatch, gguf, _base_repo(other_base))
    # Before any load has been recorded: the id says klein-9B, so the curated klein-9B base is derived
    # even though the family default is klein-4B.
    assert companion_cleanup.companion_dependents(other_base) == ["unsloth/FLUX.2-klein-9B-GGUF"]
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
        result["companions"][0]["cache_path"] == "/cache/models--black-forest-labs--FLUX.2-klein-4B"
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


def test_orphan_listing_reports_one_row_per_cache(monkeypatch):
    """A delete is scoped to a single cache, so two copies must not be pooled into one row that
    promises bytes one removal cannot deliver."""
    first = _base_repo()
    second = _repo(
        BASE_REPO,
        [("model_index.json", 460), ("vae/diffusion_pytorch_model.safetensors", 900_000)],
        cache = "/other-cache",
    )
    _install(monkeypatch, first, second)
    result = asyncio.run(companion_cleanup.orphan_companions_response())
    assert [c["cache_path"] for c in result["companions"]] == [
        "/cache/models--black-forest-labs--FLUX.2-klein-4B",
        "/other-cache/models--black-forest-labs--FLUX.2-klein-4B",
    ]
    assert [c["size_bytes"] for c in result["companions"]] == [COMPANION_BYTES, 900_460]
    assert result["total_bytes"] == COMPANION_BYTES + 900_460


def test_a_cached_mirror_and_its_upstream_do_not_pin_each_other(monkeypatch):
    """Both identities of one base resolve their own family back to the upstream id. Treating
    that as a dependency would make a cache holding both unable to delete either, forever."""
    _install(
        monkeypatch,
        _base_repo("unsloth/FLUX.1-schnell"),
        _base_repo("black-forest-labs/FLUX.1-schnell"),
    )
    assert companion_cleanup.companion_dependents("unsloth/FLUX.1-schnell") == []
    assert companion_cleanup.companion_dependents("black-forest-labs/FLUX.1-schnell") == []
    offered = [
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    ]
    assert offered == ["black-forest-labs/FLUX.1-schnell", "unsloth/FLUX.1-schnell"]


def test_the_link_trim_keeps_the_newest_not_the_alphabetically_last(monkeypatch):
    """The cap drops the OLDEST recorded checkpoints, and insertion order is the only record of
    which those are. Serialising with sort_keys made the next read alphabetical, so the trim
    evicted the lexicographically smallest instead: a link recorded seconds ago could go, and
    its non-table companion base then looked deletable while the checkpoint was still installed.
    """
    monkeypatch.setattr(companion_assets, "_MAX_LINKS", 3)
    # Recorded newest-last but named so that alphabetical order is the exact reverse.
    for name in ("unsloth/zz-GGUF", "unsloth/mm-GGUF", "unsloth/aa-GGUF"):
        assert companion_assets.record_companion_link(name, BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/zz-gguf",
        "unsloth/mm-gguf",
        "unsloth/aa-gguf",
    ]
    # A fourth evicts the oldest, which is the first recorded, not "aa".
    assert companion_assets.record_companion_link("unsloth/nn-GGUF", BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/mm-gguf",
        "unsloth/aa-gguf",
        "unsloth/nn-gguf",
    ]


def test_recording_a_second_base_refreshes_the_checkpoints_recency(monkeypatch):
    """Recording is the only recency signal there is, so a checkpoint that just resolved must
    move to the newest end rather than keep the position it was first written at."""
    monkeypatch.setattr(companion_assets, "_MAX_LINKS", 2)
    assert companion_assets.record_companion_link("unsloth/first-GGUF", BASE_REPO) is True
    assert companion_assets.record_companion_link("unsloth/second-GGUF", BASE_REPO) is True
    assert companion_assets.record_companion_link("unsloth/first-GGUF", "unsloth/other-base") is (
        True
    )
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/second-gguf",
        "unsloth/first-gguf",
    ]
    assert companion_assets.read_companion_links()["unsloth/first-gguf"] == [
        BASE_REPO,
        "unsloth/other-base",
    ]
    # "second" is the oldest now, so it is what a third checkpoint displaces.
    assert companion_assets.record_companion_link("unsloth/third-GGUF", BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/first-gguf",
        "unsloth/third-gguf",
    ]


def test_freeable_companions_only_names_bases_free_up_space_can_offer(monkeypatch):
    """The delete preview told the user to remove the asset with Free up space, but that list is
    table-only by design (a mis-recorded link must never turn an unrelated repo into a delete
    candidate), so a base reached only through a recorded link was never in it. Advertising an
    action that does nothing is worse than not advertising it."""
    link_only = "some-vendor/private-encoder"
    companion_assets.record_companion_link(GGUF_REPO, link_only)
    _install(
        monkeypatch,
        _gguf_repo(("Q2_K", Q2_K_BYTES)),
        _repo(link_only, [("model_index.json", 460)]),
    )
    impact = asyncio.run(companion_cleanup.delete_impact_response(GGUF_REPO))
    assert [c["repo_id"] for c in impact["freeable_companions"]] == []
    # It is still PROTECTED, which is the half the guard owns.
    assert companion_assets.is_companion_base(link_only) is True
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["repo_id"] for c in offered] == []


def test_reusing_an_existing_link_still_refreshes_its_recency(monkeypatch):
    """A checkpoint reloaded every day but first recorded long ago is the most-used link there
    is. Returning early because the base was already known left it at its original position, so
    the 512-link cap could throw away exactly the link that is in constant use, and for a
    card-tag-only base losing that link is the delete guard going quiet on it."""
    monkeypatch.setattr(companion_assets, "_MAX_LINKS", 2)
    assert companion_assets.record_companion_link("unsloth/old-GGUF", BASE_REPO) is True
    assert companion_assets.record_companion_link("unsloth/new-GGUF", BASE_REPO) is True
    # The old one resolves again to the SAME base: nothing new to record, but it is now the freshest
    # link, so the next checkpoint displaces the other one.
    assert companion_assets.record_companion_link("unsloth/old-GGUF", BASE_REPO) is False
    assert companion_assets.read_companion_links()["unsloth/old-gguf"] == [BASE_REPO]
    assert companion_assets.record_companion_link("unsloth/third-GGUF", BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/old-gguf",
        "unsloth/third-gguf",
    ]


def test_a_cached_community_repack_is_a_companion_identity_too(monkeypatch):
    """prefer_cached_legacy_source deliberately sends the native fetch back to a repack an
    upgraded install already holds, so on those machines the bytes protecting an installed GGUF
    sit under the OLD repo key. Expanding only the upstream/mirror pair left that copy
    unprotected: deletable after unload, and the GGUF stranded."""
    from core.inference.diffusion_families import legacy_source_repo

    mirror = "unsloth/Z-Image-Turbo-ComfyUI"
    repack = legacy_source_repo(mirror)
    assert repack, "the mirror table no longer names a legacy repack; pick another for this test"
    companion_assets.record_companion_link(GGUF_REPO, mirror)
    _install(
        monkeypatch,
        _gguf_repo(("Q2_K", Q2_K_BYTES)),
        _repo(repack, [("model_index.json", 460)]),
    )
    required = companion_assets.required_companion_bases(cache_inventory.all_hf_cache_scans())
    assert GGUF_REPO in required.get(repack.lower(), set())
    assert companion_cleanup.companion_dependents(repack) == [GGUF_REPO]
    # And it is refused as a delete while the GGUF is installed.
    assert companion_assets.is_companion_base(repack) is True


def test_a_native_component_repo_is_offered_once_nothing_needs_it(monkeypatch):
    """For an sd.cpp pick the single-file VAE and text encoder ARE the companions, and the largest
    half of the footprint. Leaving that curated table out of the offerable set made them link-only
    strangers: dropped from the delete preview and never listed by Free up space, so exactly the
    assets this cleanup exists for stayed unreclaimable."""
    from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids

    component = next(iter(sorted(sd_cpp_companion_only_repo_ids())))
    assert component in companion_assets.known_companion_base_ids()
    _install(monkeypatch, _repo(component, [("vae.safetensors", 900_000)]))
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["repo_id"].lower() for c in offered] == [component]


def test_a_chat_model_borrowed_as_an_encoder_is_never_offered(monkeypatch):
    """That table deliberately includes repos that are perfectly good chat models sd.cpp also
    borrows. Offering one as an unused asset would take away a model the user downloaded on
    purpose; holding a GGUF is what keeps it out."""
    borrowed = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    _install(monkeypatch, _repo(borrowed, [("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4_000_000)]))
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert offered == []


def test_a_family_named_only_by_the_cached_filename_still_pins_its_base(monkeypatch):
    """Both loaders detect a family from the filename as well as the repo id, so a runnable repo
    whose name says nothing was invisible to the dependency probe. That matters most on an
    upgraded install, where no links have been recorded yet and the probe is the whole guard."""
    _install(
        monkeypatch,
        _repo("some-owner/custom", [("FLUX.2-klein-4B-Q4_K_M.gguf", 2_000_000)]),
        _base_repo(),
    )
    required = companion_assets.required_companion_bases(cache_inventory.all_hf_cache_scans())
    assert "some-owner/custom" in required.get(BASE_REPO.lower(), set())
    assert companion_cleanup.companion_dependents(BASE_REPO) == ["some-owner/custom"]
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_one_full_copy_does_not_hide_an_orphaned_copy_in_another_cache(monkeypatch):
    """A delete is scoped to one cache root, which is why the listing emits one row per root.
    Pooling the denoiser test across roots let a full pipeline copy in one suppress a
    companion-only copy in another that nothing else can reclaim."""
    full = _repo(
        BASE_REPO,
        [("transformer/diffusion_pytorch_model.safetensors", 5_000_000)],
        cache = "/full-cache",
    )
    orphan = _repo(BASE_REPO, [("vae/diffusion_pytorch_model.safetensors", 900_000)])
    _install(monkeypatch, full, orphan)
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["cache_path"] for c in offered] == [
        "/cache/models--black-forest-labs--FLUX.2-klein-4B"
    ]


def test_a_preexisting_native_gguf_still_protects_its_encoder(monkeypatch):
    """The native engine never reads the diffusers base; it fetches a single-file VAE and text
    encoder from their own repos, and those repos are now offerable for deletion. With no
    recorded link -- an upgraded cache, or state loss -- the derived fallback has to name them,
    or Free up space lists the encoder of an installed checkpoint as an unused asset."""
    from core.inference.diffusion_families import (
        detect_family_for_pick,
        sd_cpp_text_encoders_for,
    )

    gguf = "Qwen-Image-Q4_K_M.gguf"
    fam = detect_family_for_pick("some-owner/custom", gguf)
    assert fam is not None and fam.sd_cpp_vae, "pick a family with a native VAE for this test"
    vae_repo = fam.sd_cpp_vae[0]
    encoder_repo = sd_cpp_text_encoders_for(fam, "some-owner/custom", gguf)[0][0]

    _install(
        monkeypatch,
        _repo("some-owner/custom", [(gguf, 2_000_000)]),
        _repo(vae_repo, [("vae.safetensors", 300_000)]),
        _repo(encoder_repo, [("encoder.gguf", 900_000)]),
    )
    required = companion_assets.required_companion_bases(cache_inventory.all_hf_cache_scans())
    assert "some-owner/custom" in required.get(vae_repo.lower(), set())
    assert "some-owner/custom" in required.get(encoder_repo.lower(), set())
    offered = [
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    ]
    assert vae_repo not in offered and encoder_repo not in offered


def test_deleting_the_companion_quant_itself_runs_the_guard(monkeypatch):
    """Native Qwen-Image opens exactly Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf inside a chat GGUF
    repo, so removing that one quant strands the image checkpoint however many siblings remain:
    none of them is a substitute for a fixed filename. The guard only ran for whole-repo deletes.
    """
    from hub.services.models import deletion

    encoder_repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    _install(
        monkeypatch,
        _repo("some-owner/qwen-image", [("Qwen-Image-Q4_K_M.gguf", 9_000_000)]),
        _repo(
            encoder_repo,
            [
                ("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4_000_000),
                ("Qwen2.5-VL-7B-Instruct-Q8_0.gguf", 8_000_000),
            ],
        ),
    )
    # The exact quant the image load opens is guarded even with a sibling still present.
    assert deletion._variant_is_a_required_companion_asset(encoder_repo, "Q4_K_M") is True
    # The sibling is nobody's asset, so an ordinary variant delete stays unguarded.
    assert deletion._variant_is_a_required_companion_asset(encoder_repo, "Q8_0") is False
    assert deletion._variant_is_a_required_companion_asset("some-owner/unrelated", "Q4_K_M") is (
        False
    )
    # ... and the guard reports the dependant, so the delete is refused rather than silently done.
    assert companion_cleanup.companion_dependents(encoder_repo) == ["some-owner/qwen-image"]


def test_a_gguf_in_a_subdirectory_still_pins_its_base(monkeypatch):
    """``CachedFileInfo.file_name`` is the basename alone. A repo whose id says nothing and whose
    GGUF sits in a subdirectory carries its family only in the directory part, so a basename-only
    scan detected no family and Free up space would offer the base out from under it."""
    _install(
        monkeypatch,
        _repo("some-owner/custom", [("FLUX.2-klein-4B/model-Q4_K_M.gguf", 2_000_000)]),
        _base_repo(),
    )
    required = companion_assets.required_companion_bases(cache_inventory.all_hf_cache_scans())
    assert "some-owner/custom" in required.get(BASE_REPO.lower(), set())
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_orphaned_native_components_do_not_hold_each_other_on_disk(monkeypatch):
    """A component repo is a companion, not a checkpoint. Several of those ids carry a family
    keyword of their own, so the derivation read a bare text-encoder fetch as an installed model
    and recorded the sibling VAE as still required: the pair survived every cleanup pass until the
    user happened to remove one of them by hand."""
    encoder = "unsloth/FLUX.2-dev-ComfyUI"
    vae = "unsloth/FLUX.2-VAE"
    _install(
        monkeypatch,
        _repo(
            encoder, [("split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors", 9_000)]
        ),
        _repo(vae, [("split_files/vae/flux2-vae.safetensors", 300_000)]),
    )
    assert companion_cleanup.companion_dependents(vae) == []
    offered = {
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }
    assert offered == {encoder, vae}


def test_a_component_repo_holding_a_checkpoint_is_still_a_checkpoint(monkeypatch):
    """The exclusion above is denoiser-gated, so a curated component id the user really did
    install a GGUF from keeps pinning what it needs."""
    encoder = "unsloth/FLUX.2-dev-ComfyUI"
    _install(
        monkeypatch,
        _repo(encoder, [("split_files/diffusion_models/flux2-dev-Q4_K_M.gguf", 8_000_000)]),
        _repo("unsloth/FLUX.2-VAE", [("split_files/vae/flux2-vae.safetensors", 300_000)]),
    )
    assert companion_cleanup.companion_dependents("unsloth/FLUX.2-VAE") == [encoder]


def test_a_borrowed_chat_repo_is_never_advertised_as_freeable(monkeypatch):
    """The delete preview must point only at rows Free up space will really show. A borrowed chat
    GGUF repo is a curated companion id but holds a denoiser, so the orphan listing skips it;
    naming it here sent the user to remove something that is never on that list."""
    borrowed = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    checkpoint = "some-owner/qwen-image"
    _install(
        monkeypatch,
        _repo(checkpoint, [("Qwen-Image-Q4_K_M.gguf", 9_000_000)]),
        _repo(borrowed, [("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4_000_000)]),
    )
    impact = asyncio.run(companion_cleanup.delete_impact_response(checkpoint))
    assert borrowed not in {c["repo_id"] for c in impact["freeable_companions"]}
    offered = {
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }
    assert borrowed not in offered


def test_an_sdxl_single_file_install_is_never_offered_as_a_leftover(monkeypatch):
    """A single_file_is_pipeline family caches its whole checkpoint as ONE top-level file, so
    there is no denoiser folder to recognise. Its repo is also a curated companion base whose
    self-dependency is dropped, so the listing called an installed SDXL an unused leftover."""
    sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
    _install(monkeypatch, _repo(sdxl, [("sd_xl_base_1.0.safetensors", 6_900_000_000)]))
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_a_variant_named_by_an_alias_spelling_still_pins_its_base(monkeypatch):
    """Family detection accepts ``flux1-dev`` for the flux.1 family, so the curated name
    ``FLUX.1-dev`` has to be matched with punctuation folded away. One dot apart, the dev base was
    offerable while the dev checkpoint was installed."""
    dev = "black-forest-labs/FLUX.1-dev"
    _install(
        monkeypatch,
        _repo("some-owner/custom", [("flux1-dev-Q4_K_M.gguf", 2_000_000)]),
        _repo(dev, [("model_index.json", 460), ("text_encoder/m.safetensors", 9_000)]),
    )
    assert companion_cleanup.companion_dependents(dev) == ["some-owner/custom"]
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_every_cached_gguf_in_one_repo_pins_its_own_base(monkeypatch):
    """One generic repo can hold checkpoints of two families in separate subdirectories, and the
    loader selects either by file name. Probing only the first left the other's base orphaned."""
    klein = "black-forest-labs/FLUX.2-klein-4B"
    dev = "black-forest-labs/FLUX.2-dev"
    _install(
        monkeypatch,
        _repo(
            "some-owner/multi",
            [("FLUX.2-klein-4B/a-Q4_K_M.gguf", 2_000_000), ("FLUX.2-dev/b-Q4_K_M.gguf", 3_000_000)],
        ),
        _repo(klein, [("model_index.json", 460)]),
        _repo(dev, [("model_index.json", 460)]),
    )
    assert companion_cleanup.companion_dependents(klein) == ["some-owner/multi"]
    assert companion_cleanup.companion_dependents(dev) == ["some-owner/multi"]


def test_copies_in_two_cache_roots_are_probed_together(monkeypatch):
    """A remembered second cache holds its own copy of the same repo id. The probe kept the first
    copy's names and skipped the rest, so a family only the second copy names lost its base."""
    klein = "black-forest-labs/FLUX.2-klein-4B"
    qwen = "Qwen/Qwen-Image"
    scans = [
        SimpleNamespace(
            repos = [
                _repo(
                    "some-owner/custom", [("FLUX.2-klein-4B-Q4_K_M.gguf", 2_000_000)], cache = "/c1"
                )
            ]
        ),
        SimpleNamespace(
            repos = [_repo("some-owner/custom", [("Qwen-Image-Q4_K_M.gguf", 9_000_000)], cache = "/c2")]
        ),
        SimpleNamespace(
            repos = [
                _repo(klein, [("model_index.json", 460)]),
                _repo(qwen, [("model_index.json", 460)]),
            ]
        ),
    ]
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: scans)
    monkeypatch.setattr(companion_cleanup.cache_inventory, "all_hf_cache_scans", lambda: scans)
    assert companion_cleanup.companion_dependents(klein) == ["some-owner/custom"]
    assert companion_cleanup.companion_dependents(qwen) == ["some-owner/custom"]


def test_a_single_file_safetensors_pick_pins_its_base(monkeypatch):
    """The loader hands a top-level single-file .safetensors to detect_family_for_pick exactly
    like a GGUF, so a cache holding only that file has a family. Probing GGUFs alone left the
    checkpoint's base reclaimable, and deleting it broke the installed pick offline."""
    dev = "black-forest-labs/FLUX.2-dev"
    _install(
        monkeypatch,
        _repo("unsloth/custom", [("FLUX.2-dev-fp8.safetensors", 9_000_000)]),
        _repo(dev, [("model_index.json", 460)]),
    )
    assert companion_cleanup.companion_dependents(dev) == ["unsloth/custom"]
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_a_legacy_component_repack_is_not_read_as_a_checkpoint(monkeypatch):
    """An upgraded install holds the component under the repack id the native fetch fell back to.
    That id matches its family just as literally, so without expanding the exclusion through the
    same mirror identities the orphaned pair kept holding each other on disk."""
    legacy = "Comfy-Org/flux2-dev"
    vae = "unsloth/FLUX.2-VAE"
    _install(
        monkeypatch,
        _repo(
            legacy, [("split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors", 9_000)]
        ),
        _repo(vae, [("split_files/vae/flux2-vae.safetensors", 300_000)]),
    )
    assert companion_cleanup.companion_dependents(vae) == []
    offered = {
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }
    assert offered == {legacy, vae}


def test_free_up_space_refuses_a_row_that_became_an_installed_model(monkeypatch):
    """The listing can be minutes old. A download of the same repo finishing in the background
    turns an orphaned companion into an installed checkpoint, and neither existing guard sees it:
    begin_delete only refuses a download still in flight, and the companion guard ignores the
    target as its own dependent. Remove would then delete the model the user just downloaded."""
    from hub.services.models import deletion

    base = BASE_REPO
    # The companion-only copy Free up space listed, now carrying a downloaded denoiser.
    _install(monkeypatch, _repo(base, [("transformer/diffusion_pytorch_model.safetensors", 9_000)]))
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deletion.delete_cached_model_response(base, only_if_orphan = True))
    assert excinfo.value.status_code == 409
    assert "no longer an unused asset" in excinfo.value.detail


def test_the_orphan_precondition_lets_a_real_orphan_through(monkeypatch):
    """The precondition only refuses; a row that is still companion-only reaches the delete, and
    it must not become a second way for an ordinary delete to fail."""
    from hub.services.models import deletion

    _install(monkeypatch, _repo(BASE_REPO, [("model_index.json", 460)]))
    try:
        deletion._delete_cached_model_blocking(BASE_REPO, None, None, only_if_orphan = True)
    except HTTPException as exc:
        assert exc.status_code != 409, exc.detail


def test_a_transformer_only_single_file_in_its_own_base_repo_is_a_checkpoint(monkeypatch):
    """Not only whole-pipeline families cache a root weight. A transformer-only single_file pick
    of a curated base repo does too, and its self-dependency is dropped, so the listing called an
    installed checkpoint a leftover and the orphan precondition would happily delete it."""
    from hub.services.models import deletion

    dev = "black-forest-labs/FLUX.2-dev"
    _install(monkeypatch, _repo(dev, [("flux2-dev-fp8.safetensors", 9_000_000)]))
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(dev, None, None, only_if_orphan = True)
    assert excinfo.value.status_code == 409


def test_the_orphan_precondition_is_scoped_to_the_cache_being_deleted(monkeypatch):
    """The listing emits one row per cache root because a delete is scoped to one. A full pipeline
    copy in another remembered cache must not veto removing the companion-only copy that was
    listed, or that row 409s forever."""
    from hub.services.models import deletion

    dev = "black-forest-labs/FLUX.2-dev"
    _install(
        monkeypatch,
        _repo(dev, [("transformer/diffusion_pytorch_model.safetensors", 9_000)], cache = "/c1"),
        _repo(dev, [("model_index.json", 460)], cache = "/c2"),
    )
    rows = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [r["cache_path"] for r in rows] == ["/c2/models--black-forest-labs--FLUX.2-dev"]
    try:
        deletion._delete_cached_model_blocking(
            dev, None, None, rows[0]["cache_path"], only_if_orphan = True
        )
    except HTTPException as exc:
        assert exc.status_code != 409, exc.detail


def test_one_checkpoints_recorded_bases_are_bounded(monkeypatch):
    """_MAX_LINKS counts checkpoints, not their bases, so a checkpoint resolved against a new
    explicit base each load grew the state file on its own, and every delete check parses and
    mirror-expands the whole list."""
    for i in range(companion_assets._MAX_BASES_PER_CHECKPOINT + 5):
        companion_assets.record_companion_link(GGUF_REPO, f"some-vendor/base-{i}")
    bases = companion_assets.read_companion_links()[GGUF_REPO.lower()]
    assert len(bases) == companion_assets._MAX_BASES_PER_CHECKPOINT
    # The most recent survive; the oldest go.
    assert bases[-1] == f"some-vendor/base-{companion_assets._MAX_BASES_PER_CHECKPOINT + 4}"
    assert "some-vendor/base-0" not in bases


def test_re_resolving_a_base_refreshes_its_place_in_the_list(monkeypatch):
    """The cap is by recency, so a base still in use must not be evicted ahead of one nothing has
    resolved since."""
    companion_assets.record_companion_link(GGUF_REPO, "some-vendor/a")
    companion_assets.record_companion_link(GGUF_REPO, "some-vendor/b")
    companion_assets.record_companion_link(GGUF_REPO, "some-vendor/a")
    assert companion_assets.read_companion_links()[GGUF_REPO.lower()] == [
        "some-vendor/b",
        "some-vendor/a",
    ]


def test_the_orphan_precondition_refuses_when_the_target_root_is_not_in_the_scan(monkeypatch):
    """An empty scoped match means the target root was not scanned, and the delete below can
    still purge that directory by path. Concluding "orphan" from copies we did not look at is the
    fail-open this precondition exists to prevent."""
    from hub.services.models import deletion

    dev = "black-forest-labs/FLUX.2-dev"
    _install(monkeypatch, _repo(dev, [("model_index.json", 460)], cache = "/c1"))
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(
            dev, None, None, "/unscanned/models--black-forest-labs--FLUX.2-dev", only_if_orphan = True
        )
    assert excinfo.value.status_code == 503


def test_the_preview_reads_a_path_qualified_variant(monkeypatch):
    """The inventory and the delete identify a row by gguf_variant_key, which for a checkpoint in
    its own directory is not the bare quant label. Comparing labels reported 0 B reclaimed and
    described the last checkpoint's companions as retained rather than freed."""
    repo = "unsloth/LTX-2.3-22B-GGUF"
    variant = "distilled/ltx-2.3-22b-distilled-Q6_K"
    _install(
        monkeypatch,
        _repo(repo, [("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", 6_000_000)]),
    )
    impact = asyncio.run(companion_cleanup.delete_impact_response(repo, variant))
    assert impact["reclaimed_bytes"] == 6_000_000


def test_the_curated_bases_carry_the_whole_mirror_table():
    """Gated and ungated alike, both sides of every pair.

    Gating decides whether a fetch may override a user's cache; it has nothing to do with whether
    a base can strand an installed checkpoint's companions, and most of the table is ungated.
    Reading only the gated half dropped Klein 4B and base-4B, HiDream Dev / Fast, SDXL Turbo and
    the Qwen bases from the curated ids, so before a companion link is recorded the guards no
    longer recognised those cached bases.
    """
    from core.inference.diffusion_families import _MIRROR_PAIRS

    curated = companion_assets._curated_base_ids()
    missing = [rid for pair in _MIRROR_PAIRS for rid in pair if rid not in curated]
    assert missing == []
