# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Companion-asset reuse, preflight and deletion when several GGUF quants share one cached base."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.services.models import cache_inventory, companion_cleanup, deletion
from hub.utils import companion_assets

GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
# Real byte counts measured from a cache holding this pair.
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
    """A base cannot be deleted while its quants are installed, or both are left unloadable."""
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
    """A GGUF caches no model card, so family alone gives the default base; the link confirms it."""
    gguf = _repo("unsloth/FLUX.2-klein-9B-GGUF", [("flux-2-klein-9b-Q4_K_M.gguf", Q4_K_M_BYTES)])
    other_base = "black-forest-labs/FLUX.2-klein-9B"
    _install(monkeypatch, gguf, _base_repo(other_base))
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
    """Only a known companion base takes the extra check, so a chat GGUF is untouched by it.

    A chat GGUF sd.cpp borrows as a text encoder (unsloth/Qwen3-8B-GGUF for FLUX.2-klein 9B, like
    unsloth/Qwen2.5-VL-7B-Instruct-GGUF for Qwen-Image) does take it, so an installed image
    checkpoint that reads it keeps it; one nobody borrows does not."""
    assert not companion_assets.is_companion_base("unsloth/Qwen3-14B-GGUF")
    assert companion_assets.is_companion_base(BASE_REPO)
    for borrowed in ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "unsloth/Qwen3-8B-GGUF"):
        assert companion_assets.is_companion_base(borrowed), borrowed


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
    assert result["companions"][0]["cache_path"] == str(
        Path("/cache/models--black-forest-labs--FLUX.2-klein-4B")
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
    assert companion_assets.record_companion_link(GGUF_REPO, BASE_REPO) is False
    assert companion_assets.read_companion_links()[GGUF_REPO.lower()] == [BASE_REPO]


def test_orphan_listing_reports_one_row_per_cache(monkeypatch):
    """Copies in two caches stay separate rows, since a delete is scoped to one cache."""
    first = _base_repo()
    second = _repo(
        BASE_REPO,
        [("model_index.json", 460), ("vae/diffusion_pytorch_model.safetensors", 900_000)],
        cache = "/other-cache",
    )
    _install(monkeypatch, first, second)
    result = asyncio.run(companion_cleanup.orphan_companions_response())
    assert [c["cache_path"] for c in result["companions"]] == [
        str(Path("/cache/models--black-forest-labs--FLUX.2-klein-4B")),
        str(Path("/other-cache/models--black-forest-labs--FLUX.2-klein-4B")),
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
    """Link cap drops oldest by insertion order; sort_keys would evict the alphabetically smallest."""
    monkeypatch.setattr(companion_assets, "_MAX_LINKS", 3)
    for name in ("unsloth/zz-GGUF", "unsloth/mm-GGUF", "unsloth/aa-GGUF"):
        assert companion_assets.record_companion_link(name, BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/zz-gguf",
        "unsloth/mm-gguf",
        "unsloth/aa-gguf",
    ]
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
    assert companion_assets.record_companion_link("unsloth/third-GGUF", BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/first-gguf",
        "unsloth/third-gguf",
    ]


def test_freeable_companions_only_names_bases_free_up_space_can_offer(monkeypatch):
    """Delete preview only names bases Free up space can offer; link-only bases must not appear there."""
    link_only = "some-vendor/private-encoder"
    companion_assets.record_companion_link(GGUF_REPO, link_only)
    _install(
        monkeypatch,
        _gguf_repo(("Q2_K", Q2_K_BYTES)),
        _repo(link_only, [("model_index.json", 460)]),
    )
    impact = asyncio.run(companion_cleanup.delete_impact_response(GGUF_REPO))
    assert [c["repo_id"] for c in impact["freeable_companions"]] == []
    assert companion_assets.is_companion_base(link_only) is True
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["repo_id"] for c in offered] == []


def test_reusing_an_existing_link_still_refreshes_its_recency(monkeypatch):
    """Re-recording an existing link refreshes its recency so the link cap never evicts a link in use."""
    monkeypatch.setattr(companion_assets, "_MAX_LINKS", 2)
    assert companion_assets.record_companion_link("unsloth/old-GGUF", BASE_REPO) is True
    assert companion_assets.record_companion_link("unsloth/new-GGUF", BASE_REPO) is True
    assert companion_assets.record_companion_link("unsloth/old-GGUF", BASE_REPO) is False
    assert companion_assets.read_companion_links()["unsloth/old-gguf"] == [BASE_REPO]
    assert companion_assets.record_companion_link("unsloth/third-GGUF", BASE_REPO) is True
    assert list(companion_assets.read_companion_links()) == [
        "unsloth/old-gguf",
        "unsloth/third-gguf",
    ]


def test_a_cached_community_repack_is_a_companion_identity_too(monkeypatch):
    """A cached legacy repack must count as a companion id, or deleting it strands the GGUF."""
    from core.inference.diffusion_families import legacy_source_repo

    repack, mirror = _synthetic_legacy_row(monkeypatch)
    assert legacy_source_repo(mirror) == repack
    companion_assets.record_companion_link(GGUF_REPO, mirror)
    _install(
        monkeypatch,
        _gguf_repo(("Q2_K", Q2_K_BYTES)),
        _repo(repack, [("model_index.json", 460)]),
    )
    required = companion_assets.required_companion_bases(cache_inventory.all_hf_cache_scans())
    assert GGUF_REPO in required.get(repack.lower(), set())
    assert companion_cleanup.companion_dependents(repack) == [GGUF_REPO]
    assert companion_assets.is_companion_base(repack) is True


def test_a_native_component_repo_is_offered_once_nothing_needs_it(monkeypatch):
    """Native sd.cpp VAE and text encoder repos are companions, so they must stay offerable for cleanup."""
    from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids

    component = next(iter(sorted(sd_cpp_companion_only_repo_ids())))
    assert component in companion_assets.known_companion_base_ids()
    _install(monkeypatch, _repo(component, [("vae.safetensors", 900_000)]))
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["repo_id"].lower() for c in offered] == [component]


def test_a_chat_model_borrowed_as_an_encoder_is_never_offered(monkeypatch):
    """Offering it as unused would take away a chat model the user chose; a held GGUF keeps it out."""
    borrowed = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    _install(monkeypatch, _repo(borrowed, [("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4_000_000)]))
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert offered == []


def test_a_family_named_only_by_the_cached_filename_still_pins_its_base(monkeypatch):
    """Family is read from the filename too, so a repo whose name says nothing must still pin its base."""
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
    """The denoiser test runs per cache root; a full copy in one must not hide an orphan copy elsewhere."""
    full = _repo(
        BASE_REPO,
        [("transformer/diffusion_pytorch_model.safetensors", 5_000_000)],
        cache = "/full-cache",
    )
    orphan = _repo(BASE_REPO, [("vae/diffusion_pytorch_model.safetensors", 900_000)])
    _install(monkeypatch, full, orphan)
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["cache_path"] for c in offered] == [
        str(Path("/cache/models--black-forest-labs--FLUX.2-klein-4B"))
    ]


def test_a_preexisting_native_gguf_still_protects_its_encoder(monkeypatch):
    """No recorded link: the derived fallback must still name a native GGUF's encoder or it looks unused."""
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
    """Deleting the companion quant itself must run the guard: its fixed filename has no substitute."""
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
    assert deletion._variant_is_a_required_companion_asset(encoder_repo, "Q4_K_M") is True
    assert deletion._variant_is_a_required_companion_asset(encoder_repo, "Q8_0") is False
    assert deletion._variant_is_a_required_companion_asset("some-owner/unrelated", "Q4_K_M") is (
        False
    )
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


_EXAMPLE_ENCODER = "unsloth/FLUX.2-dev-example-encoder"
_EXAMPLE_VAE = "unsloth/FLUX.2-dev-example-vae"
_EXAMPLE_REPACK = "some-org/FLUX.2-dev-example-repack"


def _synthetic_legacy_row(
    monkeypatch,
    mirror = "unsloth/example-repack-mirror",
    repack = "some-org/example-repack",
):
    """(repack, mirror) after adding a made-up mirror -> community repack row to the legacy table."""
    from core.inference import diffusion_families

    monkeypatch.setitem(diffusion_families._SD_CPP_LEGACY_SOURCES, mirror.lower(), repack)
    return repack, mirror


def _synthetic_flux2_dev_components(monkeypatch):
    """Points flux.2-dev's VAE and encoder at synthetic repos whose ids carry the family keyword."""
    import dataclasses

    from core.inference import diffusion_families

    families = tuple(
        dataclasses.replace(
            fam,
            sd_cpp_vae = (_EXAMPLE_VAE, "vae.safetensors"),
            sd_cpp_text_encoders = ((_EXAMPLE_ENCODER, "mistral.safetensors", "llm"),),
        )
        if fam.name == "flux.2-dev"
        else fam
        for fam in diffusion_families._FAMILIES
    )
    monkeypatch.setattr(diffusion_families, "_FAMILIES", families)
    companions = diffusion_families.sd_cpp_companion_only_repo_ids()
    assert {_EXAMPLE_ENCODER.lower(), _EXAMPLE_VAE.lower()} <= companions


def test_orphaned_native_components_do_not_hold_each_other_on_disk(monkeypatch):
    """Component repos are companions, not checkpoints: orphaned encoder and VAE must not pin each other."""
    _synthetic_flux2_dev_components(monkeypatch)
    encoder, vae = _EXAMPLE_ENCODER, _EXAMPLE_VAE
    _install(
        monkeypatch,
        _repo(encoder, [("mistral.safetensors", 9_000)]),
        _repo(vae, [("vae.safetensors", 300_000)]),
    )
    assert companion_cleanup.companion_dependents(vae) == []
    offered = {
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }
    assert offered == {encoder, vae}


def test_a_component_repo_holding_a_checkpoint_is_still_a_checkpoint(monkeypatch):
    """Denoiser-gated exclusion: a curated id with an installed GGUF still pins what it needs."""
    _synthetic_flux2_dev_components(monkeypatch)
    encoder, vae = _EXAMPLE_ENCODER, _EXAMPLE_VAE
    _install(
        monkeypatch,
        _repo(encoder, [("flux2-dev-Q4_K_M.gguf", 8_000_000)]),
        _repo(vae, [("vae.safetensors", 300_000)]),
    )
    assert companion_cleanup.companion_dependents(vae) == [encoder]


def test_a_vae_only_prequant_fetch_is_not_a_dependent(monkeypatch):
    """#11825: a VAE-only fetch from a prequant repo must not pin the base."""
    gguf = "unsloth/Qwen-Image-2.1-GGUF"
    prequant = "unsloth/Qwen-Image-2.1-FP8"
    base = "Qwen/Qwen-Image-2.1"
    _install(
        monkeypatch,
        _repo(gguf, [("qwen-image-2.1-Q4_K_M.gguf", 4_199_565_024)]),
        _repo(prequant, [("vae/qwen_image_2.1_vae_bf16.safetensors", 680_000_000)]),
        _base_repo(base),
    )
    assert companion_cleanup.companion_dependents(base) == [gguf]
    assert companion_cleanup.companion_dependents(base, ignore_repo_ids = [gguf]) == []
    impact = asyncio.run(companion_cleanup.delete_impact_response(base))
    assert impact["blocked_by"] == [gguf]


def test_a_speed_loaded_base_and_its_hosted_checkpoint_are_one_install(monkeypatch):
    """The base stays out of Free up space, yet its own delete is not refused by the hosted repo."""
    prequant = "unsloth/Qwen-Image-2.1-FP8"
    base = "Qwen/Qwen-Image-2.1"
    _install(
        monkeypatch,
        _repo(
            prequant,
            [
                ("Qwen-Image-2.1-FP8.safetensors", 7_120_000_000),
                ("vae/qwen_image_2.1_vae_bf16.safetensors", 680_000_000),
            ],
        ),
        _base_repo(base),
    )
    assert companion_cleanup.companion_dependents(base, ignore_repo_ids = [base]) == []
    assert companion_cleanup.companion_dependents(prequant) == [base]
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert offered == []
    impact = asyncio.run(companion_cleanup.delete_impact_response(base))
    assert impact["blocked_by"] == []
    assert [row["repo_id"] for row in impact["freeable_companions"]] == [prequant]


def test_a_pre_cast_encoder_fetch_is_not_a_dependent(monkeypatch):
    """Qwen-Image-2.1 defaults to the pre-cast fp8 encoder, fetched to the prequant repo's root."""
    prequant = "unsloth/Qwen-Image-2.1-FP8"
    base = "Qwen/Qwen-Image-2.1"
    _install(
        monkeypatch,
        _repo(prequant, [("Qwen-Image-2.1-text_encoder-FP8.safetensors", 9_400_000_000)]),
        _base_repo(base),
    )
    assert companion_cleanup.companion_dependents(base) == []
    impact = asyncio.run(companion_cleanup.delete_impact_response(base))
    assert impact["blocked_by"] == []


def test_a_hosted_checkpoint_and_encoder_are_freed_with_the_last_dependent(monkeypatch):
    prequant = "unsloth/Qwen-Image-2.1-FP8"
    base = "Qwen/Qwen-Image-2.1"
    _install(
        monkeypatch,
        _repo(
            prequant,
            [
                ("Qwen-Image-2.1-FP8.safetensors", 7_120_000_000),
                ("Qwen-Image-2.1-text_encoder-FP8.safetensors", 9_400_000_000),
            ],
        ),
        _base_repo(base),
    )
    assert companion_cleanup.companion_dependents(prequant) == [base]
    impact = asyncio.run(companion_cleanup.delete_impact_response(base))
    assert impact["freeable_companions"] == [
        {"repo_id": prequant, "size_bytes": 16_520_000_000, "needed_by": []}
    ]


def test_a_borrowed_chat_repo_is_never_advertised_as_freeable(monkeypatch):
    """A delete preview must name only rows Free up space shows; a borrowed chat GGUF is never listed."""
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
    """single_file_is_pipeline checkpoints are one top-level file; an installed SDXL is not an orphan."""
    sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
    _install(monkeypatch, _repo(sdxl, [("sd_xl_base_1.0.safetensors", 6_900_000_000)]))
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_a_variant_named_by_an_alias_spelling_still_pins_its_base(monkeypatch):
    """Punctuation is folded so flux1-dev and FLUX.1-dev match; a dot apart left the base offerable."""
    dev = "black-forest-labs/FLUX.1-dev"
    _install(
        monkeypatch,
        _repo("some-owner/custom", [("flux1-dev-Q4_K_M.gguf", 2_000_000)]),
        _repo(dev, [("model_index.json", 460), ("text_encoder/m.safetensors", 9_000)]),
    )
    assert companion_cleanup.companion_dependents(dev) == ["some-owner/custom"]
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_a_turbo_gguf_pins_the_turbo_index_without_a_recorded_link(monkeypatch):
    """A native Turbo load reads Qwen/Qwen-Image-2.1-Turbo's model_index.json for its sampling grid,
    so that cached index has a dependent while the GGUF is installed, link or no link."""
    turbo = "Qwen/Qwen-Image-2.1-Turbo"
    _install(
        monkeypatch,
        _repo(
            "someone/Qwen-Image-2.1-Turbo-GGUF", [("qwen_image_2.1_turbo_Q4_K_M.gguf", 2_000_000)]
        ),
        _repo(turbo, [("model_index.json", 460)]),
    )
    assert companion_cleanup.companion_dependents(turbo) == ["someone/Qwen-Image-2.1-Turbo-GGUF"]
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
    """Single-file .safetensors picks get a family like GGUFs, so their base must be pinned too."""
    dev = "black-forest-labs/FLUX.2-dev"
    _install(
        monkeypatch,
        _repo("unsloth/custom", [("FLUX.2-dev-fp8.safetensors", 9_000_000)]),
        _repo(dev, [("model_index.json", 460)]),
    )
    assert companion_cleanup.companion_dependents(dev) == ["unsloth/custom"]
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []


def test_a_legacy_component_repack_is_not_read_as_a_checkpoint(monkeypatch):
    """Upgraded installs hold components under the repack id, so the exclusion must expand mirror ids."""
    _synthetic_flux2_dev_components(monkeypatch)
    legacy, _mirror = _synthetic_legacy_row(monkeypatch, _EXAMPLE_ENCODER, _EXAMPLE_REPACK)
    vae = _EXAMPLE_VAE
    _install(
        monkeypatch,
        _repo(legacy, [("mistral.safetensors", 9_000)]),
        _repo(vae, [("vae.safetensors", 300_000)]),
    )
    assert companion_cleanup.companion_dependents(vae) == []
    offered = {
        c["repo_id"]
        for c in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }
    assert offered == {legacy, vae}


def test_free_up_space_refuses_a_row_that_became_an_installed_model(monkeypatch):
    """Free up space must refuse a row that became an installed model after the listing was taken."""
    from hub.services.models import deletion

    base = BASE_REPO
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
    """Transformer-only single_file picks of a base repo cache a root weight, so they are checkpoints."""
    from hub.services.models import deletion

    dev = "black-forest-labs/FLUX.2-dev"
    _install(monkeypatch, _repo(dev, [("flux2-dev-fp8.safetensors", 9_000_000)]))
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(dev, None, None, only_if_orphan = True)
    assert excinfo.value.status_code == 409


def test_the_orphan_precondition_is_scoped_to_the_cache_being_deleted(monkeypatch):
    """A full copy in another cache must not veto deleting the listed companion-only row, or it 409s."""
    from hub.services.models import deletion

    dev = "black-forest-labs/FLUX.2-dev"
    _install(
        monkeypatch,
        _repo(dev, [("transformer/diffusion_pytorch_model.safetensors", 9_000)], cache = "/c1"),
        _repo(dev, [("model_index.json", 460)], cache = "/c2"),
    )
    rows = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [r["cache_path"] for r in rows] == [
        str(Path("/c2/models--black-forest-labs--FLUX.2-dev"))
    ]
    try:
        deletion._delete_cached_model_blocking(
            dev, None, None, rows[0]["cache_path"], only_if_orphan = True
        )
    except HTTPException as exc:
        assert exc.status_code != 409, exc.detail


def test_one_checkpoints_recorded_bases_are_bounded(monkeypatch):
    """_MAX_LINKS counts checkpoints, not bases; a checkpoint resolved against a new base each load
    grew the state file, so the bases of one checkpoint are capped separately"""
    for i in range(companion_assets._MAX_BASES_PER_CHECKPOINT + 5):
        companion_assets.record_companion_link(GGUF_REPO, f"some-vendor/base-{i}")
    bases = companion_assets.read_companion_links()[GGUF_REPO.lower()]
    assert len(bases) == companion_assets._MAX_BASES_PER_CHECKPOINT
    assert bases[-1] == f"some-vendor/base-{companion_assets._MAX_BASES_PER_CHECKPOINT + 4}"
    assert "some-vendor/base-0" not in bases


def test_re_resolving_a_base_refreshes_its_place_in_the_list(monkeypatch):
    """Eviction is by recency, so re-resolving a base refreshes its place in the cap."""
    companion_assets.record_companion_link(GGUF_REPO, "some-vendor/a")
    companion_assets.record_companion_link(GGUF_REPO, "some-vendor/b")
    companion_assets.record_companion_link(GGUF_REPO, "some-vendor/a")
    assert companion_assets.read_companion_links()[GGUF_REPO.lower()] == [
        "some-vendor/b",
        "some-vendor/a",
    ]


def test_the_orphan_precondition_refuses_when_the_target_root_is_not_in_the_scan(monkeypatch):
    """An empty scoped match means the root was never scanned; refuse rather than call it an orphan."""
    from hub.services.models import deletion

    dev = "black-forest-labs/FLUX.2-dev"
    _install(monkeypatch, _repo(dev, [("model_index.json", 460)], cache = "/c1"))
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(
            dev, None, None, "/unscanned/models--black-forest-labs--FLUX.2-dev", only_if_orphan = True
        )
    assert excinfo.value.status_code == 503


def test_the_preview_reads_a_path_qualified_variant(monkeypatch):
    """Rows are keyed by gguf_variant_key; for a checkpoint in its own directory that is not the
    bare quant label, or the preview reports 0 B reclaimed"""
    repo = "unsloth/LTX-2.3-22B-GGUF"
    variant = "distilled/ltx-2.3-22b-distilled-Q6_K"
    _install(
        monkeypatch,
        _repo(repo, [("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", 6_000_000)]),
    )
    impact = asyncio.run(companion_cleanup.delete_impact_response(repo, variant))
    assert impact["reclaimed_bytes"] == 6_000_000


def test_the_curated_bases_carry_the_whole_mirror_table():
    """Curated base ids must cover both sides of every mirror pair, gated or not, for companion guards."""
    from core.inference.diffusion_families import _MIRROR_PAIRS

    curated = companion_assets._curated_base_ids()
    missing = [rid for pair in _MIRROR_PAIRS for rid in pair if rid not in curated]
    assert missing == []


# --------------------------------------------------------------------------------------------
# Mirror targets that are also a model's own GGUF repo (unsloth/Z-Image-GGUF holds the official Tongyi-MAI/Z-Image
# diffusers folders beside its quants).


def test_a_turbo_checkpoint_does_not_depend_on_the_base_whose_name_it_contains():
    """ "zimage" is inside "zimageturbo": a substring match made a Turbo GGUF depend on Tongyi-MAI/Z-Image too, whose
    mirror is now the Z-Image GGUF repo, so deleting that unrelated repo was refused while Turbo was installed."""
    scans = [
        SimpleNamespace(
            repos = [_repo("unsloth/Z-Image-Turbo-GGUF", [("z-image-turbo-Q4_K_M.gguf", 5_000)])]
        )
    ]
    required = companion_assets.required_companion_bases(scans)
    assert "tongyi-mai/z-image-turbo" in required
    assert "tongyi-mai/z-image" not in required
    assert "unsloth/z-image-gguf" not in required
    scans = [
        SimpleNamespace(
            repos = [
                _repo(
                    "unsloth/Qwen-Image-Edit-2511-GGUF",
                    [("qwen-image-edit-2511-Q4_K_M.gguf", 5_000)],
                )
            ]
        )
    ]
    required = companion_assets.required_companion_bases(scans)
    assert "qwen/qwen-image-edit-2511" in required
    assert "qwen/qwen-image-edit" not in required
    assert "unsloth/qwen-image-edit-gguf" not in required


def test_an_artifact_mirror_checkpoint_still_protects_its_cached_upstream_base():
    """The GGUF repo canonicalises to its upstream now, but it is a checkpoint in its own right: a cached upstream
    base it loads from must keep it as a dependent, not be folded into the GGUF's own identity."""
    scans = [
        SimpleNamespace(
            repos = [
                _repo("unsloth/Z-Image-GGUF", [("z-image-Q4_K_M.gguf", 5_000)]),
                _base_repo("Tongyi-MAI/Z-Image"),
            ]
        )
    ]
    required = companion_assets.required_companion_bases(scans)
    assert required.get("tongyi-mai/z-image") == {"unsloth/Z-Image-GGUF"}


@pytest.mark.parametrize(
    "retired", ["unsloth/Z-Image-Turbo-ComfyUI", "unsloth/flux-text-encoders", "unsloth/FLUX.2-VAE"]
)
def test_a_retired_comfy_companion_left_by_a_released_build_is_offered_for_cleanup(
    monkeypatch, retired
):
    """Retired ComfyUI repacks stay companion ids, hidden from pickers and offered by Free up space."""
    from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids

    assert retired.lower() in sd_cpp_companion_only_repo_ids()
    assert retired.lower() in companion_assets.known_companion_base_ids()
    _install(monkeypatch, _repo(retired, [("split_files/vae/ae.safetensors", 900_000)]))
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert [c["repo_id"].lower() for c in offered] == [retired.lower()]


def test_a_link_to_a_retired_sd_cli_repack_no_longer_pins_it(monkeypatch):
    """Stale sd-cli repack links must not pin their repos; the next write drops them from the file."""
    gguf = "unsloth/Z-Image-Turbo-GGUF"
    retired = "unsloth/Z-Image-Turbo-ComfyUI"
    repack = "Comfy-Org/z_image_turbo"
    for stale in (retired, repack, "Tongyi-MAI/Z-Image-Turbo"):
        assert companion_assets.record_companion_link(gguf, stale) is True
    links = companion_assets.read_companion_links()
    assert links[gguf.lower()] == ["Tongyi-MAI/Z-Image-Turbo"]

    scans = _install(
        monkeypatch,
        _repo(gguf, [("z-image-turbo-Q4_K_M.gguf", 5_000)]),
        _repo(retired, [("split_files/vae/ae.safetensors", 900_000)]),
    )
    required = companion_assets.required_companion_bases(scans)
    assert retired.lower() not in required
    assert repack.lower() not in required
    assert required["tongyi-mai/z-image-turbo"] == {gguf}
    offered = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    assert retired.lower() in [c["repo_id"].lower() for c in offered]


H3_BASE = "unsloth/MiniMax-H3"
H3_QUANT = "unsloth/MiniMax-H3-FP8"
H3_ASSETS = [
    ("MiniMax-H3-INT8-ConvRot.pt", 20_250_000_000),
    ("text_encoders/official/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", 27_140_000_000),
]


def test_h3_orphan_sidecars_are_offered_with_actual_blob_bytes(monkeypatch):
    _install(monkeypatch, _repo(H3_QUANT, H3_ASSETS))
    result = asyncio.run(companion_cleanup.orphan_companions_response())
    assert result["companions"] == [
        {
            "repo_id": H3_QUANT,
            "size_bytes": 47_390_000_000,
            "cache_path": str(Path("/cache/models--unsloth--MiniMax-H3-FP8")),
        }
    ]
    assert result["total_bytes"] == 47_390_000_000


@pytest.mark.parametrize("base", [H3_BASE, "MiniMaxAI/MiniMax-H3"])
def test_h3_speed_loaded_pipeline_protects_hosted_components_without_links(monkeypatch, base):
    # Speed load: only VAEs land in the pipeline repo; DiT and TE live in the FP8 repo.
    _install(
        monkeypatch,
        _repo(base, [("vae/diffusion_pytorch_model.safetensors", 11_100_000_000)]),
        _repo(H3_QUANT, H3_ASSETS),
    )
    assert companion_cleanup.companion_dependents(H3_QUANT) == [base]
    assert H3_QUANT not in {
        row["repo_id"]
        for row in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(H3_QUANT, None, None)
    assert excinfo.value.status_code == 400


def test_h3_delete_preview_shows_sidecars_as_separate_reclaimable_assets(monkeypatch):
    _install(
        monkeypatch,
        _repo(H3_BASE, [("vae/diffusion_pytorch_model.safetensors", 11_100_000_000)]),
        _repo(H3_QUANT, H3_ASSETS),
    )
    result = asyncio.run(companion_cleanup.delete_impact_response(H3_BASE, None))
    assert result["reclaimed_bytes"] == 11_100_000_000
    assert result["retained_companions"] == []
    assert result["freeable_companions"] == [
        {"repo_id": H3_QUANT, "size_bytes": 47_390_000_000, "needed_by": []}
    ]


def test_h3_delete_preview_retains_sidecars_for_another_cached_checkpoint(monkeypatch):
    gguf = "unsloth/MiniMax-H3-GGUF"
    _install(
        monkeypatch,
        _repo(H3_BASE, [("vae/diffusion_pytorch_model.safetensors", 11_100)]),
        _repo(H3_QUANT, H3_ASSETS),
        _repo(gguf, [("MiniMax-H3-Q4_K_M.gguf", 99_000)]),
    )
    result = asyncio.run(companion_cleanup.delete_impact_response(H3_BASE, None))
    assert result["freeable_companions"] == []
    assert result["retained_companions"] == [
        {"repo_id": H3_QUANT, "size_bytes": 47_390_000_000, "needed_by": [gguf]}
    ]


@pytest.mark.parametrize(
    "name", ["custom-model.safetensors", "transformer/model.safetensors", "custom/model.pt"]
)
def test_h3_cleanup_preserves_unknown_weights_alongside_sidecars(monkeypatch, name):
    _install(monkeypatch, _repo(H3_QUANT, [*H3_ASSETS, (name, 90_000)]))
    assert asyncio.run(companion_cleanup.orphan_companions_response())["companions"] == []
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(H3_QUANT, None, None, only_if_orphan = True)
    assert excinfo.value.status_code == 409


def test_h3_metadata_only_pipeline_does_not_pin_orphan_sidecars(monkeypatch):
    _install(monkeypatch, _repo(H3_BASE, [("model_index.json", 400)]), _repo(H3_QUANT, H3_ASSETS))
    assert companion_cleanup.companion_dependents(H3_QUANT) == []
    assert H3_QUANT in {
        row["repo_id"]
        for row in asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    }


def test_h3_legacy_conditioner_copy_is_also_a_companion(monkeypatch):
    legacy = H3_ASSETS[1][0].replace("/official/", "/")
    _install(monkeypatch, _repo(H3_QUANT, [(legacy, 27_140_000_000)]))
    assert (
        asyncio.run(companion_cleanup.orphan_companions_response())["total_bytes"] == 27_140_000_000
    )


# (hosted repo, its cached files, reading install, its files, a second install sharing a file in the repo)
_VAE = [("vae/diffusion_pytorch_model.safetensors", 300_000)]
HOSTED_CASES = {
    "image-prequant": (
        "unsloth/Z-Image-Turbo-FP8",
        [("Z-Image-Turbo-FP8.safetensors", 6_100_000)],
        "Tongyi-MAI/Z-Image-Turbo",
        [("model_index.json", 400), *_VAE, ("text_encoder/model.safetensors", 800_000)],
        ("unsloth/Z-Image-Turbo", [("model_index.json", 400), *_VAE]),
    ),
    "hosted-te": (
        "unsloth/Qwen-Image-FP8",
        [("Qwen-Image-text_encoder-FP8.safetensors", 9_400_000)],
        "unsloth/Qwen-Image-GGUF",
        [("qwen-image-Q4_K_M.gguf", 12_000_000)],
        ("unsloth/HunyuanImage-2.1-GGUF", [("hunyuanimage2.1-Q4_K_M.gguf", 11_000_000)]),
    ),
    "video-nvfp4": (
        "unsloth/HunyuanVideo-1.5-NVFP4",
        [("HunyuanVideo-1.5-NVFP4.pt", 4_800_000)],
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        [("model_index.json", 400), *_VAE],
        ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", [*_VAE]),
    ),
    "video-mirror-fp8": (
        "unsloth/Wan2.2-TI2V-5B-FP8",
        [
            ("Wan2.2-TI2V-5B-FP8.safetensors", 5_000_000),
            ("Wan2.2-TI2V-5B-text_encoder-FP8.safetensors", 6_000_000),
        ],
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        [("model_index.json", 400), *_VAE],
        ("unsloth/Wan2.2-TI2V-5B-GGUF", [("Wan2.2-TI2V-5B-Q4_K_M.gguf", 3_000_000)]),
    ),
    "minimax-h3": (
        H3_QUANT,
        H3_ASSETS,
        H3_BASE,
        [("model_index.json", 400), *_VAE],
        ("unsloth/MiniMax-H3-GGUF", [("MiniMax-H3-Q4_K_M.gguf", 99_000)]),
    ),
}
_CASE_IDS = list(HOSTED_CASES)


def _offered() -> dict:
    rows = asyncio.run(companion_cleanup.orphan_companions_response())["companions"]
    return {row["repo_id"]: row["size_bytes"] for row in rows}


@pytest.mark.parametrize("case", _CASE_IDS)
def test_hosted_component_with_no_dependent_is_offered_with_its_bytes(monkeypatch, case):
    hosted, files, *_ = HOSTED_CASES[case]
    _install(monkeypatch, _repo(hosted, files))
    assert _offered() == {hosted: sum(size for _n, size in files)}


@pytest.mark.parametrize("case", _CASE_IDS)
def test_hosted_component_stays_protected_while_its_install_remains(monkeypatch, case):
    hosted, files, dependent, dependent_files, _second = HOSTED_CASES[case]
    _install(monkeypatch, _repo(hosted, files), _repo(dependent, dependent_files))
    assert companion_cleanup.companion_dependents(hosted) == [dependent]
    assert hosted not in _offered()
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(hosted, None, None)
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("case", _CASE_IDS)
def test_deleting_the_last_install_previews_the_hosted_component_as_freeable(monkeypatch, case):
    hosted, files, dependent, dependent_files, _second = HOSTED_CASES[case]
    _install(monkeypatch, _repo(hosted, files), _repo(dependent, dependent_files))
    impact = asyncio.run(companion_cleanup.delete_impact_response(dependent, None))
    assert impact["reclaimed_bytes"] == sum(size for _n, size in dependent_files)
    assert impact["blocked_by"] == []
    assert impact["retained_companions"] == []
    assert impact["freeable_companions"] == [
        {"repo_id": hosted, "size_bytes": sum(size for _n, size in files), "needed_by": []}
    ]


@pytest.mark.parametrize("case", _CASE_IDS)
def test_a_shared_hosted_component_is_retained_for_the_other_install(monkeypatch, case):
    hosted, files, dependent, dependent_files, (second, second_files) = HOSTED_CASES[case]
    _install(
        monkeypatch,
        _repo(hosted, files),
        _repo(dependent, dependent_files),
        _repo(second, second_files),
    )
    assert companion_cleanup.companion_dependents(hosted) == sorted([dependent, second])
    impact = asyncio.run(companion_cleanup.delete_impact_response(dependent, None))
    assert impact["freeable_companions"] == []
    assert impact["retained_companions"] == [
        {"repo_id": hosted, "size_bytes": sum(size for _n, size in files), "needed_by": [second]}
    ]
    assert hosted not in _offered()


@pytest.mark.parametrize("case", _CASE_IDS)
@pytest.mark.parametrize("extra", ["custom-model.safetensors", "transformer/model.safetensors"])
def test_a_hosted_repo_with_an_unknown_weight_is_preserved(monkeypatch, case, extra):
    hosted, files, *_ = HOSTED_CASES[case]
    _install(monkeypatch, _repo(hosted, [*files, (extra, 90_000)]))
    assert hosted not in _offered()
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(hosted, None, None, only_if_orphan = True)
    assert excinfo.value.status_code == 409


def test_a_shared_repo_is_pinned_only_by_the_families_whose_files_are_cached(monkeypatch):
    """unsloth/Qwen-Image-FP8 serves HunyuanImage-2.1 its encoder, but only Qwen-Image's DiT is cached there."""
    hosted = "unsloth/Qwen-Image-FP8"
    gguf = "unsloth/HunyuanImage-2.1-GGUF"
    dit = [("Qwen-Image-FP8.safetensors", 20_000_000)]
    _install(monkeypatch, _repo(hosted, dit), _repo(gguf, [("hunyuanimage2.1-Q4_K_M.gguf", 9)]))
    assert companion_cleanup.companion_dependents(hosted) == []
    assert _offered() == {hosted: 20_000_000}
    encoder = [("Qwen-Image-text_encoder-INT8-ConvRot.safetensors", 7_000_000)]
    _install(
        monkeypatch,
        _repo(hosted, [*dit, *encoder]),
        _repo(gguf, [("hunyuanimage2.1-Q4_K_M.gguf", 9)]),
    )
    assert companion_cleanup.companion_dependents(hosted) == [gguf]
    assert _offered() == {}


def test_a_gguf_install_does_not_pin_a_hosted_denoiser(monkeypatch):
    """A GGUF pick brings its own denoiser; only a speed load beside the base reads the hosted one."""
    hosted = "unsloth/Z-Image-Turbo-FP8"
    _install(
        monkeypatch,
        _repo(hosted, [("Z-Image-Turbo-FP8.safetensors", 6_100_000)]),
        _repo("unsloth/Z-Image-Turbo-GGUF", [("z-image-turbo-Q4_K_M.gguf", 3_000_000)]),
    )
    assert companion_cleanup.companion_dependents(hosted) == []
    assert hosted in _offered()


def test_a_speed_loaded_image_base_is_never_offered_as_unused(monkeypatch):
    """The base minus its denoiser plus the hosted DiT is the install: neither half is an unused leftover."""
    base = "Tongyi-MAI/Z-Image-Turbo"
    _install(
        monkeypatch,
        _repo(base, [("model_index.json", 400), *_VAE]),
        _repo("unsloth/Z-Image-Turbo-FP8", [("Z-Image-Turbo-FP8.safetensors", 6_100_000)]),
    )
    assert _offered() == {}
    _install(monkeypatch, _repo(base, [("model_index.json", 400), *_VAE]))
    assert base in _offered()


def test_a_scoped_delete_of_one_speed_load_copy_retains_the_hosted_components(monkeypatch):
    from hub.utils import hf_cache_state

    base, hosted = "Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo-FP8"
    copies = [_repo(base, [("model_index.json", 400), *_VAE], cache = c) for c in ("/a", "/b")]
    monkeypatch.setattr(
        hf_cache_state, "resolve_delete_target_root", lambda *a: Path(a[2]).parent.resolve()
    )
    _install(monkeypatch, *copies, _repo(hosted, [("Z-Image-Turbo-FP8.safetensors", 6_100_000)]))
    impact = asyncio.run(
        companion_cleanup.delete_impact_response(base, None, cache_path = copies[0].repo_path)
    )
    assert impact["freeable_companions"] == []
    assert impact["retained_companions"] == [
        {"repo_id": hosted, "size_bytes": 6_100_000, "needed_by": [base]}
    ]


def test_the_orphan_recheck_refuses_a_base_that_became_a_speed_load(monkeypatch):
    base = "Tongyi-MAI/Z-Image-Turbo"
    _install(
        monkeypatch,
        _repo(base, [("model_index.json", 400), *_VAE]),
        _repo("unsloth/Z-Image-Turbo-FP8", [("Z-Image-Turbo-FP8.safetensors", 6_100_000)]),
    )
    with pytest.raises(HTTPException) as excinfo:
        deletion._delete_cached_model_blocking(base, None, None, only_if_orphan = True)
    assert excinfo.value.status_code == 409


_HOSTED_21 = "unsloth/Qwen-Image-2.1-FP8"
_HOSTED_21_FILES = [
    ("Qwen-Image-2.1-FP8.safetensors", 7_120_000),
    ("Qwen-Image-2.1-text_encoder-FP8.safetensors", 9_400_000),
    ("vae/qwen_image_2.1_vae_bf16.safetensors", 680_000),
]
_CONFIG_ONLY_21 = [
    ("model_index.json", 400),
    ("transformer/config.json", 300),
    ("text_encoder/config.json", 300),
    ("vae/config.json", 300),
    ("tokenizer/tokenizer.json", 9_000),
]


def test_a_fully_hosted_speed_load_is_an_install(monkeypatch):
    """Hosted DiT, encoder and VAE: the base keeps only configs and the tokenizer, and still loads."""
    base = "Qwen/Qwen-Image-2.1"
    _install(monkeypatch, _repo(base, _CONFIG_ONLY_21), _repo(_HOSTED_21, _HOSTED_21_FILES))
    assert companion_cleanup.companion_dependents(_HOSTED_21) == [base]
    assert _offered() == {}
    impact = asyncio.run(companion_cleanup.delete_impact_response(base, None))
    assert impact["blocked_by"] == []
    assert [row["repo_id"] for row in impact["freeable_companions"]] == [_HOSTED_21]
    for repo_id, refusal in ((base, 409), (_HOSTED_21, 400)):
        with pytest.raises(HTTPException) as excinfo:
            deletion._delete_cached_model_blocking(repo_id, None, None, only_if_orphan = True)
        assert excinfo.value.status_code == refusal


def test_a_config_only_base_with_no_hosted_weights_stays_an_orphan(monkeypatch):
    base = "Qwen/Qwen-Image-2.1"
    _install(monkeypatch, _repo(base, _CONFIG_ONLY_21))
    assert base in _offered()


def test_a_config_only_base_missing_a_hosted_component_is_kept(monkeypatch):
    """No hosted VAE cached: the next load refetches the dense vae/, so the speed load is still an install."""
    base = "Qwen/Qwen-Image-2.1"
    _install(monkeypatch, _repo(base, _CONFIG_ONLY_21), _repo(_HOSTED_21, _HOSTED_21_FILES[:2]))
    assert _offered() == {}


_HV_TE = (
    "unsloth/HunyuanVideo-1.5-FP8",
    [("HunyuanVideo-1.5-text_encoder-FP8.safetensors", 7_000_000)],
)
_HV_BASE = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"


@pytest.mark.parametrize(
    "pick",
    ["hunyuanvideo1.5_480p_t2v.safetensors", "split_files/hunyuanvideo1.5_480p_t2v.safetensors"],
)
def test_a_video_single_file_checkpoint_pins_its_hosted_encoder(monkeypatch, pick):
    checkpoint = "some-owner/custom-video"
    _install(
        monkeypatch,
        _repo(checkpoint, [(pick, 16_000_000)]),
        _repo(*_HV_TE),
        _repo(_HV_BASE, [("model_index.json", 400), *_VAE]),
    )
    assert companion_cleanup.companion_dependents(_HV_TE[0]) == [checkpoint]
    assert _offered() == {}


def _sd_cpp_repos(repo_id: str, gguf: str) -> list:
    from core.inference.diffusion_families import detect_family_for_pick

    fam = detect_family_for_pick(repo_id, gguf)
    specs = sorted(companion_assets._sd_cpp_component_specs(fam, repo_id, gguf))
    by_repo: dict[str, list] = {}
    for repo, filename in specs:
        by_repo.setdefault(repo, []).append((filename, 1_000))
    return [_repo(repo, files) for repo, files in by_repo.items()]


def _shape(name):
    """Every cached repo one install reads, by install shape: nothing here may be offered while it is cached."""
    base_files = [("model_index.json", 400), *_VAE, ("text_encoder/model.safetensors", 800_000)]
    klein_te = (
        "unsloth/FLUX.2-klein-4B-FP8",
        [("FLUX.2-klein-4B-text_encoder-FP8.safetensors", 9)],
    )
    wan_mirror = "unsloth/Wan2.2-TI2V-5B-FP8"
    wan_base = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    wan_te = (wan_mirror, [("Wan2.2-TI2V-5B-text_encoder-FP8.safetensors", 9)])
    shapes = {
        "diffusers-base-image": [
            _repo(BASE_REPO, [*base_files, ("transformer/model.safetensors", 9)]),
            _repo(*klein_te),
        ],
        "diffusers-base-video": [
            _repo(wan_base, [("model_index.json", 400), *_VAE, ("transformer/m.safetensors", 9)]),
            _repo(*wan_te),
        ],
        "speed-load-image": [
            _repo("Tongyi-MAI/Z-Image-Turbo", base_files),
            _repo(
                "unsloth/Z-Image-Turbo-FP8",
                [
                    ("Z-Image-Turbo-FP8.safetensors", 9),
                    ("Z-Image-Turbo-text_encoder-FP8.safetensors", 9),
                ],
            ),
        ],
        "speed-load-video": [
            _repo(wan_base, [("model_index.json", 400), *_VAE]),
            _repo(wan_mirror, [("Wan2.2-TI2V-5B-FP8.safetensors", 9), *wan_te[1]]),
        ],
        "fully-hosted-image": [
            _repo("Qwen/Qwen-Image-2.1", _CONFIG_ONLY_21),
            _repo(_HOSTED_21, _HOSTED_21_FILES),
        ],
        "single-file-image": [
            _repo("some-owner/custom", [("flux-2-klein-4b.safetensors", 9)]),
            _repo(BASE_REPO, base_files),
            _repo(*klein_te),
        ],
        "single-file-video": [
            _repo("some-owner/custom-video", [("hunyuanvideo1.5_480p_t2v.safetensors", 9)]),
            _repo(_HV_BASE, [("model_index.json", 400), *_VAE]),
            _repo(*_HV_TE),
        ],
        "gguf-image": [
            _repo(GGUF_REPO, [("flux-2-klein-4b-Q4_K_M.gguf", 9)]),
            _repo(BASE_REPO, base_files),
            _repo(*klein_te),
        ],
        "gguf-video": [
            _repo("unsloth/Wan2.2-TI2V-5B-GGUF", [("Wan2.2-TI2V-5B-Q4_K_M.gguf", 9)]),
            _repo(wan_base, [("model_index.json", 400), *_VAE]),
            _repo(*wan_te),
        ],
        "sd-cpp-image": [
            _repo("unsloth/Z-Image-Turbo-GGUF", [("z-image-turbo-Q4_K_M.gguf", 9)]),
            *_sd_cpp_repos("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
        ],
        "modular-h3": [
            _repo(H3_BASE, [("modular_model_index.json", 400), *_VAE]),
            _repo(H3_QUANT, H3_ASSETS),
        ],
        "modular-h3-gguf": [
            _repo("unsloth/MiniMax-H3-GGUF", [("MiniMax-H3-Q4_K_M.gguf", 9)]),
            _repo(H3_QUANT, H3_ASSETS[1:]),
        ],
    }
    return shapes[name]


INSTALL_SHAPES = [
    "diffusers-base-image",
    "diffusers-base-video",
    "speed-load-image",
    "speed-load-video",
    "fully-hosted-image",
    "single-file-image",
    "single-file-video",
    "gguf-image",
    "gguf-video",
    "sd-cpp-image",
    "modular-h3",
    "modular-h3-gguf",
]


@pytest.mark.parametrize("shape", INSTALL_SHAPES)
def test_nothing_an_install_reads_is_offered_while_it_is_cached(monkeypatch, shape):
    repos = _shape(shape)
    _install(monkeypatch, *repos)
    assert _offered() == {}
    for repo in repos:
        if repo.repo_id.lower() in companion_assets._hosted_component_table():
            assert companion_cleanup.companion_dependents(repo.repo_id) != []


def test_each_cache_root_contributes_its_own_nested_video_pick(monkeypatch):
    checkpoint = "some-owner/custom-video"
    wan_te = ("unsloth/Wan2.2-TI2V-5B-FP8", [("Wan2.2-TI2V-5B-text_encoder-FP8.safetensors", 9)])
    _install(
        monkeypatch,
        _repo(checkpoint, [("split_files/hunyuanvideo1.5_480p_t2v.safetensors", 9)]),
        _repo(checkpoint, [("split_files/wan2.2-ti2v-5b.safetensors", 9)], cache = "/other-cache"),
        _repo(*_HV_TE),
        _repo(*wan_te),
    )
    assert companion_cleanup.companion_dependents(_HV_TE[0]) == [checkpoint]
    assert companion_cleanup.companion_dependents(wan_te[0]) == [checkpoint]
    assert _offered() == {}


def _split_roots(repos: list) -> list:
    """Each repo's files split across two cache roots (a repo with one file goes whole to the second root)."""
    out = []
    for repo in repos:
        revision = repo.revisions[0]
        files = [
            (Path(f.file_path).relative_to(Path(revision.snapshot_path)).as_posix(), f.size_on_disk)
            for f in revision.files
        ]
        first, second = files[: len(files) // 2], files[len(files) // 2 :]
        if first:
            out.append(_repo(repo.repo_id, first))
        out.append(_repo(repo.repo_id, second, cache = "/other-cache"))
    return out


@pytest.mark.parametrize("shape", INSTALL_SHAPES)
def test_nothing_an_install_reads_is_offered_across_two_roots(monkeypatch, shape):
    repos = _split_roots(_shape(shape))
    _install(monkeypatch, *repos)
    # A curated base copy with no denoiser in its own root stays a per-root orphan row, as before; hosted files never.
    hosted = companion_assets._hosted_component_table()
    assert not {r.lower() for r in _offered()} & set(hosted)
    for repo in repos:
        if repo.repo_id.lower() in hosted:
            assert companion_cleanup.companion_dependents(repo.repo_id) != []
