# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every component a VIDEO load fetches comes from an unsloth/* repo.

The video families name vendor bases (Wan-AI, Lightricks, MiniMaxAI) as their logical identity; the fetch sites swap
each for the existing unsloth repo that carries its official files byte-identical at upstream paths (the model's -FP8
repo, or -GGUF; diffusion_families._VIDEO_MIRROR_PAIRS) exactly as the image side does. Only a licence-blocked repo,
or a model with no unsloth repo at all, may stay outside unsloth/*.
"""

from __future__ import annotations

import types

import pytest

from core.inference import video as video_mod
from core.inference import video_families as vf
from core.inference.diffusion_families import (
    _GATED_MIRROR_PAIRS,
    _MIRROR_PAIRS,
    _UNGATED_MIRROR_PAIRS,
    _VIDEO_GATED_MIRROR_PAIRS,
    _VIDEO_MIRROR_PAIRS,
    _VIDEO_UNGATED_MIRROR_PAIRS,
    canonical_base,
    mirror_repo,
    prefer_cached_legacy_source,
    prefer_ungated_mirror,
    upstream_is_gated,
)

# The ONLY ids a video load may fetch from outside unsloth/*, each with the licence clause that blocks a mirror.
# HunyuanVideo-1.5 (Tencent Hunyuan Community License): "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION,
# UNITED KINGDOM AND SOUTH KOREA AND IS EXPRESSLY LIMITED TO THE TERRITORY" and "You must not use, reproduce, modify,
# distribute, or display the Tencent Hunyuan Works ... outside the Territory". A public Hub repo cannot honour that.
LICENCE_BLOCKED = frozenset(
    {
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-480p_t2v",
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-720p_t2v",
    }
)

# No unsloth repo exists for these (no -FP8, no -GGUF), so nothing can carry their files yet.
NO_UNSLOTH_REPO = frozenset(
    {
        "wan-ai/wan2.1-t2v-1.3b-diffusers",
        "wan-ai/wan2.1-t2v-14b-diffusers",
    }
)

EXPECTED_VIDEO_PAIRS = {
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "unsloth/Wan2.2-TI2V-5B-FP8",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "unsloth/Wan2.2-T2V-A14B-FP8",
    "Lightricks/LTX-2": "unsloth/LTX-2-FP8",
    "Lightricks/LTX-2.3": "unsloth/LTX-2.3-FP8",
    "Lightricks/LTX-2.5-Diffusers": "unsloth/LTX-2.5-FP8",
    "Lightricks/LTX-2.5": "unsloth/LTX-2.5-GGUF",
    "MiniMaxAI/MiniMax-H3": "unsloth/MiniMax-H3",
}


@pytest.fixture
def empty_cache(monkeypatch, tmp_path):
    """A fresh install: nothing cached under any id, mirrors enabled."""
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(tmp_path))
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "other"))
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)
    return tmp_path


def test_video_mirror_pairs_round_trip(empty_cache):
    assert dict(_VIDEO_MIRROR_PAIRS) == EXPECTED_VIDEO_PAIRS
    for upstream, mirror in _VIDEO_MIRROR_PAIRS:
        assert mirror.startswith("unsloth/")
        assert mirror_repo(upstream) == mirror
        assert mirror_repo(upstream.lower()) == mirror
        assert canonical_base(mirror) == upstream
        assert canonical_base(mirror.upper()) == upstream
        assert prefer_ungated_mirror(upstream) == mirror
        assert video_mod._video_fetch_repo(upstream) == mirror
    for upstream, _mirror in _VIDEO_GATED_MIRROR_PAIRS:
        assert upstream_is_gated(upstream)
    for upstream, _mirror in _VIDEO_UNGATED_MIRROR_PAIRS:
        assert not upstream_is_gated(upstream)


def test_video_mirror_opt_out_keeps_upstream(empty_cache, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    for upstream, mirror in _VIDEO_MIRROR_PAIRS:
        assert video_mod._video_fetch_repo(upstream) == upstream
        assert video_mod._video_fetch_repo(mirror) == upstream
        assert video_mod._video_listing_repo(upstream) == upstream


def test_video_pairs_stay_out_of_the_image_companion_table():
    """The image companion table (companion_assets reads it) is unchanged; video ids only feed the id lookups."""
    image_ids = {rid.lower() for pair in _MIRROR_PAIRS for rid in pair}
    assert not image_ids & {rid.lower() for pair in _VIDEO_MIRROR_PAIRS for rid in pair}
    assert len(_MIRROR_PAIRS) == len(_GATED_MIRROR_PAIRS) + len(_UNGATED_MIRROR_PAIRS) == 25


def _video_fetch_ids() -> dict[str, str]:
    """Every Hub id a video load can fetch from, mapped to where it came from."""
    from core.inference import video_ltx2, video_ltx25, video_minimax_h3, video_minimax_h3_te

    ids: dict[str, str] = {}
    for fam in vf._FAMILIES:
        ids[fam.base_repo] = f"{fam.name}.base_repo"
        if fam.gguf_repo:
            ids[fam.gguf_repo] = f"{fam.name}.gguf_repo"
        for _scheme, repo in fam.prequant_repos:
            ids[repo] = f"{fam.name}.prequant_repos"
        for base, _scheme, repo in fam.prequant_variant_repos:
            ids[base] = f"{fam.name}.prequant_variant_repos(base)"
            ids[repo] = f"{fam.name}.prequant_variant_repos"
        for _scheme, _component, repo in fam.te_prequant_repos:
            ids[repo] = f"{fam.name}.te_prequant_repos"
    ids[video_ltx2.LTX23_PREQUANT_BASE] = "video_ltx2.LTX23_PREQUANT_BASE"
    ids[video_ltx2.LTX23_EXTRAS_REPO] = "video_ltx2.LTX23_EXTRAS_REPO"
    ids[video_ltx25.LTX25_SINGLE_FILE_REPO] = "video_ltx25.LTX25_SINGLE_FILE_REPO"
    ids[video_minimax_h3.H3_GGUF_REPO] = "video_minimax_h3.H3_GGUF_REPO"
    ids[video_minimax_h3.H3_COMPONENT_REPO] = "video_minimax_h3.H3_COMPONENT_REPO"
    ids[video_minimax_h3_te.H3_TE_QUANT_REPO] = "video_minimax_h3_te.H3_TE_QUANT_REPO"
    return ids


def test_every_video_fetch_id_resolves_to_unsloth(empty_cache):
    """Walks the video family table: any fetch id that does not land on unsloth/* fails, bar the licence-blocked."""
    offenders = {}
    for rid, where in _video_fetch_ids().items():
        fetched = prefer_cached_legacy_source(video_mod._video_fetch_repo(rid) or rid)
        if fetched.lower().startswith("unsloth/"):
            continue
        if rid.lower() in LICENCE_BLOCKED | NO_UNSLOTH_REPO:
            continue
        offenders[rid] = (where, fetched)
    assert offenders == {}


def test_licence_blocked_allowlist_is_exact(empty_cache):
    """The allowlist names only what the table still fetches from outside unsloth/*: a stale entry fails here."""
    outside = {
        rid.lower()
        for rid in _video_fetch_ids()
        if not (video_mod._video_fetch_repo(rid) or rid).lower().startswith("unsloth/")
    }
    assert outside == LICENCE_BLOCKED | NO_UNSLOTH_REPO


def test_prequant_tables_fold_a_mirror_onto_its_upstream():
    """The loaders now receive the mirror id; the variant tables are keyed on the upstream."""
    for upstream, mirror in _VIDEO_MIRROR_PAIRS:
        assert vf._prequant_base_key(mirror) == vf._prequant_base_key(upstream) == upstream.lower()
    for fam in vf._FAMILIES:
        for base, scheme, repo in fam.prequant_variant_repos:
            mirror = mirror_repo(base)
            if mirror:
                assert vf.video_family_prequant_repo(fam, scheme, mirror) == vf.video_family_prequant_repo(
                    fam, scheme, base
                )


def test_ltx25_single_file_release_recognised_under_its_mirror():
    from core.inference.video_ltx25 import is_ltx25_single_file_base

    assert is_ltx25_single_file_base("Lightricks/LTX-2.5")
    assert is_ltx25_single_file_base("unsloth/LTX-2.5-GGUF")
    assert not is_ltx25_single_file_base("unsloth/LTX-2.5-FP8")


def test_a_gguf_pick_from_the_ltx25_gguf_repo_keeps_the_diffusers_base():
    """That repo now also carries the split release's files, but a GGUF picked from it is still a pack load."""
    from core.inference.video_families import detect_video_family, resolve_video_base_repo
    from core.inference.video_ltx25 import is_ltx25_single_file_base, ltx25_default_base

    assert ltx25_default_base("unsloth/LTX-2.5-GGUF") is None
    assert not is_ltx25_single_file_base("unsloth/LTX-2.5-GGUF", fold_mirror = False)
    fam = detect_video_family("unsloth/LTX-2.5-GGUF")
    assert resolve_video_base_repo(fam, None, "unsloth/LTX-2.5-GGUF") == "Lightricks/LTX-2.5-Diffusers"
    # A DiT picked from the official single-file release still assembles from that release.
    assert ltx25_default_base("Lightricks/LTX-2.5") == "Lightricks/LTX-2.5"


def test_the_fp8_repos_still_resolve_their_own_prequant_rows():
    """The -FP8 repos are both a hosted-checkpoint repo and now a base's file carrier: folding their id onto the
    upstream base must not change which checkpoint repo a scheme resolves to."""
    for fam in vf._FAMILIES:
        for scheme, repo in fam.prequant_repos:
            assert vf.video_family_prequant_repo(fam, scheme, fam.base_repo) == vf.video_family_prequant_repo(
                fam, scheme, mirror_repo(fam.base_repo) or fam.base_repo
            )


def test_no_video_fetch_reads_a_comfy_or_third_party_repack(tmp_path, monkeypatch):
    """Even with the old Comfy-Org / QuantStack repack fully cached, the H3 VAEs, the H3 conditioner and the Wan GGUFs
    come from unsloth/*."""
    from core.inference.diffusion_families import legacy_source_repo
    from core.inference.video_minimax_h3 import H3_AUDIO_VAE, H3_VIDEO_VAE, h3_component_source
    from core.inference.video_minimax_h3_te import h3_te_quant_source

    monkeypatch.setattr("core.inference.diffusion_families._upstream_is_cached", lambda *_a, **_k: True)
    for repo in ("unsloth/MiniMax-H3-GGUF", "unsloth/MiniMax-H3-FP8", "unsloth/Wan2.2-TI2V-5B-GGUF"):
        assert legacy_source_repo(repo) is None
        assert prefer_cached_legacy_source(repo, ("x.safetensors",)) == repo
    assert h3_component_source(H3_VIDEO_VAE) == h3_component_source(H3_AUDIO_VAE) == "unsloth/MiniMax-H3-GGUF"
    assert h3_te_quant_source("int8") == "unsloth/MiniMax-H3-FP8"
    from core.inference import video_minimax_h3, video_minimax_h3_te

    assert not hasattr(video_minimax_h3, "H3_LEGACY_COMPONENT_REPO")
    assert not hasattr(video_minimax_h3_te, "H3_LEGACY_TE_QUANT_REPO")


def test_ltx23_official_file_is_eligible_from_its_mirror():
    from core.inference.video_ltx2 import LTX23_PREQUANT_SOURCE_REPOS

    assert "unsloth/ltx-2.3-fp8" in LTX23_PREQUANT_SOURCE_REPOS


def _sib(name, size):
    return types.SimpleNamespace(rfilename = name, size = size, lfs = None, blob_id = None)


class _RecordingApi:
    calls: list = []

    def __init__(self, token = None):
        pass

    def model_info(self, repo, files_metadata = True, **_kw):
        type(self).calls.append(repo)
        return types.SimpleNamespace(
            sha = "0" * 40,
            siblings = [
                _sib("model_index.json", 1),
                _sib("scheduler/scheduler_config.json", 1),
                _sib("vae/config.json", 1),
                _sib("vae/diffusion_pytorch_model.safetensors", 2),
                _sib("transformer/config.json", 1),
                _sib("transformer/diffusion_pytorch_model.safetensors", 3),
                # The carrier repo's OWN files, which a base load must never stage.
                _sib("Model-FP8.safetensors", 9),
                _sib("Model-FP8.pt", 9),
                _sib("samples/fp8_1.png", 9),
            ],
        )


@pytest.mark.parametrize("upstream", ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Lightricks/LTX-2"])
def test_predownload_base_pulls_every_file_from_the_mirror(empty_cache, monkeypatch, upstream):
    import huggingface_hub

    _RecordingApi.calls = []
    fetched: list = []
    names: list = []

    def _download(repo, name, token, **_kw):
        fetched.append(repo)
        names.append(name)
        path = empty_cache / "snap" / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingApi)
    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    backend = video_mod.VideoBackend()
    root = backend._predownload_base(upstream, None, "pipeline")
    assert root is not None
    assert fetched and set(fetched) == {EXPECTED_VIDEO_PAIRS[upstream]}
    assert set(_RecordingApi.calls) == {EXPECTED_VIDEO_PAIRS[upstream]}
    assert not {"Model-FP8.safetensors", "Model-FP8.pt", "samples/fp8_1.png"} & set(names)


def test_checkpoint_pick_from_a_mirrored_repo_downloads_from_the_mirror(empty_cache, monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, name, token, **kw: (seen.append(repo), str(empty_cache / name))[1],
    )
    video_mod.VideoBackend._resolve_checkpoint_path("Lightricks/LTX-2.3", "ltx-2.3-22b-dev.safetensors", None)
    assert seen == ["unsloth/LTX-2.3-FP8"]


def test_in_flight_load_claims_both_sides_of_a_mirror_pair():
    backend = video_mod.VideoBackend()
    backend._loading = video_mod._VideoLoadingState(
        repo_id = "unsloth/Wan2.2-TI2V-5B-GGUF", base_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    )
    ids = backend.loading_repo_ids()
    assert "Wan-AI/Wan2.2-TI2V-5B-Diffusers" in ids
    assert "unsloth/Wan2.2-TI2V-5B-FP8" in ids
