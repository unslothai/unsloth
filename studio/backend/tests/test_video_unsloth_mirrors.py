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
                assert vf.video_family_prequant_repo(
                    fam, scheme, mirror
                ) == vf.video_family_prequant_repo(fam, scheme, base)


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
    assert (
        resolve_video_base_repo(fam, None, "unsloth/LTX-2.5-GGUF") == "Lightricks/LTX-2.5-Diffusers"
    )
    # A DiT picked from the official single-file release still assembles from that release.
    assert ltx25_default_base("Lightricks/LTX-2.5") == "Lightricks/LTX-2.5"


def test_the_fp8_repos_still_resolve_their_own_prequant_rows():
    """The -FP8 repos are both a hosted-checkpoint repo and now a base's file carrier: folding their id onto the
    upstream base must not change which checkpoint repo a scheme resolves to."""
    for fam in vf._FAMILIES:
        for scheme, repo in fam.prequant_repos:
            assert vf.video_family_prequant_repo(
                fam, scheme, fam.base_repo
            ) == vf.video_family_prequant_repo(
                fam, scheme, mirror_repo(fam.base_repo) or fam.base_repo
            )


def test_no_video_fetch_reads_a_comfy_or_third_party_repack(tmp_path, monkeypatch):
    """Even with the old Comfy-Org / QuantStack repack fully cached, the H3 VAEs, the H3 conditioner and the Wan GGUFs
    come from unsloth/*."""
    from core.inference.diffusion_families import legacy_source_repo
    from core.inference.video_minimax_h3 import H3_AUDIO_VAE, H3_VIDEO_VAE, h3_component_source
    from core.inference.video_minimax_h3_te import h3_te_quant_source

    monkeypatch.setattr(
        "core.inference.diffusion_families._upstream_is_cached", lambda *_a, **_k: True
    )
    for repo in (
        "unsloth/MiniMax-H3-GGUF",
        "unsloth/MiniMax-H3-FP8",
        "unsloth/Wan2.2-TI2V-5B-GGUF",
    ):
        assert legacy_source_repo(repo) is None
        assert prefer_cached_legacy_source(repo, ("x.safetensors",)) == repo
    assert (
        h3_component_source(H3_VIDEO_VAE)
        == h3_component_source(H3_AUDIO_VAE)
        == "unsloth/MiniMax-H3-GGUF"
    )
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

    def model_info(
        self,
        repo,
        files_metadata = True,
        **_kw,
    ):
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
    video_mod.VideoBackend._resolve_checkpoint_path(
        "Lightricks/LTX-2.3", "ltx-2.3-22b-dev.safetensors", None
    )
    assert seen == ["unsloth/LTX-2.3-FP8"]


def test_in_flight_load_claims_both_sides_of_a_mirror_pair():
    backend = video_mod.VideoBackend()
    backend._loading = video_mod._VideoLoadingState(
        repo_id = "unsloth/Wan2.2-TI2V-5B-GGUF", base_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    )
    ids = backend.loading_repo_ids()
    assert "Wan-AI/Wan2.2-TI2V-5B-Diffusers" in ids
    assert "unsloth/Wan2.2-TI2V-5B-FP8" in ids


def test_h3_vaes_are_read_from_the_official_provenance_paths():
    """The vae/ copies came from the Comfy-Org repack; the build reads the vae/official/ files made from MiniMaxAI's
    weights (byte-identical, so released builds reading vae/ keep working)."""
    from core.inference.video_minimax_h3 import H3_AUDIO_VAE, H3_VIDEO_VAE

    assert H3_VIDEO_VAE == "vae/official/minimax_h3_video_vae_fp16.safetensors"
    assert H3_AUDIO_VAE == "vae/official/minimax_h3_audio_vae_fp32.safetensors"


def _partial_upstream_cache(root, repo, files):
    """An upstream snapshot holding ``files`` (some weight, not the whole component set)."""
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents = True, exist_ok = True)
    (folder / "refs" / "main").write_text("a" * 40)
    for name in files:
        path = folder / "snapshots" / ("a" * 40) / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x" * 10)
    return folder


def test_assembly_reads_the_repo_the_predownload_staged(empty_cache, monkeypatch):
    """A partial upstream cache: the exact-name check stages into the mirror, and that choice reaches assembly
    (the name-less default would flip back to the partial upstream)."""
    import huggingface_hub

    upstream = "Lightricks/LTX-2"
    _partial_upstream_cache(empty_cache, upstream, ["vae/diffusion_pytorch_model.safetensors"])
    assert (
        video_mod._video_fetch_repo(upstream) == upstream
    )  # the default assembly would re-decide to

    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingApi)

    def _download(repo, name, token, **_kw):
        path = empty_cache / "snap" / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    backend = video_mod.VideoBackend()
    choice: dict = {}
    assert backend._predownload_base(upstream, None, "pipeline", fetch_choice = choice)
    assert choice == {"repo": "unsloth/LTX-2-FP8"}

    # _run_load forwards it, and load_pipeline / the H3 modular loader take it ahead of the name-less default.
    import inspect

    for fn in (
        video_mod.VideoBackend.load_pipeline,
        video_mod.VideoBackend._load_h3_modular_pipeline,
    ):
        assert "_base_fetch_repo" in inspect.signature(fn).parameters
    src = inspect.getsource(video_mod.VideoBackend._run_load)
    assert 'kwargs["_base_fetch_repo"] = base_fetch_choice.get("repo")' in src
    assert "fetch_choice = base_fetch_choice" in src




def test_no_mirror_keeps_an_explicit_checkpoint_in_its_repo(empty_cache, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    seen: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, name, token, **kw: (seen.append(repo), str(empty_cache / name))[1],
    )
    video_mod.VideoBackend._resolve_checkpoint_path(
        "unsloth/LTX-2.5-GGUF", "ltx-2.5-Q4_K_M.gguf", None
    )
    assert seen == ["unsloth/LTX-2.5-GGUF"]
    # Base components still fold back onto the upstream under the opt-out.
    assert video_mod._video_fetch_repo("unsloth/LTX-2.5-FP8") == "Lightricks/LTX-2.5-Diffusers"


def test_ltx25_estimate_reads_shard_indexes_from_the_listing_mirror(empty_cache, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingApi)
    seen: list = []

    def _plan_shards(info, base, hf_token):
        seen.append(base)
        return None

    backend = video_mod.VideoBackend()
    monkeypatch.setattr(backend, "_ltx25_plan_shards", _plan_shards)
    monkeypatch.setattr(backend, "_te_prequant_hub_files", lambda *a, **k: ())
    monkeypatch.setattr(backend, "_h3_te_quant_hub_files", lambda *a, **k: ((), ()))
    backend._estimate_download_bytes(
        "Lightricks/LTX-2.5-Diffusers", None, "Lightricks/LTX-2.5-Diffusers", None, "pipeline"
    )
    assert seen == ["unsloth/LTX-2.5-FP8"]


def _snapshot(root, repo, files):
    """A cached revision of ``repo`` under ``root`` holding ``files`` (name -> bytes)."""
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents = True, exist_ok = True)
    (folder / "refs" / "main").write_text("b" * 40)
    for name, data in files.items():
        path = folder / "snapshots" / ("b" * 40) / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(data)
    return folder


_INDEX = (
    b'{"_class_name": "LTX2Pipeline", "vae": ["diffusers", "AutoencoderKL"],'
    b' "transformer": ["diffusers", "LTX2Transformer"], "scheduler": ["diffusers", "S"]}'
)
_COMPLETE = {
    "model_index.json": _INDEX,
    "vae/diffusion_pytorch_model.safetensors": b"v",
    "transformer/diffusion_pytorch_model.safetensors": b"t",
    "scheduler/scheduler_config.json": b"{}",
}


def test_nameless_choice_prefers_the_complete_mirror_over_a_partial_upstream(
    empty_cache, monkeypatch
):
    upstream, mirror = "Lightricks/LTX-2", "unsloth/LTX-2-FP8"
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _snapshot(
        empty_cache,
        upstream,
        {"model_index.json": _INDEX, "vae/diffusion_pytorch_model.safetensors": b"v"},
    )
    _snapshot(empty_cache, mirror, _COMPLETE)
    # Offline assembly has no names: the complete mirror wins over the partial upstream.
    assert video_mod._video_fetch_repo(upstream) == mirror
    # An equally complete upstream keeps being used.
    _snapshot(empty_cache, upstream, _COMPLETE)
    assert video_mod._video_fetch_repo(upstream) == upstream


def test_named_choice_reuses_an_upstream_complete_in_the_fallback_root(empty_cache):
    upstream = "Lightricks/LTX-2"
    _snapshot(empty_cache / "other", upstream, _COMPLETE)
    names = list(_COMPLETE)
    assert video_mod._video_fetch_repo(upstream, files = names) == upstream
    assert (
        video_mod._video_fetch_repo(upstream, files = names + ["missing.safetensors"])
        == "unsloth/LTX-2-FP8"
    )






def test_h3_staging_index_is_read_from_the_mirror(empty_cache, monkeypatch):
    seen: list = []

    def _download(repo, name, token, **_kw):
        seen.append(repo)
        path = empty_cache / name
        path.write_text(
            '{"text_encoder": ["x", "Y", {"pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3"}]}'
        )
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    assert (
        video_mod.VideoBackend._h3_te_base_index_source("MiniMaxAI/MiniMax-H3", None)
        == "MiniMaxAI/MiniMax-H3"
    )
    assert seen == ["unsloth/MiniMax-H3"]






def test_h3_component_specs_follow_the_selected_mirror(empty_cache):
    spec = lambda src, sub = None: types.SimpleNamespace(  # noqa: E731
        pretrained_model_name_or_path = src, subfolder = sub, type_hint = object
    )
    pipe = types.SimpleNamespace(
        _component_specs = {
            "transformer": spec("MiniMaxAI/MiniMax-H3", "transformer"),
            "vae": spec("MiniMaxAI/MiniMax-H3", "vae"),
            "text_encoder": spec("someone/other-encoder", "text_encoder"),
            "scheduler": spec("unsloth/MiniMax-H3", "scheduler"),
        }
    )
    got = video_mod._h3_mirrored_component_sources(
        pipe, "MiniMaxAI/MiniMax-H3", "unsloth/MiniMax-H3", local_files_only = False
    )
    assert got == {"transformer": "unsloth/MiniMax-H3", "vae": "unsloth/MiniMax-H3"}
    # An upstream kept for its complete cache, or a local base, redirects nothing.
    assert not video_mod._h3_mirrored_component_sources(
        pipe, "MiniMaxAI/MiniMax-H3", str(empty_cache), local_files_only = False
    )


def test_h3_modular_load_passes_the_mirrored_sources_to_load_components():
    import inspect

    src = inspect.getsource(video_mod.VideoBackend._load_h3_modular_pipeline)
    assert "_h3_mirrored_component_sources(" in src
    assert src.index("_h3_mirrored_component_sources(") < src.index("pipe.load_components(")






def test_a_pipeline_load_of_the_ltx25_carrier_is_refused():
    with pytest.raises(ValueError, match = "single files"):
        video_mod._detect_load_family("unsloth/LTX-2.5-GGUF", None, None, None, "pipeline")
    with pytest.raises(ValueError, match = "single files"):
        video_mod._detect_load_family("Lightricks/LTX-2.5", None, None, None, "pipeline")


# ── the load's fetch plan: one immutable record per role, read by every fetch site and by progress ────────────────


def _blob(root, repo, key, data, incomplete = False):
    folder = root / ("models--" + repo.replace("/", "--")) / "blobs"
    folder.mkdir(parents = True, exist_ok = True)
    (folder / (key + (".incomplete" if incomplete else ""))).write_bytes(data)


def _plan(repo, files, revision = None):
    return video_mod._PlannedFetch(repo, revision, tuple(files))


def _progress_of(backend, plan, expected):
    backend._loading = video_mod._VideoLoadingState(
        repo_id = "x/ckpt", base_repo = "x/base", expected_bytes = expected
    )
    backend._loading.fetch_plan = plan
    return backend.load_progress()


def test_progress_reads_planned_files_by_blob_key_in_the_plan_repo(empty_cache, monkeypatch):
    upstream, mirror = "Lightricks/LTX-2", "unsloth/LTX-2-FP8"
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _blob(empty_cache, upstream, "k-vae", b"u" * 30)  # a partial upstream cache the load does not read
    _blob(empty_cache, mirror, "k-carrier-fp8", b"c" * 900)  # the carrier's own root checkpoint
    _blob(empty_cache, mirror, "k-old-transformer", b"o" * 50)  # a superseded revision's same-named file
    _blob(empty_cache, mirror, "k-unrelated", b"z" * 500, incomplete = True)  # an abandoned unrelated partial
    _blob(empty_cache, mirror, "k-index", b"i" * 2)
    _blob(empty_cache, mirror, "k-transformer", b"t" * 7, incomplete = True)  # the planned shard, in flight
    plan = {
        "base": _plan(
            mirror,
            [
                ("model_index.json", "k-index", 2),
                ("vae/diffusion_pytorch_model.safetensors", "k-vae", 30),
                ("transformer/diffusion_pytorch_model.safetensors", "k-transformer", 100),
            ],
        )
    }
    progress = _progress_of(video_mod.VideoBackend(), plan, 132)
    assert progress["phase"] == "downloading"
    assert progress["downloaded_bytes"] == 2 + 7


def test_progress_reads_the_fallback_cache_root(empty_cache, monkeypatch):
    """Fetches pass reuse_other_cache_root; an upstream complete only in the import-time root counts."""
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _blob(empty_cache / "other", "Lightricks/LTX-2", "k-a", b"a" * 40)
    plan = {"base": _plan("Lightricks/LTX-2", [("a.safetensors", "k-a", 40)])}
    assert _progress_of(video_mod.VideoBackend(), plan, 40)["phase"] == "finalizing"


def test_progress_counts_a_carrier_checkpoint_and_its_mirrored_base_per_role(empty_cache, monkeypatch):
    upstream, carrier = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "unsloth/Wan2.2-TI2V-5B-FP8"
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _blob(empty_cache, carrier, "k-fp8", b"c" * 40)
    _blob(empty_cache, upstream, "k-vae", b"v" * 3)
    plan = {
        "checkpoint": _plan(carrier, [("Wan-FP8.safetensors", "k-fp8", 40)]),
        "base": _plan(upstream, [("vae/diffusion_pytorch_model.safetensors", "k-vae", 3)]),
    }
    assert _progress_of(video_mod.VideoBackend(), plan, 1000)["downloaded_bytes"] == 43


def test_progress_without_a_blob_key_reads_the_pinned_revision(empty_cache, monkeypatch):
    mirror = "unsloth/LTX-2-FP8"
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _snapshot(empty_cache, mirror, {"model_index.json": b"{}"})  # revision "b" * 40
    plan_old = {"base": _plan(mirror, [("model_index.json", None, 2)], revision = "a" * 40)}
    plan_new = {"base": _plan(mirror, [("model_index.json", None, 2)], revision = "b" * 40)}
    backend = video_mod.VideoBackend()
    assert _progress_of(backend, plan_old, 100)["downloaded_bytes"] == 0
    assert _progress_of(backend, plan_new, 100)["downloaded_bytes"] == 2


class _KeyedApi(_RecordingApi):
    revisions: list = []

    def model_info(self, repo, files_metadata = True, revision = None, **_kw):
        type(self).revisions.append(revision)
        info = super().model_info(repo, files_metadata)
        for sib in info.siblings:
            sib.blob_id = "git-" + sib.rfilename
            if sib.rfilename.endswith(".safetensors"):
                sib.lfs = types.SimpleNamespace(sha256 = "lfs-" + sib.rfilename)
        return info


def test_estimate_settles_one_plan_per_role(empty_cache, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _KeyedApi)
    backend = video_mod.VideoBackend()
    monkeypatch.setattr(backend, "_te_prequant_hub_files", lambda *a, **k: ())
    monkeypatch.setattr(backend, "_h3_te_quant_hub_files", lambda *a, **k: ((), ()))
    planned: dict = {}
    backend._estimate_download_bytes(
        "Lightricks/LTX-2", None, "Lightricks/LTX-2", None, "pipeline", planned_files = planned
    )
    base = planned["base"]
    assert base.repo == "unsloth/LTX-2-FP8" and base.revision == "0" * 40
    files = {name: (key, size) for name, key, size in base.files}
    assert files["transformer/diffusion_pytorch_model.safetensors"] == (
        "lfs-transformer/diffusion_pytorch_model.safetensors",
        3,
    )
    assert files["model_index.json"] == ("git-model_index.json", 1)
    assert "Model-FP8.safetensors" not in files
    # An upstream complete in the cache keeps being read, pinned to its own snapshot (not the mirror listing).
    _snapshot(empty_cache, "Lightricks/LTX-2", {name: b"x" for name in files})
    planned.clear()
    backend._estimate_download_bytes(
        "Lightricks/LTX-2", None, "Lightricks/LTX-2", None, "pipeline", planned_files = planned
    )
    # ...pinned to ITS snapshot's commit, with that snapshot's own sizes (the mirror listing says 3 for the DiT).
    assert planned["base"].repo == "Lightricks/LTX-2" and planned["base"].revision == "b" * 40
    sizes = {name: size for name, _key, size in planned["base"].files}
    assert sizes["transformer/diffusion_pytorch_model.safetensors"] == 1


def test_predownload_follows_the_plan_repo_and_revision(empty_cache, monkeypatch):
    import huggingface_hub

    _KeyedApi.revisions = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _KeyedApi)
    calls: list = []

    def _download(repo, name, token, revision = None, **_kw):
        calls.append((repo, revision))
        path = empty_cache / "snap" / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    backend = video_mod.VideoBackend()
    choice: dict = {}
    backend._predownload_base(
        "Lightricks/LTX-2",
        None,
        "pipeline",
        fetch_choice = choice,
        fetch_repo = "unsloth/LTX-2-FP8",
        revision = "c" * 40,
    )
    assert calls and set(calls) == {("unsloth/LTX-2-FP8", "c" * 40)}
    assert _KeyedApi.revisions == ["c" * 40]
    assert choice == {"repo": "unsloth/LTX-2-FP8", "revision": "c" * 40}


def test_a_superseded_worker_cannot_rewrite_the_active_plan(empty_cache, monkeypatch):
    """Only _run_load writes fetch_plan, under its token check; a fetch site has no way to patch it."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _KeyedApi)
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, name, token, **kw: (
            (empty_cache / "snap" / name).parent.mkdir(parents = True, exist_ok = True),
            (empty_cache / "snap" / name).write_bytes(b"x"),
            str(empty_cache / "snap" / name),
        )[2],
    )
    backend = video_mod.VideoBackend()
    active = {"base": _plan("unsloth/Wan2.2-TI2V-5B-FP8", [("a", "k", 1)])}
    backend._loading = video_mod._VideoLoadingState(repo_id = "n/new", base_repo = "n/new")
    backend._loading.fetch_plan = dict(active)
    backend._predownload_base("Lightricks/LTX-2", None, "pipeline")  # the stale worker's base pull
    assert backend._loading.fetch_plan == active
    assert not hasattr(video_mod.VideoBackend, "_note_load_fetch_repo")



def test_completeness_needs_weights_beside_a_model_config(empty_cache, monkeypatch):
    """An interrupted upstream pull that left every component's config but no weights does not tie a full mirror."""
    upstream, mirror = "Lightricks/LTX-2", "unsloth/LTX-2-FP8"
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _snapshot(
        empty_cache,
        upstream,
        {
            "model_index.json": _INDEX,
            "vae/config.json": b"{}",
            "transformer/config.json": b"{}",
            "scheduler/scheduler_config.json": b"{}",
        },
    )
    _snapshot(
        empty_cache,
        mirror,
        {**_COMPLETE, "vae/config.json": b"{}", "transformer/config.json": b"{}"},
    )
    assert video_mod._video_cached_completeness(upstream) == 1  # the scheduler alone
    assert video_mod._video_fetch_repo(upstream) == mirror


def test_assembly_opens_the_pinned_checkpoint_and_partner(empty_cache, monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, name, token, revision = None, **kw: (
            seen.append((repo, name, revision)),
            str(empty_cache / name),
        )[1],
    )
    pin = ("unsloth/Wan2.2-T2V-A14B-GGUF", "c" * 40)
    video_mod.VideoBackend._resolve_checkpoint_path(
        "unsloth/Wan2.2-T2V-A14B-GGUF", "HighNoise/x-HighNoise-Q4_K_M.gguf", None, fetch = pin
    )
    video_mod.VideoBackend._resolve_moe_partner_path(
        "unsloth/Wan2.2-T2V-A14B-GGUF", "HighNoise/x-HighNoise-Q4_K_M.gguf", None, fetch = pin
    )
    assert {(repo, rev) for repo, _name, rev in seen} == {pin}
    assert len(seen) == 2 and seen[0][1] != seen[1][1]


# ── invariant: a load touches the Hub only at the (repo, revision) pairs its own plan names ─────────────────────────

_GGUF_REPO = "unsloth/Wan2.2-TI2V-5B-GGUF"
_GGUF_FILE = "Wan2.2-TI2V-5B-Q4_K_M.gguf"
_WAN_UP, _WAN_MIRROR = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "unsloth/Wan2.2-TI2V-5B-FP8"
_WAN_FILES = {
    "model_index.json": 1,
    "scheduler/scheduler_config.json": 1,
    "tokenizer/tokenizer.json": 1,
    "vae/config.json": 1,
    "vae/diffusion_pytorch_model.safetensors": 2,
    "text_encoder/config.json": 1,
    "text_encoder/model.safetensors": 5,
    "transformer/config.json": 1,
    "transformer/diffusion_pytorch_model.safetensors": 9,
}


class _HubRecorder:
    listings: list = []
    downloads: list = []
    config_reads: list = []
    # Repos whose main advances after their first listing (a later unpinned listing sees a new commit).
    moving: tuple = (_GGUF_REPO,)

    def __init__(self, token = None):
        pass

    def model_info(self, repo, files_metadata = True, revision = None, **_kw):
        type(self).listings.append((repo, revision))
        # The checkpoint repo's main advances after the first listing: a re-plan must keep the fetched commit and size.
        moved = (
            revision is None
            and repo in type(self).moving
            and any(r == repo for r, _ in type(self).listings[:-1])
        )
        files = (
            {_GGUF_FILE: 70 if moved else 7}
            if repo == _GGUF_REPO
            else dict(_WAN_FILES, **{"Wan-FP8.safetensors": 99})
        )
        return types.SimpleNamespace(
            sha = revision
            or (("h" if moved else "g") * 40 if repo == _GGUF_REPO else ("n" if moved else "m") * 40),
            siblings = [
                types.SimpleNamespace(
                    rfilename = n, size = sz, blob_id = "git-" + n, lfs = types.SimpleNamespace(sha256 = "lfs-" + n)
                )
                for n, sz in files.items()
            ],
        )


def _assembly_config_reads(base, empty_cache, monkeypatch):
    """The base config reads assembly makes by hub id, through the real readers, recorded as (repo, revision)."""
    import diffusers
    import transformers

    from core.inference import (
        diffusion_comfy_components,
        diffusion_prequant,
        video_ltx2,
        video_ltx25,
    )

    def _record(repo, *_a, revision = None, **_kw):
        _HubRecorder.config_reads.append((repo, revision))
        return {}

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_record))
    monkeypatch.setattr(diffusers.LTX2VideoTransformer3DModel, "load_config", staticmethod(_record))
    monkeypatch.setattr(diffusers.AutoencoderKL, "load_config", staticmethod(_record))
    monkeypatch.setattr("core.inference.diffusion_krea2.remap_rope_parameters", lambda *_a, **_k: None)
    # encoder + VAE configs (supplied components), a prequant DiT's config, the LTX-2.3 config, an LTX-2.5 companion
    diffusion_comfy_components._component_class_and_config(
        "text_encoder", "T5EncoderModel", base = base, hf_token = None, local_files_only = False, cache_dir = None
    )
    diffusion_comfy_components._component_class_and_config(
        "vae", "AutoencoderKL", base = base, hf_token = None, local_files_only = False, cache_dir = None
    )
    diffusion_prequant._load_transformer_config(
        types.SimpleNamespace(load_config = _record), base, None, None, str(empty_cache / "ckpt")
    )
    video_ltx2._LTX23PrequantConfig(base).load_config(base)
    before = len(_HubRecorder.downloads)
    video_ltx25.ltx25_companion_path(base, "companion.safetensors", None)
    _HubRecorder.config_reads.extend((r, rev) for r, rev, _ in _HubRecorder.downloads[before:])
    # The LTX-2.5 encoder worker is another process: the pin rides in its config.
    monkeypatch.setattr(video_ltx25, "ensure_text_encoder_sidecar", lambda: str(empty_cache))
    monkeypatch.setattr(video_ltx25.Ltx25TextEncoderClient, "start", lambda self: None)
    client = video_ltx25.start_text_encoder(
        repo = base,
        hf_token = None,
        cache_dir = None,
        local_files_only = False,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
    )
    _HubRecorder.config_reads.append((client.config["repo"], client.config.get("revision")))
    # The LTX-2.5 DiT's single-file config.
    import diffusers.loaders.single_file_utils as sfu

    monkeypatch.setattr(sfu, "load_single_file_checkpoint", lambda *_a, **_k: {})
    monkeypatch.setattr(
        diffusers.LTX2VideoTransformer3DModel,
        "from_single_file",
        staticmethod(lambda _state, config = None, revision = None, **_k: _record(config, revision = revision)),
    )
    video_ltx25.load_ltx25_transformer(
        str(empty_cache / "dit.safetensors"), base_repo = base, torch_dtype = None, is_gguf = False, hf_token = None
    )
    # The LTX-2.3 DiT's single-file config.
    video_ltx2.load_ltx23_transformer({}, base_repo = base, torch_dtype = None, is_gguf = False, hf_token = None)


def _walk_a_load(
    empty_cache,
    monkeypatch,
    *,
    te_fetch_falls_back,
    predownload_leaves_early = False,
    carrier_checkpoint = False,
    top_up = False,
):
    import huggingface_hub

    _HubRecorder.listings, _HubRecorder.downloads, _HubRecorder.config_reads = [], [], []
    _HubRecorder.moving = (_WAN_MIRROR,) if carrier_checkpoint else (_GGUF_REPO,)
    monkeypatch.setattr(huggingface_hub, "HfApi", _HubRecorder)
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))

    def _download(repo, name, token, revision = None, **_kw):
        _HubRecorder.downloads.append((repo, revision, name))
        path = empty_cache / "dl" / repo.replace("/", "--") / (revision or "main") / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    # The upstream already holds every base file EXCEPT the dense encoder weights: enough for the reduced plan only.
    _snapshot(empty_cache, _WAN_UP, {n: b"x" for n in _WAN_FILES if n != "text_encoder/model.safetensors"})
    backend = video_mod.VideoBackend()
    # The estimate expects a hosted pre-cast encoder; the real fetch then finds it (or falls back).
    monkeypatch.setattr(backend, "_te_prequant_sources", lambda *a, **k: {"text_encoder": object()})
    monkeypatch.setattr(backend, "_te_prequant_hub_files", lambda *a, **k: {"text_encoder": []})
    monkeypatch.setattr(
        backend, "_fetch_te_prequant", lambda *a, **k: () if te_fetch_falls_back else ("text_encoder",)
    )
    monkeypatch.setattr(video_mod, "_assert_pick_is_not_speech", lambda *a, **k: None)
    captured: dict = {}

    def _assemble(**kwargs):
        captured.update(kwargs)
        # What assembly does with the plan: re-open the checkpoint where the prefetch pinned it.
        import inspect

        pin = (
            {"fetch": kwargs.get("_checkpoint_fetch")}
            if "fetch" in inspect.signature(backend._resolve_checkpoint_path).parameters
            else {}
        )
        backend._resolve_checkpoint_path(kwargs["repo_id"], kwargs["gguf_filename"], None, **pin)
        _assembly_config_reads(kwargs["_base_fetch_repo"], empty_cache, monkeypatch)
        if top_up:
            # An injection failed: load_pipeline restores the skipped dense weights through this, with its plan.
            backend._top_up_dense_base(
                kwargs["base_repo"],
                None,
                "single_file",
                base_local_dir = kwargs.get("_base_local_dir"),
                fetch_repo = kwargs["_base_fetch_repo"],
                revision = kwargs["_base_fetch_revision"],
                local_files_only = False,
            )

    monkeypatch.setattr(backend, "load_pipeline", _assemble)
    if predownload_leaves_early:
        # No scoped pull (a non-diffusers layout, a metadata failure): fetch_choice is never filled.
        monkeypatch.setattr(backend, "_predownload_base", lambda *a, **k: None)
    backend._load_token = 1
    repo_id, filename = (_WAN_MIRROR, "Wan-FP8.safetensors") if carrier_checkpoint else (_GGUF_REPO, _GGUF_FILE)
    backend._loading = video_mod._VideoLoadingState(repo_id = repo_id, base_repo = _WAN_UP)
    loading = backend._loading
    backend._run_load(repo_id = repo_id, gguf_filename = filename, _load_token = 1)
    assert loading.error is None, loading.error
    captured["_expected_bytes"] = loading.expected_bytes
    return loading.fetch_plan, captured


@pytest.mark.parametrize(
    "te_fetch_falls_back, predownload_leaves_early", [(False, False), (True, False), (True, True)]
)
def test_every_hub_call_of_a_load_matches_its_plan(
    empty_cache, monkeypatch, te_fetch_falls_back, predownload_leaves_early
):
    plan, captured = _walk_a_load(
        empty_cache,
        monkeypatch,
        te_fetch_falls_back = te_fetch_falls_back,
        predownload_leaves_early = predownload_leaves_early,
    )
    pinned = {(p.repo, p.revision) for p in plan.values()}
    assert {(repo, rev) for repo, rev, _ in _HubRecorder.downloads} <= pinned
    # Listings are read from the listing repos only (the mirror, never the vendor id), at the plan's commit or before
    # the plan existed.
    assert {repo for repo, _ in _HubRecorder.listings} <= {_GGUF_REPO, _WAN_MIRROR}
    revisions = {p.revision for p in plan.values()} | {None}
    assert {rev for _, rev in _HubRecorder.listings} <= revisions
    assert captured["_checkpoint_fetch"] == (plan["checkpoint"].repo, plan["checkpoint"].revision)
    # The total is exactly what the kept plan sums to, after every re-plan.
    assert captured["_expected_bytes"] == sum(size for p in plan.values() for _n, _k, size in p.files)
    # Every base config read during assembly hits the plan's base (repo, revision).
    assert len(_HubRecorder.config_reads) == 8
    assert set(_HubRecorder.config_reads) == {(plan["base"].repo, plan["base"].revision)}
    assert captured["_base_fetch_repo"] == plan["base"].repo
    assert captured["_base_fetch_revision"] == plan["base"].revision
    if te_fetch_falls_back:
        # The dense encoder is back in the pull, which the partial upstream cannot serve: re-planned onto the mirror.
        assert plan["base"].repo == _WAN_MIRROR and plan["base"].revision == "m" * 40
        if not predownload_leaves_early:
            assert any(name == "text_encoder/model.safetensors" for _, _, name in _HubRecorder.downloads)
    else:
        assert plan["base"].repo == _WAN_UP and plan["base"].revision == "b" * 40
    assert plan["checkpoint"] == video_mod._PlannedFetch(
        _GGUF_REPO, "g" * 40, ((_GGUF_FILE, "lfs-" + _GGUF_FILE, 7),)
    )


def test_the_ltx25_encoder_worker_takes_the_plan_revision(empty_cache, monkeypatch):
    from core.inference import video_ltx25

    monkeypatch.setattr(video_ltx25, "ensure_text_encoder_sidecar", lambda: str(empty_cache))
    monkeypatch.setattr(video_ltx25.Ltx25TextEncoderClient, "start", lambda self: None)
    common = dict(hf_token = None, cache_dir = None, local_files_only = False, device = "cpu", dtype = "bfloat16")
    pinned = video_ltx25.start_text_encoder(
        repo = "unsloth/LTX-2.5-FP8", revision = "r" * 40, resident = False, **common
    )
    assert pinned.config["revision"] == "r" * 40
    unpinned = video_ltx25.start_text_encoder(repo = "unsloth/LTX-2.5-FP8", resident = False, **common)
    assert "revision" not in unpinned.config
    import inspect

    src = inspect.getsource(video_mod.VideoBackend.load_pipeline)
    assert "revision = None if _base_local_dir else _base_fetch_revision" in src


def test_an_upstream_split_across_cache_roots_is_not_reused(empty_cache):
    upstream = "Lightricks/LTX-2"
    names = list(_COMPLETE)
    half = len(names) // 2
    _snapshot(empty_cache, upstream, {n: _COMPLETE[n] for n in names[:half]})
    _snapshot(empty_cache / "other", upstream, {n: _COMPLETE[n] for n in names[half:]})
    # Every name is cached, but no ONE root holds the set: assembly could not open it as one snapshot.
    assert video_mod._video_fetch_repo(upstream, files = names) == "unsloth/LTX-2-FP8"


def test_ltx25_shard_indexes_are_read_at_the_listing_commit(empty_cache, monkeypatch):
    from core.inference import video_ltx25

    seen: list = []

    def _download(repo, name, token, revision = None, **_kw):
        seen.append((repo, name, revision))
        path = empty_cache / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text('{"weight_map": {}}')
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(video_ltx25, "is_ltx25_pack", lambda names: True)
    info = types.SimpleNamespace(
        sha = "s" * 40, siblings = [_sib("transformer/diffusion_pytorch_model.safetensors.index.json", 1)]
    )
    video_mod.VideoBackend._ltx25_plan_shards(info, "unsloth/LTX-2.5-FP8", None)
    assert seen == [("unsloth/LTX-2.5-FP8", "transformer/diffusion_pytorch_model.safetensors.index.json", "s" * 40)]


def test_expected_total_counts_identical_blobs_once(empty_cache, monkeypatch):
    plan = {
        "base": _plan(
            "unsloth/LTX-2-FP8",
            [("a/model.safetensors", "same", 10), ("b/model.safetensors", "same", 10), ("c.json", "other", 1)],
        )
    }
    assert video_mod._plan_total(plan) == 11
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    _blob(empty_cache, "unsloth/LTX-2-FP8", "same", b"x" * 10)
    _blob(empty_cache, "unsloth/LTX-2-FP8", "other", b"y")
    progress = _progress_of(video_mod.VideoBackend(), plan, video_mod._plan_total(plan))
    assert progress["phase"] == "finalizing"


@pytest.mark.parametrize("te_fetch_falls_back", [False, True])
def test_roles_sharing_a_repo_share_one_commit(
    empty_cache, monkeypatch, te_fetch_falls_back
):
    """An FP8 pick from the carrier that also mirrors the base: both roles list that repo once and pin one commit,
    even though its main moves mid-load."""
    plan, captured = _walk_a_load(
        empty_cache,
        monkeypatch,
        te_fetch_falls_back = te_fetch_falls_back,
        carrier_checkpoint = True,
    )
    by_repo: dict = {}
    for entry in plan.values():
        by_repo.setdefault(entry.repo, set()).add(entry.revision)
    assert all(len(revs) == 1 for revs in by_repo.values()), by_repo
    assert plan["checkpoint"].repo == _WAN_MIRROR and plan["checkpoint"].revision == "m" * 40
    pinned = {(p.repo, p.revision) for p in plan.values()}
    assert {(repo, rev) for repo, rev, _ in _HubRecorder.downloads} <= pinned
    assert captured["_expected_bytes"] == video_mod._plan_total(plan)


def test_load_pipeline_tops_up_through_the_plan():
    import inspect

    src = inspect.getsource(video_mod.VideoBackend.load_pipeline)
    call = src[src.index("self._top_up_dense_base(") :][:600]
    assert "fetch_repo = _base_fetch_repo" in call and "revision = _base_fetch_revision" in call


@pytest.mark.parametrize("te_fetch_falls_back", [False, True])
def test_the_dense_top_up_after_a_failed_injection_stays_on_the_plan(empty_cache, monkeypatch, te_fetch_falls_back):
    plan, _captured = _walk_a_load(
        empty_cache, monkeypatch, te_fetch_falls_back = te_fetch_falls_back, top_up = True
    )
    pinned = {(p.repo, p.revision) for p in plan.values()}
    assert {(repo, rev) for repo, rev, _ in _HubRecorder.downloads} <= pinned
    # The top-up pulled the whole unskipped set again, from the plan's base repo at its commit.
    base = (plan["base"].repo, plan["base"].revision)
    assert sum(1 for repo, rev, _ in _HubRecorder.downloads if (repo, rev) == base) >= len(_WAN_FILES) - 3


def test_a_cached_upstream_plan_describes_its_own_snapshot(empty_cache, monkeypatch):
    """The mirror's file changed after the upstream was cached: the plan (and so progress) reads the upstream blob."""
    upstream = "Lightricks/LTX-2"
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(empty_cache))
    folder = empty_cache / ("models--" + upstream.replace("/", "--"))
    (folder / "blobs").mkdir(parents = True)
    (folder / "blobs" / "old-key").write_bytes(b"o" * 12)
    snap = folder / "snapshots" / ("c" * 40)
    snap.mkdir(parents = True)
    (snap / "model.safetensors").symlink_to(folder / "blobs" / "old-key")
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("c" * 40)
    info = types.SimpleNamespace(
        sha = "m" * 40,
        siblings = [
            types.SimpleNamespace(
                rfilename = "model.safetensors", size = 99, blob_id = None, lfs = types.SimpleNamespace(sha256 = "new-key")
            )
        ],
    )
    entry = video_mod._plan_fetch(upstream, "unsloth/LTX-2-FP8", info, ["model.safetensors"], None)
    assert entry == video_mod._PlannedFetch(upstream, "c" * 40, (("model.safetensors", "old-key", 12),))
    assert _progress_of(video_mod.VideoBackend(), {"base": entry}, 12)["phase"] == "finalizing"
