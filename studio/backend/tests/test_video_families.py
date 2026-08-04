# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video family registry: detection, shape snapping, generation defaults.
Pure-module tests: no torch, no network."""

import pytest

from core.inference.video_families import (
    VIDEO_CANCELLED_MSG,
    VIDEO_NOT_LOADED_MSG,
    default_video_generation_params,
    detect_video_family,
    resolve_video_base_repo,
    snap_num_frames,
    snap_video_size,
    supported_video_family_names,
)


@pytest.mark.parametrize(
    "repo_id",
    [
        "unsloth/LTX-2.3-GGUF",
        "Lightricks/LTX-2",
        "Lightricks/LTX-2.3-fp8",
        "lightricks/ltx-2.3",
        "some/dir/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
    ],
)
def test_detect_ltx2(repo_id):
    fam = detect_video_family(repo_id)
    assert fam is not None and fam.name == "ltx-2"
    assert fam.pipeline_class == "LTX2Pipeline"
    assert fam.has_audio is True


def test_detect_override_and_unknown():
    assert detect_video_family("x", override = "ltx-2").name == "ltx-2"
    assert detect_video_family("x", override = "ltx2").name == "ltx-2"
    assert detect_video_family("x", override = "nope") is None
    # A short alias must not match inside an unrelated word.
    assert detect_video_family("someorg/deluxtreme-model") is None


@pytest.mark.parametrize(
    "repo_id",
    [
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "wan-ai/wan2.2-ti2v-5b-diffusers",
        "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "some/dir/wan2.2-ti2v-5b-Q4_K_M.gguf",
    ],
)
def test_detect_wan_ti2v_5b(repo_id):
    # The TI2V-5B repo ids route to the single-DiT Wan family (no MoE, no audio).
    fam = detect_video_family(repo_id)
    assert fam is not None and fam.name == "wan2.2-ti2v-5b"
    assert fam.pipeline_class == "WanPipeline"
    assert fam.transformer_class == "WanTransformer3DModel"
    assert fam.is_moe is False
    assert fam.cfg2_kwarg is None
    assert fam.has_audio is False
    assert fam.frame_step == 4  # Wan VAE temporal factor is 4 (4k+1)


@pytest.mark.parametrize(
    "repo_id",
    [
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "wan-ai/wan2.2-t2v-a14b-diffusers",
        "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "some/dir/wan2.2-t2v-a14b-Q4_K_M.gguf",
    ],
)
def test_detect_wan_t2v_a14b(repo_id):
    # The A14B repo ids route to the dual-expert MoE family: a second DiT plus a second guidance kwarg (guidance_scale_2).
    fam = detect_video_family(repo_id)
    assert fam is not None and fam.name == "wan2.2-t2v-a14b"
    assert fam.pipeline_class == "WanPipeline"
    assert fam.transformer2_class == "WanTransformer3DModel"
    assert fam.is_moe is True
    assert fam.cfg2_kwarg == "guidance_scale_2"
    assert fam.has_audio is False
    assert fam.frame_step == 4


def test_detect_wan_overrides():
    # Short aliases the picker / GGUF filenames use resolve to the right family.
    assert detect_video_family("x", override = "wan2.2-5b").name == "wan2.2-ti2v-5b"
    assert detect_video_family("x", override = "wan-ti2v").name == "wan2.2-ti2v-5b"
    assert detect_video_family("x", override = "wan2.2-14b").name == "wan2.2-t2v-a14b"
    assert detect_video_family("x", override = "wan-t2v").name == "wan2.2-t2v-a14b"


def test_wan_and_ltx_do_not_cross_route():
    # LTX ids must never resolve to a Wan family and vice versa (separate engines).
    assert detect_video_family("Lightricks/LTX-2").name == "ltx-2"
    assert detect_video_family("unsloth/LTX-2.3-GGUF").name == "ltx-2"
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").name == "wan2.2-ti2v-5b"


def test_sentinels_are_video_specific():
    # The routes match these EXACTLY for 409s, so they must not collide with the image sentinels.
    assert "video" in VIDEO_NOT_LOADED_MSG.lower()
    assert "video" in VIDEO_CANCELLED_MSG.lower()


def test_resolve_base_repo():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert resolve_video_base_repo(fam, None) == "Lightricks/LTX-2"
    assert resolve_video_base_repo(fam, "  ") == "Lightricks/LTX-2"
    assert resolve_video_base_repo(fam, "other/base") == "other/base"


def test_snap_num_frames_lattice():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    # Valid counts are k * 8 + 1: on-lattice values pass through, everything else floors to the previous point, never below 1.
    assert snap_num_frames(fam, 121) == 121
    assert snap_num_frames(fam, 120) == 113
    assert snap_num_frames(fam, 122) == 121
    assert snap_num_frames(fam, 1) == 1
    assert snap_num_frames(fam, 0) == 1
    assert snap_num_frames(fam, 9) == 9


def test_snap_video_size_multiple():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert snap_video_size(fam, 768, 512) == (768, 512)
    assert snap_video_size(fam, 1000, 700) == (992, 672)
    # Never snaps to zero: the floor is one multiple.
    assert snap_video_size(fam, 1, 1) == (32, 32)


def test_generation_defaults_distilled_vs_dev():
    # The distilled checkpoints run few-step with CFG off while the dev-config base repo wants the full schedule, so the filename wins.
    assert default_video_generation_params(
        "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf", "Lightricks/LTX-2"
    ) == (8, 1.0)
    assert default_video_generation_params(None, "Lightricks/LTX-2") == (40, 4.0)
    assert default_video_generation_params("unknown/thing") == (40, 4.0)


def test_supported_names():
    assert supported_video_family_names() == (
        "ltx-2",
        "wan2.2-ti2v-5b",
        "wan2.2-t2v-a14b",
        "hunyuanvideo-1.5",
        "hunyuanvideo-1.5-720p",
    )


def test_wan_snap_num_frames_4k_plus_1():
    # Wan's temporal factor is 4, so valid counts are 4k+1 (not LTX-2's 8k+1).
    fam = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert fam.frame_step == 4
    assert snap_num_frames(fam, 81) == 81  # 4*20 + 1, on-lattice
    assert snap_num_frames(fam, 121) == 121  # 4*30 + 1
    assert snap_num_frames(fam, 120) == 117  # floors to 4*29 + 1
    assert snap_num_frames(fam, 3) == 1  # below the first stride floors to 1
    assert snap_num_frames(fam, 5) == 5  # 4*1 + 1


def test_wan_snap_video_size_16():
    # Wan patchifies at spatial factor 8 * patch 2 = 16; sizes floor to /16.
    fam = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert fam.resolution_multiple == 16
    # A14B's native 720p is the true 16:9 1280x720, NOT the 1280x704 that TI2V-5B's /32 VAE floors to, and it is the default preset.
    assert fam.resolution_presets[0] == (1280, 720)
    assert snap_video_size(fam, 1280, 720) == (1280, 720)  # native 720p, on-grid
    assert snap_video_size(fam, 1000, 700) == (992, 688)


def test_wan_generation_defaults():
    # Both Wan families default to the pipeline's 50 steps / CFG 5.0.
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-TI2V-5B-Diffusers") == (50, 5.0)
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-T2V-A14B-Diffusers") == (50, 5.0)
    # A GGUF filename carrying the family name still lands on the Wan defaults.
    assert default_video_generation_params(
        "wan2.2-ti2v-5b-Q4_K_M.gguf", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    ) == (50, 5.0)


def test_generation_defaults_fallback_honors_family():
    # When no identifier names a known variant (an opaque local path under an explicit family_override), the resolved family's own default is used, not the hardcoded LTX 40/4.0.
    assert default_video_generation_params("/models/my-clip", "/models/my-clip") == (40, 4.0)
    assert default_video_generation_params(
        "/models/my-clip", "/models/my-clip", fallback = (50, 5.0)
    ) == (50, 5.0)
    # A recognised token still wins over the fallback.
    assert default_video_generation_params("wan2.2-ti2v-5b", fallback = (8, 1.0)) == (50, 5.0)


def test_generation_defaults_wan_is_segment_not_substring():
    # "wan" must match as a name segment, not a raw substring, so an opaque repo/path ("swan", "taiwan") does not pick up Wan's 50-step/CFG-5 schedule.
    assert default_video_generation_params(
        "user/swan-video", "Lightricks/LTX-2", fallback = (40, 4.0)
    ) == (40, 4.0)
    assert default_video_generation_params(
        "taiwan-clips.gguf", "user/taiwan-clips", fallback = (40, 4.0)
    ) == (40, 4.0)
    # Genuine Wan identifiers (segment-initial, with a version suffix or separator) still match.
    assert default_video_generation_params("wan2.2-ti2v-5b-Q4_K_M.gguf") == (50, 5.0)
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-T2V-A14B") == (50, 5.0)
    # An "ltxv" style name still resolves to LTX (trailing letters stay free).
    assert default_video_generation_params("ltxv-2.3-distilled") == (8, 1.0)
    assert default_video_generation_params("Lightricks/LTXV-2.3") == (40, 4.0)


def test_wan_size_tables_present():
    ti2v = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    a14b = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert ti2v.bf16_components_gb is not None and a14b.bf16_components_gb is not None
    # bf16-RESIDENT transformer sizes: the Wan transformers ship FP32 on disk (TI2V index 20.0 GB, A14B 57.15 GB per expert),
    # so the table holds the halved bf16 sizes (ti2v ~10.0, a14b ~57.2). The fp32 sums over-budget the plan ~2x.
    assert ti2v.bf16_components_gb[0] == 10.0
    assert a14b.bf16_components_gb[0] == 57.2
    # The A14B DiT total (two experts) still dwarfs the single TI2V-5B DiT.
    assert a14b.bf16_components_gb[0] > ti2v.bf16_components_gb[0] * 3
    # A portrait preset is offered for the 5B (a vertical option per the task).
    assert any(h > w for (w, h) in ti2v.resolution_presets)


def test_wan_ti2v_5b_snaps_to_32_not_16():
    # TI2V-5B's VAE is 16x spatial with a patch of 2, so WanPipeline floors H/W to 32. The backend must snap to /32 too, or a /16-only request renders at another size.
    fam = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert fam.resolution_multiple == 32
    assert snap_video_size(fam, 1280, 720) == (1280, 704)  # 720 is /16 but not /32 -> floors to 704
    assert snap_video_size(fam, 1280, 704) == (1280, 704)  # on-grid preset unchanged
    # A14B keeps /16 (its VAE is the Wan2.1 8x VAE, so 8*2 = 16).
    a14b = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert a14b.resolution_multiple == 16


def test_wan_families_force_vae_fp32():
    # Wan's VAE decodes in float32 (diffusers loads AutoencoderKLWan at fp32 while the pipe runs bf16), so the loader pins it
    # back to fp32 for these families to avoid banding / black frames. LTX-2's VAE is bf16-native.
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").vae_force_fp32 is True
    assert detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers").vae_force_fp32 is True
    assert detect_video_family("unsloth/LTX-2.3-GGUF").vae_force_fp32 is False


def test_family_size_table_present():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert fam.bf16_components_gb is not None
    transformer_gb, text_encoder_gb, companions_gb = fam.bf16_components_gb
    # RESIDENT bf16 figures: a 37.8 GB DiT and the Gemma3-12B TE at ~24.4 GB once cast (the ~49 GB fp32 hub store never sits on
    # device). Regressing to the download size would push auto planning to offload on cards that fit.
    assert transformer_gb > text_encoder_gb > 20.0
    assert text_encoder_gb < 30.0
    assert companions_gb > 0.0


def test_hv15_detection_and_flags():
    fam = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    assert fam is not None and fam.name == "hunyuanvideo-1.5"
    # CFG lives on the guider component (no guidance kwarg in __call__), and the HV15 VAE compresses 16x spatial / 4x temporal.
    assert fam.guidance_via_guider is True
    assert fam.frame_step == 4 and fam.resolution_multiple == 16
    assert fam.has_audio is False
    assert detect_video_family("x/y", override = "hv15") is fam
    # The incompatible HunyuanVideo 1.0 repos must NOT be claimed: their model_index pins HunyuanVideoPipeline.
    assert detect_video_family("hunyuanvideo-community/HunyuanVideo") is None


def test_hv15_generation_defaults():
    # The community repacks ship a guider with guidance_scale 6.0 and the pipeline's own 50-step schedule.
    assert default_video_generation_params(
        None, "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    ) == (50, 6.0)


def test_hv15_720p_checkpoints_never_route_to_the_480p_family():
    """Every 720p repack must land on the 720p family, not just the aliased t2v path.

    The tier is baked into the weights: the two repacks ship transformer target_size 640 vs 960
    and scheduler shift 5.0 vs 9.0, and their bucket lists are disjoint. The 480p family also
    supplies the base repo the VAE and text encoder come from, so a 720p checkpoint routed there
    runs the whole pipeline off-tier.
    """
    for repo_id in (
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
        "QuantStack/HunyuanVideo-1.5-GGUF/720p_t2v-Q4_K_M.gguf",
    ):
        fam = detect_video_family(repo_id)
        assert fam is not None and fam.name == "hunyuanvideo-1.5-720p", repo_id
        assert fam.base_repo.endswith("720p_t2v"), repo_id
    # The 480p repacks are untouched.
    for repo_id in (
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
    ):
        assert detect_video_family(repo_id).name == "hunyuanvideo-1.5", repo_id


def test_video_resolution_presets_are_upstream_sanctioned():
    """No preset may be a size its checkpoint was never trained for.

    HV15 presets must be real buckets of their own tier (generate_crop_size_list(base_size=640)
    for 480p, 960 for 720p); Wan2.2 TI2V-5B must offer only the two shapes upstream's
    SUPPORTED_SIZES asserts on.
    """

    def buckets(base_size, patch_size = 16):
        num_patches = round((base_size / patch_size) ** 2)
        out, wp, hp = [], num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= 4.0:
                out.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return out

    for name, base_size in (("hunyuanvideo-1.5", 640), ("hunyuanvideo-1.5-720p", 960)):
        fam = detect_video_family("x/y", override = name)
        allowed = buckets(base_size)
        for size in fam.resolution_presets:
            assert size in allowed, f"{name}: {size} is not a bucket of this tier"

    ti2v = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert set(ti2v.resolution_presets) == {(1280, 704), (704, 1280)}
