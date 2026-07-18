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
    # The A14B repo ids route to the dual-expert MoE family: a second DiT + a second
    # guidance kwarg (guidance_scale_2, verified present in diffusers 0.39).
    fam = detect_video_family(repo_id)
    assert fam is not None and fam.name == "wan2.2-t2v-a14b"
    assert fam.pipeline_class == "WanPipeline"
    assert fam.transformer2_class == "WanTransformer3DModel"
    assert fam.is_moe is True
    assert fam.cfg2_kwarg == "guidance_scale_2"
    assert fam.has_audio is False
    assert fam.frame_step == 4


@pytest.mark.parametrize(
    "repo_id",
    [
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "wan-ai/wan2.2-i2v-a14b-diffusers",
        "Wan-AI/Wan2.2-I2V-A14B",
        "some/dir/wan2.2-i2v-a14b-Q4_K_M.gguf",
    ],
)
def test_detect_wan_i2v_a14b(repo_id):
    # The I2V repo ids route to the image-conditioned dual-expert family: the pipeline is
    # WanImageToVideoPipeline and generate() requires a source image. Same DiT pair /
    # second guidance kwarg as T2V-A14B; boundary_ratio (0.9) lives in the pipeline config.
    fam = detect_video_family(repo_id)
    assert fam is not None and fam.name == "wan2.2-i2v-a14b"
    assert fam.pipeline_class == "WanImageToVideoPipeline"
    assert fam.transformer2_class == "WanTransformer3DModel"
    assert fam.is_moe is True
    assert fam.cfg2_kwarg == "guidance_scale_2"
    assert fam.image_conditioned is True
    assert fam.has_audio is False
    assert fam.frame_step == 4
    assert fam.vae_force_fp32 is True
    # Same shipped expert layout as T2V: the table holds the halved bf16-resident sizes.
    assert fam.bf16_components_gb == (57.2, 11.4, 0.5)


def test_i2v_does_not_claim_t2v_ids_and_vice_versa():
    # "wan2.2-i2v" must not swallow the T2V/TI2V ids, and the generic wan families stay
    # text-only (image_conditioned False), so the image gate can't misfire on them.
    assert detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers").name == "wan2.2-t2v-a14b"
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").name == "wan2.2-ti2v-5b"
    assert detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers").image_conditioned is False
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").image_conditioned is False


def test_wan_i2v_generation_defaults():
    # The I2V card recipe (40 steps, CFG 3.5) must beat the generic "wan" 50/5.0 key, and
    # the T2V ids must keep 50/5.0.
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-I2V-A14B-Diffusers") == (40, 3.5)
    assert default_video_generation_params("wan2.2-i2v-a14b-Q4_K_M.gguf") == (40, 3.5)
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-T2V-A14B-Diffusers") == (50, 5.0)
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-TI2V-5B-Diffusers") == (50, 5.0)


def test_detect_wan_overrides():
    # Short aliases the picker / GGUF filenames use resolve to the right family.
    assert detect_video_family("x", override = "wan2.2-5b").name == "wan2.2-ti2v-5b"
    assert detect_video_family("x", override = "wan-ti2v").name == "wan2.2-ti2v-5b"
    assert detect_video_family("x", override = "wan2.2-14b").name == "wan2.2-t2v-a14b"
    assert detect_video_family("x", override = "wan-t2v").name == "wan2.2-t2v-a14b"
    assert detect_video_family("x", override = "wan-i2v").name == "wan2.2-i2v-a14b"
    assert detect_video_family("x", override = "wan2.2-i2v").name == "wan2.2-i2v-a14b"


def test_wan_and_ltx_do_not_cross_route():
    # LTX ids must never resolve to a Wan family and vice versa (separate engines).
    assert detect_video_family("Lightricks/LTX-2").name == "ltx-2"
    assert detect_video_family("unsloth/LTX-2.3-GGUF").name == "ltx-2"
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").name == "wan2.2-ti2v-5b"


def test_sentinels_are_video_specific():
    # The routes match these EXACTLY for 409s; they must not collide with the
    # image sentinels or a video 409 would be mis-attributed.
    assert "video" in VIDEO_NOT_LOADED_MSG.lower()
    assert "video" in VIDEO_CANCELLED_MSG.lower()


def test_resolve_base_repo():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert resolve_video_base_repo(fam, None) == "Lightricks/LTX-2"
    assert resolve_video_base_repo(fam, "  ") == "Lightricks/LTX-2"
    assert resolve_video_base_repo(fam, "other/base") == "other/base"


def test_snap_num_frames_lattice():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    # Valid counts are k * 8 + 1: on-lattice values pass through, everything
    # else floors to the previous lattice point, never below 1.
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
    # The distilled checkpoints run few-step with CFG off; the dev-config base
    # repo wants the full schedule. The picked filename wins over the base repo.
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
        "wan2.2-i2v-a14b",
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
    # A14B's native 720p is the true 16:9 1280x720 (720 = 45*16 renders exactly on the /16 grid),
    # NOT the 1280x704 that TI2V-5B's /32 VAE floors to. The default preset is that native 720p.
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
    # When no identifier names a known variant (a Wan model loaded from an opaque local path under
    # an explicit family_override), the fallback -- the resolved family's own default -- is used,
    # not the hardcoded LTX 40/4.0. Without a family fallback a Wan model would wrongly run 40/4.0.
    assert default_video_generation_params("/models/my-clip", "/models/my-clip") == (40, 4.0)
    assert default_video_generation_params(
        "/models/my-clip", "/models/my-clip", fallback = (50, 5.0)
    ) == (50, 5.0)
    # A recognised token still wins over the fallback.
    assert default_video_generation_params("wan2.2-ti2v-5b", fallback = (8, 1.0)) == (50, 5.0)


def test_generation_defaults_wan_is_segment_not_substring():
    # "wan" must match as a name segment, not a raw substring, so an opaque non-Wan
    # repo/path whose name merely contains the letters "wan" ("swan", "taiwan") does
    # NOT silently pick up Wan's 50-step/CFG-5 schedule ahead of its canonical base repo.
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
    # bf16-RESIDENT transformer sizes. The Wan transformers ship FP32 on disk (safetensors
    # headers are F32; TI2V index 20.0 GB = 5B x 4, A14B 57.15 GB per expert = 14.3B x 4), so the
    # table must hold the HALVED bf16-resident sizes -- ti2v ~10.0, a14b two experts ~57.2 -- NOT
    # the fp32 on-disk sums (20.0 / 114.3). A regression to those over-budgets the plan ~2x and
    # forces needless offload on an 80 GB GPU.
    assert ti2v.bf16_components_gb[0] == 10.0
    assert a14b.bf16_components_gb[0] == 57.2
    # The A14B DiT total (two experts) still dwarfs the single TI2V-5B DiT.
    assert a14b.bf16_components_gb[0] > ti2v.bf16_components_gb[0] * 3
    # A portrait preset is offered for the 5B (a vertical option per the task).
    assert any(h > w for (w, h) in ti2v.resolution_presets)


def test_wan_ti2v_5b_snaps_to_32_not_16():
    # TI2V-5B's VAE is 16x spatial (vae/config.json scale_factor_spatial=16) and the transformer
    # patch is 2, so WanPipeline floors H/W to 16*2 = 32 (pipeline_wan.py:505). The backend must
    # snap to /32 too, or a /16-but-not-/32 request (e.g. 720) is recorded but rendered at 704,
    # desyncing gallery metadata from the actual clip.
    fam = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert fam.resolution_multiple == 32
    assert snap_video_size(fam, 1280, 720) == (1280, 704)  # 720 is /16 but not /32 -> floors to 704
    assert snap_video_size(fam, 1280, 704) == (1280, 704)  # on-grid preset unchanged
    # A14B keeps /16 (its VAE is the Wan2.1 8x VAE, so 8*2 = 16).
    a14b = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert a14b.resolution_multiple == 16


def test_wan_families_force_vae_fp32():
    # Wan's VAE decodes in float32 (diffusers loads AutoencoderKLWan at torch.float32 while the
    # pipe runs bf16); the loader pins the VAE back to fp32 for these families to avoid banding /
    # black frames. LTX-2 keeps the default (its VAE is bf16-native).
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").vae_force_fp32 is True
    assert detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers").vae_force_fp32 is True
    assert detect_video_family("unsloth/LTX-2.3-GGUF").vae_force_fp32 is False


def test_family_size_table_present():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert fam.bf16_components_gb is not None
    transformer_gb, text_encoder_gb, companions_gb = fam.bf16_components_gb
    # The Gemma3-27B text encoder outweighs the DiT itself; a table that lost
    # that would let auto planning under-reserve by ~50 GB.
    assert text_encoder_gb > transformer_gb > 20.0
    assert companions_gb > 0.0


def test_hv15_detection_and_flags():
    fam = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    assert fam is not None and fam.name == "hunyuanvideo-1.5"
    # CFG lives on the guider component (no guidance kwarg in __call__), and the
    # HV15 VAE compresses 16x spatial / 4x temporal.
    assert fam.guidance_via_guider is True
    assert fam.frame_step == 4 and fam.resolution_multiple == 16
    assert fam.has_audio is False
    assert detect_video_family("x/y", override = "hv15") is fam
    # The incompatible HunyuanVideo 1.0 repos must NOT be claimed: their
    # model_index pins HunyuanVideoPipeline, which this family cannot load.
    assert detect_video_family("hunyuanvideo-community/HunyuanVideo") is None


def test_hv15_generation_defaults():
    # The community repacks ship a guider with guidance_scale 6.0 and the
    # pipeline's own 50-step schedule.
    assert default_video_generation_params(
        None, "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    ) == (50, 6.0)


def test_video_prequant_repo_wiring():
    # Every wired repo resolves through the shared family_prequant_repo helper (duck-typed
    # over VideoFamily), and only the schemes the family's deny table allows are wired:
    # Wan carries int8 + fp8 (condition_embedder excluded at build), HunyuanVideo-1.5 is
    # int8-only (fp8 measured broken), and the 720p repack has its OWN repo (different
    # trained weights than 480p).
    from core.inference.diffusion_families import family_prequant_repo

    wan5b = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert family_prequant_repo(wan5b, "int8") == "unsloth/Wan2.2-TI2V-5B-FP8"
    assert family_prequant_repo(wan5b, "fp8") == "unsloth/Wan2.2-TI2V-5B-FP8"
    a14b = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert family_prequant_repo(a14b, "int8") == "unsloth/Wan2.2-T2V-A14B-FP8"
    i2v = detect_video_family("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    assert family_prequant_repo(i2v, "fp8") == "unsloth/Wan2.2-I2V-A14B-FP8"
    hv480 = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    assert family_prequant_repo(hv480, "int8") == "unsloth/HunyuanVideo-1.5-480p-FP8"
    assert family_prequant_repo(hv480, "fp8") is None
    hv720 = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v")
    assert family_prequant_repo(hv720, "int8") == "unsloth/HunyuanVideo-1.5-720p-FP8"
    assert family_prequant_repo(hv720, "int8") != family_prequant_repo(hv480, "int8")
    # LTX-2 base carries int8 + fp8 (both gate-validated); the 2.3 distilled weights get
    # their OWN repo via the variant table, keyed on the lowercased 2.3 base, because a
    # checkpoint baked from the base LTX-2 DiT fails base_model_id validation against 2.3.
    ltx = detect_video_family("Lightricks/LTX-2")
    assert family_prequant_repo(ltx, "int8") == "unsloth/LTX-2-FP8"
    assert family_prequant_repo(ltx, "fp8") == "unsloth/LTX-2-FP8"
    assert family_prequant_repo(ltx, "fp8", "Lightricks/LTX-2.3") == "unsloth/LTX-2.3-FP8"
    assert family_prequant_repo(ltx, "int8", "Lightricks/LTX-2.3") == "unsloth/LTX-2.3-FP8"
    # An unknown LTX variant falls back to the family default (the loader's base_model_id
    # check then refuses a mismatched checkpoint and dense-quantises).
    assert family_prequant_repo(ltx, "fp8", "someone/ltx-finetune") == "unsloth/LTX-2-FP8"


def test_video_prequant_dual_expert_resolution():
    # The A14B repo carries the whole expert pair per scheme; the second expert's file
    # takes the -2 suffix and the transformer_2_ legacy fallback.
    from core.inference.diffusion_prequant import resolve_prequant_source

    fam = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    first = resolve_prequant_source(fam, "int8", expert = "transformer")
    second = resolve_prequant_source(fam, "int8", expert = "transformer_2")
    assert first.location == second.location == "unsloth/Wan2.2-T2V-A14B-FP8"
    assert first.filename == "Wan2.2-T2V-A14B-INT8.pt"
    assert second.filename == "Wan2.2-T2V-A14B-INT8-2.pt"
    assert second.fallback_filename == "transformer_2_int8.pt"


def test_video_prequant_ltx_variant_resolution():
    # The 2.3 base selects the 2.3 repo (its checkpoints stamp base_model_id
    # Lightricks/LTX-2.3); without a base the family default (base LTX-2 weights) resolves.
    from core.inference.diffusion_prequant import resolve_prequant_source

    fam = detect_video_family("Lightricks/LTX-2")
    base = resolve_prequant_source(fam, "fp8")
    assert base.location == "unsloth/LTX-2-FP8"
    assert base.filename == "LTX-2-FP8.pt"
    v23 = resolve_prequant_source(fam, "int8", base_repo = "Lightricks/LTX-2.3")
    assert v23.location == "unsloth/LTX-2.3-FP8"
    assert v23.filename == "LTX-2.3-INT8.pt"
