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
    assert snap_video_size(fam, 1280, 704) == (1280, 704)  # on-grid preset
    assert snap_video_size(fam, 1000, 700) == (992, 688)


def test_wan_generation_defaults():
    # Both Wan families default to the pipeline's 50 steps / CFG 5.0.
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-TI2V-5B-Diffusers") == (50, 5.0)
    assert default_video_generation_params(None, "Wan-AI/Wan2.2-T2V-A14B-Diffusers") == (50, 5.0)
    # A GGUF filename carrying the family name still lands on the Wan defaults.
    assert default_video_generation_params(
        "wan2.2-ti2v-5b-Q4_K_M.gguf", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    ) == (50, 5.0)


def test_wan_size_tables_present():
    ti2v = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    a14b = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert ti2v.bf16_components_gb is not None and a14b.bf16_components_gb is not None
    # The A14B DiT total (two experts) dwarfs the single TI2V-5B DiT.
    assert a14b.bf16_components_gb[0] > ti2v.bf16_components_gb[0] * 3
    # A portrait preset is offered for the 5B (a vertical option per the task).
    assert any(h > w for (w, h) in ti2v.resolution_presets)


def test_family_size_table_present():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert fam.bf16_components_gb is not None
    transformer_gb, text_encoder_gb, companions_gb = fam.bf16_components_gb
    # The Gemma3-27B text encoder outweighs the DiT itself; a table that lost
    # that would let auto planning under-reserve by ~50 GB.
    assert text_encoder_gb > transformer_gb > 20.0
    assert companions_gb > 0.0
