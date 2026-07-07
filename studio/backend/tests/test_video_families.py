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
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers") is None  # V3 family
    # A short alias must not match inside an unrelated word.
    assert detect_video_family("someorg/deluxtreme-model") is None


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
    assert supported_video_family_names() == ("ltx-2",)


def test_family_size_table_present():
    fam = detect_video_family("unsloth/LTX-2.3-GGUF")
    assert fam.bf16_components_gb is not None
    transformer_gb, text_encoder_gb, companions_gb = fam.bf16_components_gb
    # The Gemma3-27B text encoder outweighs the DiT itself; a table that lost
    # that would let auto planning under-reserve by ~50 GB.
    assert text_encoder_gb > transformer_gb > 20.0
    assert companions_gb > 0.0
