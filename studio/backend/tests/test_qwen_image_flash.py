# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""nvidia/Qwen-Image-Flash (a 4-step DMD2 student of Qwen-Image): runs its own recipe (4 steps, CFG 1) on its shipped
static shift-3 grid, and never seeds its denoiser from the teacher's hosted quantized artifacts, even though its card
tags Qwen/Qwen-Image as the base. CPU only, no network."""

from __future__ import annotations

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference import diffusion_families as df

FLASH = "nvidia/Qwen-Image-Flash"


@pytest.fixture
def fam():
    family = df.detect_family(FLASH)
    assert family is not None and family.name == "qwen-image"
    return family


def test_flash_runs_its_own_recipe_and_the_base_keeps_its_own():
    assert df.default_generation_params(FLASH) == (4, 1.0)
    assert df.default_generation_params("/models/qwen_image_flash") == (4, 1.0)
    assert df.default_generation_params("Qwen/Qwen-Image") == (20, 4.0)
    assert df.default_generation_params("Qwen/Qwen-Image-2512") == (50, 4.0)


def test_flash_keeps_its_shipped_shift_3_schedule(fam):
    # None = keep the shipped scheduler (static shift 3.0); the family's 3.1 still applies to the base.
    assert df.comfy_flow_shift_for(fam, FLASH) is None
    assert df.comfy_flow_shift_for(fam, "Qwen/Qwen-Image") == 3.1


def test_flash_never_takes_the_teacher_s_hosted_artifacts(fam):
    for scheme in ("int8", "fp8"):
        assert df.family_prequant_repo(fam, scheme, base_repo = FLASH) is None
        assert (
            df.family_prequant_repo(fam, scheme, base_repo = "Qwen/Qwen-Image")
            == "unsloth/Qwen-Image-FP8"
        )


def test_flash_resolves_to_itself_despite_its_card_tag(fam, monkeypatch):
    monkeypatch.setattr(diffusion_mod, "_hf_base_model", lambda *a, **k: "Qwen/Qwen-Image")
    monkeypatch.setattr(diffusion_mod, "_remember_companion_base", lambda *a, **k: None)
    assert diffusion_mod._resolve_base_repo(FLASH, None, fam, None) == FLASH
    # A plain fine-tune whose name spells no curated variant still follows its card tag.
    assert (
        diffusion_mod._resolve_base_repo("someone/qwen-image-anime", None, fam, None)
        == "Qwen/Qwen-Image"
    )
    # An explicit base from the caller always wins.
    assert (
        diffusion_mod._resolve_base_repo(FLASH, "Qwen/Qwen-Image", fam, None) == "Qwen/Qwen-Image"
    )


def test_flash_is_a_trusted_full_pipeline_pick():
    # Full pipelines load only from trusted repos; Flash is a curated variant, so its pipeline pick is allowed.
    assert diffusion_mod._is_trusted_diffusion_repo(FLASH)
    assert not diffusion_mod._is_trusted_diffusion_repo("someone/qwen-image-flash-copy")


@pytest.mark.parametrize(
    "family_id, repo_id, tag, expected, own_fp8",
    [
        # Turbo is a distill of 2.1 whose card tags 2.1: a non-pipeline pick (GGUF / single file / planner) now
        # resolves to Turbo and Turbo's own FP8 build, not 2.1's.
        (
            "Qwen/Qwen-Image-2.1",
            "/models/qwen_image_2.1_turbo",
            "Qwen/Qwen-Image-2.1",
            "Qwen/Qwen-Image-2.1-Turbo",
            "unsloth/Qwen-Image-2.1-Turbo-FP8",
        ),
        # MOSS-Image-Base (a named variant, excluded from the SFT artifacts) tagged as the SFT model: its own FP8.
        (
            "OpenMOSS-Team/MOSS-Image-SFT",
            "/models/moss_image_base",
            "OpenMOSS-Team/MOSS-Image-SFT",
            "OpenMOSS-Team/MOSS-Image-Base",
            "unsloth/MOSS-Image-Base-FP8",
        ),
    ],
)
def test_a_named_variant_tagged_as_its_teacher_resolves_to_its_own_weights(
    monkeypatch, family_id, repo_id, tag, expected, own_fp8
):
    family = df.detect_family(family_id)
    monkeypatch.setattr(diffusion_mod, "_hf_base_model", lambda *a, **k: tag)
    monkeypatch.setattr(diffusion_mod, "_remember_companion_base", lambda *a, **k: None)
    base = diffusion_mod._resolve_base_repo(repo_id, None, family, None)
    assert base == expected
    assert df.family_prequant_repo(family, "fp8", base_repo = base) == own_fp8
    # The teacher itself still resolves to itself.
    assert diffusion_mod._resolve_base_repo(family_id, None, family, None) == family_id
