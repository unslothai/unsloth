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
