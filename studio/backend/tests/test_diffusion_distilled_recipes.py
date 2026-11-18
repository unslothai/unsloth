# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Distilled checkpoints resolve their publishers' steps / guidance, not their base family's recipe."""

from __future__ import annotations

import pytest

from core.inference import diffusion as dmod
from core.inference import diffusion_families as df

# (repo_id, file, base_repo) -> (steps, guidance)
_HUNYUAN_DISTILLED = [
    ("hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers", None, None),
    ("tencent/HunyuanImage-2.1", "dit/hunyuanimage2.1-distilled.safetensors", None),
    ("tencent/HunyuanImage-2.1", "dit/hunyuanimage2.1-distilled_fp8.safetensors", None),
    (
        "Comfy-Org/Hunyuan-Image-2.1",
        "split_files/diffusion_models/hunyuanimage2.1_distilled_bf16.safetensors",
        "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    ),
]
_ANIMA_TURBO = [
    ("circlestone-labs/Anima", "split_files/diffusion_models/anima-turbo-v1.0.safetensors"),
    ("circlestone-labs/Anima", "split_files/diffusion_models/anima-turbo-v1.1.safetensors"),
    ("/models/anima", "anima-turbo-v1.1.safetensors"),
    ("someone/Anima-1.0-Turbo", None),
]
_CHROMA_FLASH = [
    ("lodestones/Chroma1-Flash", None, None),
    ("lodestones/Chroma1-Flash", "Chroma1-HD-Flash.safetensors", "lodestones/Chroma1-HD"),
    ("silveroxides/Chroma1-Flash-GGUF", "Chroma1-HD-Flash-Q8_0.gguf", "lodestones/Chroma1-HD"),
]


def _fam(name):
    return df.detect_family("", override = name)


def _status_recipe(repo, gguf, base, family):
    got = dmod._generation_defaults_for(repo, gguf, base, family = _fam(family))
    return got["steps"], got["guidance"]


@pytest.mark.parametrize("repo, gguf, base", _HUNYUAN_DISTILLED)
def test_hunyuanimage_distilled_runs_tencents_8_step_recipe(repo, gguf, base):
    # tencent/HunyuanImage-2.1 README: distilled 8 steps, guidance 3.25 (base 50).
    assert _status_recipe(repo, gguf, base, "hunyuanimage-2.1") == (8, 3.25)
    assert df.default_generation_params(gguf, repo, base) == (8, 3.25)


def test_hunyuanimage_base_keeps_its_recipe():
    base = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"
    assert _status_recipe(base, None, None, "hunyuanimage-2.1") == (50, 3.25)
    assert df.default_generation_params(
        "split_files/diffusion_models/hunyuanimage2.1_bf16.safetensors",
        "Comfy-Org/Hunyuan-Image-2.1",
    ) == (50, 3.25)


@pytest.mark.parametrize("repo, gguf", _ANIMA_TURBO)
def test_anima_turbo_runs_cfg_1_and_10_steps(repo, gguf):
    # circlestone-labs/Anima README: "Use at CFG 1 and 8-12 steps".
    assert _status_recipe(repo, gguf, None, "anima") == (10, 1.0)


@pytest.mark.parametrize(
    "repo, gguf",
    [
        ("circlestone-labs/Anima", "split_files/diffusion_models/anima-base-v1.0.safetensors"),
        ("circlestone-labs/Anima", "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors"),
        ("/models/x", "oneObsession_anima29BV2.safetensors"),
        ("circlestone-labs/Anima-Base-v1.0-Diffusers", None),
    ],
)
def test_other_anima_files_keep_the_base_recipe(repo, gguf):
    assert _status_recipe(repo, gguf, None, "anima") == (30, 4.0)


def test_turbo_outside_anima_is_not_read_as_anima_turbo():
    assert df.default_generation_params("Tongyi-MAI/Z-Image-Turbo") == (8, 0.0)
    assert df.default_generation_params(
        "animagineXL40_v4Opt.safetensors", "stabilityai/stable-diffusion-xl-base-1.0"
    ) == (25, 7.0)


@pytest.mark.parametrize("repo, gguf, base", _CHROMA_FLASH)
def test_chroma1_flash_runs_cfg_1(repo, gguf, base):
    # lodestones/Chroma1-Flash commit 31f035ba04: "flash rev-0.1 use heun 8 steps CFG=1" (16 Euler steps here).
    assert _status_recipe(repo, gguf, base, "chroma") == (16, 1.0)
    assert df.default_generation_params(gguf, repo, base) == (16, 1.0)


def test_chroma1_hd_keeps_its_recipe():
    assert _status_recipe("lodestones/Chroma1-HD", None, None, "chroma") == (26, 3.5)


def test_load_time_lookups_name_the_turbo_file_too():
    # The load-time step-cache and offload pricing lookups pass the same names as the status recipe.
    ids = dmod._family_default_ids(
        _fam("anima"),
        "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
        "circlestone-labs/Anima",
    )
    assert df.default_generation_params(*ids, "circlestone-labs/Anima") == (10, 1.0)
    assert dmod._family_default_ids(_fam("anima"), None, "circlestone-labs/Anima") == (
        _fam("anima").base_repo,
    )
