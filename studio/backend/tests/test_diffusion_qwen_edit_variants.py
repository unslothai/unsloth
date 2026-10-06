# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Edit variants share one family (and the 2511 companion repo) but not one transformer config or
ComfyUI shift: only the 2511 config sets zero_cond_t, and the 2509 and original templates sample at shift 3."""

from __future__ import annotations

import pytest

from core.inference.diffusion_families import (
    comfy_flow_shift_for,
    detect_family,
    detect_family_for_pick,
    transformer_config_overrides_for,
)

_OFF = {"zero_cond_t": False}


@pytest.mark.parametrize(
    "identifiers, expected",
    [
        # (gguf filename, repo id, base repo), as the loader passes them.
        (("qwen-image-edit-2509-Q6_K.gguf", "unsloth/Qwen-Image-Edit-2509-GGUF", "Qwen/Qwen-Image-Edit-2511"), _OFF),
        ((None, "unsloth/Qwen-Image-Edit-2509-GGUF", "Qwen/Qwen-Image-Edit-2511"), _OFF),
        ((None, "/models/local", "Qwen/Qwen-Image-Edit-2509"), _OFF),
        # ComfyUI-style file names under a folder that names nothing.
        (("qwen_image_edit_2509_Q4_K_M.gguf", "/models/unet", "Qwen/Qwen-Image-Edit-2511"), _OFF),
        (("qwen-image-edit-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-GGUF", "Qwen/Qwen-Image-Edit-2511"), _OFF),
        ((None, "QuantStack/Qwen-Image-Edit-GGUF", "Qwen/Qwen-Image-Edit"), _OFF),
        (("qwen-image-edit-2511-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-2511-GGUF", "Qwen/Qwen-Image-Edit-2511"), {}),
        (("qwen_image_edit_2511_bf16.gguf", "/models/unet", "Qwen/Qwen-Image-Edit-2511"), {}),
        # A file that names no variant keeps the family base (2511) config, as before.
        (("model.gguf", "/models/unet", "Qwen/Qwen-Image-Edit-2511"), {}),
    ],
)
def test_qwen_edit_transformer_config_per_variant(identifiers, expected):
    fam = detect_family("qwen-image-edit")
    assert fam.base_repo == "Qwen/Qwen-Image-Edit-2511"
    assert transformer_config_overrides_for(fam, *identifiers) == expected


def test_variant_lookup_returns_a_fresh_dict():
    fam = detect_family("qwen-image-edit")
    first = transformer_config_overrides_for(fam, "qwen-image-edit-2509-Q6_K.gguf")
    first["zero_cond_t"] = True
    assert transformer_config_overrides_for(fam, "qwen-image-edit-2509-Q6_K.gguf") == _OFF


@pytest.mark.parametrize(
    "repo_id, gguf_filename",
    [
        ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q4_K_M.gguf"),
        ("unsloth/Qwen-Image-GGUF", "qwen-image-Q4_K_M.gguf"),
        ("unsloth/FLUX.1-Kontext-dev-GGUF", "flux1-kontext-dev-Q4_K_M.gguf"),
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    ],
)
def test_other_families_get_no_config_override(repo_id, gguf_filename):
    fam = detect_family_for_pick(repo_id, gguf_filename)
    assert fam is not None and fam.name != "qwen-image-edit"
    assert transformer_config_overrides_for(fam, gguf_filename, repo_id, fam.base_repo) == {}


def test_qwen_edit_flow_shift_per_variant():
    # Comfy-Org templates: image_qwen_image_edit and image_qwen_image_edit_2509 use ModelSamplingAuraFlow 3,
    # image_qwen_image_edit_2511 uses 3.1.
    fam = detect_family("qwen-image-edit")
    base = fam.base_repo
    assert comfy_flow_shift_for(fam, "qwen-image-edit-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-GGUF", base) == 3.0
    assert comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-Edit", None) == 3.0
    assert comfy_flow_shift_for(fam, "qwen-image-edit-2509-Q6_K.gguf", "unsloth/Qwen-Image-Edit-2509-GGUF", base) == 3.0
    assert comfy_flow_shift_for(fam, "qwen-image-edit-2511-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-2511-GGUF", base) == 3.1
    assert comfy_flow_shift_for(fam, None, "unsloth/Qwen-Image-Edit-2511", None) == 3.1
    # Nothing named: the family base (2511) decides, as before.
    assert comfy_flow_shift_for(fam, "model.gguf", "/models/unet", base) == 3.1
    assert comfy_flow_shift_for(fam, None, None, None) == 3.1
