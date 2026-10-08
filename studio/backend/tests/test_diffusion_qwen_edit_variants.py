# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Edit variants: only 2511 sets zero_cond_t; 2509 and the original sample at shift 3."""

from __future__ import annotations

import pytest

from core.inference.diffusion_families import (
    comfy_flow_shift_for,
    detect_family,
    detect_family_for_pick,
    transformer_config_overrides_for,
    transformer_variant_differs_from_base,
)

_OFF = {"zero_cond_t": False}


@pytest.mark.parametrize(
    "identifiers, expected",
    [
        (
            (
                "qwen-image-edit-2509-Q6_K.gguf",
                "unsloth/Qwen-Image-Edit-2509-GGUF",
                "Qwen/Qwen-Image-Edit-2511",
            ),
            _OFF,
        ),
        ((None, "unsloth/Qwen-Image-Edit-2509-GGUF", "Qwen/Qwen-Image-Edit-2511"), _OFF),
        ((None, "/models/local", "Qwen/Qwen-Image-Edit-2509"), _OFF),
        (("qwen_image_edit_2509_Q4_K_M.gguf", "/models/unet", "Qwen/Qwen-Image-Edit-2511"), _OFF),
        (
            (
                "qwen-image-edit-Q4_K_M.gguf",
                "unsloth/Qwen-Image-Edit-GGUF",
                "Qwen/Qwen-Image-Edit-2511",
            ),
            _OFF,
        ),
        ((None, "QuantStack/Qwen-Image-Edit-GGUF", "Qwen/Qwen-Image-Edit"), _OFF),
        (
            (
                "qwen-image-edit-2511-Q4_K_M.gguf",
                "unsloth/Qwen-Image-Edit-2511-GGUF",
                "Qwen/Qwen-Image-Edit-2511",
            ),
            {},
        ),
        (("qwen_image_edit_2511_bf16.gguf", "/models/unet", "Qwen/Qwen-Image-Edit-2511"), {}),
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
    # Comfy-Org image_qwen_image_edit{,_2509} templates: ModelSamplingAuraFlow 3; _2511: 3.1.
    fam = detect_family("qwen-image-edit")
    base = fam.base_repo
    assert (
        comfy_flow_shift_for(
            fam, "qwen-image-edit-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-GGUF", base
        )
        == 3.0
    )
    assert comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-Edit", None) == 3.0
    assert (
        comfy_flow_shift_for(
            fam, "qwen-image-edit-2509-Q6_K.gguf", "unsloth/Qwen-Image-Edit-2509-GGUF", base
        )
        == 3.0
    )
    assert (
        comfy_flow_shift_for(
            fam, "qwen-image-edit-2511-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-2511-GGUF", base
        )
        == 3.1
    )
    assert comfy_flow_shift_for(fam, None, "unsloth/Qwen-Image-Edit-2511", None) == 3.1
    assert comfy_flow_shift_for(fam, "model.gguf", "/models/unet", base) == 3.1
    assert comfy_flow_shift_for(fam, None, None, None) == 3.1


@pytest.mark.parametrize(
    "gguf_filename, shift",
    [
        ("qwen_image_edit_2509_Q4_K_M.gguf", 3.0),
        ("Qwen_Image_Edit-Q4_0.gguf", 3.0),
        ("qwen_image_edit_2511_Q4_K_M.gguf", 3.1),
    ],
)
def test_comfy_style_names_pick_their_shift(gguf_filename, shift):
    fam = detect_family("qwen-image-edit")
    assert comfy_flow_shift_for(fam, gguf_filename, "/models/unet", fam.base_repo) == shift


@pytest.mark.parametrize(
    "base, gguf_filename, differs",
    [
        ("Qwen/Qwen-Image-Edit-2511", "qwen-image-edit-2509-Q6_K.gguf", True),
        ("Qwen/Qwen-Image-Edit-2511", "qwen_image_edit_Q4_K_M.gguf", True),
        ("Qwen/Qwen-Image-Edit-2511", "qwen-image-edit-2511-Q4_K_M.gguf", False),
        ("Qwen/Qwen-Image-Edit-2511", "model.gguf", False),
        ("Qwen/Qwen-Image-Edit-2509", "qwen-image-edit-2509-Q6_K.gguf", False),
        ("/models/qwen-edit-pipeline", "qwen-image-edit-2509-Q6_K.gguf", False),
    ],
)
def test_variant_differs_from_base(base, gguf_filename, differs):
    fam = detect_family("qwen-image-edit")
    assert (
        transformer_variant_differs_from_base(fam, base, gguf_filename, "/models/unet") is differs
    )


def test_other_families_never_differ_from_base():
    fam = detect_family("qwen-image")
    assert not transformer_variant_differs_from_base(
        fam, fam.base_repo, "qwen-image-edit-2509-Q6_K.gguf"
    )


@pytest.mark.parametrize(
    "gguf_filename, stages_transformer",
    [("qwen-image-edit-2509-Q4_K_M.gguf", False), ("qwen-image-edit-2511-Q4_K_M.gguf", True)],
)
def test_download_plan_skips_the_2511_transformer_for_a_variant(
    monkeypatch, gguf_filename, stages_transformer
):
    import core.inference.diffusion as dmod

    monkeypatch.setattr(dmod, "_resolve_base_repo", lambda *a, **k: "Qwen/Qwen-Image-Edit-2511")
    monkeypatch.setattr(dmod, "_assert_base_repo_accessible", lambda *a, **k: None)
    monkeypatch.setattr(
        dmod.DiffusionBackend, "_te_prequant_plan_files", staticmethod(lambda *a, **k: {})
    )
    monkeypatch.setattr(
        dmod.DiffusionBackend, "_dense_quant_prefetch_decision", lambda self, *a, **k: True
    )
    asked = []

    def estimate(
        *a,
        include_transformer = False,
        **k,
    ):
        asked.append(include_transformer((), ("transformer/x.safetensors",)))
        return 0, []

    monkeypatch.setattr(dmod.DiffusionBackend, "_estimate_download_bytes", staticmethod(estimate))
    dmod.DiffusionBackend().download_plan(
        "unsloth/Qwen-Image-Edit-GGUF", gguf_filename = gguf_filename, transformer_quant = "fp8"
    )
    assert asked == [stages_transformer]
