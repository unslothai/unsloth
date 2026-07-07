# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the pure per-family inference-info helper.

Covers ``family_inference_infos()``: every auto-policy family appears, the component sizes
round-trip, and the quant estimates order correctly (quantised < bf16, nvfp4 < int8). No
torch / diffusers / GPU."""

from __future__ import annotations

from core.inference.diffusion_auto_policy import _FAMILY_BF16_GB, _QUANT_STEADY_FACTOR
from core.inference.diffusion_inference_info import family_inference_infos


def test_covers_every_auto_policy_family():
    infos = family_inference_infos()
    names = {info["family"] for info in infos}
    assert names == set(_FAMILY_BF16_GB), "info must list exactly the auto-policy families"
    # One entry per family, in registry order.
    assert [info["family"] for info in infos] == list(_FAMILY_BF16_GB)


def test_each_family_reports_all_schemes():
    for info in family_inference_infos():
        estimated = info["estimated_resident_gb"]
        assert set(estimated) == {"bf16", *_QUANT_STEADY_FACTOR}
        # Every reported value is a float rounded to one decimal.
        for value in estimated.values():
            assert isinstance(value, float)
            assert round(value, 1) == value


def test_component_sizes_match_the_table():
    infos = {info["family"]: info for info in family_inference_infos()}
    for name, (transformer, text_encoders, vae) in _FAMILY_BF16_GB.items():
        info = infos[name]
        assert info["transformer_bf16_gb"] == round(transformer, 1)
        assert info["text_encoders_bf16_gb"] == round(text_encoders, 1)
        assert info["vae_bf16_gb"] == round(vae, 1)


def test_quantised_estimate_is_below_bf16():
    # A quantised transformer is smaller than bf16, so its resident estimate must be too
    # (the companions are shared, and every steady factor is < 1).
    for info in family_inference_infos():
        estimated = info["estimated_resident_gb"]
        for scheme in _QUANT_STEADY_FACTOR:
            assert estimated[scheme] < estimated["bf16"], f"{info['family']} {scheme}"


def test_nvfp4_is_below_int8():
    # nvfp4 packs two params per byte (~0.33x) vs int8's one byte per param (~0.55x), so
    # nvfp4's estimate is the smaller of the two on every family.
    for info in family_inference_infos():
        estimated = info["estimated_resident_gb"]
        assert estimated["nvfp4"] < estimated["int8"], info["family"]
