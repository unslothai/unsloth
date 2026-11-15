# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 on fp16-only GPUs (float32 compute): the 16-bit activation table, scaled 2x, prices the plan instead of
the flat 8 GiB, so a GGUF denoiser that fits stays resident instead of streaming every step."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    DeviceMemory,
    calibrated_image_activation,
    plan_diffusion_memory,
)

# Planner inputs (MiB) logged by a real FLUX.3 klein Q4_K_M load on a Kaggle T4 (float32 compute, dense encoder).
F3_GGUF_T4 = dict(model_dense_mib = 20_920, companion_dense_mib = 17_249, text_encoder_dense_mib = 16_928)
GIB = 1024


def _t4():
    # safe_device_budget_mib 12_758 on that load
    return DeviceMemory("cuda", "cuda:0", "discrete_vram", 14_806, 15_095)


def _plan(act, memory = None):
    return plan_diffusion_memory(
        target = types.SimpleNamespace(
            device = "cuda", backend = "cuda", supports_model_cpu_offload = True
        ),
        device_memory = memory or _t4(),
        runtime_headroom_mib = 8192,
        calibrated_activation = act,
        **F3_GGUF_T4,
    )


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dm.FP32_CALIBRATED_ENV, raising = False)


@pytest.mark.parametrize("family", sorted(dm.FP32_CALIBRATED_FAMILIES))
@pytest.mark.parametrize("max_speed", (False, True))
def test_float32_compute_doubles_every_column(family, max_speed):
    half = calibrated_image_activation(family, max_speed = max_speed)
    full = calibrated_image_activation(family, max_speed = max_speed, compute_bytes = 4)
    assert half == calibrated_image_activation(family, max_speed = max_speed, compute_bytes = 2)
    for name in (
        "text_encoder_mib",
        "denoise_mib",
        "decode_mib",
        "tiled_decode_mib",
        "max_canvas_mib",
    ):
        assert abs(getattr(full, name) - 2 * getattr(half, name)) <= 2
    assert full.resident_only == half.resident_only


def test_other_families_stay_unmeasured_at_float32():
    for family in dm._MEASURED_IMAGE_ACTIVATION_MIB:
        if family in dm.FP32_CALIBRATED_FAMILIES:
            continue
        assert calibrated_image_activation(family, compute_bytes = 4) is None
        assert calibrated_image_activation(family) is not None


def test_kill_switch_restores_the_flat_plan(monkeypatch):
    monkeypatch.setenv(dm.FP32_CALIBRATED_ENV, "0")
    assert calibrated_image_activation("flux.3-klein", compute_bytes = 4) is None
    assert calibrated_image_activation("flux.3-klein") is not None


def test_t4_gguf_keeps_the_denoiser_resident():
    flat = _plan(None)
    assert flat.offload_policy == OFFLOAD_GROUP and flat.stream_transformer
    act = calibrated_image_activation("flux.3-klein", compute_bytes = 4)
    plan = _plan(act)
    assert not plan.stream_transformer
    assert plan.stream_text_encoders
    assert plan.estimates["calibrated_headroom_mib"] == act.headroom(plan.vae_tiling)


def test_t4_under_a_6_gib_budget_still_streams():
    memory = DeviceMemory("cuda", "cuda:0", "discrete_vram", 6 * GIB, 15_095)
    act = calibrated_image_activation("flux.3-klein", compute_bytes = 4)
    assert _plan(act, memory).stream_transformer


def test_calibrated_activation_routes_float32_flux3_only(monkeypatch):
    import torch

    import core.inference.diffusion as d
    import core.inference.diffusion_fp16_guard as guard
    from core.inference.diffusion_families import detect_family

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    monkeypatch.setattr(guard, "fp16_promotes_to_fp32", lambda fam: True)
    flux3 = detect_family("unsloth/flux-3-image-klein-GGUF")
    zimage = detect_family("unsloth/Z-Image-Turbo-GGUF")
    assert flux3.name == "flux.3-klein" and flux3.fp16_incompatible

    def on(dtype):
        return types.SimpleNamespace(backend = "cuda", vendor = "nvidia", dtype = dtype)

    for dtype in (torch.float16, torch.float32):
        act = d._calibrated_activation(flux3, on(dtype), fp32_scaled = True)
        assert act == calibrated_image_activation("flux.3-klein", compute_bytes = 4)
        # the post-load phase seeds (the default) read the table unscaled, so they keep a float32 load unmeasured
        assert d._calibrated_activation(flux3, on(dtype)) is None
        assert d._calibrated_activation(zimage, on(dtype), fp32_scaled = True) is None
    assert d._calibrated_activation(flux3, on(torch.bfloat16)) == calibrated_image_activation(
        "flux.3-klein"
    )
    monkeypatch.setenv(dm.FP32_CALIBRATED_ENV, "0")
    assert d._calibrated_activation(flux3, on(torch.float32), fp32_scaled = True) is None
    assert d._calibrated_activation(flux3, on(torch.bfloat16)) is not None


def test_only_the_load_plan_reads_the_scaled_table():
    import ast
    import inspect

    import core.inference.diffusion as d

    src = ast.unparse(ast.parse(inspect.getsource(d.DiffusionBackend)))
    assert "calibrated_activation=_calibrated_activation(fam, target, fp32_scaled=True)" in src
    assert src.count("fp32_scaled=True") == 1
