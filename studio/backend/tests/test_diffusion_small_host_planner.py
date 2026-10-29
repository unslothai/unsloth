# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Planner inputs the small-host route depends on, kept apart from ``test_diffusion_small_host.py`` so they run
against trees without that module: Qwen-Image computes in fp32 on an fp16-only card (fp16 renders NaN / black), and
the dense measured activation peak scales with that compute dtype. CPU only."""

from __future__ import annotations

import pytest
import torch

import core.inference.diffusion_memory as dm


def test_qwen_image_is_fp16_incompatible():
    from core.inference.diffusion import _resolve_diffusion_compute_dtype
    from core.inference.diffusion_families import detect_family
    for name in ("qwen-image", "qwen-image-edit"):
        fam = detect_family("", override = name)
        assert fam.fp16_incompatible, name
        assert _resolve_diffusion_compute_dtype(fam, torch.float16) is torch.float32
        assert _resolve_diffusion_compute_dtype(fam, torch.bfloat16) is torch.bfloat16


def test_dense_measured_peak_scales_with_fp32_compute(monkeypatch):
    for name in (dm.MEASURED_ACTIVATION_ENV, "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE"):
        monkeypatch.delenv(name, raising = False)
    fp16 = dm.measured_image_runtime_mib("qwen-image", "off", dense_transformer_mib = 19525)
    fp32 = dm.measured_image_runtime_mib(
        "qwen-image", "off", dense_transformer_mib = 19525, compute_bytes = 4
    )
    # measured: 4248 MiB above the weights at fp16, 8480 at fp32
    assert fp16 == 5120 and fp32 >= 8480 * 1.15


@pytest.mark.parametrize("bf16_table", [True, False])
def test_dense_table_places_bf16_only_from_its_own_measurements(monkeypatch, bf16_table):
    """bf16 is priced only by its own table; with it off a bf16 card keeps the flat plan."""
    import types

    for name in (
        dm.MEASURED_ACTIVATION_ENV,
        "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
        dm.PARTIAL_RESIDENT_ENV,
        dm.MEASURED_ACTIVATION_BF16_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(
        dm,
        "_loaded_component_mib",
        lambda pipe: {
            "transformer": (22700, "dit"),
            "text_encoder_2": (9346, "text_encoder"),
            "vae": (160, "other"),
        },
    )
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 24000, 24564)
    plan = dm.MemoryPlan(
        requested_mode = dm.MEMORY_MODE_AUTO,
        offload_policy = dm.OFFLOAD_GROUP,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = memory,
        estimates = {"safe_device_budget_mib": 21500, "base_overhead_mib": 2048},
        stream_text_encoders = True,
    )
    if not bf16_table:
        monkeypatch.setenv(dm.MEASURED_ACTIVATION_BF16_ENV, "0")
    for dtype, kept in ((torch.bfloat16, bf16_table), (torch.float16, True)):
        pipe = types.SimpleNamespace(
            transformer = types.SimpleNamespace(dtype = dtype), _unsloth_dense_tier_seeds = True
        )
        out = dm.refine_plan_from_loaded_weights(pipe, plan, family = "flux.1", speed_mode = "off")
        assert bool(out.resident_transformer_mib) is kept, dtype
        assert bool(out.estimates.get("measured_dense_bf16")) is (
            dtype == torch.bfloat16 and bf16_table
        )
