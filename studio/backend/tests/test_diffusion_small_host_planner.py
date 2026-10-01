# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Planner inputs the small-host route depends on, kept apart from ``test_diffusion_small_host.py`` so they run
against trees without that module: Qwen-Image computes in fp32 on an fp16-only card (fp16 renders NaN / black), and
the dense measured activation peak scales with that compute dtype. CPU only."""

from __future__ import annotations

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
