# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory mode Low VRAM never keeps more on the card than Balanced.

Whole-module offload holds the whole denoiser for every step, which is above Balanced's streamed blocks whenever the
denoiser outweighs the text encoders: Cosmos3 Nano (no text encoder) measured 18.94 GiB under Low VRAM against
8.95 GiB under Balanced on a 96 GB card. Low VRAM now streams the denoiser and the text encoders."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as diffusion_memory
from core.inference.diffusion_memory import (
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_LOW_VRAM,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    DeviceMemory,
    apply_memory_plan,
    plan_diffusion_memory,
    torchao_offload_plan,
)
from core.inference.video_families import _FAMILIES

MIB_PER_GB = 1000**3 / 1024**2


def _target():
    return types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)


def _card(total_mib):
    return DeviceMemory("cuda", "cuda", "discrete_vram", total_mib, total_mib)


def _plan(
    mode,
    dit,
    te,
    vae,
    *,
    total_mib = 24 * 1024,
    headroom = 4096,
):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = _card(total_mib),
        model_dense_mib = dit + te + vae,
        runtime_headroom_mib = headroom,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        requested_mode = mode,
    )


def _weights_on_card_mib(plan, dit, te, vae):
    """The largest weight set a plan holds on the card at once. Whole-module offload onloads one component whole;
    group offload keeps the companions it does not stream (the streamed block window is the same in both modes)."""
    if plan.offload_policy == OFFLOAD_MODEL:
        return max(dit, te, vae)
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_transformer
    return vae + (0 if plan.stream_text_encoders else te)


# (denoiser per expert, text encoders, VAE) in GB from each family's bf16 table; a MoE onloads one expert at a time.
_VIDEO_SIZES = {
    fam.name: (
        fam.bf16_components_gb[0] / (2 if getattr(fam, "is_moe", False) else 1),
        fam.bf16_components_gb[1],
        fam.bf16_components_gb[2],
    )
    for fam in _FAMILIES
    if fam.bf16_components_gb and not getattr(fam, "modular_workflow", None)
}
_IMAGE_SIZES = {"flux.1-like": (23.8, 9.8, 0.2), "sdxl-like": (5.1, 1.6, 0.2)}


def _mib(gb):
    return int(gb * MIB_PER_GB)


@pytest.mark.parametrize("scale", [1.0, 0.5], ids = ["bf16", "int8"])
@pytest.mark.parametrize("name", sorted(_VIDEO_SIZES) + sorted(_IMAGE_SIZES))
def test_low_vram_holds_no_more_than_balanced(name, scale):
    dit_gb, te_gb, vae_gb = {**_VIDEO_SIZES, **_IMAGE_SIZES}[name]
    dit, te, vae = _mib(dit_gb * scale), _mib(te_gb), _mib(vae_gb)
    low = _plan(MEMORY_MODE_LOW_VRAM, dit, te, vae)
    balanced = _plan(MEMORY_MODE_BALANCED, dit, te, vae)
    assert _weights_on_card_mib(low, dit, te, vae) <= _weights_on_card_mib(balanced, dit, te, vae)


def test_cosmos3_nano_low_vram_streams_the_int8_denoiser():
    """The measured case: Nano's int8 DiT (about 15 GB) with no text encoder stayed whole on the card."""
    dit, te, vae = _mib(30.4 / 2), 0, _mib(1.4)
    plan = _plan(MEMORY_MODE_LOW_VRAM, dit, te, vae, total_mib = 96 * 1024)
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_transformer and plan.stream_text_encoders
    assert plan.vae_tiling and plan.vae_slicing
    assert plan.estimates["low_vram_streamed"] == 1
    assert any("low_vram" in reason and "streamed" in reason for reason in plan.reasons)


def test_low_vram_streams_the_text_encoders_that_balanced_keeps():
    dit, te, vae = _mib(14.3), _mib(11.4), _mib(0.5)
    low = _plan(MEMORY_MODE_LOW_VRAM, dit, te, vae)
    balanced = _plan(MEMORY_MODE_BALANCED, dit, te, vae)
    assert balanced.offload_policy == OFFLOAD_GROUP and not balanced.stream_text_encoders
    assert low.offload_policy == OFFLOAD_GROUP and low.stream_text_encoders


def test_low_vram_without_a_size_split_still_streams():
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _card(80000),
        model_dense_mib = 1000,
        runtime_headroom_mib = 1000,
        requested_mode = MEMORY_MODE_LOW_VRAM,
    )
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_text_encoders


def test_a_torchao_scheme_that_cannot_stream_keeps_whole_module_offload(monkeypatch):
    """Before streaming was the default, low_vram served such a scheme under whole-module offload; it still does."""
    monkeypatch.setattr(diffusion_memory, "torchao_scheme_streams", lambda *a, **k: False)
    dit, te, vae = _mib(8.0), _mib(4.0), _mib(0.3)
    low = _plan(MEMORY_MODE_LOW_VRAM, dit, te, vae)
    placed = torchao_offload_plan(low, "fp8")
    assert placed is not None and placed.offload_policy == OFFLOAD_MODEL
    assert not placed.stream_text_encoders and placed.vae_tiling
    # Balanced is unchanged: a scheme that cannot stream has no placement under group offload.
    assert torchao_offload_plan(_plan(MEMORY_MODE_BALANCED, dit, te, vae), "fp8") is None


class _UNetPipe:
    def __init__(self):
        self.unet = object()
        self.transformer = None
        self.calls: list[str] = []

    def enable_model_cpu_offload(self, device = None):
        self.calls.append("model_offload")

    def enable_vae_tiling(self):
        self.calls.append("vae_tiling")

    def enable_vae_slicing(self):
        self.calls.append("vae_slicing")


def test_a_unet_takes_the_whole_module_tier_under_low_vram():
    """Group offload streams transformer blocks only; a U-Net keeps low_vram's whole-module tier."""
    pipe = _UNetPipe()
    plan = _plan(MEMORY_MODE_LOW_VRAM, _mib(5.1), _mib(1.6), _mib(0.2))
    effective, tiled = apply_memory_plan(pipe, plan, device = "cuda")
    assert effective == OFFLOAD_MODEL and tiled is True
    assert "model_offload" in pipe.calls
    assert pipe._unsloth_applied_plan.offload_policy == OFFLOAD_MODEL
