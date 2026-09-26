# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured activation planning: faster tiers only, more VRAM never slower, flat estimate everywhere else."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_FAST,
    MEMORY_MODE_LOW_VRAM,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    CalibratedImageActivation,
    DeviceMemory,
    calibrated_image_activation,
    plan_diffusion_memory,
)

# Planner inputs logged by real loads (MiB): transformer + companions, companions, text encoders.
QWEN21_GGUF = dict(model_dense_mib = 17_897, companion_dense_mib = 10_247, text_encoder_dense_mib = 8_959)
ZIMAGE_BF16 = dict(model_dense_mib = 31_326, companion_dense_mib = 7_820, text_encoder_dense_mib = 7_629)
FLUX1_BF16 = dict(model_dense_mib = 32_184, companion_dense_mib = 9_478, text_encoder_dense_mib = 9_318)
SDXL = dict(model_dense_mib = 13_235, companion_dense_mib = 13_232, text_encoder_dense_mib = 3_119)
QWEN21_ACT = CalibratedImageActivation(
    text_encoder_mib = 1_498,
    denoise_mib = 794,
    decode_mib = 9_172,
    tiled_decode_mib = 590,
    max_canvas_mib = 2_960,
)
SMALL_VAE_ACT = CalibratedImageActivation(
    text_encoder_mib = 1_200,
    denoise_mib = 900,
    decode_mib = 3_000,
    tiled_decode_mib = 3_000,
    max_canvas_mib = 3_600,
)
GIB = 1024


def _target():
    return types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)


def _card(total_gib, kind = "discrete_vram"):
    # a desktop plus the CUDA context already hold 1.2 GiB, as on a real card
    total = int(total_gib * GIB)
    return DeviceMemory("cuda", "cuda", kind, total - 1_229, total)


def _plan(
    total_gib,
    sizes,
    act,
    mode = None,
    explicit = False,
    kind = "discrete_vram",
):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = _card(total_gib, kind),
        runtime_headroom_mib = 8192,
        requested_mode = mode,
        explicit_offload = explicit,
        calibrated_activation = act,
        **sizes,
    )


def _rank(plan, sizes):
    """Speed rank (0 fastest); whole-module offload of a component over budget is streamed after load."""
    budget = plan.estimates["safe_device_budget_mib"]
    transformer = sizes["model_dense_mib"] - sizes["companion_dense_mib"]
    if plan.offload_policy == OFFLOAD_NONE:
        return 0
    if plan.offload_policy == OFFLOAD_GROUP and not plan.stream_transformer:
        return 2 if plan.vae_tiling else 1
    if plan.offload_policy == OFFLOAD_MODEL:
        if max(transformer, sizes["text_encoder_dense_mib"]) > budget:
            return 6
        return 4 if plan.vae_tiling else 3
    return 5


def test_16gb_qwen_image_21_keeps_the_transformer_resident_instead_of_streaming_every_step():
    flat = _plan(16, QWEN21_GGUF, None)
    assert flat.offload_policy == OFFLOAD_GROUP and flat.stream_transformer
    plan = _plan(16, QWEN21_GGUF, QWEN21_ACT)
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_transformer is False and plan.stream_text_encoders is True
    assert plan.vae_tiling is True and plan.vae_slicing is True
    assert "tiled" in plan.reasons[-1]


def test_14gb_qwen_image_21_offloads_whole_modules_instead_of_streaming_every_step():
    flat = _plan(15, QWEN21_GGUF, None)
    assert flat.offload_policy == OFFLOAD_GROUP and flat.stream_transformer
    plan = _plan(15, QWEN21_GGUF, QWEN21_ACT)
    assert plan.offload_policy == OFFLOAD_MODEL and plan.vae_tiling is True


def test_12gb_keeps_tiling_when_the_measured_decode_does_not_fit():
    flat = _plan(12, QWEN21_GGUF, None)
    plan = _plan(12, QWEN21_GGUF, QWEN21_ACT)
    assert flat.offload_policy == plan.offload_policy == OFFLOAD_MODEL
    assert flat.vae_tiling is plan.vae_tiling is True
    assert plan == flat


def test_whole_module_offload_untiles_a_decode_that_fits():
    flat = _plan(12, QWEN21_GGUF, None)
    plan = _plan(12, QWEN21_GGUF, SMALL_VAE_ACT)
    assert flat.offload_policy == plan.offload_policy == OFFLOAD_MODEL
    assert flat.vae_tiling is True and plan.vae_tiling is False


def test_whole_module_offload_untiles_when_the_resident_transformer_does_not_fit():
    act = CalibratedImageActivation(1_000, 800, 3_000, 600, 2_000)
    sizes = dict(model_dense_mib = 23_897, companion_dense_mib = 10_247, text_encoder_dense_mib = 8_959)
    plan = _plan(19, sizes, act)
    assert plan.offload_policy == OFFLOAD_MODEL and plan.vae_tiling is False


@pytest.mark.parametrize(
    "sizes, act",
    [
        (QWEN21_GGUF, QWEN21_ACT),
        (QWEN21_GGUF, SMALL_VAE_ACT),
        (ZIMAGE_BF16, SMALL_VAE_ACT),
        (FLUX1_BF16, SMALL_VAE_ACT),
        (SDXL, SMALL_VAE_ACT),
        (ZIMAGE_BF16, CalibratedImageActivation(1_000, 800, 30_000, 600, 1_000)),
        (QWEN21_GGUF, CalibratedImageActivation(1_000, 800, 3_000, 600, 9_000)),
    ],
)
def test_more_vram_is_never_slower_and_never_slower_than_the_flat_plan(sizes, act):
    last = None
    for step in range(4 * 8, 96 * 8 + 1):
        gib = step / 8
        flat = _plan(gib, sizes, None)
        plan = _plan(gib, sizes, act)
        rank = _rank(plan, sizes)
        assert rank <= _rank(flat, sizes), (gib, plan.reasons)
        if last is not None:
            assert rank <= last[1], (gib, last, plan.reasons)
        last = (gib, rank)


def test_a_largest_canvas_denoise_that_would_not_fit_keeps_the_transformer_off_the_resident_tiers():
    # a resident transformer holds the VAE too while it denoises, and a 2048 canvas cannot be tiled there
    act = CalibratedImageActivation(1_000, 800, 3_000, 600, 6_000)
    for gib in (16, 17):
        assert _plan(gib, QWEN21_GGUF, act).offload_policy == OFFLOAD_MODEL
    plan = _plan(18, QWEN21_GGUF, act)
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_transformer is False


def test_an_unknown_free_reading_keeps_the_flat_plan():
    card = DeviceMemory("cuda", "cuda", "discrete_vram", None, 16 * GIB)
    kw = dict(target = _target(), device_memory = card, runtime_headroom_mib = 8192, **QWEN21_GGUF)
    assert plan_diffusion_memory(calibrated_activation = QWEN21_ACT, **kw) == plan_diffusion_memory(
        **kw
    )


@pytest.mark.parametrize(
    "mode, explicit",
    [
        (MEMORY_MODE_LOW_VRAM, False),
        (MEMORY_MODE_BALANCED, False),
        (MEMORY_MODE_FAST, False),
        (None, True),
    ],
)
def test_explicit_modes_keep_the_flat_plan(mode, explicit):
    for gib in (6, 8, 12, 16, 20, 24, 48):
        assert _plan(gib, QWEN21_GGUF, QWEN21_ACT, mode, explicit) == _plan(
            gib, QWEN21_GGUF, None, mode, explicit
        )


def test_unified_memory_keeps_the_flat_plan():
    for gib in (8, 16, 32):
        assert _plan(gib, QWEN21_GGUF, QWEN21_ACT, kind = "unified_memory") == _plan(
            gib, QWEN21_GGUF, None, kind = "unified_memory"
        )


def test_an_unknown_split_keeps_the_flat_plan():
    sizes = dict(QWEN21_GGUF, companion_dense_mib = None, text_encoder_dense_mib = None)
    for gib in (8, 16, 24):
        assert _plan(gib, sizes, QWEN21_ACT) == _plan(gib, sizes, None)


def test_only_measured_families_are_calibrated():
    assert calibrated_image_activation(None) is None
    assert calibrated_image_activation("hidream-i1") is None
    assert calibrated_image_activation("qwen-image-edit") is None
    assert calibrated_image_activation("qwen-image") is None
    assert calibrated_image_activation("sdxl") is None
    for family, tiers in dm._MEASURED_IMAGE_ACTIVATION_MIB.items():
        for max_speed, raw in zip((False, True), tiers):
            act = calibrated_image_activation(family, max_speed = max_speed)
            assert act.tiled_decode_mib <= act.decode_mib
            assert act.headroom(False) >= max(raw[:4]) and act.max_canvas_mib >= raw[4]
        # the max tier never plans for less than the default tiers
        assert all(m >= d for m, d in zip(tiers[1], tiers[0]))
    assert calibrated_image_activation("qwen-image-2.1") == calibrated_image_activation(
        "qwen-image-2.1", max_speed = True
    )


def test_16gb_qwen_image_21_at_the_max_speed_tier_offloads_whole_modules_untiled():
    # max-autotune holds about 9.6 GB for a 2048 denoise, which a resident transformer could not add on a 16 GB card
    act = calibrated_image_activation("qwen-image-2.1", max_speed = True)
    plan = _plan(16, QWEN21_GGUF, act)
    assert plan.offload_policy == OFFLOAD_MODEL and plan.vae_tiling is False
    assert plan == _plan(16, QWEN21_GGUF, act)


def test_the_shipped_qwen_image_21_figures_never_stream_the_transformer_on_16gb():
    act = calibrated_image_activation("qwen-image-2.1", max_speed = False)
    # a desktop and the CUDA context holding 1.2 GB: whole-module offload with the measured decode untiled
    plan = _plan(16, QWEN21_GGUF, act)
    assert plan.offload_policy == OFFLOAD_MODEL and plan.vae_tiling is False
    # a headless card with only the CUDA context: the transformer stays resident, the text encoders stream
    card = DeviceMemory("cuda", "cuda", "discrete_vram", 16 * GIB - 512, 16 * GIB)
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = card,
        runtime_headroom_mib = 8192,
        calibrated_activation = act,
        **QWEN21_GGUF,
    )
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_transformer is False
    assert plan.vae_tiling is True
    for gib in (18, 20):
        plan = _plan(gib, QWEN21_GGUF, act)
        assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_transformer is False


def test_the_requested_speed_tier_picks_the_figures(monkeypatch):
    import inspect

    import core.inference.diffusion as d

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    fam = types.SimpleNamespace(name = "qwen-image-2.1")
    nvidia = types.SimpleNamespace(backend = "cuda", vendor = "nvidia")
    default = calibrated_image_activation("qwen-image-2.1", max_speed = False)
    maxed = calibrated_image_activation("qwen-image-2.1", max_speed = True)
    assert default != maxed
    # outside a load nothing says which tier will run, so plan for the largest
    assert d._calibrated_activation(fam, nvidia) == maxed
    seen = []

    @d._plans_at_requested_speed
    def load(**kwargs):
        seen.append(d._calibrated_activation(fam, nvidia))

    cases = [(None, default), ("off", default), ("eager", default), ("default", default)]
    cases += [("max", maxed), (" MAX ", maxed), ("bogus", maxed)]
    for mode, want in cases:
        load(speed_mode = mode)
        assert seen.pop() == want, mode
    load()
    assert seen.pop() == default
    assert d._calibrated_activation(fam, nvidia) == maxed
    # both load entry points carry the speed tier into their plans, and keep their signatures
    for fn in (
        d.DiffusionBackend.load_pipeline,
        d.DiffusionBackend._pipeline_planned_denoiser_scheme,
    ):
        assert "speed_mode" in inspect.signature(fn).parameters
        assert fn.__wrapped__ is not None


def test_only_nvidia_with_a_confirmed_subquadratic_kernel_is_calibrated(monkeypatch):
    import core.inference.diffusion as d

    fam = types.SimpleNamespace(name = "qwen-image-2.1")
    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    for backend, vendor in (
        ("cuda", "amd"),
        ("cuda", None),
        ("xpu", "intel"),
        ("mps", "apple"),
        ("cpu", None),
    ):
        assert (
            d._calibrated_activation(fam, types.SimpleNamespace(backend = backend, vendor = vendor))
            is None
        )
    nvidia = types.SimpleNamespace(backend = "cuda", vendor = "nvidia")
    assert d._calibrated_activation(types.SimpleNamespace(name = "hidream-i1"), nvidia) is None
    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: False)
    assert d._calibrated_activation(fam, nvidia) is None

    def boom(target):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", boom)
    assert d._calibrated_activation(fam, nvidia) is None


def test_an_explicit_auto_mode_ignores_the_legacy_offload_flag():
    # a supplied memory mode wins over cpu_offload, so auto + cpu_offload plans exactly like auto
    for gib in (12, 16, 20):
        assert _plan(gib, QWEN21_GGUF, QWEN21_ACT, "auto", True) == _plan(
            gib, QWEN21_GGUF, QWEN21_ACT, "auto"
        )
    assert _plan(16, QWEN21_GGUF, QWEN21_ACT, "auto", True) != _plan(
        16, QWEN21_GGUF, None, "auto", True
    )
