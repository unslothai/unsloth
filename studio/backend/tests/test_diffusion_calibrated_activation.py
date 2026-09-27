# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
QWEN21_GGUF_BF16_TE = dict(
    model_dense_mib = 25_627, companion_dense_mib = 17_977, text_encoder_dense_mib = 16_689
)
LARGE_TE = dict(model_dense_mib = 19_288, companion_dense_mib = 12_288, text_encoder_dense_mib = 11_000)
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
    "sizes, act, strict",
    [
        (QWEN21_GGUF, QWEN21_ACT, True),
        (QWEN21_GGUF, SMALL_VAE_ACT, True),
        (ZIMAGE_BF16, SMALL_VAE_ACT, True),
        (FLUX1_BF16, SMALL_VAE_ACT, True),
        (SDXL, SMALL_VAE_ACT, True),
        (ZIMAGE_BF16, CalibratedImageActivation(1_000, 800, 30_000, 600, 1_000), True),
        (QWEN21_GGUF, CalibratedImageActivation(1_000, 800, 3_000, 600, 9_000), False),
    ],
)
def test_more_vram_is_never_slower_and_never_slower_than_the_flat_plan(sizes, act, strict):
    last = None
    for step in range(4 * 8, 96 * 8 + 1):
        gib = step / 8
        flat = _plan(gib, sizes, None)
        plan = _plan(gib, sizes, act)
        rank = _rank(plan, sizes)
        assert rank <= _rank(flat, sizes), (gib, plan.reasons)
        if last is not None:
            assert rank <= last[1] or (not strict and plan == flat), (gib, last, plan.reasons)
        last = (gib, rank)


@pytest.mark.parametrize("max_speed", [False, True])
@pytest.mark.parametrize("family", ["qwen-image-2.1", "flux.1", "flux.2-klein", "z-image"])
@pytest.mark.parametrize(
    "sizes", [QWEN21_GGUF, QWEN21_GGUF_BF16_TE, LARGE_TE, ZIMAGE_BF16, FLUX1_BF16]
)
def test_a_module_taken_off_streaming_fits_beside_its_own_phase(sizes, family, max_speed):
    act = calibrated_image_activation(family, max_speed = max_speed)
    transformer = sizes["model_dense_mib"] - sizes["companion_dense_mib"]
    te = sizes["text_encoder_dense_mib"]
    others = sizes["companion_dense_mib"] - te
    for step in range(4 * 8, 96 * 8 + 1):
        gib = step / 8
        flat = _plan(gib, sizes, None)
        plan = _plan(gib, sizes, act)
        if plan == flat or flat.offload_policy != OFFLOAD_GROUP or not flat.stream_transformer:
            continue
        free = _card(gib).free_mib - dm.DEFAULT_BASE_OVERHEAD_MIB
        assert transformer + act.max_canvas_mib <= free, (gib, plan.reasons)
        if plan.offload_policy == OFFLOAD_MODEL:
            assert te + act.text_encoder_mib <= free and others + act.tiled_decode_mib <= free, (
                gib,
                plan.reasons,
            )


def test_a_largest_canvas_denoise_that_would_not_fit_keeps_the_transformer_off_the_resident_tiers():
    act = CalibratedImageActivation(1_000, 800, 3_000, 600, 6_000)
    assert _plan(16, QWEN21_GGUF, act) == _plan(16, QWEN21_GGUF, None)
    assert _plan(17, QWEN21_GGUF, act).offload_policy == OFFLOAD_MODEL
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
        assert all(m >= d for m, d in zip(tiers[1], tiers[0]))
    assert calibrated_image_activation("qwen-image-2.1") == calibrated_image_activation(
        "qwen-image-2.1", max_speed = True
    )


def test_a_text_encoder_that_cannot_run_whole_keeps_streaming():
    act = calibrated_image_activation("qwen-image-2.1", max_speed = False)
    assert _plan(15, LARGE_TE, act) == _plan(15, LARGE_TE, None)


def test_16gb_qwen_image_21_at_the_max_speed_tier_keeps_streaming_the_transformer():
    act = calibrated_image_activation("qwen-image-2.1", max_speed = True)
    assert _plan(16, QWEN21_GGUF, act) == _plan(16, QWEN21_GGUF, None)
    plan = _plan(22, QWEN21_GGUF, act)
    assert plan.offload_policy == OFFLOAD_MODEL and plan.vae_tiling is False


def test_the_shipped_qwen_image_21_figures_never_stream_the_transformer_on_16gb():
    act = calibrated_image_activation("qwen-image-2.1", max_speed = False)
    plan = _plan(16, QWEN21_GGUF, act)
    assert plan.offload_policy == OFFLOAD_MODEL and plan.vae_tiling is False
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


def test_an_fp32_promoted_family_keeps_the_flat_plan(monkeypatch):
    import torch

    import core.inference.diffusion as d
    from core.inference.diffusion_families import detect_family

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    zimage = detect_family("unsloth/Z-Image-Turbo-GGUF")
    assert zimage.name == "z-image" and zimage.fp16_incompatible
    flux = types.SimpleNamespace(name = "flux.1")

    def on(dtype):
        return types.SimpleNamespace(backend = "cuda", vendor = "nvidia", dtype = dtype)

    assert d._calibrated_activation(zimage, on(torch.float16)) is None
    assert d._calibrated_activation(zimage, on(torch.float32)) is None
    assert d._calibrated_activation(zimage, on(torch.bfloat16)) is not None
    assert d._calibrated_activation(flux, on(torch.float16)) is not None


def test_a_reference_family_the_guard_cannot_size_keeps_the_flat_plan(monkeypatch):
    # FLUX.2 Klein takes up to four ~1 MP references the generation guard never counts (no reference_resolutions),
    # so a promoted tier would have no headroom for them. Qwen-Image-2.1 declares them, so the guard sizes it.
    import core.inference.diffusion as d
    from core.inference.diffusion_families import detect_family

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    nvidia = types.SimpleNamespace(backend = "cuda", vendor = "nvidia")
    klein = detect_family("black-forest-labs/FLUX.2-klein-4B")
    assert klein.name == "flux.2-klein" and klein.reference and not klein.reference_resolutions
    assert d._calibrated_activation(klein, nvidia) is None
    qwen = detect_family("Qwen/Qwen-Image-2.1")
    assert qwen.name == "qwen-image-2.1" and qwen.reference_resolutions
    assert d._calibrated_activation(qwen, nvidia) is not None


def test_an_explicit_auto_mode_ignores_the_legacy_offload_flag():
    for gib in (12, 16, 20):
        assert _plan(gib, QWEN21_GGUF, QWEN21_ACT, "auto", True) == _plan(
            gib, QWEN21_GGUF, QWEN21_ACT, "auto"
        )
    assert _plan(16, QWEN21_GGUF, QWEN21_ACT, "auto", True) != _plan(
        16, QWEN21_GGUF, None, "auto", True
    )


def _resident_after_placement(plan, sizes):
    if plan.offload_policy == OFFLOAD_NONE:
        return sizes["model_dense_mib"]
    if plan.offload_policy == OFFLOAD_GROUP and not plan.stream_transformer:
        return sizes["model_dense_mib"] - sizes["text_encoder_dense_mib"]
    return 0


@pytest.mark.parametrize("max_speed", [False, True])
@pytest.mark.parametrize("tile_side", [256, dm.DEFAULT_VAE_TILE_SIDE])
@pytest.mark.parametrize("sizes", [QWEN21_GGUF, QWEN21_GGUF_BF16_TE])
def test_generation_guard_never_refuses_the_calibrated_2048_canvas_on_a_promoted_tier(
    max_speed, tile_side, sizes
):
    # The promoted tier places the transformer resident, which lowers the free VRAM the guard reads; the 2048 canvas
    # it was sized for must still run (tiled), never come back as a 400.
    act = calibrated_image_activation("qwen-image-2.1", max_speed = max_speed)
    promoted = 0
    for step in range(10 * 4, 48 * 4 + 1):
        gib = step / 4
        plan = _plan(gib, sizes, act)
        if "calibrated_headroom_mib" not in plan.estimates:
            continue
        promoted += 1
        total = int(gib * GIB)
        free = max(0, total - 1_229 - _resident_after_placement(plan, sizes))
        verdict = dm.raise_on_image_activation_shortfall(
            device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
            width = 2048,
            height = 2048,
            family = "qwen-image-2.1",
            vae_tile_side = tile_side,
            vae_sliced = True,
        )
        assert verdict.action in (dm.ACTIVATION_RUN, dm.ACTIVATION_TILE), (gib, plan.reasons)
    assert promoted > 0


def _guard(
    free,
    total,
    *,
    calibrated,
    tile_side = 256,
    **kw,
):
    return dm.image_activation_verdict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        family = kw.pop("family", "qwen-image-2.1"),
        vae_tile_side = tile_side,
        vae_sliced = True,
        calibrated_placement = calibrated,
        **kw,
    )


def _fits(verdict):
    overhead = dm.DEFAULT_BASE_OVERHEAD_MIB
    if verdict.needed_mib + overhead <= verdict.budget_mib:
        return True
    return verdict.tiled_needed_mib is not None and (
        verdict.tiled_needed_mib + overhead <= verdict.budget_mib
    )


# Six 512px references at the Qwen-Image-2.1 condition weight: ~5980 MiB untiled, under the flat 8192 MiB plan.
SIX_REFS = dict(width = 512, height = 512, condition_pixels = int(6 * 512 * 512 * 0.32))


def test_references_on_a_calibrated_tier_are_checked_against_the_free_budget():
    total = 16 * GIB
    free = 4_266 + 1_229
    flat = _guard(free, total, calibrated = False, **SIX_REFS)
    assert flat.action != dm.ACTIVATION_REFUSE  # the flat plan budgeted 8192 MiB for it
    verdict = _guard(free, total, calibrated = True, **SIX_REFS)
    assert verdict.action == dm.ACTIVATION_REFUSE
    assert "balanced memory mode" in verdict.message
    assert "fewer input images" in verdict.message


def test_references_on_a_calibrated_tier_run_when_they_fit():
    verdict = _guard(20 * GIB, 24 * GIB, calibrated = True, **SIX_REFS)
    assert verdict.action == dm.ACTIVATION_RUN


@pytest.mark.parametrize("sizes", [QWEN21_GGUF, QWEN21_GGUF_BF16_TE])
def test_conditioned_requests_on_every_promoted_tier_never_skip_the_budget(sizes):
    act = calibrated_image_activation("qwen-image-2.1", max_speed = True)
    for step in range(10 * 4, 48 * 4 + 1):
        gib = step / 4
        plan = _plan(gib, sizes, act)
        if "calibrated_headroom_mib" not in plan.estimates:
            continue
        total = int(gib * GIB)
        free = max(0, total - 1_229 - _resident_after_placement(plan, sizes))
        for kw in (SIX_REFS, dict(width = 1024, height = 1024, controlnet = True)):
            verdict = _guard(free, total, calibrated = True, **kw)
            assert verdict.action == dm.ACTIVATION_REFUSE or _fits(verdict), (gib, kw)


@pytest.mark.parametrize("family", ["flux.1", "qwen-image-2.1"])
def test_controlnet_on_a_calibrated_tier_budgets_its_forward_and_residuals(family):
    total = 24 * GIB
    free = 6 * GIB
    kw = dict(width = 1024, height = 1024, family = family)
    plain = _guard(free, total, calibrated = True, **kw)
    assert plain.action != dm.ACTIVATION_REFUSE
    verdict = _guard(free, total, calibrated = True, controlnet = True, **kw)
    assert verdict.action == dm.ACTIVATION_REFUSE
    assert verdict.needed_mib > plain.needed_mib
    assert "with ControlNet" in verdict.message
    assert "without ControlNet" in verdict.message
    assert "balanced memory mode" in verdict.message
    assert "fewer input images" not in verdict.message
    # plenty of room: runs
    roomy = _guard(40 * GIB, 48 * GIB, calibrated = True, controlnet = True, **kw)
    assert roomy.action == dm.ACTIVATION_RUN


@pytest.mark.parametrize("free", [1_500, 3_000, 5_000, 8_000, 12_000, 30_000])
@pytest.mark.parametrize("tile_side", [None, 256, dm.DEFAULT_VAE_TILE_SIDE])
@pytest.mark.parametrize(
    "kw",
    [
        dict(width = 512, height = 512),
        dict(width = 1024, height = 1024),
        dict(width = 2048, height = 2048),
        dict(width = 1024, height = 1024, batch_size = 2),
        dict(width = 1344, height = 768, family = "flux.1"),
    ],
)
def test_unconditioned_requests_on_a_calibrated_tier_are_unchanged(free, tile_side, kw):
    flat = _guard(free, 16 * GIB, calibrated = False, tile_side = tile_side, **dict(kw))
    cal = _guard(free, 16 * GIB, calibrated = True, tile_side = tile_side, **dict(kw))
    assert cal == flat


@pytest.mark.parametrize("free", [1_500, 3_000, 5_000, 8_000, 12_000, 30_000])
@pytest.mark.parametrize(
    "kw",
    [
        SIX_REFS,
        dict(width = 1024, height = 1024, condition_pixels = 1024 * 1024),
        dict(width = 1024, height = 1024),
    ],
)
def test_flat_tiers_ignore_the_controlnet_flag(free, kw):
    # Flat tiers keep today's verdict: the flat plan already budgets conditioned work.
    base = _guard(free, 16 * GIB, calibrated = False, **dict(kw))
    with_cn = _guard(free, 16 * GIB, calibrated = False, controlnet = True, **dict(kw))
    assert with_cn.action == base.action
    assert (with_cn.needed_mib, with_cn.tiled_needed_mib) == (
        base.needed_mib,
        base.tiled_needed_mib,
    )


def test_generate_passes_the_calibrated_placement_and_controlnet_to_the_guard():
    import inspect

    import core.inference.diffusion as d

    src = inspect.getsource(d)
    assert 'calibrated_placement = bool(getattr(state, "calibrated_placement", False)),' in src
    assert 'controlnet = workflow == "controlnet" and control_pil is not None,' in src
