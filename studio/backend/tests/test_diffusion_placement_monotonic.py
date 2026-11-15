# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""More free VRAM never keeps less of the denoiser resident: the text encoders give way to the denoiser."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_measured_peaks as dmp
import core.inference.diffusion_memory as dm

GIB = 1024
STEP_MIB = 256

# family -> (8-bit denoiser MiB, dense denoiser MiB, text encoders MiB, VAE MiB)
FAMILIES = {
    "flux.1": (11420, 22700, {"text_encoder": 235, "text_encoder_2": 9083}, 160),
    "flux.2-klein": (3800, 7400, {"text_encoder": 7680}, 160),
    "qwen-image": (19500, 38900, {"text_encoder": 15800}, 242),
    "qwen-image-2.1": (19500, 38900, {"text_encoder": 15800}, 242),
    "z-image": (5900, 11740, {"text_encoder": 7670}, 160),
    "wan": (13600, 27000, {"text_encoder": 10800}, 242),
}
# quant -> (8-bit weights, torchao, fp8-stored, dense compute bytes)
QUANTS = {
    "int8": (True, True, False, None),
    "fp8_stored": (True, False, True, None),
    "fp16": (False, False, False, 2),
}
FLUX_INT8_LOADED = {
    "transformer": (11420, "dit"),
    "text_encoder": (235, "text_encoder"),
    "text_encoder_2": (9083, "text_encoder"),
    "vae": (160, "other"),
}
FLUX_INT8_FLAT = dict(
    model_dense_mib = 22019,
    companion_dense_mib = 9536,
    text_encoder_dense_mib = 9346,
    runtime_headroom_mib = 8192,
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch, tmp_path):
    for env in (
        dm.MEASURED_ACTIVATION_ENV,
        dm.PARTIAL_RESIDENT_ENV,
        dm.RESIDENT_DIT_ENV,
        dm.STREAMED_RESIDENCY_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setenv(dmp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    monkeypatch.setattr(dm, "_vram_budget_fraction", lambda: 1.0)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 1296)


def _discrete(free_mib: int, total_mib: int) -> "dm.DeviceMemory":
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", free_mib, total_mib)


def _patch_weights(
    monkeypatch,
    loaded,
    *,
    torchao,
    fp8_stored = False,
    compute = None,
    learned = None,
):
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: torchao)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(loaded))
    monkeypatch.setattr(dm, "_denoiser_compute_bytes", lambda pipe: compute)
    monkeypatch.setattr(dm, "_holds_fp8_stored", lambda module: fp8_stored and module is not None)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_packed", lambda pipe: fp8_stored)
    monkeypatch.setattr(dmp, "lookup", lambda **kw: learned)
    monkeypatch.setattr(
        dmp,
        "lookup_bucket",
        lambda **kw: None if learned is None else (learned, 1024 * 1024, False),
    )


def _plan(
    free_mib,
    total_mib,
    flat,
    *,
    family,
    speed_mode = "default",
    torchao = True,
    calibrated = None,
    compute = None,
):
    import torch

    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _discrete(free_mib, total_mib),
        calibrated_activation = calibrated,
        **flat,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL and torchao:
        plan = dm.torchao_streaming_plan(plan)
    dtype = torch.float16 if compute == 2 else torch.bfloat16
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = dtype))
    refined = dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed_mode)
    return dm.prefer_whole_module_offload(pipe, refined)


def _denoiser_key(plan, dit_mib):
    """(per-step streaming, -resident denoiser MiB); None for whole-module offload (not a residency tier)."""
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        return None
    if plan.offload_policy == dm.OFFLOAD_NONE or not plan.stream_transformer:
        return (0, -dit_mib)
    resident = int(plan.resident_transformer_mib or 0)
    return (0, -dit_mib) if resident >= dit_mib else (1, -resident)


@pytest.mark.parametrize("gib", (24, 48, 80))
@pytest.mark.parametrize("learned", (None, 3072))
@pytest.mark.parametrize("quant", sorted(QUANTS))
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_more_free_vram_never_keeps_less_of_the_denoiser(monkeypatch, family, quant, learned, gib):
    q8, dense, encoders, vae = FAMILIES[family]
    eight, torchao, fp8_stored, compute = QUANTS[quant]
    dit = q8 if eight else dense
    te = sum(encoders.values())
    loaded = {
        "transformer": (dit, "dit"),
        **{name: (mib, "text_encoder") for name, mib in encoders.items()},
        "vae": (vae, "other"),
    }
    _patch_weights(
        monkeypatch,
        loaded,
        torchao = torchao,
        fp8_stored = fp8_stored,
        compute = compute,
        learned = learned,
    )
    flat = dict(
        model_dense_mib = dit + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None, family = family),
    )
    total = gib * GIB
    previous = None
    for free in range(4 * GIB, min(80, gib) * GIB + 1, STEP_MIB):
        plan = _plan(free, total, flat, family = family, torchao = torchao, compute = compute)
        key = _denoiser_key(plan, dit)
        if key is None:
            continue
        if previous is not None:
            assert key <= previous[1], (free, previous, key, plan.reasons)
        previous = (free, key)


@pytest.mark.parametrize("gib, frac", [(24, 0.9), (32, 0.7), (48, 0.5), (80, 0.3)])
def test_int8_flux_keeps_the_denoiser_resident_past_the_encoder_floor(monkeypatch, gib, frac):
    """Past ~19.9 GiB of budget the flat plan keeps the 9.3 GiB encoders resident; main then held 5.3 GiB of the
    11.4 GiB int8 denoiser instead of all of it. The denoiser now stays whole and the encoders stream."""
    _patch_weights(monkeypatch, FLUX_INT8_LOADED, torchao = True)
    total = gib * GIB
    free = int(total * frac)
    plan = _plan(free, total, FLUX_INT8_FLAT, family = "flux.1")
    assert plan.offload_policy == dm.OFFLOAD_GROUP
    assert plan.stream_text_encoders
    assert plan.resident_transformer_mib == 11420
    assert plan.estimates["denoiser_first_resident_te_dit_mib"] < 11420
    assert any("encoders once per prompt" in reason for reason in plan.reasons)
    below = _plan(20 * GIB, total, FLUX_INT8_FLAT, family = "flux.1")
    assert below.resident_transformer_mib == 11420 and below.stream_text_encoders
    assert "denoiser_first_resident_te_dit_mib" not in below.estimates


def test_encoders_stay_resident_when_they_cost_the_denoiser_nothing(monkeypatch):
    """Room for the whole denoiser beside the resident encoders: the encoders-resident plan is kept."""
    _patch_weights(monkeypatch, FLUX_INT8_LOADED, torchao = True)
    total = 96 * GIB
    for free in range(20 * GIB, 60 * GIB, STEP_MIB):
        plan = _plan(free, total, FLUX_INT8_FLAT, family = "flux.1")
        if (
            plan.offload_policy == dm.OFFLOAD_GROUP
            and plan.stream_transformer
            and not plan.stream_text_encoders
        ):
            assert plan.resident_transformer_mib == 11420
            assert "denoiser_first_resident_te_dit_mib" not in plan.estimates


@pytest.mark.parametrize("mode", ["fast", "balanced", "low_vram"])
def test_explicit_modes_are_untouched(monkeypatch, mode):
    _patch_weights(monkeypatch, FLUX_INT8_LOADED, torchao = True)
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _discrete(22118, 24 * GIB),
        requested_mode = mode,
        **FLUX_INT8_FLAT,
    )
    refined = dm.refine_plan_from_loaded_weights(
        object(), plan, family = "flux.1", speed_mode = "default"
    )
    assert refined is plan


def _speed_key(plan, dit_mib):
    """Speed key (tier, -resident MiB): 0 resident, 1 fitting offload, 2 streamed per step, 3 oversize."""
    if plan.offload_policy == dm.OFFLOAD_NONE or (
        plan.offload_policy == dm.OFFLOAD_GROUP and not plan.stream_transformer
    ):
        return (0, -dit_mib)
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        fits = dit_mib <= int(plan.estimates.get("safe_device_budget_mib") or 0)
        return (1, 0) if fits else (3, 0)
    resident = int(plan.resident_transformer_mib or 0)
    return (0, -dit_mib) if resident >= dit_mib else (2, -resident)


@pytest.mark.parametrize("calibrated", (False, True))
@pytest.mark.parametrize("learned", (None, 3072))
@pytest.mark.parametrize("quant", ("bf16",))
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_more_free_vram_never_trades_whole_module_offload_for_streaming(
    monkeypatch, family, quant, learned, calibrated
):
    """Below the streamed tiers' floor whole-module offload holds the whole denoiser; a few hundred MiB more used to
    stream it every step instead (FLUX.2-klein-4B bf16 on B200: 0.34 s -> 0.60 s per 4-step 1024^2 image)."""
    q8, dense, encoders, vae = FAMILIES[family]
    eight, torchao, fp8_stored, compute = {**QUANTS, "bf16": (False, False, False, None)}[quant]
    dit = q8 if eight else dense
    te = sum(encoders.values())
    loaded = {
        "transformer": (dit, "dit"),
        **{name: (mib, "text_encoder") for name, mib in encoders.items()},
        "vae": (vae, "other"),
    }
    _patch_weights(
        monkeypatch,
        loaded,
        torchao = torchao,
        fp8_stored = fp8_stored,
        compute = compute,
        learned = learned,
    )
    flat = dict(
        model_dense_mib = dit + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None, family = family),
    )
    act = dm.calibrated_image_activation(family) if calibrated else None
    if calibrated and act is None:
        pytest.skip("no calibrated activation for this family")
    previous = None
    for free in range(4 * GIB, 24 * GIB + 1, STEP_MIB):
        plan = _plan(
            free, 24 * GIB, flat, family = family, torchao = torchao, calibrated = act, compute = compute
        )
        key = _speed_key(plan, dit)
        if previous is not None:
            assert key <= previous[1], (free, previous, key, plan.reasons)
        previous = (free, key)


def test_bf16_klein_keeps_whole_module_offload_past_the_streamed_floor(monkeypatch):
    q8, dense, encoders, vae = FAMILIES["flux.2-klein"]
    te = sum(encoders.values())
    loaded = {
        "transformer": (dense, "dit"),
        "text_encoder": (te, "text_encoder"),
        "vae": (vae, "other"),
    }
    _patch_weights(monkeypatch, loaded, torchao = False)
    flat = dict(
        model_dense_mib = dense + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
    )
    below = _plan(12288, 24 * GIB, flat, family = "flux.2-klein", torchao = False)
    above = _plan(12544, 24 * GIB, flat, family = "flux.2-klein", torchao = False)
    assert below.offload_policy == dm.OFFLOAD_MODEL
    assert "whole_module_kept_below_floor_mib" not in below.estimates
    assert above.offload_policy == dm.OFFLOAD_MODEL
    assert above.estimates["whole_module_kept_below_floor_mib"] == 10400
    assert any("uploading it once per call" in reason for reason in above.reasons)
    # The resident-transformer tier still wins once it fits.
    resident = _plan(20 * GIB, 24 * GIB, flat, family = "flux.2-klein", torchao = False)
    assert resident.offload_policy == dm.OFFLOAD_GROUP and not resident.stream_transformer


def test_streamed_plans_that_pin_the_whole_denoiser_or_hold_torchao_are_kept(monkeypatch):
    """A refined group plan already holding the whole denoiser beats whole-module offload; a torchao denoiser's
    whole-module tier is streaming, so its ladder is unchanged."""
    q8, dense, encoders, vae = FAMILIES["flux.2-klein"]
    te = sum(encoders.values())
    loaded = {
        "transformer": (q8, "dit"),
        "text_encoder": (te, "text_encoder"),
        "vae": (vae, "other"),
    }
    flat = dict(
        model_dense_mib = q8 + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
    )
    _patch_weights(monkeypatch, loaded, torchao = False, fp8_stored = True, learned = 3072)
    pinned = _plan(14 * GIB, 24 * GIB, flat, family = "flux.2-klein", torchao = False)
    assert pinned.offload_policy == dm.OFFLOAD_GROUP and pinned.resident_transformer_mib == q8
    _patch_weights(monkeypatch, loaded, torchao = True)
    streamed = _plan(14 * GIB, 24 * GIB, flat, family = "flux.2-klein", torchao = True)
    assert streamed.offload_policy == dm.OFFLOAD_GROUP
    assert "whole_module_kept_below_floor_mib" not in streamed.estimates


def test_bf16_partial_residency_gives_way_to_whole_module_offload(monkeypatch):
    """A learned peak pins 4.7 of the 7.4 GB bf16 denoiser past the floor; whole-module offload still holds all of it
    (B200, 12.25 GiB free: 0.37 s whole-module vs 0.73 s with 5.5 GB pinned)."""
    q8, dense, encoders, vae = FAMILIES["flux.2-klein"]
    te = sum(encoders.values())
    loaded = {
        "transformer": (dense, "dit"),
        "text_encoder": (te, "text_encoder"),
        "vae": (vae, "other"),
    }
    flat = dict(
        model_dense_mib = dense + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
    )
    _patch_weights(monkeypatch, loaded, torchao = False, learned = 3072)
    plan = _plan(12544, 24 * GIB, flat, family = "flux.2-klein", torchao = False)
    assert plan.offload_policy == dm.OFFLOAD_MODEL
    # the same learned peak on an fp16 denoiser (the T4 dtype) keeps its partial residency
    _patch_weights(monkeypatch, loaded, torchao = False, compute = 2, learned = 3072)
    fp16 = _plan(12544, 24 * GIB, flat, family = "flux.2-klein", torchao = False, compute = 2)
    assert fp16.offload_policy == dm.OFFLOAD_GROUP and 0 < fp16.resident_transformer_mib < dense


@pytest.mark.parametrize(
    "family, quant",
    [("flux.2-klein", "fp16"), ("flux.2-klein", "fp8_stored"), ("z-image", "fp8_stored")],
)
def test_unmeasured_denoisers_keep_the_streamed_plan(monkeypatch, family, quant):
    """fp16 (the T4 dtype, where stock whole-module offload measured slower than streaming), fp8-stored and packed
    denoisers keep the streamed group plan past the floor: only bf16 was measured faster whole-module."""
    q8, dense, encoders, vae = FAMILIES[family]
    eight, torchao, fp8_stored, compute = QUANTS[quant]
    dit = q8 if eight else dense
    te = sum(encoders.values())
    loaded = {
        "transformer": (dit, "dit"),
        "text_encoder": (te, "text_encoder"),
        "vae": (vae, "other"),
    }
    _patch_weights(monkeypatch, loaded, torchao = torchao, fp8_stored = fp8_stored, compute = compute)
    flat = dict(
        model_dense_mib = dit + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None, family = family),
    )
    plan = _plan(12544, 24 * GIB, flat, family = family, torchao = torchao, compute = compute)
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_transformer
    assert "whole_module_kept_below_floor_mib" not in plan.estimates


# (GiB/s, TFLOPS) probes reproducing measured klein-4B bf16 1024^2 per-step transfer / compute.
KLEIN_DIT_MIB = 7393
CARDS = {
    "b200": (0.142, 0.0296),
    "a100": (0.541, 0.150),
    "rtx_pro_6000": (0.1415, 0.0985),
}


def _card_probe(card):
    transfer, compute = CARDS[card]
    flops = 2.0 * KLEIN_DIT_MIB * 2**20 / 2.0 * (1024 // 16) ** 2
    return (KLEIN_DIT_MIB / 1024.0 / transfer, dm._DIT_COMPUTE_EFFICIENCY * flops / compute / 1e12)


def _klein_above_floor(monkeypatch, *, compute = None):
    q8, dense, encoders, vae = FAMILIES["flux.2-klein"]
    te = sum(encoders.values())
    loaded = {
        "transformer": (KLEIN_DIT_MIB, "dit"),
        "text_encoder": (te, "text_encoder"),
        "vae": (vae, "other"),
    }
    _patch_weights(monkeypatch, loaded, torchao = False, compute = compute)
    flat = dict(
        model_dense_mib = KLEIN_DIT_MIB + te + vae,
        companion_dense_mib = te + vae,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
    )
    return flat


def _priced(
    monkeypatch,
    free_mib,
    flat,
    *,
    steps,
    card,
    compute = None,
):
    import torch

    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: _card_probe(card))
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _discrete(free_mib, 24 * GIB),
        **flat,
    )
    dtype = torch.float16 if compute == 2 else torch.bfloat16
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = dtype))
    refined = dm.refine_plan_from_loaded_weights(
        pipe, plan, family = "flux.2-klein", speed_mode = "default"
    )
    return refined, dm.prefer_whole_module_offload(pipe, refined, steps = steps)


@pytest.mark.parametrize(
    "card, steps, whole",
    [
        ("b200", 2, True),
        ("b200", 4, True),
        ("b200", 8, True),
        ("a100", 4, True),
        ("a100", 8, True),
        ("rtx_pro_6000", 2, False),
        ("rtx_pro_6000", 4, False),
        ("rtx_pro_6000", 8, True),
    ],
)
def test_whole_module_offload_is_kept_only_where_it_is_cheaper(monkeypatch, card, steps, whole):
    """B200 0.336 vs 0.595 s and A100 2.336 vs 3.429 s at 4 steps (whole-module faster); RTX PRO 6000 0.762 vs 0.743 s
    at 4 steps (streamed faster), 1.022 vs 1.181 s at 8, 0.561 vs 0.458 s at 2."""
    flat = _klein_above_floor(monkeypatch)
    refined, plan = _priced(monkeypatch, 12544, flat, steps = steps, card = card)
    assert refined.offload_policy == dm.OFFLOAD_GROUP and refined.stream_transformer
    model_s, group_s = plan.estimates["whole_module_cost_s"], plan.estimates["streamed_cost_s"]
    assert (model_s < group_s) is whole
    if whole:
        assert plan.offload_policy == dm.OFFLOAD_MODEL
        assert plan.estimates["whole_module_kept_below_floor_mib"] == 10400
    else:
        assert plan.offload_policy == dm.OFFLOAD_GROUP
        assert "whole_module_kept_below_floor_mib" not in plan.estimates
        assert replace_fields(plan) == replace_fields(refined)
        assert any("streaming kept over whole-module offload" in r for r in plan.reasons)


def replace_fields(plan):
    return dm.replace(plan, reasons = (), estimates = {})


def test_cost_model_fits_the_measured_crossovers():
    """Per-card crossover of whole-module vs streamed (steps where they cost the same): RTX PRO 6000 measured 4.2-4.35."""

    def crossover(card):
        h2d, tflops = _card_probe(card)
        (w1, s1), (w2, _) = (
            dm._whole_module_offload_costs(
                KLEIN_DIT_MIB, steps = n, h2d_gib_s = h2d, bf16_tflops = tflops
            )
            for n in (1, 2)
        )
        compute = w2 - w1
        return (w1 - compute) / (s1 - compute)

    assert crossover("b200") <= 2.0
    assert crossover("a100") <= 4.0
    assert 4.0 < crossover("rtx_pro_6000") < 5.0


def test_unpriced_or_kill_switch_keep_their_meaning(monkeypatch):
    """No steps or no measurement: the unconditional switch (#15129). ``1`` always switches, ``0`` never does."""
    flat = _klein_above_floor(monkeypatch)
    _, unpriced = _priced(monkeypatch, 12544, flat, steps = None, card = "rtx_pro_6000")
    assert unpriced.offload_policy == dm.OFFLOAD_MODEL
    assert "whole_module_cost_s" not in unpriced.estimates
    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: None)
    q8, dense, encoders, vae = FAMILIES["flux.2-klein"]
    import torch

    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = torch.bfloat16))
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _discrete(12544, 24 * GIB),
        **flat,
    )
    refined = dm.refine_plan_from_loaded_weights(
        pipe, plan, family = "flux.2-klein", speed_mode = "default"
    )
    assert dm.prefer_whole_module_offload(pipe, refined, steps = 4).offload_policy == dm.OFFLOAD_MODEL
    monkeypatch.setenv(dm.WHOLE_MODULE_OFFLOAD_ENV, "1")
    _, forced = _priced(monkeypatch, 12544, flat, steps = 4, card = "rtx_pro_6000")
    assert (
        forced.offload_policy == dm.OFFLOAD_MODEL and "whole_module_cost_s" not in forced.estimates
    )
    monkeypatch.setenv(dm.WHOLE_MODULE_OFFLOAD_ENV, "0")
    refined, off = _priced(monkeypatch, 12544, flat, steps = 4, card = "b200")
    assert off is refined


@pytest.mark.parametrize("card", sorted(CARDS))
@pytest.mark.parametrize("steps", (2, 4, 8))
def test_plans_outside_the_window_are_unchanged_by_pricing(monkeypatch, card, steps):
    """Below the floor, at the resident tiers and on an fp16 denoiser the priced call returns the same plan."""
    for compute in (None, 2):
        flat = _klein_above_floor(monkeypatch, compute = compute)
        for free in range(4 * GIB, 24 * GIB + 1, STEP_MIB):
            refined, priced = _priced(
                monkeypatch, free, flat, steps = steps, card = card, compute = compute
            )
            import torch

            dtype = torch.float16 if compute == 2 else torch.bfloat16
            pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = dtype))
            unpriced = dm.prefer_whole_module_offload(pipe, refined)
            if "whole_module_kept_below_floor_mib" not in unpriced.estimates:
                assert priced == unpriced == refined, (free, compute)


@pytest.mark.parametrize(
    "probe, decisions",
    [
        # real probe readings: B200, and RTX PRO 6000 (measured streamed at 2 and 4 steps, whole at 8)
        ((51.5, 1490.0), {2: True, 4: True, 8: True}),
        ((52.97, 309.0), {2: False, 4: False, 8: True}),
    ],
)
def test_real_probe_readings_make_the_measured_call(probe, decisions):
    for steps, whole in decisions.items():
        model_s, group_s = dm._whole_module_offload_costs(
            7438, steps = steps, h2d_gib_s = probe[0], bf16_tflops = probe[1]
        )
        assert (model_s < group_s) is whole, (probe, steps, model_s, group_s)
