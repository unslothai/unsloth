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
):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _discrete(free_mib, total_mib),
        calibrated_activation = calibrated,
        **flat,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL and torchao:
        plan = dm.torchao_streaming_plan(plan)
    return dm.refine_plan_from_loaded_weights(
        types.SimpleNamespace(transformer = object()), plan, family = family, speed_mode = speed_mode
    )


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
        plan = _plan(free, total, flat, family = family, torchao = torchao)
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
