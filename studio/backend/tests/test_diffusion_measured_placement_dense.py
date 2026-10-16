# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense-denoiser placement planner tests from a FLUX.2-klein-4B 15 GB T4 load; CPU only."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

KLEIN_LOADED = {
    "transformer": (7393, "dit"),
    "text_encoder": (7673, "text_encoder"),
    "vae": (160, "other"),
}
KLEIN_FLAT = dict(model_dense_mib = 15258, companion_dense_mib = 7820, text_encoder_dense_mib = 7629)
T4_TOTAL = 15095
KLEIN_HEADROOM = 3072  # 2455 x 1.15, rounded up to 256 MiB
ENVS = (
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
    "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
)


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _flat_plan(
    budget_mib: int,
    total_mib: int = T4_TOTAL,
    mode = None,
):
    return dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
        requested_mode = mode,
        **KLEIN_FLAT,
    )


@pytest.fixture
def klein_pipe(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(KLEIN_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    import torch

    # fp16 on the T4: the compute dtype the dense eager table was measured at.
    return types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = torch.float16))


def _refine(
    pipe,
    plan,
    family = "flux.2-klein",
    speed = "off",
):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed)


def test_dense_headroom_values(klein_pipe, monkeypatch):
    m = dm.measured_image_runtime_mib
    assert m("flux.2-klein", "off", dense_transformer_mib = 7393) == KLEIN_HEADROOM
    assert m("flux.1", "off", dense_transformer_mib = 22680) == 2816
    assert m("qwen-image", "off", dense_transformer_mib = 38968) == 5120
    assert m("flux.2-klein", "off", dense_transformer_mib = 7393, width = 2048, height = 2048) == 11520
    assert m("flux.2-klein", "default", dense_transformer_mib = 7393) is None
    assert m("sdxl", "off", dense_transformer_mib = 4900) is None
    assert m("flux.2-klein", "off", dense_transformer_mib = 17300) is None
    assert m("flux.2-klein", "off") is None
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE", "0")
    assert m("flux.2-klein", "off", dense_transformer_mib = 7393) is None
    assert m("qwen-image-2.1", "default") == 2304
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION", "0")
    assert m("flux.2-klein", "off", dense_transformer_mib = 7393) is None


def test_t4_klein_transformer_resident(klein_pipe):
    plan = _flat_plan(12758)
    assert plan.estimates["resident_transformer_floor_mib"] == 17869
    assert plan.offload_policy == dm.OFFLOAD_GROUP
    assert plan.stream_transformer and plan.stream_text_encoders
    new = _refine(klein_pipe, plan)
    assert (new.offload_policy, new.stream_transformer, new.stream_text_encoders) == (
        dm.OFFLOAD_GROUP,
        True,
        True,
    )
    assert new.resident_transformer_mib == 7393
    assert new.estimates["measured_runtime_headroom_mib"] == KLEIN_HEADROOM
    assert new.estimates["measured_dense_transformer_mib"] == 7393
    assert 7393 + 160 + KLEIN_HEADROOM + dm.DEFAULT_BASE_OVERHEAD_MIB <= 12758


@pytest.mark.parametrize("env", ENVS[:2])
def test_kill_switches_restore_flat_plan(klein_pipe, monkeypatch, env):
    plan = _flat_plan(12758)
    monkeypatch.setenv(env, "0")
    assert _refine(klein_pipe, plan) is plan


def test_large_card_and_other_tiers_untouched(klein_pipe):
    resident = _flat_plan(160000, 183359)
    assert resident.offload_policy == dm.OFFLOAD_NONE
    assert _refine(klein_pipe, resident) is resident
    plan = _flat_plan(12758)
    assert _refine(klein_pipe, plan, speed = "default") is plan
    assert _refine(klein_pipe, plan, family = "sdxl") is plan
    balanced = _flat_plan(12758, mode = "balanced")
    assert _refine(klein_pipe, balanced) is balanced


def test_bigger_dense_dit_untouched(klein_pipe, monkeypatch):
    monkeypatch.setattr(
        dm,
        "_loaded_component_mib",
        lambda pipe: {**KLEIN_LOADED, "transformer": (17300, "dit")},
    )
    plan = _flat_plan(12758)
    assert _refine(klein_pipe, plan) is plan


@pytest.mark.parametrize("budget", list(range(3000, 30001, 250)))
def test_every_budget_fits_measured_need(klein_pipe, budget):
    """Only resident room is ever added, and kept bytes + measured peak x margin + base overhead fit the budget."""
    plan = _flat_plan(budget, max(8188, budget + 2400))
    new = _refine(klein_pipe, plan)
    if new is plan:
        return
    assert (new.offload_policy, new.stream_transformer, new.stream_text_encoders) == (
        plan.offload_policy,
        plan.stream_transformer,
        plan.stream_text_encoders,
    )
    kept = (
        160
        + (7393 if not new.stream_transformer else int(new.resident_transformer_mib or 0))
        + int(new.resident_text_encoder_mib or 0)
        + (0 if new.stream_text_encoders else 7673)
    )
    assert int(new.resident_transformer_mib or 0) <= 7393
    assert kept + KLEIN_HEADROOM + dm.DEFAULT_BASE_OVERHEAD_MIB <= budget


def test_dense_dit_never_takes_the_whole_resident_tier(klein_pipe):
    """The whole-resident tier's slack was measured on the int8 route; a dense DiT keeps the partial room."""
    new = _refine(klein_pipe, _flat_plan(11000, 13400))
    assert "resident_dit_slack_mib" not in new.estimates
    assert "encode_resident_transformer_mib" not in new.estimates
    assert int(new.resident_transformer_mib or 0) < 7393


def test_dense_request_extra_releases_for_oversized(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    pipe = types.SimpleNamespace(
        _unsloth_measured_reserve = (KLEIN_HEADROOM, "flux.2-klein", "off", 7393)
    )
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    assert dm.measured_request_extra_mib(pipe, width = 2048, height = 2048) == 11520 - KLEIN_HEADROOM
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, batch_size = 2) > 0
    q21 = types.SimpleNamespace(_unsloth_measured_reserve = (2304, "qwen-image-2.1", "default"))
    assert dm.measured_request_extra_mib(q21, width = 2048, height = 2048) == 8704 - 2304


def test_gguf_denoiser_keeps_the_flat_plan(klein_pipe):
    import torch

    class GGUFParameter(torch.nn.Parameter):
        pass

    dit = torch.nn.Linear(4, 4)
    dit.weight = GGUFParameter(dit.weight.data, requires_grad = False)
    klein_pipe.transformer = types.SimpleNamespace(dtype = torch.float16, parameters = dit.parameters)
    # GGUF dequantizes a whole Linear per forward: a transient the dense table never measured.
    plan = _flat_plan(12758)
    assert _refine(klein_pipe, plan) is plan


def test_small_host_int8_denoiser_keeps_the_flat_plan(klein_pipe):
    import torch

    import core.inference.diffusion_small_host as sh

    dit = torch.nn.Sequential(torch.nn.Linear(2048, 2048)).to(torch.bfloat16)
    sh.quantize_int8_weight_(dit, compute_dtype = torch.float16, work_device = "cpu")
    klein_pipe.transformer = types.SimpleNamespace(dtype = torch.float16, parameters = dit.parameters)
    plan = _flat_plan(12758)
    assert _refine(klein_pipe, plan) is plan
