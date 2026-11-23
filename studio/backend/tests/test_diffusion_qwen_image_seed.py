# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image / Qwen-Image-2512 text-to-image placement seeds; CPU only, sizes from 16 / 12 GiB load logs."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

LOADED = {
    "transformer": (32491, "dit"),  # prequant cache line: 31.73 GiB
    "text_encoder": (8559, "text_encoder"),
    "vae": (242, "other"),
}
# "estimates={'model_dense_mib': 30603, 'companion_dense_mib': 9151, 'text_encoder_dense_mib': 8864, ...}"
FLAT = dict(model_dense_mib = 30603, companion_dense_mib = 9151, text_encoder_dense_mib = 8864)
WINDOW = 1620  # "streams 60 offload groups with event-fenced prefetch (2 ahead, 1620 MiB window)"
RESIDENT_HEADROOM = 7168  # seed x 1.15, rounded up to 256 MiB
STREAMED_HEADROOM = (
    3072  # Qwen-Image t2i's own streamed figure (2560) x 1.15, rounded up to 256 MiB
)
EDIT_STREAMED_HEADROOM = 4608  # the Qwen-Image-Edit figure (3824), which t2i borrowed before
BUDGETS = [(9550, 12288), (13638, 16376)]
ENVS = (
    "MEASURED_ACTIVATION_ENV",
    "PARTIAL_RESIDENT_ENV",
    "MEASURED_ACTIVATION_QWEN_EDIT_ENV",
    "MEASURED_ACTIVATION_QWEN_IMAGE_ENV",
    "MEASURED_ACTIVATION_STREAMED_ENV",
    "STREAMED_RESIDENCY_ENV",
    "RESIDENT_DIT_ENV",
    "QWEN_IMAGE_STREAMED_SEED_ENV",
)


def _clear(monkeypatch):
    for env in ENVS:
        name = getattr(dm, env, None)
        if name:
            monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_LEARNED_PEAKS", "0")


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _plan(budget_mib: int, total_mib: int):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "qwen-image"
        ),
        requested_mode = None,
        **FLAT,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def pipe(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda p: dict(LOADED), raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda p: True)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda p: WINDOW)
    return object()


def _refine(
    pipe,
    plan,
    family = "qwen-image",
    speed = "default",
):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed)


def _wide_room(memory, headroom: int) -> int:
    return (
        int(memory.free_mib)
        - dm._resident_dit_slack_mib(memory)
        - LOADED["vae"][0]
        - headroom
        - WINDOW
    )


def test_seed_values(monkeypatch):
    _clear(monkeypatch)
    m = dm.measured_image_runtime_mib
    assert m("qwen-image", "default") == RESIDENT_HEADROOM
    assert m("qwen-image", "max") == RESIDENT_HEADROOM
    assert m("qwen-image", "default", streamed = True) == STREAMED_HEADROOM
    assert m("qwen-image", "default", width = 1328, height = 1328) > RESIDENT_HEADROOM
    assert m("qwen-image", "off") is None
    assert m("qwen-image", "off", dense_transformer_mib = 38968) == 5120
    # GGUF / packed denoisers (no static seed) still answer from a learned peak alone
    assert m("qwen-image", "default", static_seed = False) is None
    assert m("qwen-image-edit", "default") == RESIDENT_HEADROOM
    assert m("qwen-image-edit", "default", streamed = True) == EDIT_STREAMED_HEADROOM


@pytest.mark.parametrize("budget,total", BUDGETS)
def test_budget_keeps_part_of_the_denoiser_resident(pipe, budget, total):
    plan = _plan(budget, total)
    assert plan.stream_transformer and plan.offload_policy in (
        dm.OFFLOAD_GROUP,
        dm.OFFLOAD_STREAMING,
    )
    assert not plan.resident_transformer_mib
    new = _refine(pipe, plan)
    memory = plan.device_memory
    resident = new.resident_transformer_mib
    assert resident and resident == _wide_room(memory, STREAMED_HEADROOM)
    assert resident < LOADED["transformer"][0]
    assert new.estimates["measured_streamed_headroom_mib"] == STREAMED_HEADROOM
    assert new.estimates["measured_runtime_headroom_mib"] == RESIDENT_HEADROOM
    # the encode streams the resident groups back out (the 16 GiB Edit overrun)
    assert new.estimates["encode_resident_transformer_mib"] == 0
    assert resident + 2560 + WINDOW + LOADED["vae"][0] + dm._resident_dit_slack_mib(memory) <= int(
        memory.free_mib
    )
    assert new.offload_policy == plan.offload_policy
    assert new.stream_text_encoders == plan.stream_text_encoders


@pytest.mark.parametrize("budget,total", BUDGETS)
def test_kill_switch_restores_the_all_streamed_plan(pipe, monkeypatch, budget, total):
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_QWEN_IMAGE_ENV, "0")
    assert dm.measured_image_runtime_mib("qwen-image", "default") is None
    assert dm.measured_image_runtime_mib("qwen-image", "default", streamed = True) is None
    plan = _plan(budget, total)
    assert _refine(pipe, plan) is plan
    assert dm.measured_image_runtime_mib("qwen-image-edit", "default") == RESIDENT_HEADROOM


def test_streamed_kill_switch_falls_back_to_the_resident_seed(pipe, monkeypatch):
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_STREAMED_ENV, "0")
    plan = _plan(13638, 16376)
    new = _refine(pipe, plan)
    assert new.resident_transformer_mib == _wide_room(plan.device_memory, RESIDENT_HEADROOM)
    assert "measured_streamed_headroom_mib" not in new.estimates


@pytest.mark.parametrize("speed", ["off", "eager"])
def test_unseeded_tiers_keep_the_flat_plan(pipe, speed):
    plan = _plan(13638, 16376)
    assert _refine(pipe, plan, speed = speed) is plan


def test_dense_bf16_denoiser_is_not_placed_from_the_int8_seed(pipe, monkeypatch):
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda p: False)
    plan = _plan(13638, 16376)
    assert _refine(pipe, plan) is plan


def test_t2i_streamed_figure_is_its_own_and_below_the_edit_figure(monkeypatch):
    _clear(monkeypatch)
    assert dm._MEASURED_STREAMED_IMAGE_PEAK_MIB["qwen-image"] == 2560
    assert dm._MEASURED_STREAMED_IMAGE_PEAK_MIB["qwen-image-edit"] == 3824
    m = dm.measured_image_runtime_mib
    assert m("qwen-image", "default", streamed = True, width = 1328, height = 1328) > STREAMED_HEADROOM
    assert m("qwen-image", "default", streamed = True, width = 1328, height = 1328) < m(
        "qwen-image-edit", "default", streamed = True, width = 1328, height = 1328
    )
    assert m("qwen-image", "default") == RESIDENT_HEADROOM


def test_t2i_streamed_kill_switch_restores_the_borrowed_figure(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(dm.QWEN_IMAGE_STREAMED_SEED_ENV, "0")
    m = dm.measured_image_runtime_mib
    assert m("qwen-image", "default", streamed = True) == EDIT_STREAMED_HEADROOM
    assert m("qwen-image", "default") == RESIDENT_HEADROOM
    assert m("qwen-image-edit", "default", streamed = True) == EDIT_STREAMED_HEADROOM
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_STREAMED_ENV, "0")
    assert m("qwen-image", "default", streamed = True) == RESIDENT_HEADROOM
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_QWEN_IMAGE_ENV, "0")
    assert m("qwen-image", "default", streamed = True) is None


@pytest.mark.parametrize("budget,total", BUDGETS)
def test_t2i_figure_keeps_more_resident_than_the_borrowed_one(pipe, monkeypatch, budget, total):
    plan = _plan(budget, total)
    new = _refine(pipe, plan)
    monkeypatch.setenv(dm.QWEN_IMAGE_STREAMED_SEED_ENV, "0")
    old = _refine(pipe, plan)
    memory = plan.device_memory
    # kill switch == the placement before this change, byte for byte
    assert old.resident_transformer_mib == _wide_room(memory, EDIT_STREAMED_HEADROOM)
    assert old.estimates["measured_streamed_headroom_mib"] == EDIT_STREAMED_HEADROOM
    gained = new.resident_transformer_mib - old.resident_transformer_mib
    assert gained == EDIT_STREAMED_HEADROOM - STREAMED_HEADROOM
    assert new.resident_transformer_mib + STREAMED_HEADROOM + WINDOW + LOADED["vae"][
        0
    ] + dm._resident_dit_slack_mib(memory) == int(memory.free_mib)
    assert new.estimates["encode_resident_transformer_mib"] == 0


def test_learned_peak_above_the_t2i_figure_still_raises_it(monkeypatch):
    _clear(monkeypatch)
    import core.inference.diffusion_measured_peaks as mp

    monkeypatch.setattr(mp, "lookup", lambda **kw: 4000)
    assert dm.measured_image_runtime_mib("qwen-image", "default", quant = "q", streamed = True) == 4608
    monkeypatch.setattr(
        mp, "lookup", lambda **kw: 1907
    )  # the learned 16 GiB render: never lowers the figure
    assert (
        dm.measured_image_runtime_mib("qwen-image", "default", quant = "q", streamed = True)
        == STREAMED_HEADROOM
    )
