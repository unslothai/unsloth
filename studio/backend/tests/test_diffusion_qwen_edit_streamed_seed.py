# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Edit-2511 streamed-tier placement seed; CPU only, sizes from 12 / 16 GiB load logs."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

# Qwen-Image-Edit-2511 auto (int8 DiT, int8 encoder, bf16 VAE) as loaded on an RTX PRO 6000 capped at 12 / 16 GiB.
LOADED = {
    "transformer": (32496, "dit"),
    "text_encoder": (8559, "text_encoder"),
    "vae": (242, "other"),
}
FLAT = dict(model_dense_mib = 38956, companion_dense_mib = 16062, text_encoder_dense_mib = 15820)
WINDOW = 1620  # "streams 60 offload groups with event-fenced prefetch (2 ahead, 1620 MiB window)"
RESIDENT_HEADROOM = 7168  # 6046 x 1.15, rounded up to 256 MiB
STREAMED_HEADROOM = 4608  # 3824 x 1.15, rounded up to 256 MiB
ENVS = (
    "MEASURED_ACTIVATION_ENV",
    "PARTIAL_RESIDENT_ENV",
    "MEASURED_ACTIVATION_QWEN_EDIT_ENV",
    "MEASURED_ACTIVATION_STREAMED_ENV",
    "STREAMED_RESIDENCY_ENV",
    "RESIDENT_DIT_ENV",
)


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _plan(budget_mib: int, total_mib: int):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "qwen-image-edit"
        ),
        requested_mode = None,
        **FLAT,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def pipe(monkeypatch):
    for env in ENVS:
        name = getattr(dm, env, None)
        if name:
            monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_LEARNED_PEAKS", "0")
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda p: dict(LOADED), raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda p: True)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda p: WINDOW)
    return object()


def _refine(pipe, plan):
    return dm.refine_plan_from_loaded_weights(
        pipe, plan, family = "qwen-image-edit", speed_mode = "default"
    )


def _wide_room(memory, headroom: int) -> int:
    return (
        int(memory.free_mib)
        - dm._resident_dit_slack_mib(memory)
        - LOADED["vae"][0]
        - headroom
        - WINDOW
    )


def test_streamed_seed_value():
    assert dm.measured_image_runtime_mib("qwen-image-edit", "default") == RESIDENT_HEADROOM
    assert (
        dm.measured_image_runtime_mib("qwen-image-edit", "default", streamed = True)
        == STREAMED_HEADROOM
    )
    # scales with the output pixels like the resident seed
    assert dm.measured_image_runtime_mib(
        "qwen-image-edit", "default", width = 1328, height = 1328, streamed = True
    ) > dm.measured_image_runtime_mib("qwen-image-edit", "default", streamed = True)
    # families without a streamed entry keep their seed; eager tiers stay unseeded
    assert dm.measured_image_runtime_mib(
        "qwen-image-2.1", "default", streamed = True
    ) == dm.measured_image_runtime_mib("qwen-image-2.1", "default")
    assert dm.measured_image_runtime_mib("qwen-image-edit", "off", streamed = True) is None


@pytest.mark.parametrize("budget,total", [(9550, 12288), (13638, 16376)])
def test_streamed_placement_keeps_more_groups_resident(pipe, budget, total):
    plan = _plan(budget, total)
    assert plan.stream_transformer
    new = _refine(pipe, plan)
    memory = plan.device_memory
    resident = new.resident_transformer_mib
    assert resident == _wide_room(memory, STREAMED_HEADROOM)
    # the old placement subtracted the resident seed, which already carries a window pad, beside the window
    assert resident - _wide_room(memory, RESIDENT_HEADROOM) == RESIDENT_HEADROOM - STREAMED_HEADROOM
    assert new.estimates["measured_streamed_headroom_mib"] == STREAMED_HEADROOM
    # the whole-render headroom (multi-GPU splits read it) is unchanged
    assert new.estimates["measured_runtime_headroom_mib"] == RESIDENT_HEADROOM
    # never above free memory: resident + raw streamed peak + window + VAE + slack
    assert resident + 3824 + WINDOW + LOADED["vae"][0] + dm._resident_dit_slack_mib(memory) <= int(
        memory.free_mib
    )


def test_streamed_seed_kill_switch(pipe, monkeypatch):
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_STREAMED_ENV, "0")
    plan = _plan(9550, 12288)
    new = _refine(pipe, plan)
    assert new.resident_transformer_mib == _wide_room(plan.device_memory, RESIDENT_HEADROOM)
    assert "measured_streamed_headroom_mib" not in new.estimates
    assert (
        dm.measured_image_runtime_mib("qwen-image-edit", "default", streamed = True)
        == RESIDENT_HEADROOM
    )


def test_qwen_edit_kill_switch_still_drops_the_streamed_seed(pipe, monkeypatch):
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_QWEN_EDIT_ENV, "0")
    assert dm.measured_image_runtime_mib("qwen-image-edit", "default", streamed = True) is None
    plan = _plan(9550, 12288)
    assert _refine(pipe, plan) is plan


def _reserve(streamed: bool):
    headroom = STREAMED_HEADROOM if streamed else RESIDENT_HEADROOM
    base = (
        headroom,
        "qwen-image-edit",
        "default",
        None,
        None,
        None,
        None,
        None,
        True,
        False,
        False,
        False,
    )
    return base + ((True,) if streamed else ())


def test_request_guard_prices_on_the_streamed_scale(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_LEARNED_PEAKS", "0")
    for env in ENVS:
        name = getattr(dm, env, None)
        if name:
            monkeypatch.delenv(name, raising = False)
    pipe = types.SimpleNamespace(_unsloth_measured_reserve = _reserve(True))
    # the request the load was placed for keeps every resident group
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    # a larger one streams the difference back out
    big = dm.measured_request_extra_mib(pipe, width = 1328, height = 1328)
    assert (
        big
        == dm.measured_image_runtime_mib(
            "qwen-image-edit", "default", width = 1328, height = 1328, streamed = True
        )
        - STREAMED_HEADROOM
    )
    assert big > 0
    # a second source image is priced by the flat allowance on top
    assert (
        dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, condition_pixels = 1024 * 1024)
        > big
    )
    # a placement made from the resident seed is priced as before
    old = types.SimpleNamespace(_unsloth_measured_reserve = _reserve(False))
    assert dm.measured_request_extra_mib(old, width = 1024, height = 1024) == 0


def test_vae_prebuild_child_is_priced_against_the_streamed_runtime(monkeypatch):
    # 16 GiB load: 7.5 GiB free after the resident groups land, but the streamed seed (4608) and the prefetch window
    # (1620) are still to come, so a child would take the room the first render needs (OOM in the child before).
    import torch
    from torch.torch_version import TorchVersion

    from core.inference import diffusion_vae_prebuild as prebuild

    spawned = []
    monkeypatch.delenv("UNSLOTH_DIFFUSION_VAE_PREBUILD", raising = False)
    monkeypatch.setattr(prebuild, "_spawn", lambda job, logger: spawned.append(job) or True)
    monkeypatch.setattr(prebuild, "plan", lambda pipe: {"device": 0})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(
        torch,
        "__version__",
        TorchVersion(str(torch.__version__).split("+")[0] + "+cu130"),
        raising = False,
    )
    monkeypatch.setattr(dm, "user_budget_reserve_bytes", lambda device, total: 0)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: WINDOW)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (7680 << 20, 16384 << 20))
    first_start = types.SimpleNamespace(hit = False)

    def reserve(streamed):
        return (
            (STREAMED_HEADROOM if streamed else RESIDENT_HEADROOM, "qwen-image-edit")
            + (None,) * 10
            + (streamed,)
        )

    assert (
        prebuild.maybe_kick(
            types.SimpleNamespace(_unsloth_measured_reserve = reserve(True)), first_start
        )
        is False
    )
    assert spawned == []
    # resident-seed placement (and the kill switch, which places from it) keeps the free-memory gate unchanged
    assert (
        prebuild.maybe_kick(
            types.SimpleNamespace(_unsloth_measured_reserve = reserve(False)), first_start
        )
        is True
    )
    assert prebuild.maybe_kick(types.SimpleNamespace(), first_start) is True
    # room for the child on top of the pending streamed runtime: it still runs
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (16384 << 20, 24576 << 20))
    assert (
        prebuild.maybe_kick(
            types.SimpleNamespace(_unsloth_measured_reserve = reserve(True)), first_start
        )
        is True
    )
    assert len(spawned) == 3


@pytest.mark.parametrize("budget,total", [(9550, 12288), (13638, 16376)])
def test_streamed_seed_encode_streams_every_resident_group(pipe, monkeypatch, budget, total):
    # 16 GiB load on hardware: the encode kept the flat room (3989 MiB) on top of the streamed tier's wider residency,
    # held 3782 MiB through the prompt encode (base 1080) and overran the budget (allocator OOM, then a retry)
    new = _refine(pipe, _plan(budget, total))
    assert new.estimates["measured_streamed_headroom_mib"] == STREAMED_HEADROOM
    assert new.estimates["encode_resident_transformer_mib"] == 0
    # install_encode_release then streams the whole residency for the encode: the encode state is the all-streamed one
    surplus = new.resident_transformer_mib - new.estimates["encode_resident_transformer_mib"]
    assert surplus == new.resident_transformer_mib
    # the streamed-seed kill switch keeps the resident-seed placement and its flat encode room unchanged
    monkeypatch.setenv(dm.MEASURED_ACTIVATION_STREAMED_ENV, "0")
    plan = _plan(budget, total)
    old = _refine(pipe, plan)
    assert "measured_streamed_headroom_mib" not in old.estimates
    flat = old.estimates["encode_resident_transformer_mib"]
    assert flat > 0
    # during the encode the streamed tier never holds more than the resident-seed placement does (base: proven on
    # hardware at both budgets); the extra residency it pins for the denoise streams out first
    extra = new.resident_transformer_mib - old.resident_transformer_mib
    assert extra > 0
    assert new.estimates["encode_resident_transformer_mib"] + extra <= max(flat, extra)
    assert new.estimates["encode_resident_transformer_mib"] <= max(0, flat - extra)
