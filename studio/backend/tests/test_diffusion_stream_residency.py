# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Partial residency of a streamed torchao denoiser; planner inputs are what real FLUX.1-dev / Qwen-Image-2.1 int8
auto loads logged."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

FLUX_LOADED = {
    "transformer": (11420, "dit"),
    "text_encoder": (235, "text_encoder"),
    "text_encoder_2": (9083, "text_encoder"),
    "vae": (160, "other"),
}
FLUX_FLAT = dict(
    model_dense_mib = 22019,
    companion_dense_mib = 9536,
    text_encoder_dense_mib = 9346,
    runtime_headroom_mib = 8192,
)
FLUX_WINDOW = 1296  # 3 x the largest int8 double block (the logged prefetcher window)

Q21_LOADED = {
    "transformer": (6922, "dit"),
    "text_encoder": (8960, "text_encoder"),
    "vae": (645, "other"),
}
Q21_FLAT = dict(
    model_dense_mib = 19630,
    companion_dense_mib = 12182,
    text_encoder_dense_mib = 10847,
    runtime_headroom_mib = 8192,
)
Q21_WINDOW = 624


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _plan(flat: dict, budget_mib: int, total_mib: int):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        **flat,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def stub(monkeypatch):
    for env in (
        dm.MEASURED_ACTIVATION_ENV,
        dm.PARTIAL_RESIDENT_ENV,
        dm.RESIDENT_DIT_ENV,
        "UNSLOTH_DIFFUSION_STREAMED_RESIDENCY",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)

    def use(loaded: dict, window: int):
        monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(loaded))
        monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: window, raising = False)
        return object()

    return use


def _refine(pipe, plan, family):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = "default")


def test_flux_and_z_image_have_a_measured_peak(monkeypatch):
    monkeypatch.delenv(dm.MEASURED_ACTIVATION_ENV, raising = False)
    # 2666 MiB untiled VAE decode x 1.15, rounded up to 256 MiB; linear in pixels above 1 MP
    for family in ("flux.1", "z-image"):
        assert dm.measured_image_runtime_mib(family, "default") == 3072
        assert dm.measured_image_runtime_mib(family, "max") == 3072
        assert dm.measured_image_runtime_mib(family, "default", width = 2048, height = 2048) == 12288
        assert dm.measured_image_runtime_mib(family, "off") is None


def test_flux_int8_16gb_keeps_most_blocks_resident(stub):
    """The logged 16 GB FLUX.1-dev plan streamed all 58 groups."""
    pipe = stub(FLUX_LOADED, FLUX_WINDOW)
    plan = _plan(FLUX_FLAT, 13668, 16384)
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_text_encoders
    assert plan.stream_transformer
    new = _refine(pipe, plan, "flux.1")
    free = plan.device_memory.free_mib
    slack = dm._resident_dit_slack_mib(plan.device_memory)
    room = free - slack - 160 - 3072 - FLUX_WINDOW
    assert new.offload_policy == dm.OFFLOAD_GROUP and new.stream_transformer
    assert new.resident_transformer_mib == room
    assert 8000 < room < 11420
    assert new.resident_text_encoder_mib is None
    assert new.estimates["encode_resident_transformer_mib"] == 13668 - 3072 - 2048 - 160
    assert new.estimates["stream_window_mib"] == FLUX_WINDOW
    assert new.estimates["measured_runtime_headroom_mib"] == 3072


def test_flux_int8_12gb_partial(stub):
    pipe = stub(FLUX_LOADED, FLUX_WINDOW)
    plan = _plan(FLUX_FLAT, 9572, 12288)
    new = _refine(pipe, plan, "flux.1")
    free = plan.device_memory.free_mib
    room = free - dm._resident_dit_slack_mib(plan.device_memory) - 160 - 3072 - FLUX_WINDOW
    assert new.resident_transformer_mib == room > 4000


def test_q21_int8_8gb_drops_the_double_reserve(stub):
    pipe = stub(Q21_LOADED, Q21_WINDOW)
    plan = _plan(Q21_FLAT, 5476, 8192)
    assert plan.offload_policy == dm.OFFLOAD_STREAMING
    new = _refine(pipe, plan, "qwen-image-2.1")
    free = plan.device_memory.free_mib
    flat_room = 5476 - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 645
    room = free - 1024 - 645 - 2304 - Q21_WINDOW
    assert flat_room == 479  # what the 8 GB load logged
    assert new.resident_transformer_mib == room
    assert room > 5 * flat_room
    assert new.estimates["encode_resident_transformer_mib"] == flat_room


def test_never_past_free_memory(stub):
    for loaded, flat, window, family, budgets in (
        (
            FLUX_LOADED,
            FLUX_FLAT,
            FLUX_WINDOW,
            "flux.1",
            ((13668, 16384), (9572, 12288), (5500, 8192)),
        ),
        (
            Q21_LOADED,
            Q21_FLAT,
            Q21_WINDOW,
            "qwen-image-2.1",
            ((5476, 8192), (4000, 6144), (9000, 12288)),
        ),
    ):
        pipe = stub(loaded, window)
        other = sum(m for m, r in loaded.values() if r == "other")
        for budget, total in budgets:
            plan = _plan(flat, budget, total)
            new = _refine(pipe, plan, family)
            kept = new.resident_transformer_mib or 0
            headroom = new.estimates.get("measured_runtime_headroom_mib", 0)
            whole = kept == loaded["transformer"][0]
            need = kept + other + headroom + dm._resident_dit_slack_mib(plan.device_memory)
            assert need + (0 if whole else window) <= plan.device_memory.free_mib, (family, budget)


def test_whole_tier_still_wins_when_it_fits(stub):
    pipe = stub(Q21_LOADED, Q21_WINDOW)
    new = _refine(pipe, _plan(Q21_FLAT, 9550, 12288), "qwen-image-2.1")
    assert new.resident_transformer_mib == 6922
    assert "stream_window_mib" not in new.estimates


def test_kill_switch_and_unsized_window_keep_the_flat_room(stub, monkeypatch):
    plan = _plan(Q21_FLAT, 5476, 8192)
    flat_room = 5476 - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 645
    pipe = stub(Q21_LOADED, -1)
    assert _refine(pipe, plan, "qwen-image-2.1").resident_transformer_mib == flat_room
    pipe = stub(Q21_LOADED, Q21_WINDOW)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_STREAMED_RESIDENCY", "0")
    new = _refine(pipe, plan, "qwen-image-2.1")
    assert new.resident_transformer_mib == flat_room
    assert "encode_resident_transformer_mib" not in new.estimates


def test_dense_denoisers_keep_the_flat_room(stub, monkeypatch):
    """The dense eager table was measured resident, not streamed: its partial room is unchanged."""
    pipe = stub(
        {
            "transformer": (22700, "dit"),
            "text_encoder_2": (9083, "text_encoder"),
            "vae": (160, "other"),
        },
        1296,
    )
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    monkeypatch.setattr(dm, "_denoiser_compute_bytes", lambda pipe: 2)
    plan = _plan(dict(FLUX_FLAT, model_dense_mib = 32000), 13668, 16384)
    new = dm.refine_plan_from_loaded_weights(pipe, plan, family = "flux.1", speed_mode = "off")
    assert "stream_window_mib" not in new.estimates


def test_stream_window_is_depth_plus_one_largest_blocks(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREFETCH_DEPTH", raising = False)

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2048, 2048)  # top level: not a streamed block
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1024, 1024, bias = False)] * 1)
            self.single = torch.nn.ModuleList([torch.nn.Linear(512, 1024, bias = False)])

    pipe = types.SimpleNamespace(
        components = {"transformer": DiT(), "vae": torch.nn.Linear(4096, 4096)}
    )
    # largest block 4 MiB (fp32 1024x1024), depth 2 -> 3 in flight
    assert dm._stream_window_mib(pipe) == 12
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PREFETCH_DEPTH", "4")
    assert dm._stream_window_mib(pipe) == 20
    assert dm._stream_window_mib(types.SimpleNamespace(components = {})) == -1
