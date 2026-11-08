# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 streamed partial placement sized from the denoise phase, its resident blocks lent to the VAE decode.
CPU only; sizes from an L4 6 GiB int8 load log ("1028 MiB of the 6144 MiB transformer ... measured peak 2816 MiB")."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

LOADED = {
    "transformer": (6144, "dit"),
    "text_encoder": (4619, "text_encoder"),
    "vae": (160, "other"),
}
FLAT = dict(model_dense_mib = 10583, companion_dense_mib = 4779, text_encoder_dense_mib = 4619)
WINDOW = 703
WHOLE_RENDER_HEADROOM = 2816  # untiled decode 2437 x 1.15, rounded up to 256 MiB
DENOISE_HEADROOM = 1280  # denoise 901 x 1.15, rounded up to 256 MiB
ENVS = (
    "MEASURED_ACTIVATION_ENV",
    "PARTIAL_RESIDENT_ENV",
    "MEASURED_ACTIVATION_STREAMED_ENV",
    "STREAMED_RESIDENCY_ENV",
    "RESIDENT_DIT_ENV",
    "STREAMED_DECODE_LEND_ENV",
)


def _clear(monkeypatch):
    for env in ENVS:
        name = getattr(dm, env, None)
        if name:
            monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_LEARNED_PEAKS", "0")


def _memory(budget_mib: int = 6144, total_mib: int = 23034) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _plan(
    budget_mib: int = 6144,
    total_mib: int = 23034,
    family = "flux.3-klein",
):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None, family = family),
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
    return types.SimpleNamespace(_unsloth_phase_seeds = True)


def _refine(
    pipe,
    plan,
    family = "flux.3-klein",
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


def test_seed_columns(monkeypatch):
    _clear(monkeypatch)
    m = dm.measured_image_runtime_mib
    assert m("flux.3-klein", "default", phase_seeds = True) == WHOLE_RENDER_HEADROOM
    assert m("flux.3-klein", "default", phase_seeds = True, seed_phase = 1) == DENOISE_HEADROOM
    assert m("flux.3-klein-base", "max", phase_seeds = True, seed_phase = 1) == DENOISE_HEADROOM
    assert (
        dm._streamed_decode_lend_seed(
            types.SimpleNamespace(_unsloth_phase_seeds = True), "flux.3-klein", "default"
        )
        == 2437
    )
    assert (
        dm._streamed_decode_lend_seed(
            types.SimpleNamespace(_unsloth_phase_seeds = True), "flux.3-klein-base", "max"
        )
        == 2512
    )
    # unmeasured families, unseeded pipes and unseeded tiers keep the whole-render seed
    for pipe, family, speed in (
        (types.SimpleNamespace(_unsloth_phase_seeds = True), "flux.2-klein", "default"),
        (types.SimpleNamespace(_unsloth_phase_seeds = True), "lumina-2", "default"),
        (types.SimpleNamespace(_unsloth_phase_seeds = False), "flux.3-klein", "default"),
        (types.SimpleNamespace(_unsloth_phase_seeds = True), "flux.3-klein", "off"),
    ):
        assert dm._streamed_decode_lend_seed(pipe, family, speed) is None


@pytest.mark.parametrize("family", ["flux.3-klein", "flux.3-klein-base"])
def test_budget_sizes_the_resident_blocks_from_the_denoise_phase(pipe, family):
    plan = _plan(family = family)
    assert plan.stream_transformer and not plan.resident_transformer_mib
    new = _refine(pipe, plan, family = family)
    memory = plan.device_memory
    resident = new.resident_transformer_mib
    assert resident == _wide_room(memory, DENOISE_HEADROOM)
    assert resident > _wide_room(memory, WHOLE_RENDER_HEADROOM) + 1500
    assert new.estimates["measured_streamed_headroom_mib"] == DENOISE_HEADROOM
    assert new.estimates["encode_resident_transformer_mib"] == 0
    assert new.estimates["measured_streamed_seed_phase"] == 1
    decode = 2437 if family == "flux.3-klein" else 2512
    lend = dm.decode_lend_from_plan(new)
    assert lend == (resident, decode, 1024 * 1024, False)
    # the decode's need at 1024^2 is the measured decode x margin; release_for_decode streams only the shortfall
    assert dm.decode_need_mib(lend, width = 1024, height = 1024) == -(-decode * 115 // 100)
    assert dm.decode_need_mib(lend, width = 2048, height = 2048) > dm.decode_need_mib(
        lend, width = 1024, height = 1024
    )


def test_kill_switch_restores_the_whole_render_seed(pipe, monkeypatch):
    monkeypatch.setenv(dm.STREAMED_DECODE_LEND_ENV, "0")
    plan = _plan()
    new = _refine(pipe, plan)
    assert new.resident_transformer_mib == _wide_room(plan.device_memory, WHOLE_RENDER_HEADROOM)
    assert "decode_lend_transformer_mib" not in new.estimates
    assert "measured_streamed_seed_phase" not in new.estimates
    assert dm.decode_lend_from_plan(new) is None


def test_other_families_are_unchanged(pipe, monkeypatch):
    plan = _plan(family = "flux.2-klein")
    new = _refine(pipe, plan, family = "flux.2-klein")
    assert "decode_lend_transformer_mib" not in (new.estimates or {})
    monkeypatch.setenv(dm.STREAMED_DECODE_LEND_ENV, "0")
    again = _refine(pipe, plan, family = "flux.2-klein")
    assert again.resident_transformer_mib == new.resident_transformer_mib


def _reserve(pipe, new):
    pipe._unsloth_measured_reserve = (
        new.estimates["measured_streamed_headroom_mib"],
        "flux.3-klein",
        "default",
        None,
        None,
        None,
        None,
        False,
        True,
        False,
        False,
        False,
        True,
        new.estimates.get("measured_streamed_seed_phase"),
    )


def test_requests_price_the_denoise_phase_not_the_decode(pipe):
    new = _refine(pipe, _plan())
    _reserve(pipe, new)
    # a 1024^2 render needs nothing beyond the placement: the decode borrows through release_for_decode
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, cfg = False) == 0
    # a bigger canvas still streams resident blocks for the denoise
    assert dm.measured_request_extra_mib(pipe, width = 2048, height = 2048, cfg = False) > 0
    # a reserve written by an older load (13 fields) keeps the whole-render pricing
    pipe._unsloth_measured_reserve = pipe._unsloth_measured_reserve[:13]
    assert (
        dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, cfg = False)
        == WHOLE_RENDER_HEADROOM - DENOISE_HEADROOM
    )
