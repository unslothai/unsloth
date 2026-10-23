# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The discrete VRAM reserve is fixed (ComfyUI's ``minimum_inference_memory``, floored at 2 GiB), not a share of the card.

Cards up to 20 GiB already sat on the old 2 GiB floor, so their plans are unchanged. CPU only.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_memory as dm

GIB = 1024
CARD_TOTALS_MIB = (
    8 * GIB,
    10 * GIB,
    12 * GIB,
    15095,  # T4 as torch reports it
    16 * GIB,
    20 * GIB,
    24 * GIB,
    32 * GIB,
    40 * GIB,
    48 * GIB,
    80 * GIB,
    96 * GIB,
    141 * GIB,
    182272,  # B200
)
FREE_FRACTIONS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.12)


def _old_reserve_mib(total_mib: int) -> int:
    return max(2048, min(int(total_mib * 0.10), 4096))


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("total", CARD_TOTALS_MIB)
def test_reserve_matrix_never_grows_and_small_cards_keep_theirs(monkeypatch, platform, total):
    monkeypatch.setattr(sys, "platform", platform)
    new = dm._reserve_mib("discrete_vram", total)
    old = _old_reserve_mib(total)
    assert new == 2048
    assert new <= old
    if total <= 20 * GIB:
        assert new == old
    for frac in FREE_FRACTIONS:
        free = int(total * frac)
        memory = dm.DeviceMemory("cuda", "cuda:0", "discrete_vram", free, total)
        budget = dm._safe_device_budget_mib(memory)
        assert budget == max(0, free - new)
        assert budget >= max(0, free - old)
        # an explicit fast load keeps its own 2 GiB minimum
        assert dm._fast_device_budget_mib(memory) == max(0, free - 2048)


def test_comfy_minimum_inference_memory(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert dm.comfy_min_inference_mib(8 * GIB) == 1200
    monkeypatch.setattr(sys, "platform", "win32")
    assert dm.comfy_min_inference_mib(8 * GIB) == 1400
    assert dm.comfy_min_inference_mib(16 * GIB) == 1500
    assert dm.discrete_reserve_mib(16 * GIB) == 2048


# (transformer, text encoder, other companions) MiB: klein-4B, Z-Image, Q21 int8, FLUX.1 fp8, SDXL-sized
_LOADS = (
    (7393, 7673, 160),
    (11739, 7672, 160),
    (9800, 9100, 480),
    (11500, 9500, 160),
    (4900, 1600, 160),
)


def _plan(total: int, free: int, dit: int, te: int, other: int, mode: str):
    return dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        requested_mode = mode,
        model_dense_mib = dit + te + other,
        companion_dense_mib = te + other,
        text_encoder_dense_mib = te,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None),
    )


@pytest.mark.parametrize("total", [t for t in CARD_TOTALS_MIB if t <= 20 * GIB])
@pytest.mark.parametrize("mode", ["auto", "fast"])
def test_small_card_plans_are_identical_to_the_old_reserve(monkeypatch, total, mode):
    for frac in FREE_FRACTIONS:
        free = int(total * frac)
        for dit, te, other in _LOADS:
            new = _plan(total, free, dit, te, other, mode)
            with monkeypatch.context() as m:
                m.setattr(
                    dm, "discrete_reserve_mib", lambda total_mib = None: _old_reserve_mib(total_mib)
                )
                old = _plan(total, free, dit, te, other, mode)
            assert new == old, (total, free, dit, te, mode)


@pytest.mark.parametrize("total", [t for t in CARD_TOTALS_MIB if t > 20 * GIB])
def test_big_card_plans_only_move_toward_resident(monkeypatch, total):
    order = {
        dm.OFFLOAD_NONE: 0,
        "group": 1,
        "model": 1,
        dm.OFFLOAD_STREAMING: 2,
        "sequential": 3,
    }
    for frac in FREE_FRACTIONS:
        free = int(total * frac)
        for dit, te, other in _LOADS:
            new = _plan(total, free, dit, te, other, "auto")
            with monkeypatch.context() as m:
                m.setattr(
                    dm, "discrete_reserve_mib", lambda total_mib = None: _old_reserve_mib(total_mib)
                )
                old = _plan(total, free, dit, te, other, "auto")
            assert (
                new.estimates["safe_device_budget_mib"] >= old.estimates["safe_device_budget_mib"]
            )
            assert order.get(new.offload_policy, 9) <= order.get(old.offload_policy, 9), (
                total,
                free,
                dit,
                te,
            )
