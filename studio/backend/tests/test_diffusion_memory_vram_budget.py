# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The image memory planner honours the user's VRAM Budget setting (#8178, #10695). Hermetic: no GPU."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    DeviceMemory,
    OFFLOAD_NONE,
    plan_diffusion_memory,
    total_capacity_budget_mib,
)
from utils import vram_budget_settings

GiB = 1024


@pytest.fixture
def fraction(monkeypatch):
    state = {"value": vram_budget_settings.VRAM_FRACTION_DEFAULT}
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: state["value"])
    return state


def _card(free, total, kind = "discrete_vram"):
    return DeviceMemory("cuda", "cuda", kind, free, total)


def _plan(memory, model_mib):
    return plan_diffusion_memory(
        target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True),
        device_memory = memory,
        model_dense_mib = model_mib,
        runtime_headroom_mib = 1 * GiB,
        companion_dense_mib = 2 * GiB,
    )


def test_lower_budget_shrinks_the_planner_budget(fraction):
    card = _card(22 * GiB, 24 * GiB)
    default = dm._safe_device_budget_mib(card)
    fraction["value"] = 0.80
    assert dm._safe_device_budget_mib(card) == 22 * GiB - int(24 * GiB * 0.20)
    assert dm._safe_device_budget_mib(card) < default


def test_lower_budget_moves_a_resident_load_to_offload(fraction):
    card = _card(22 * GiB, 24 * GiB)
    model = 12 * GiB  # + 1 GiB headroom + 2 GiB base overhead = 15 GiB resident
    assert _plan(card, model).offload_policy == OFFLOAD_NONE
    fraction["value"] = 0.70
    plan = _plan(card, model)
    assert plan.offload_policy != OFFLOAD_NONE
    assert plan.estimates["safe_device_budget_mib"] == 22 * GiB - int(24 * GiB * 0.30)


def test_fast_and_capacity_budgets_follow_the_setting(fraction):
    card = _card(22 * GiB, 24 * GiB)
    fast, capacity = dm._fast_device_budget_mib(card), total_capacity_budget_mib(card)
    fraction["value"] = 0.60
    reserve = int(24 * GiB * 0.40)
    assert dm._fast_device_budget_mib(card) == 22 * GiB - reserve < fast
    assert total_capacity_budget_mib(card) == int((24 * GiB - reserve) * 0.85) < capacity


@pytest.mark.parametrize(
    "free, total, kind",
    [
        (22 * GiB, 24 * GiB, "discrete_vram"),
        (150 * GiB, 192 * GiB, "discrete_vram"),  # 3% of 192 GiB exceeds the 4 GiB discrete cap
        (20 * GiB, 32 * GiB, "unified_memory"),
        (10 * GiB, 16 * GiB, "system_memory"),
    ],
)
@pytest.mark.parametrize("value", [None, 0.97, 0.99, 1.0])
def test_default_or_higher_budget_is_unchanged(fraction, free, total, kind, value):
    card = _card(free, total, kind)
    if value is not None:
        fraction["value"] = value
    reserve = dm._reserve_mib(kind, total)
    assert dm._safe_device_budget_mib(card) == free - reserve
    assert dm._fast_device_budget_mib(card) == free - max(2048, reserve // 2)
    assert total_capacity_budget_mib(card) == int((total - reserve) * 0.85)


def test_unified_reserve_is_only_ever_tightened(fraction):
    mac = _card(20 * GiB, 32 * GiB, "unified_memory")
    default = dm._safe_device_budget_mib(mac)
    fraction["value"] = 0.85  # below the 20% unified reserve already kept
    assert dm._safe_device_budget_mib(mac) == default
    fraction["value"] = 0.50
    assert dm._safe_device_budget_mib(mac) == 20 * GiB - 16 * GiB


def test_unreadable_setting_keeps_the_planner_budget(monkeypatch):
    card = _card(22 * GiB, 24 * GiB)
    default = dm._safe_device_budget_mib(card)

    def boom():
        raise RuntimeError("settings store unavailable")

    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", boom)
    assert dm._safe_device_budget_mib(card) == default
