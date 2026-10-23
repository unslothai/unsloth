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


def _card(
    free,
    total,
    kind = "discrete_vram",
):
    return DeviceMemory("cuda", "cuda", kind, free, total)


def _plan(memory, model_mib):
    return plan_diffusion_memory(
        target = types.SimpleNamespace(
            device = "cuda", backend = "cuda", supports_model_cpu_offload = True
        ),
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


# Placement paths outside the one-card planner charge the same reserve.


def test_encoder_only_secondary_card_honours_the_budget(fraction, monkeypatch):
    import core.inference.diffusion_multigpu as mg
    from core.inference.diffusion_layer_split_plan import secondary_reserve_mib

    monkeypatch.setattr(mg, "_free_total_mib", lambda ordinal: (22 * GiB, 24 * GiB))
    assert mg.encoder_card_limit_mib(1) == 22 * GiB - secondary_reserve_mib()
    fraction["value"] = 0.50
    assert mg.encoder_card_limit_mib(1) == 22 * GiB - 12 * GiB


def test_cfg_replica_is_declined_when_it_would_break_the_budget(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_multigpu as mg

    class Pipe:
        transformer = torch.nn.Linear(1024, 1024)  # 4 MiB of weights

    probes = {
        o: mg.DeviceProbe(ordinal = o, name = "card", total_mib = 24 * GiB, free_mib = 13 * GiB)
        for o in (0, 1)
    }
    monkeypatch.setattr(mg, "cfg_parallel_decline", lambda pipe, **kw: None)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(
        mg, "probe_devices", lambda ordinals, **kw: {o: probes[o] for o in ordinals}
    )
    armed, declined = Pipe(), Pipe()
    try:
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,))):
            assert mg.install_cfg_parallel(armed, offload_policy = "none") is None
            fraction["value"] = (
                0.50  # keeps 12 GiB of the 13 GiB free: the 4 MiB replica still fits
            )
            mg.uninstall_cfg_parallel(armed)
            assert mg.install_cfg_parallel(armed, offload_policy = "none") is None
            fraction["value"] = 0.45  # keeps 13.2 GiB: no room left for the replica
            reason = mg.install_cfg_parallel(declined, offload_policy = "none")
        assert reason is not None and reason.startswith("replica needs")
    finally:
        mg.uninstall_cfg_parallel(armed)
        mg.uninstall_cfg_parallel(declined)


def test_video_h3_free_reading_keeps_the_budget_free(fraction, monkeypatch):
    import core.inference.video as vid
    from utils import hardware

    monkeypatch.setattr(
        hardware, "trusted_mem_get_info", lambda *a, **k: (22 * GiB << 20, 24 * GiB << 20)
    )
    assert vid._h3_free_device_bytes("cuda") == 22 * GiB << 20
    fraction["value"] = 0.50
    assert vid._h3_free_device_bytes("cuda") == 10 * GiB << 20
