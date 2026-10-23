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


def test_default_request_tiles_under_a_lowered_budget_but_is_never_refused(fraction):
    card = _card(22 * GiB, 24 * GiB)

    def verdict(tile_side):
        return dm.image_activation_verdict(
            device_memory = card, width = None, height = None, vae_tile_side = tile_side
        ).action

    assert verdict(512) == dm.ACTIVATION_RUN  # default budget: untouched
    fraction["value"] = 0.40  # 7,783 MiB budget: under the 8,192 MiB default estimate + overhead
    assert verdict(512) == dm.ACTIVATION_TILE
    assert (
        verdict(None) == dm.ACTIVATION_RUN
    )  # nothing to tile: the load budgeted this size, so still not refused


def test_calibrated_tier_does_not_promote_past_a_lowered_budget(fraction):
    from core.inference.diffusion_memory import OFFLOAD_MODEL, CalibratedImageActivation

    act = CalibratedImageActivation(
        text_encoder_mib = 1_498,
        denoise_mib = 794,
        decode_mib = 9_172,
        tiled_decode_mib = 590,
        max_canvas_mib = 2_960,
    )
    total = 25 * GiB
    card = DeviceMemory("cuda", "cuda", "discrete_vram", total - 1_229, total)

    def plan():
        return plan_diffusion_memory(
            target = types.SimpleNamespace(
                device = "cuda", backend = "cuda", supports_model_cpu_offload = True
            ),
            device_memory = card,
            runtime_headroom_mib = 8192,
            calibrated_activation = act,
            model_dense_mib = 17_897,
            companion_dense_mib = 10_247,
            text_encoder_dense_mib = 8_959,
        )

    fraction["value"] = 0.50
    p = plan()
    budget = p.estimates["safe_device_budget_mib"]
    assert budget == total - 1_229 - int(total * 0.50)
    transformer = 17_897 - 10_247
    # whole-module offload uploads the transformer beside the max-canvas activation: it must fit the budget
    if p.offload_policy == OFFLOAD_MODEL:
        assert transformer + act.max_canvas_mib + p.estimates["base_overhead_mib"] <= budget


@pytest.mark.parametrize(
    "memory",
    [
        DeviceMemory("mps", "mps", "unified_memory", 20 * GiB, 32 * GiB),
        DeviceMemory(
            "mps", "mps", "unified_memory", 20 * GiB, 32 * GiB, metal_working_set_mib = 28 * GiB
        ),
        DeviceMemory("cpu", "cpu", "system_memory", 10 * GiB, 16 * GiB),
    ],
)
def test_metal_and_cpu_ignore_the_budget_like_the_llm_path(fraction, memory):
    safe, fast, capacity = (
        dm._safe_device_budget_mib(memory),
        dm._fast_device_budget_mib(memory),
        total_capacity_budget_mib(memory),
    )
    fraction["value"] = 0.30
    assert dm._safe_device_budget_mib(memory) == safe
    assert dm._fast_device_budget_mib(memory) == fast
    assert total_capacity_budget_mib(memory) == capacity


def test_cfg_replica_picks_the_card_the_budget_leaves_room_on(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_multigpu as mg

    class Pipe:
        transformer = torch.nn.Linear(51_200, 51_200, bias = False, device = "meta")  # 10 GiB fp32

    probes = {
        0: mg.DeviceProbe(ordinal = 0, name = "card", total_mib = 24 * GiB, free_mib = 23 * GiB),
        1: mg.DeviceProbe(ordinal = 1, name = "card", total_mib = 192 * GiB, free_mib = 60 * GiB),
        2: mg.DeviceProbe(ordinal = 2, name = "card", total_mib = 24 * GiB, free_mib = 23 * GiB),
    }
    monkeypatch.setattr(mg, "cfg_parallel_decline", lambda pipe, **kw: None)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(
        mg, "probe_devices", lambda ordinals, **kw: {o: probes[o] for o in ordinals}
    )
    fraction["value"] = (
        0.50  # cuda:1 must keep 96 GiB free (none usable), cuda:2 keeps 12 GiB (11 GiB usable)
    )
    pipe = Pipe()
    try:
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))):
            assert mg.install_cfg_parallel(pipe, offload_policy = "none") is None
        assert str(mg._CFG_ROUTERS[pipe].secondary) == "cuda:2"
    finally:
        mg.uninstall_cfg_parallel(pipe)


def test_measured_residency_promotion_keeps_the_budget_free(fraction, monkeypatch):
    # the logged Qwen-Image-2.1 int8 auto load on a 12 GiB card, refined from its loaded weights
    for env in (dm.MEASURED_ACTIVATION_ENV, dm.PARTIAL_RESIDENT_ENV, dm.RESIDENT_DIT_ENV):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMED_RESIDENCY", raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    loaded = {
        "transformer": (6922, "dit"),
        "text_encoder": (8960, "text_encoder"),
        "vae": (645, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(loaded))
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 624, raising = False)
    card = DeviceMemory("cuda", "cuda", "discrete_vram", 11_598, 12 * GiB)

    def refined():
        plan = plan_diffusion_memory(
            target = types.SimpleNamespace(supports_model_cpu_offload = True),
            device_memory = card,
            model_dense_mib = 19630,
            companion_dense_mib = 12182,
            text_encoder_dense_mib = 10847,
            runtime_headroom_mib = 8192,
        )
        if plan.offload_policy == dm.OFFLOAD_MODEL:
            plan = dm.torchao_streaming_plan(plan)
        return dm.refine_plan_from_loaded_weights(
            object(), plan, family = "qwen-image-2.1", speed_mode = "default"
        )

    default = refined()
    fraction["value"] = 0.50
    capped = refined()
    reserve = int(12 * GiB * 0.50)
    resident = capped.resident_transformer_mib or 0
    # whatever stays resident, the user's share of the card stays free beside it
    assert resident + 645 + capped.estimates["measured_runtime_headroom_mib"] + reserve <= 11_598
    assert resident < (default.resident_transformer_mib or 0)


def test_companion_card_is_ranked_by_what_the_budget_leaves(fraction, monkeypatch):
    import core.inference.diffusion_multigpu as mg

    probes = {
        0: mg.DeviceProbe(ordinal = 0, name = "a", total_mib = 24 * GiB, free_mib = 23 * GiB),
        1: mg.DeviceProbe(ordinal = 1, name = "b", total_mib = 192 * GiB, free_mib = 60 * GiB),
        2: mg.DeviceProbe(ordinal = 2, name = "a", total_mib = 24 * GiB, free_mib = 23 * GiB),
    }
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))
    assert mg.best_secondary(selection, probes) == 1  # default budget: most free wins, unchanged
    fraction["value"] = 0.50  # cuda:1 keeps 96 GiB free (nothing usable), cuda:2 has 11 GiB usable
    assert mg.best_secondary(selection, probes) == 2
    readings = {1: (60 * GiB, 192 * GiB), 2: (23 * GiB, 24 * GiB)}
    monkeypatch.setattr(mg, "_free_total_mib", lambda o: readings[o])
    automatic = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2), automatic = True)
    assert mg.best_secondary(automatic) == 2


def test_companion_card_rank_uses_the_reserve_the_limit_enforces(fraction):
    import core.inference.diffusion_multigpu as mg

    probes = {
        0: mg.DeviceProbe(ordinal = 0, name = "a", total_mib = 24 * GiB, free_mib = 23 * GiB),
        1: mg.DeviceProbe(ordinal = 1, name = "a", total_mib = 24 * GiB, free_mib = 20 * GiB),
        2: mg.DeviceProbe(ordinal = 2, name = "b", total_mib = 192 * GiB, free_mib = 26 * GiB),
    }
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))
    assert mg.best_secondary(selection, probes) == 2  # default: most free
    fraction["value"] = 0.96  # just below the default: the planner reserves still bind
    limits = {
        o: dm._safe_device_budget_mib(
            DeviceMemory("cuda", "cuda", "discrete_vram", probes[o].free_mib, probes[o].total_mib)
        )
        for o in (1, 2)
    }
    assert limits[2] > limits[1]
    assert mg.best_secondary(selection, probes) == 2


def test_video_h3_capacity_probe_charges_the_same_reserve_as_placement(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.video as vid
    from utils import hardware

    total = 80 * GiB << 20
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (total, total))
    monkeypatch.setattr(hardware, "trusted_mem_get_info", lambda *a, **k: (total, total))
    assert vid._h3_device_capacity_bytes("cuda") == total
    fraction["value"] = 0.50
    # an idle card: what the plan sees before the download equals what placement sees after it
    assert vid._h3_device_capacity_bytes("cuda") == vid._h3_free_device_bytes("cuda") == total // 2
