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
    assert vid._h3_device_capacity_bytes("cuda") == vid._h3_free_device_bytes("cuda") == total // 2


def _share_of_card_reserve(monkeypatch):
    monkeypatch.setattr(
        dm,
        "discrete_reserve_mib",
        lambda total_mib = None: max(2048, min(int((total_mib or 0) * 0.10), 4096)),
    )


def test_companion_card_rank_uses_the_fast_limit_in_fast_mode(fraction, monkeypatch):
    import core.inference.diffusion_multigpu as mg

    _share_of_card_reserve(monkeypatch)

    probes = {
        0: mg.DeviceProbe(ordinal = 0, name = "a", total_mib = 24 * GiB, free_mib = 23 * GiB),
        1: mg.DeviceProbe(ordinal = 1, name = "a", total_mib = 24 * GiB, free_mib = 20 * GiB),
        2: mg.DeviceProbe(ordinal = 2, name = "b", total_mib = 192 * GiB, free_mib = 26010),
    }
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))
    fraction["value"] = 0.96
    fast = {
        o: dm._fast_device_budget_mib(
            DeviceMemory("cuda", "cuda", "discrete_vram", probes[o].free_mib, probes[o].total_mib)
        )
        for o in (1, 2)
    }
    assert fast[1] > fast[2]
    assert mg.best_secondary(selection, probes) == 2
    assert mg.best_secondary(selection, probes, mode = "fast") == 1


def test_video_h3_residency_budget_keeps_the_reserve_deficit(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.video as vid
    import core.inference.video_minimax_h3_residency as res
    from utils import hardware

    monkeypatch.setattr(
        hardware, "trusted_mem_get_info", lambda *a, **k: (2 * GiB << 20, 24 * GiB << 20)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 9 * GiB << 20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 8 * GiB << 20)
    monkeypatch.setattr(res, "h3_phase_need_gb", lambda *a, **k: 0.0)
    residency = types.SimpleNamespace(resident_bytes = lambda: 8 * GiB << 20)

    def budget():
        return vid._h3_residency_budget_bytes(
            residency, "cuda", width = 64, height = 64, frames = 1, te_streamed = True, te_scheme = None
        )

    assert budget() == 11 * GiB << 20  # default budget: free + cache + resident
    fraction["value"] = 0.50  # keeps 12 GiB free: the card is already 10 GiB inside it
    assert budget() == -(1 * GiB << 20)
    assert (
        vid._h3_free_device_bytes("cuda") == 0
    )  # the plain free reading still never goes negative


def test_video_h3_residency_budget_does_not_count_graph_held_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.video as vid
    import core.inference.video_minimax_h3_residency as res

    monkeypatch.setattr(vid, "_h3_signed_free_device_bytes", lambda device: 1000 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 600 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 500 * 2**20)
    monkeypatch.setattr(res, "h3_phase_need_gb", lambda *a, **k: 0.0)
    residency = types.SimpleNamespace(resident_bytes = lambda: 0)
    budget = vid._h3_residency_budget_bytes(
        residency, "cuda", width = 64, height = 64, frames = 1, te_streamed = True, te_scheme = None
    )
    assert budget == 1100 * 2**20


def test_video_request_available_reading_charges_the_reserve(fraction, monkeypatch):
    import core.inference.video as vid
    from utils import hardware

    monkeypatch.setattr(
        hardware, "trusted_mem_get_info", lambda *a, **k: (40 * GiB << 20, 80 * GiB << 20)
    )
    cuda = types.SimpleNamespace(memory_reserved = lambda device = None: 2 * GiB << 20)
    dev = types.SimpleNamespace(type = "cuda", index = 0)
    assert vid._request_available_bytes(dev, cuda) == 42 * GiB << 20
    fraction["value"] = 0.80  # keeps 16 GiB of the 80 GiB card free
    assert vid._request_available_bytes(dev, cuda) == (42 * GiB - int(80 * GiB * (1 - 0.80))) << 20
    assert vid._user_reserve_bytes(types.SimpleNamespace(type = "mps"), 80 * GiB << 20) == 0
    assert vid._user_reserve_bytes("cpu", 80 * GiB << 20) == 0


def test_every_video_request_guard_reads_through_the_reserve():
    """No request guard (H3 tiers with or without residency, the multi-GPU split, the conventional offload floor) sums raw free + allocator cache on its own any more."""
    import ast
    import inspect

    import core.inference.video as vid

    tree = ast.parse(inspect.getsource(vid))
    raw = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and {getattr(node.left, "id", None), getattr(node.right, "id", None)}
        == {"free_bytes", "reserved_bytes"}
    ]
    assert raw == []
    generate = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and "MiniMax-H3 needs about" in ast.get_source_segment(inspect.getsource(vid), n)
        and n.name != "_request_available_bytes"
    )
    calls = [
        n
        for n in ast.walk(generate)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_request_available_bytes"
    ]
    assert len(calls) == 3


def test_video_load_placement_guard_charges_the_reserve(fraction):
    import core.inference.video as vid

    card = _card(22 * GiB, 24 * GiB)
    assert vid._planned_free_less_user_reserve_mib(card) == 22 * GiB
    assert vid._planned_free_less_user_reserve_mib(_card(None, 24 * GiB)) is None
    fraction["value"] = 0.75
    assert vid._planned_free_less_user_reserve_mib(card) == 22 * GiB - 6 * GiB
    mac = DeviceMemory("mps", "mps", "unified_memory", 20 * GiB, 32 * GiB)
    assert vid._planned_free_less_user_reserve_mib(mac) == 20 * GiB


def test_h3_native_card_free_charges_the_reserve(fraction, monkeypatch):
    import contextlib

    import core.inference.video as vid
    from utils import hardware

    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda **k: {
            "devices": [{"visible_ordinal": 0, "vram_total_gb": 24.0, "vram_used_gb": 4.0}]
        },
    )
    from utils.hardware import gpu_query

    monkeypatch.setattr(gpu_query, "fresh_reads", contextlib.nullcontext)
    gib = 1024**3
    assert vid._h3_card_memory_bytes("cuda", 0) == (20 * gib, 24 * gib)
    fraction["value"] = 0.50
    assert vid._h3_card_memory_bytes("cuda", 0) == (8 * gib, 24 * gib)
    assert vid._h3_card_free_bytes("cuda", 0) == 8 * gib


def test_quant_staging_keeps_the_reserve_free(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.video as V

    moved = []

    class Block(torch.nn.Linear):
        def to(self, *a, **k):
            moved.append(self)
            return self

    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(
        [Block(1024, 1024, bias = False).to(torch.bfloat16) for _ in range(4)]
    )
    moved.clear()
    block = 1024 * 1024 * 2
    total = 64 * GiB << 20
    free = (
        V.DEFAULT_BASE_OVERHEAD_MIB * 1024 * 1024
        + V._STAGE_LINEAR_WORKSPACE * block
        + 4 * block
        + (1 * GiB << 20)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("utils.hardware.trusted_mem_get_info", lambda d, module = None: (free, total))
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda")
    assert len(V._stage_denoiser_for_quant(dit, target)) == 4
    moved.clear()
    fraction["value"] = 0.99  # 0.64 GiB reserve: the 1 GiB spare covers it, all four still stage
    assert len(V._stage_denoiser_for_quant(dit, target)) == 4
    fraction["value"] = 0.90  # 6.4 GiB reserve: nothing fits beside it
    assert V._stage_denoiser_for_quant(dit, target) == []


def test_image_layer_split_card_guard_charges_the_reserve(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion as D
    from core.inference import diffusion_layer_split_plan as lsp

    from . import test_diffusion_layer_split_plan as T

    for env in (T.mg.MULTIGPU_ENV, T.mg.CFG_PARALLEL_ENV, lsp.LAYER_SPLIT_ENV):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(T.mg, "_capability", lambda o: None)
    monkeypatch.setattr(lsp, "_is_rocm", lambda: False)
    monkeypatch.setattr(T.mg, "_rocm_torch", lambda: False)
    eng = T._Engine()
    monkeypatch.setattr(lsp, "engine_available", lambda: True)
    monkeypatch.setattr(lsp, "place_layer_split", eng.place)
    monkeypatch.setattr(lsp, "undo_layer_split", eng.undo)
    plan, pipe = T._layer_plan(monkeypatch, eng)
    lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    monkeypatch.setattr(D, "estimate_image_runtime_mib", lambda **k: 4096)
    working, reserve = lsp.applied_layer_split(pipe).working_mib(1)
    free = working + reserve + 512
    monkeypatch.setattr(
        D,
        "reclaimable_snapshot_device_memory",
        lambda target: DeviceMemory("cuda", "cuda", "discrete_vram", free, 24 * GiB),
    )
    kw = {"width": 1024, "height": 1024, "family": "flux2"}
    assert D._guard_layer_cards(pipe, object(), kw) == 1
    fraction["value"] = 0.90  # 2.4 GiB of cuda:1 kept free: the 512 MiB spare is not enough
    with pytest.raises(ValueError, match = "cuda:1"):
        D._guard_layer_cards(pipe, object(), kw)


def test_denoiser_hold_between_calls_keeps_the_reserve_free(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_multistage as MS

    param = types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda"))
    dit = types.SimpleNamespace(parameters = lambda: iter([param]))
    monkeypatch.setattr(MS, "_denoisers", lambda pipe: [dit])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (4 * GiB << 20, 24 * GiB << 20))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    assert MS.companions_fit_beside_denoiser(object(), ()) is True  # 2 GiB margin within 4 GiB free
    fraction["value"] = 0.90  # 2.4 GiB kept free leaves 1.6 GiB, under the margin
    assert MS.companions_fit_beside_denoiser(object(), ()) is False


def test_cfg_batch_doubled_call_keeps_the_reserve_free(fraction, monkeypatch):
    from core.inference import diffusion_cfg_batch as cb

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device = None: (10 * GiB << 20, 24 * GiB << 20)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 2 * GiB << 20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 1 * GiB << 20)
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 0)
    monkeypatch.setattr(cb, "_torch", lambda: torch)
    assert cb._usable_free_mib("cuda:0") == 11 * GiB
    fraction["value"] = 0.75
    assert cb._usable_free_mib("cuda:0") == 11 * GiB - 6 * GiB


def test_primary_card_rank_uses_free_less_the_reserve(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_device as dd

    cards = {0: (20 * GiB << 20, 80 * GiB << 20), 1: (14 * GiB << 20, 16 * GiB << 20)}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda o: cards[o])
    assert dd.rank_cuda_ordinals([0, 1]) == [0, 1]
    fraction["value"] = (
        0.80  # cuda:0 keeps 16 GiB (4 GiB left), cuda:1 keeps 3.2 GiB (10.8 GiB left)
    )
    assert dd.rank_cuda_ordinals([0, 1]) == [1, 0]


def test_primary_card_rank_uses_the_reserve_the_planner_enforces(fraction, monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_device as dd

    _share_of_card_reserve(monkeypatch)
    cards = {0: (12 * GiB << 20, 40 * GiB << 20), 1: (11 * GiB << 20, 16 * GiB << 20)}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda o: cards[o])
    assert dd.rank_cuda_ordinals([0, 1]) == [0, 1]
    fraction["value"] = 0.96
    safe = {o: dm._safe_device_budget_mib(_card(f >> 20, t >> 20)) for o, (f, t) in cards.items()}
    assert safe[1] > safe[0]
    assert dd.rank_cuda_ordinals([0, 1]) == [1, 0]


@pytest.fixture
def no_pins(monkeypatch):
    monkeypatch.setattr(dm, "_LOAD_BUDGET_FRACTIONS", {})


def test_generation_guards_keep_the_load_time_budget(fraction, no_pins):
    """Editing the Budget while a pipeline stays loaded must not move its generation-time reserve."""

    @dm.vram_budget_scope("image", load = True)
    def load():
        assert dm._user_budget_reserve_mib(10 * GiB) == 5 * GiB
        fraction["value"] = (
            0.70  # edited mid-load: this load still plans with the value it began with
        )
        assert dm._user_budget_reserve_mib(10 * GiB) == 5 * GiB
        dm.commit_load_budget_fraction("image")

    @dm.vram_budget_scope("image")
    def generate():
        return dm._user_budget_reserve_mib(10 * GiB), dm.user_budget_reserve_share()

    fraction["value"] = 0.50
    load()
    fraction["value"] = 0.75
    assert generate() == (5 * GiB, pytest.approx(0.50))
    assert dm._user_budget_reserve_mib(10 * GiB) == 5 * GiB
    load_again = dm.vram_budget_scope("image", load = True)(
        lambda: (dm._user_budget_reserve_mib(10 * GiB), dm.commit_load_budget_fraction("image"))[0]
    )
    assert load_again() == 2560
    assert generate()[0] == 2560
    dm.release_load_budget_fraction("image")
    fraction["value"] = 0.625
    assert dm._user_budget_reserve_mib(10 * GiB) == 3840
    fraction["value"] = vram_budget_settings.VRAM_FRACTION_DEFAULT
    assert generate() == (0, 0.0)


def test_each_backend_reads_its_own_snapshot(fraction, no_pins):
    for owner, value in (("image", 0.50), ("video", 0.75)):
        fraction["value"] = value
        dm.vram_budget_scope(owner, load = True)(lambda o = owner: dm.commit_load_budget_fraction(o))()
    fraction["value"] = vram_budget_settings.VRAM_FRACTION_DEFAULT
    image = dm.vram_budget_scope("image")(lambda: dm._user_budget_reserve_mib(10 * GiB))
    video = dm.vram_budget_scope("video")(lambda: dm._user_budget_reserve_mib(10 * GiB))
    assert (image(), video()) == (5 * GiB, 2560)
    assert dm._user_budget_reserve_mib(10 * GiB) == 5 * GiB


def test_default_budget_reserve_stays_zero_with_a_pipeline_loaded(fraction, no_pins):
    dm.vram_budget_scope("image", load = True)(lambda: dm.commit_load_budget_fraction("image"))()
    assert dm._user_budget_reserve_mib(10 * GiB) == 0 and dm.user_budget_reserve_share() == 0.0


def _scopes(fn, seen = None):
    """Every ``vram_budget_scope`` wrapper in ``fn``'s decorator chain, followed through ``__wrapped__`` and through closures (not every decorator here uses functools.wraps)."""
    seen = set() if seen is None else seen
    if not callable(fn) or id(fn) in seen or not hasattr(fn, "__code__"):
        return []
    seen.add(id(fn))
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
    found = []
    if fn.__code__ is dm.vram_budget_scope("x")(lambda: None).__code__:
        found.append((cells["owner"].cell_contents, cells["load"].cell_contents))
    for inner in [getattr(fn, "__wrapped__", None)] + [c.cell_contents for c in cells.values()]:
        found += _scopes(inner, seen)
    return found


def test_backend_entry_points_run_under_their_budget_scope():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend
    for cls, owner, loads, gens in (
        (
            DiffusionBackend,
            "image",
            ("begin_load", "_run_load", "download_plan", "load_pipeline"),
            ("generate",),
        ),
        (
            VideoBackend,
            "video",
            ("begin_load", "_run_load", "_run_load_h3_native", "download_plan", "load_pipeline"),
            ("begin_generate", "generate"),
        ),
    ):
        for name in loads:
            assert (owner, True) in _scopes(getattr(cls, name)), (cls.__name__, name)
        for name in gens:
            assert (owner, False) in _scopes(getattr(cls, name)), (cls.__name__, name)


def test_denoiser_hold_fit_does_not_count_graph_held_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_multistage as MS

    param = types.SimpleNamespace(device = torch.device("cuda", 0))
    dit = types.SimpleNamespace(parameters = lambda: iter([param]))
    monkeypatch.setattr(MS, "_denoisers", lambda pipe: [dit])
    monkeypatch.setattr(MS, "HOLD_MARGIN_BYTES", 1200 * 2**20)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1000 * 2**20, 24 * GiB << 20))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 600 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 500 * 2**20)
    # 1000 free + 100 reusable = 1100 MiB < 1200 need; counting the 500 graph-held gave 1600.
    assert MS.companions_fit_beside_denoiser(object(), ()) is False
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 0)
    assert MS.companions_fit_beside_denoiser(object(), ()) is True  # no graph pool: unchanged


def test_cfg_replica_free_does_not_count_graph_held_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_multigpu as mg

    seen = []
    monkeypatch.setattr(mg, "_free_total_mib", lambda ordinal: (1000, 24 * GiB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 600 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        dm, "_graph_held_cache_bytes", lambda device = None: seen.append(device) or 500 * 2**20
    )
    router = types.SimpleNamespace(secondary = torch.device("cuda", 1))
    assert mg.CfgParallelRouter._secondary_free_total_mib(router) == (1100, 24 * GiB)
    assert seen == [1]  # the replica card's pools, not the primary's
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 0)
    assert mg.CfgParallelRouter._secondary_free_total_mib(router) == (1600, 24 * GiB)
