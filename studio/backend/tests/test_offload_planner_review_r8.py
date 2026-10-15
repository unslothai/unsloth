# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The lazy mode a restored CPU fallback inherits, the prompt-cache clamp a manual placement must
not own, and where a fitter placement's LIVE cache sits on a sliding-window model."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_planner import _fit_fallback_placement  # noqa: E402
from test_offload_cost_gate import GIB, dense_layout, gated  # noqa: E402


# ------------------------------------------------ finding 1: the direct CPU replay's lazy mode


@pytest.mark.parametrize(
    "emitted, extras, expected",
    [
        ("off", None, None),  # Studio's device-specific choice leaves with the devices
        ("on", None, None),
        (None, ["-lzm", "off"], "off"),  # the user's own flag stays
        (None, ["-lzm", "on"], "on"),
    ],
)
def test_a_restored_cpu_fallback_hands_studios_lazy_mode_back_to_auto(
    tmp_path, monkeypatch, emitted, extras, expected
):
    """A LoadRequest carrying cpu_fallback replays the Vulkan argv on no device without a crash
    first. Round 7 stripped Studio's -lzm on the crash recovery only, so this replay kept an
    iGPU's "off" and held the whole per-layer table resident on the CPU."""
    from test_llama_cpp_placement import _launch_with_vulkan_cpu_replay, _vulkan_cpu_replay_backend

    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 2.0, free_mib = 24 * 1024, avail_mib = 64 * 1024
    )
    backend._studio_lazy_mode = lambda **_kw: emitted
    backend._planner_may_run = lambda *a, **k: True
    kwargs = {"extra_args": tuple(extras)} if extras else {}
    captured = _launch_with_vulkan_cpu_replay(
        backend, gguf, crash = False, cpu_fallback = True, **kwargs
    )
    replay = captured["cmds"][-1]
    assert "--device" in replay and replay[replay.index("--device") + 1] == "none", replay
    modes = [replay[i + 1] for i, tok in enumerate(replay) if tok == "-lzm"]
    assert modes == ([] if expected is None else [expected]), replay


# ------------------------------------------------ finding 2: manual placement owns its own fit


@pytest.mark.parametrize("smart_offload", [True, False])
def test_a_manual_placement_keeps_the_prompt_cache_the_flag_off_launch_has(
    tmp_path, monkeypatch, smart_offload
):
    """Manual placement empties the GPU pool to bypass the planner, so a clamp that still thought
    the planner owned the fit credited zero VRAM: a 22 GiB model, 18 GiB of host RAM and -ngl 99
    emitted --cache-ram 0 with the flag on and nothing with it off."""
    from core.inference.offload_planner import Plan
    from test_offload_planner_launch import _launch_with

    cmd, _b, _seen = _launch_with(
        tmp_path,
        monkeypatch,
        Plan(reason = "declined"),
        owns = smart_offload,
        n_ctx = 8192,
        n_parallel = 4,
        avail_mib = 18 * 1024,
        model_mib = 22 * 1024,
        gpu_memory_mode = "manual",
        gpu_layers = 99,
    )
    assert "--cache-ram" not in cmd, cmd


# ------------------------------------------------ findings 3 / 4: the fitter's live cache split


_N, _WINDOW, _RESERVED, _SLOTS, _UBATCH = 62, 1024, 131072, 4, 512
_LIVE = 8192 + 256
_CELL = 16 * 1024  # bytes per layer per cell


def _is_swa(layer: int) -> bool:
    """Gemma 3: five window layers, then one full-context layer."""
    return layer % 6 != 5


def _weights_at(ctx: int, slots: int) -> tuple[int, ...]:
    ctx = max(256, int(ctx))
    window = min(ctx, _WINDOW * max(1, slots) + _UBATCH)
    return tuple(window if _is_swa(i) else ctx for i in range(_N))


def _bytes_at(ctx: int, slots: int) -> int:
    return sum(_weights_at(ctx, slots)) * _CELL


def _gemma3(moe: bool):
    layout = replace(dense_layout(n_blocks = _N), arch = "gemma3", has_swa = True)
    layout = replace(layout, n_ctx_train = _RESERVED)
    if moe:
        layout = replace(layout, is_moe = True)
    return layout


def _place(layout, *, with_live_weights: bool):
    opts = gated(
        kv_unified = True,
        n_parallel = _SLOTS,
        workload_prompt_tokens = 8192,
        workload_generated_tokens = 256,
        kv_bytes_at = _bytes_at,
        kv_swa_bytes_at = lambda c, s: sum(w for i, w in enumerate(_weights_at(c, s)) if _is_swa(i))
        * _CELL,
        kv_layer_weights_at = _weights_at if with_live_weights else None,
    )
    floor = _bytes_at(_RESERVED, _SLOTS)
    args = dict(quantised = False, kv_bytes_floor = floor, kv_on_host = False, n_seq = _SLOTS)
    # Enough short that whole leading layers leave, all of them window layers.
    from core.inference.offload_planner import all_resident_bytes

    resident = all_resident_bytes(
        layout,
        _RESERVED,
        kv_quantised = False,
        kv_bytes_floor = floor,
        kv_on_host = False,
        n_seq = _SLOTS,
        trust_floor = True,
    )
    budget = resident - int((1.1 if not layout.is_moe else 62 * 0.2 + 4.5 * 0.045) * GIB)
    return _fit_fallback_placement(
        layout, opts, budget, _RESERVED, kv_layer_weights = _weights_at(_RESERVED, _SLOTS), **args
    )


def _moved_layers(placement, layout) -> int:
    whole = sum(g.bytes_total for g in placement.host_groups if g.name == "layers")
    return round(whole / layout.blocks[0].resident_bytes)


@pytest.mark.parametrize("moe", [False, True], ids = ["dense", "moe"])
def test_the_fitters_live_cache_is_split_by_the_live_prefix(moe):
    """Window layers hold a window at any context, full-context layers grow with it: the reserved
    vector gives five moved window layers 1.5% of the cache, the 8448-token request puts 4.7% of
    its traffic on them. Memory freed keeps the reserved split; the traffic follows the live one."""
    layout = _gemma3(moe)
    placement = _place(layout, with_live_weights = True)
    assert placement is not None and placement.kv_host_bytes > 0
    moved = _moved_layers(placement, layout)
    assert 1 <= moved < 6, moved  # leading window layers only
    live = _weights_at(_LIVE, 1)
    live_total = min(_bytes_at(_RESERVED, _SLOTS), _bytes_at(_LIVE, 1))
    expected = int(live_total * sum(live[:moved]) / sum(live))
    assert placement.kv_host_bytes == expected, (placement.kv_host_bytes, expected)
    # The gap the reserved vector left: a third of the traffic.
    reserved = _weights_at(_RESERVED, _SLOTS)
    stale = int(live_total * sum(reserved[:moved]) / sum(reserved))
    assert expected > 3 * stale


def test_the_live_split_moves_the_gate_from_decline_to_accept():
    """The undercharge made the fitter look cheaper than it is: on this card the gate declined a
    3-block spill (1026 ms against a 1026 ms fitter) that wins once the moved window layers carry
    their live share (1514 ms)."""
    from core.inference.offload_planner import ContextPolicy, plan_placement

    layout = _gemma3(False)

    def plan(with_live_weights):
        opts = gated(
            kv_unified = True,
            n_parallel = _SLOTS,
            min_parallel = _SLOTS,
            workload_prompt_tokens = 8192,
            workload_generated_tokens = 256,
            context_policy = ContextPolicy.NEVER_REDUCE,
            kv_bytes_at = _bytes_at,
            kv_swa_bytes_at = lambda c, s: sum(
                w for i, w in enumerate(_weights_at(c, s)) if _is_swa(i)
            )
            * _CELL,
            kv_layer_weights_at = _weights_at if with_live_weights else None,
            kv_bytes_floor_by_parallel = {_SLOTS: _bytes_at(_RESERVED, _SLOTS)},
        )
        return plan_placement(
            layout,
            [44096 * 1024 * 1024],
            128 * GIB,
            _RESERVED,
            opts = opts,
            kv_layer_weights = _weights_at(_RESERVED, _SLOTS),
        )

    stale, live = plan(False), plan(True)
    assert stale.declined_by_gate and not stale.spilled_blocks, stale.reason
    assert not live.declined_by_gate and live.spilled_blocks, live.reason
    assert live.predicted_fit_request_ms > stale.predicted_fit_request_ms


def test_without_a_live_vector_the_fitter_keeps_the_reserved_split():
    """Every caller that supplies no live vector (flag off never reaches here) prices as before."""
    layout = _gemma3(False)
    placement = _place(layout, with_live_weights = False)
    assert placement is not None
    moved = _moved_layers(placement, layout)
    reserved = _weights_at(_RESERVED, _SLOTS)
    live_total = min(_bytes_at(_RESERVED, _SLOTS), _bytes_at(_LIVE, 1))
    assert placement.kv_host_bytes == int(live_total * sum(reserved[:moved]) / sum(reserved))


def test_the_live_vector_reaches_the_planner_through_the_seam():
    """The launch hands the planner the callable alongside kv_bytes_at, built on the same
    geometry, and the seam forwards it into PlanOptions."""
    import inspect

    from core.inference import llama_cpp as llama_mod

    src = inspect.getsource(llama_mod.LlamaCppBackend.load_model)
    assert '"kv_layer_weights_at": (' in src
    seam = inspect.getsource(llama_mod.LlamaCppBackend._planned_tensor_spill)
    assert "kv_layer_weights_at = (" in seam
