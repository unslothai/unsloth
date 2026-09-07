# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The spill plan's decisions on the real launch path.

The planner is patched to return a hand-built Plan, so these cases are about what
load_model DOES with one: which argv values it rewrites in place, what it records so
a retry can put them back, which flags follow, and that a launch the planner does
not own is untouched. Dense Qwen3.8-27B metadata, a card the weights outgrow, so
the load reaches the --fit arm where the planner is consulted.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference.offload_planner import Plan  # noqa: E402
from test_llama_cpp_placement import _backend, _launch  # noqa: E402

MIB = 1024 * 1024
CARD_MIB = 12 * 1024
NATIVE_CTX = 262144

DENSE = {
    "_architecture": "qwen3",
    "_vocab_size": 248320,
    "_n_layers": 64,
    "_n_kv_heads": 8,
    "_n_heads": 32,
    "_embedding_length": 5120,
    "_kv_key_length": 128,
    "_kv_value_length": 128,
    "_key_length_mla": None,
    "_context_length": NATIVE_CTX,
}


def _launch_with(tmp_path, monkeypatch, plan, *, owns = True, n_ctx = 0, n_parallel = 4, caps = None):
    """Launch a load the planner is consulted on, returning (cmd, backend, seen inputs)."""
    if owns:
        monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    else:
        monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
    memory = [(0, CARD_MIB, CARD_MIB)]
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    # 30 GiB of weights on a 12 GiB card: nothing fits, the fit arm is reached.
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024 * MIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
        **(caps or {}),
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    seen = {}

    def fake_plan(inputs, *, extra_args = None, env = None):
        seen["inputs"] = inputs
        seen["extra_args"] = extra_args
        return plan

    backend._planned_tensor_spill = fake_plan
    launched = _launch(backend, gguf, speculative_type = "off", n_ctx = n_ctx, n_parallel = n_parallel)
    return launched["cmd"], backend, seen


def _flag(cmd, name, default = None):
    return cmd[cmd.index(name) + 1] if name in cmd else default


def test_a_plan_that_lowers_the_slots_rewrites_parallel_in_place_and_records_the_old_value(
    tmp_path, monkeypatch
):
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "--parallel") == "1"
    assert cmd.count("--parallel") == 1
    assert cmd[-4:] != ["--fit", "on"] and _flag(cmd, "--fit") == "off"
    assert backend._spill_plan_restore.get("--parallel") == "4"
    assert backend._spill_plan_flags == ["-ngl", "-1", "--fit", "off"]
    # The local is rebound too, not only the token: the context integrity flags
    # below the consumer read it, and --kv-unified is a multi-slot flag.
    assert "--kv-unified" not in cmd
    four, _b, _s = _launch_with(tmp_path, monkeypatch, Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,)))
    assert "--kv-unified" in four


def test_a_plan_that_keeps_the_slots_records_nothing(tmp_path, monkeypatch):
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "--parallel") == "4"
    assert "--parallel" not in backend._spill_plan_restore
    assert "-ot" in cmd


def test_a_plan_at_a_larger_context_raises_c_and_the_published_ceiling(tmp_path, monkeypatch):
    """Auto settled for 8192 before the planner was asked at the context it
    wanted; a plan that serves the larger one is launched at it, and /status
    publishes it, with the fitter's own value kept for the revocation."""
    plan = Plan(changed = True, n_ctx = 65536, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "-c") == "65536"
    assert backend._spill_plan_restore.get("-c") == "8192"
    assert backend._effective_context_length == 65536
    assert backend._max_context_length >= 65536
    # And the planner WAS asked at the pre-cap context, not at 8192.
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX
    assert seen["inputs"]["context_policy_fit_only"] is True
    assert seen["inputs"]["min_ctx"] == 8192


def test_an_explicit_context_is_never_reduced_and_never_raised(tmp_path, monkeypatch):
    plan = Plan(changed = True, n_ctx = 16384, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan, n_ctx = 16384)
    assert seen["inputs"]["n_ctx"] == 16384
    assert seen["inputs"]["context_policy_fit_only"] is False
    assert _flag(cmd, "-c") == "16384"
    assert "-c" not in backend._spill_plan_restore


def test_the_floor_map_covers_every_slot_count_down_to_one(tmp_path, monkeypatch):
    plan = Plan(reason = "declined")
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    floors = seen["inputs"]["kv_bytes_floor_by_parallel"]
    assert sorted(floors) == [1, 2, 3, 4]
    assert floors[4] == seen["inputs"]["kv_cache_bytes"]
    # Non-decreasing, not strictly: this launch plans a UNIFIED cache (one stream
    # shared by every slot), so fewer slots do not shrink it and rung 1 has only
    # the per-slot recurrent state and compute terms to win there.
    assert floors[1] <= floors[2] <= floors[3] <= floors[4]
    assert seen["inputs"]["min_parallel"] == 1
    assert seen["inputs"]["n_parallel"] == 4
    assert 512 <= seen["inputs"]["workload_prompt_tokens"] <= 8192


def test_a_typed_parallel_pins_the_floor_of_the_slot_rung(tmp_path, monkeypatch):
    """A --parallel the user typed is theirs: the planner may not walk below it."""
    plan = Plan(reason = "declined")
    memory = [(0, CARD_MIB, CARD_MIB)]
    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024 * MIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {"supports_kv_unified": True}
    seen = {}

    def fake_plan(inputs, *, extra_args = None, env = None):
        seen["inputs"] = inputs
        return plan

    backend._planned_tensor_spill = fake_plan
    _launch(backend, gguf, speculative_type = "off", n_parallel = 4, extra_args = ["--parallel", "4"])
    assert seen["inputs"]["min_parallel"] == 4
    assert sorted(seen["inputs"]["kv_bytes_floor_by_parallel"]) == [4]


def test_the_cache_ram_clamp_is_emitted_on_the_fallback_and_rewritten_by_the_plan(
    tmp_path, monkeypatch
):
    declined_cmd, _b, seen = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    assert _flag(declined_cmd, "--fit") == "on"
    # 64 GiB of host RAM leaves the default whole.
    assert _flag(declined_cmd, "--cache-ram") == "8192"
    assert seen["inputs"]["cache_ram_default_mib"] == 8192
    assert seen["inputs"]["cache_ram_user_set"] is False

    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 1024)
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "--cache-ram") == "1024"
    assert cmd.count("--cache-ram") == 1
    assert backend._spill_plan_restore.get("--cache-ram") == "8192"


def test_a_load_mode_the_plan_chose_rides_the_fit_record(tmp_path, monkeypatch):
    """The pair reaches the argv only through _fit_load_mode_flags, so every
    retry that strips the fit's load mode strips the plan's too."""
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), load_mode_none = True)
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True})
    assert _flag(cmd, "--load-mode") == "none"
    assert backend._fit_load_mode_flags and "--load-mode" in backend._fit_load_mode_flags
    assert "--load-mode" not in backend._spill_plan_flags

    mmap_plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), load_mode_none = False)
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, mmap_plan, caps = {"supports_load_mode": True})
    assert "--load-mode" not in cmd


def test_a_launch_the_planner_does_not_own_is_untouched(tmp_path, monkeypatch):
    """Flag off: no clamp, no floor map, nothing priced for the rungs, and the
    fitter's argv verbatim. The planner method itself declines on the flag
    (covered in the seam suite); here it is patched to a decline so that what
    load_model prices AROUND it is what is under test."""
    plan = Plan(reason = "declined")
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan, owns = False)
    assert _flag(cmd, "--fit") == "on"
    assert _flag(cmd, "--parallel") == "4"
    assert seen["inputs"]["n_ctx"] == 8192
    assert seen["inputs"]["kv_bytes_floor_by_parallel"] == {}
    assert seen["inputs"]["min_parallel"] == 4
    assert seen["inputs"]["mmproj_movable"] is False
    assert seen["inputs"]["draft_droppable"] is False
    assert "--cache-ram" not in cmd


def test_a_planner_owned_launch_on_windows_carries_the_clamp_and_not_the_tuning_zero(
    tmp_path, monkeypatch
):
    """#5692's Windows full-offload tuning emits --cache-ram 0, and #10382 skips it
    on a shared pool. Neither reaches a launch the planner owns: the tuning keys
    on fully_gpu_offloaded, which only the "model fits, force every layer on"
    branch sets, and a planner launch (spill or --fit on fallback) never takes
    that branch. So the argv carries exactly one --cache-ram, the planner's host
    RAM clamp, and no trailing zero for last-wins to prefer."""
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, _backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert cmd.count("--cache-ram") == 1, cmd
    assert _flag(cmd, "--cache-ram") == "8192"
    assert "--ctx-checkpoints" not in cmd
