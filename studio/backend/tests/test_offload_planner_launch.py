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


def _launch_with(
    tmp_path,
    monkeypatch,
    plan,
    *,
    owns = True,
    n_ctx = 0,
    n_parallel = 4,
    caps = None,
    avail_mib = 64 * 1024,
    **load_kwargs,
):
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
    backend._available_system_memory_mib = lambda: avail_mib
    seen = {}

    def fake_plan(
        inputs,
        *,
        extra_args = None,
        env = None,
    ):
        seen["inputs"] = inputs
        seen["extra_args"] = extra_args
        return plan

    backend._planned_tensor_spill = fake_plan
    launched = _launch(
        backend,
        gguf,
        speculative_type = "off",
        n_ctx = n_ctx,
        n_parallel = n_parallel,
        **load_kwargs,
    )
    return launched["cmd"], backend, seen


def _flag(
    cmd,
    name,
    default = None,
):
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
    four, _b, _s = _launch_with(
        tmp_path,
        monkeypatch,
        Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,)),
    )
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

    def fake_plan(
        inputs,
        *,
        extra_args = None,
        env = None,
    ):
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
    # 64 GiB of host RAM leaves the default whole, and a default that does not
    # bind is NOT written out: it is llama-server's own value, and emitting it
    # made every flag-on argv differ from flag-off, planned or declined alike.
    assert "--cache-ram" not in declined_cmd
    assert seen["inputs"]["cache_ram_default_mib"] == 8192
    assert seen["inputs"]["cache_ram_user_set"] is False

    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 1024
    )
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "--cache-ram") == "1024"
    assert cmd.count("--cache-ram") == 1
    assert backend._spill_plan_restore.get("--cache-ram") is None


def test_the_cache_ram_clamp_is_emitted_on_the_fallback_only_when_it_binds(tmp_path, monkeypatch):
    """A host with too little RAM for the model's spill plus the 8 GiB default
    gets the bound; the flag then carries the value the load-mode rule priced."""
    declined_cmd, _b, seen = _launch_with(
        tmp_path, monkeypatch, Plan(reason = "declined"), avail_mib = 12 * 1024
    )
    got = _flag(declined_cmd, "--cache-ram")
    assert got is not None and 0 <= int(got) < 8192, declined_cmd
    assert seen["inputs"]["cache_ram_default_mib"] == int(got)


def test_a_load_mode_the_plan_chose_rides_the_fit_record(tmp_path, monkeypatch):
    """The pair reaches the argv only through _fit_load_mode_flags, so every
    retry that strips the fit's load mode strips the plan's too."""
    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), load_mode_none = True
    )
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True})
    assert _flag(cmd, "--load-mode") == "none"
    assert backend._fit_load_mode_flags and "--load-mode" in backend._fit_load_mode_flags
    assert "--load-mode" not in backend._spill_plan_flags

    mmap_plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), load_mode_none = False
    )
    cmd, backend, _ = _launch_with(
        tmp_path, monkeypatch, mmap_plan, caps = {"supports_load_mode": True}
    )
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
    that branch. So the argv carries the planner's host RAM clamp when it binds
    and nothing otherwise, and never a trailing zero for last-wins to prefer."""
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, _backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert "--cache-ram" not in cmd, cmd
    bound = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 2048
    )
    cmd, _backend, _ = _launch_with(tmp_path, monkeypatch, bound)
    assert cmd.count("--cache-ram") == 1, cmd
    assert _flag(cmd, "--cache-ram") == "2048"
    assert "--ctx-checkpoints" not in cmd


def _launch_crash_then_ok(
    tmp_path,
    monkeypatch,
    plan,
    *,
    caps = None,
    **load_kwargs,
):
    """The planned launch crashes at startup; the revocation retry comes up healthy.

    Mirrors the real recovery in _spawn_and_wait: the first child exits on a signal,
    _revoke_spill_plan hands placement back to llama.cpp, and the second child is the
    one the session serves.
    """
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024 * MIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
        **(caps or {}),
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    backend._planned_tensor_spill = lambda inputs, **_kw: plan

    real_popen = subprocess.Popen
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        cmds.append(list(cmd))
        crashed = "off" == (cmd[cmd.index("--fit") + 1] if "--fit" in cmd else "")
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": -11 if crashed else None,
                "poll": lambda self: -11 if crashed else None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        launched = cmds[-1] if cmds else []
        backend._stdout_lines = []
        return not ("--fit" in launched and launched[launched.index("--fit") + 1] == "off")

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                speculative_type = "off",
                n_parallel = 4,
                **load_kwargs,
            )
        )
    return cmds, backend


def test_a_revoked_plan_commits_the_slots_the_child_that_answered_launched_with(
    tmp_path, monkeypatch
):
    """The revocation puts the fitter's --parallel back in the argv, so the state
    committed after the retry has to follow it.

    _commit_effective_parallel_slots drives request admission, the slot-save loop
    and the micro-batch recorded beside it, and nothing re-reads the slot count from
    the server the way _reconcile_effective_ctx_with_server re-reads the context.
    Left at the plan's reduced value, a four-slot child is served as a one-slot one.
    """
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan)

    assert len(cmds) == 2, cmds
    assert cmds[0][cmds[0].index("--parallel") + 1] == "1"
    assert cmds[1][cmds[1].index("--parallel") + 1] == "4"
    assert backend.effective_parallel_slots == 4


def test_the_planner_admits_against_ram_the_launch_has_already_spent(tmp_path, monkeypatch):
    """Plan.host_bytes is the embeddings plus the spilled weights and nothing else,
    yet the seam replaces the fit-derived load mode with the plan's. Host RAM this
    launch spends outside that figure has to reach the planner, or --load-mode none
    is decided against RAM that is not free -- an OOM kill where mmap would only
    have paged (llama.cpp #22629 is the same overcommit for --cache-ram alone).

    A --cache-ram the USER typed is the sharpest of the three: the rewrite below
    refuses to clamp it, so the plan's own clamp cannot pay for it."""
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    # Nothing unusual: the clamp owns the prompt cache and there is no drafter.
    assert seen["inputs"]["host_ram_unpriced_bytes"] == 0

    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, cache_ram = 20000)
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= 20000 * MIB


def test_the_batch_floor_follows_the_slots_the_plan_lowered(tmp_path, monkeypatch):
    """--batch-size is raised to max(slots, 2) when it is emitted, and llama.cpp
    derives the micro-batch from the emitted value. Rung 1 priced the cache and
    draft tables at the floor for the REDUCED slot count, so the flag has to
    follow the slots or the child runs a larger micro-batch than the pinned plan
    reserved for. The fitter's value is recorded so a revocation restores it."""
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, n_batch = 1)
    assert _flag(cmd, "--parallel") == "1"
    assert _flag(cmd, "--batch-size") == "2"
    assert backend._spill_plan_restore.get("--batch-size") == "4"

    # A batch already above every floor is untouched and not recorded.
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, n_batch = 512)
    assert _flag(cmd, "--batch-size") == "512"
    assert "--batch-size" not in backend._spill_plan_restore

    cmds, _backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan, n_batch = 1)
    assert _flag(cmds[0], "--batch-size") == "2"
    assert _flag(cmds[1], "--batch-size") == "4"


def test_a_context_only_shrink_plan_is_launched_at_the_context_it_proved(tmp_path, monkeypatch):
    """The auto path capped -c at 8192 before the planner was asked at the
    context it wanted. A plan that fits every tensor at a shorter context than
    requested spills nothing and moves no knob, and was discarded for exactly that,
    launching the child at the cap. It is Unsloth's placement all the same: every
    layer on the GPU, pinned, at the context the planner proved."""
    plan = Plan(changed = True, n_ctx = 12288)
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX
    assert _flag(cmd, "-c") == "12288"
    assert _flag(cmd, "--fit") == "off" and _flag(cmd, "-ngl") == "-1"
    assert backend._spill_plan_restore.get("-c") == "8192"
    assert backend._effective_context_length == 12288

    # At the requested context with nothing spilled it is llama.cpp's own launch.
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, Plan(changed = True, n_ctx = NATIVE_CTX))
    assert _flag(cmd, "--fit") == "on"
    assert backend._spill_plan_flags == []


def test_a_plan_that_moves_no_weight_is_all_on_the_gpu_for_the_mlock_gate(tmp_path, monkeypatch):
    """A plan that fits by lowering the slots, moving the projector or dropping the
    draft emits -ngl -1 --fit off with every layer on the card, the same launch the
    fits branch makes, but the Model Memory gate was told the weights sit in host
    RAM and would page-lock a full host copy of a model that needed those rungs to
    fit at all. A plan that spills weights really does leave some in host RAM."""
    from core.inference.llama_cpp import LlamaCppBackend

    seen = []
    real = LlamaCppBackend._weights_in_host_memory

    def spy(self, **kw):
        seen.append(bool(kw.get("fully_gpu_offloaded")))
        return real(self, **kw)

    monkeypatch.setattr(LlamaCppBackend, "_weights_in_host_memory", spy)
    _launch_with(tmp_path, monkeypatch, Plan(changed = True, n_ctx = 8192, n_parallel = 1))
    assert seen and seen[0] is True, seen
    seen.clear()
    _launch_with(
        tmp_path,
        monkeypatch,
        Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,)),
    )
    assert seen and seen[0] is False, seen


def test_the_layout_cache_identity_covers_every_shard(tmp_path, monkeypatch):
    """The layout of a split GGUF is read from every shard, so a sibling shard
    replaced while shard 1 is untouched has to invalidate the cached layout, or a
    stale complete one undercounts most of the model."""
    import types

    from core.inference import offload_layout
    from core.inference.llama_cpp import LlamaCppBackend

    first = tmp_path / "m-00001-of-00002.gguf"
    second = tmp_path / "m-00002-of-00002.gguf"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    reads = []
    monkeypatch.setattr(
        offload_layout, "layout_from_gguf", lambda path, **kw: reads.append(path) or object()
    )
    self = types.SimpleNamespace()
    a = LlamaCppBackend._tensor_spill_layout(self, str(first), all_shards = True)
    b = LlamaCppBackend._tensor_spill_layout(self, str(first), all_shards = True)
    assert a is b and len(reads) == 1
    second.write_bytes(b"22")
    c = LlamaCppBackend._tensor_spill_layout(self, str(first), all_shards = True)
    assert c is not a and len(reads) == 2


def test_a_plan_does_not_outlive_the_load_that_made_it(tmp_path, monkeypatch):
    """A knob-only plan leaves -ngl -1 --fit off on record. The next load's fits
    branch emits the same four tokens, so a retry there would strip them as if
    they were a plan and write the previous model's --parallel back into a
    command that never had one."""
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    _cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert backend._spill_plan_flags and backend._spill_plan_restore
    gguf = backend._gguf_path
    backend._planned_tensor_spill = lambda inputs, **_kw: None
    _launch(backend, gguf, speculative_type = "off", n_ctx = 16384, n_parallel = 4)
    assert backend._spill_plan_flags == [] and backend._spill_plan_restore == {}
    backend._spill_plan_flags, backend._spill_plan_restore = ["-ngl", "-1"], {"--parallel": "4"}
    backend.unload_model()
    assert backend._spill_plan_flags == [] and backend._spill_plan_restore == {}


def test_a_clamp_the_plan_appended_leaves_with_the_plan(tmp_path, monkeypatch):
    """On a roomy host the fallback carries no --cache-ram, so the plan's clamp is
    appended rather than rewritten. The revocation restored only values that were
    there before, so the clamp survived the plan it belonged to and the fitter
    placement ran with a prompt cache it never asked to shrink."""
    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 1024
    )
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "--cache-ram") == "1024"
    assert backend._spill_plan_restore.get("--cache-ram", "missing") is None
    reverted = backend._drop_tensor_spill(list(cmd), "test")
    assert reverted != cmd and "--cache-ram" not in reverted
    assert reverted[-2:] == ["--fit", "on"]

    # Where the fallback had its own bound, that value comes back.
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, avail_mib = 12 * 1024)
    before = backend._spill_plan_restore.get("--cache-ram")
    assert before is not None and _flag(cmd, "--cache-ram") == "1024"
    reverted = backend._drop_tensor_spill(list(cmd), "test")
    assert _flag(reverted, "--cache-ram") == before


def test_the_workload_prompt_is_the_whole_window_under_a_unified_cache(tmp_path, monkeypatch):
    """Studio appends --kv-unified on every multi-slot launch the build supports it
    on, and under a unified cache a single request may fill all of n_ctx. Pricing
    the prompt at n_ctx / slots under-charged the spill's prefill by the slot
    count, on exactly the loads Studio starts."""
    plan = Plan(changed = True, n_ctx = 4096, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _b, seen = _launch_with(tmp_path, monkeypatch, plan, n_ctx = 4096)
    assert seen["inputs"]["kv_unified"] is True
    assert seen["inputs"]["workload_prompt_tokens"] == 4096
    # The user turned the unified cache off: four private windows of a quarter each.
    _cmd, _b, seen = _launch_with(
        tmp_path, monkeypatch, plan, n_ctx = 4096, extra_args = ["--no-kv-unified"]
    )
    assert seen["inputs"]["kv_unified"] is False
    assert seen["inputs"]["n_parallel"] == 4
    assert seen["inputs"]["workload_prompt_tokens"] == 1024


def test_a_revoked_plan_takes_its_load_mode_with_it(tmp_path, monkeypatch):
    """Once a plan is taken the fit's --load-mode is replaced by the plan's,
    priced against the plan's host side. The revocation handed placement back to
    the fitter, which can put far more on the host, and left "none" in the retry
    argv, so the pageable fallback became a host-RAM OOM."""
    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), load_mode_none = True
    )
    cmds, backend = _launch_crash_then_ok(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert len(cmds) == 2, cmds
    assert _flag(cmds[0], "--load-mode") == "none"
    assert "--load-mode" not in cmds[1], cmds[1]
    assert cmds[1][-2:] == ["--fit", "on"]


def test_a_cache_ram_typed_into_the_extras_is_the_one_the_launch_prices(tmp_path, monkeypatch):
    """Extras are appended after every emitted flag and llama.cpp is last-wins, so
    a typed --cache-ram is what the child runs with. Priced as unset, the launch
    charged nothing for it, appended a clamp the extras then overrode, and let an
    unbounded prompt cache eat the RAM --load-mode none reserved for the weights."""
    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 1024
    )
    cmd, backend, seen = _launch_with(
        tmp_path, monkeypatch, plan, avail_mib = 12 * 1024, extra_args = ["--cache-ram", "-1"]
    )
    assert seen["inputs"]["cache_ram_user_set"] is True
    # -1 is charged as the default the cache is likely to reach, never as zero.
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= 8192 * MIB
    assert [i for i, a in enumerate(cmd) if a == "--cache-ram"] == [len(cmd) - 2]
    assert _flag(cmd, "--cache-ram") == "-1"
    assert "--cache-ram" not in backend._spill_plan_restore
