# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The spill plan's decisions on the real launch path."""

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
    # 30 GiB on the 12 GiB card: nothing fits, so the fit arm is reached. Named so a case can
    # state its own spill instead.
    model_mib = 30 * 1024,
    speculative_type = "off",
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
    backend._get_gguf_size_bytes = lambda _path: model_mib * MIB
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
        speculative_type = speculative_type,
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
    """Auto settled for 8192 before the planner was asked at the context it wanted; a plan that
    serves the larger one is launched at it, with the fitter's value kept for the revocation."""
    plan = Plan(changed = True, n_ctx = 65536, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "-c") == "65536"
    assert backend._spill_plan_restore.get("-c") == "8192"
    assert backend._effective_context_length == 65536
    assert backend._max_context_length >= 65536
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
    """A host with too little RAM for the spill plus the 8 GiB default gets the bound, and the
    flag carries the value the load-mode rule priced."""
    declined_cmd, _b, seen = _launch_with(
        tmp_path, monkeypatch, Plan(reason = "declined"), avail_mib = 12 * 1024
    )
    got = _flag(declined_cmd, "--cache-ram")
    assert got is not None and 0 <= int(got) < 8192, declined_cmd
    assert seen["inputs"]["cache_ram_default_mib"] == int(got)


def test_a_load_mode_the_plan_chose_rides_the_fit_record(tmp_path, monkeypatch):
    """The pair reaches the argv only through _fit_load_mode_flags, so every retry that strips
    the fit's load mode strips the plan's too."""
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
    """Flag off: no clamp, no floor map, nothing priced for the rungs, and the fitter's argv
    verbatim. The planner is patched to a decline so what load_model prices AROUND it is tested."""
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
    """#5692's Windows full-offload tuning emits --cache-ram 0 and #10382 skips it on a shared
    pool, and neither reaches a launch the planner owns: both key on fully_gpu_offloaded, which
    a planner launch never sets. So the argv never carries a trailing zero for last-wins."""
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
    """The planned launch crashes at startup and the revocation retry comes up healthy, mirroring
    _spawn_and_wait, where _revoke_spill_plan hands placement back to llama.cpp."""
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
    """The revocation puts the fitter's --parallel back in the argv, so the state committed after
    the retry has to follow it: nothing re-reads the slot count from the server."""
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan)

    assert len(cmds) == 2, cmds
    assert cmds[0][cmds[0].index("--parallel") + 1] == "1"
    assert cmds[1][cmds[1].index("--parallel") + 1] == "4"
    assert backend.effective_parallel_slots == 4


def test_the_planner_admits_against_ram_the_launch_has_already_spent(tmp_path, monkeypatch):
    """Plan.host_bytes is the embeddings plus the spilled weights and nothing else, yet the seam
    replaces the fit-derived load mode with the plan's. Host RAM spent outside that figure has to
    reach the planner, or --load-mode none is decided against RAM that is not free."""
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert seen["inputs"]["host_ram_unpriced_bytes"] == 0

    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, cache_ram = 20000)
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= 20000 * MIB


def test_the_batch_floor_follows_the_slots_the_plan_lowered(tmp_path, monkeypatch):
    """--batch-size is raised to max(slots, 2) when emitted and llama.cpp derives the micro-batch
    from the emitted value, so the flag must follow rung 1's reduced slot count."""
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, n_batch = 1)
    assert _flag(cmd, "--parallel") == "1"
    assert _flag(cmd, "--batch-size") == "2"
    assert backend._spill_plan_restore.get("--batch-size") == "4"

    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, n_batch = 512)
    assert _flag(cmd, "--batch-size") == "512"
    assert "--batch-size" not in backend._spill_plan_restore

    cmds, _backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan, n_batch = 1)
    assert _flag(cmds[0], "--batch-size") == "2"
    assert _flag(cmds[1], "--batch-size") == "4"


def test_a_context_only_shrink_plan_is_launched_at_the_context_it_proved(tmp_path, monkeypatch):
    """The auto path capped -c before the planner was asked at the context it wanted."""
    plan = Plan(changed = True, n_ctx = 12288)
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX
    assert _flag(cmd, "-c") == "12288"
    assert _flag(cmd, "--fit") == "off" and _flag(cmd, "-ngl") == "-1"
    assert backend._spill_plan_restore.get("-c") == "8192"
    assert backend._effective_context_length == 12288

    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, Plan(changed = True, n_ctx = NATIVE_CTX))
    assert _flag(cmd, "--fit") == "on"
    assert backend._spill_plan_flags == []


def test_a_plan_that_moves_no_weight_is_all_on_the_gpu_for_the_mlock_gate(tmp_path, monkeypatch):
    """A plan that fits by lowering the slots, moving the projector or dropping the draft emits
    -ngl -1 --fit off with every layer on the card, but the Model Memory gate was told the weights
    sit in host RAM and would page-lock a full host copy of a model that fits."""
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
    """The layout of a split GGUF is read from every shard, so a sibling shard replaced while
    shard 1 is untouched has to invalidate the cached layout."""
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
    """A knob-only plan leaves -ngl -1 --fit off on record and the next load's fits branch emits
    the same four tokens, so a retry there would write the previous model's --parallel in."""
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
    """On a roomy host the fallback carries no --cache-ram, so the plan's clamp is appended
    rather than rewritten and the revocation, which restores only prior values, left it."""
    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 1024
    )
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan)
    assert _flag(cmd, "--cache-ram") == "1024"
    assert backend._spill_plan_restore.get("--cache-ram", "missing") is None
    reverted = backend._drop_tensor_spill(list(cmd), "test")
    assert reverted != cmd and "--cache-ram" not in reverted
    assert reverted[-2:] == ["--fit", "on"]

    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, plan, avail_mib = 12 * 1024)
    before = backend._spill_plan_restore.get("--cache-ram")
    assert before is not None and _flag(cmd, "--cache-ram") == "1024"
    reverted = backend._drop_tensor_spill(list(cmd), "test")
    assert _flag(reverted, "--cache-ram") == before


def test_the_workload_prompt_is_the_whole_window_under_a_unified_cache(tmp_path, monkeypatch):
    """Studio appends --kv-unified on every multi-slot launch the build supports it on, and under
    a unified cache a single request may fill all of n_ctx, so pricing the prompt at n_ctx / slots
    under-charged the spill's prefill."""
    plan = Plan(changed = True, n_ctx = 4096, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _b, seen = _launch_with(tmp_path, monkeypatch, plan, n_ctx = 4096)
    assert seen["inputs"]["kv_unified"] is True
    assert seen["inputs"]["workload_prompt_tokens"] == 4096
    _cmd, _b, seen = _launch_with(
        tmp_path, monkeypatch, plan, n_ctx = 4096, extra_args = ["--no-kv-unified"]
    )
    assert seen["inputs"]["kv_unified"] is False
    assert seen["inputs"]["n_parallel"] == 4
    assert seen["inputs"]["workload_prompt_tokens"] == 1024


def test_a_revoked_plan_takes_its_load_mode_with_it(tmp_path, monkeypatch):
    """Once a plan is taken the fit's --load-mode is replaced by the plan's. The revocation hands
    placement back to the fitter, which can put far more on the host, so leaving "none" in the
    retry argv turned a pageable fallback into a host-RAM OOM."""
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
    """Extras are appended after every emitted flag and llama.cpp is last-wins, so a typed
    --cache-ram is what the child runs with; priced as unset, the launch charged nothing."""
    plan = Plan(
        changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,), cache_ram_mib = 1024
    )
    cmd, backend, seen = _launch_with(
        tmp_path, monkeypatch, plan, avail_mib = 12 * 1024, extra_args = ["--cache-ram", "-1"]
    )
    assert seen["inputs"]["cache_ram_user_set"] is True
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= 8192 * MIB
    assert [i for i, a in enumerate(cmd) if a == "--cache-ram"] == [len(cmd) - 2]
    assert _flag(cmd, "--cache-ram") == "-1"
    assert "--cache-ram" not in backend._spill_plan_restore


def test_the_fit_footprint_prices_the_cache_ram_the_extras_carry(tmp_path, monkeypatch):
    """The fit's own load-mode footprint charged the panel's --cache-ram or the auto bound, never
    one typed into the extras, so a ceiling above the default was under-counted."""
    from core.inference.llama_cpp import LlamaCppBackend

    seen_kw = {}
    real = LlamaCppBackend._fit_derived_load_mode
    orig = _backend

    def hooked(*a, **k):
        backend, gguf = orig(*a, **k)

        def wrapped(**kw):
            seen_kw.update(kw)
            return real(backend, **kw)

        backend._fit_derived_load_mode = wrapped
        return backend, gguf

    monkeypatch.setitem(globals(), "_backend", hooked)
    plan = Plan(changed = False, n_ctx = 8192)
    _launch_with(tmp_path, monkeypatch, plan, extra_args = ["--cache-ram", "16384"])
    assert seen_kw.get("prompt_cache_bytes") == 16384 * MIB


def test_the_flag_off_fit_footprint_carries_no_prompt_cache_term(tmp_path, monkeypatch):
    """Flag-off is byte-identical to a tree with no prompt_cache_bytes term: the term arrives
    with the clamp that pays for it, so it is charged on the planner's path and nowhere else."""
    from core.inference.llama_cpp import LlamaCppBackend

    seen_kw = {}
    real = LlamaCppBackend._fit_derived_load_mode
    orig = _backend

    def hooked(*a, **k):
        backend, gguf = orig(*a, **k)

        def wrapped(**kw):
            seen_kw.update(kw)
            return real(backend, **kw)

        backend._fit_derived_load_mode = wrapped
        return backend, gguf

    monkeypatch.setitem(globals(), "_backend", hooked)
    _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"), owns = False)
    assert seen_kw.get("prompt_cache_bytes") == 0
    assert seen_kw.get("prompt_cache_unbounded") is False

    seen_kw.clear()
    _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"), owns = True)
    assert seen_kw.get("prompt_cache_bytes") > 0, "not vacuous: the planner still charges it"


def test_the_flag_off_load_mode_matches_main_on_a_spill_the_host_holds(tmp_path, monkeypatch):
    """A 22 GiB model on the 12 GiB card is a 10 GiB spill that 18 GiB of free RAM holds, and the
    fit emits --load-mode none. Charging the default prompt cache on top pushed it back to mmap."""
    caps = {"supports_load_mode": True}
    off, _backend_off, _s = _launch_with(
        tmp_path,
        monkeypatch,
        Plan(reason = "declined"),
        owns = False,
        avail_mib = 18 * 1024,
        model_mib = 22 * 1024,
        caps = caps,
    )
    assert _flag(off, "--load-mode") == "none", off

    on, _backend_on, _s = _launch_with(
        tmp_path,
        monkeypatch,
        Plan(reason = "declined"),
        owns = True,
        avail_mib = 18 * 1024,
        model_mib = 22 * 1024,
        caps = caps,
    )
    assert "--load-mode" not in on, on


def test_the_snapshot_carries_the_cache_estimator_as_a_callable(tmp_path, monkeypatch):
    """The floor map is one measurement per slot count at ONE context and the ladder also walks
    the context, so the estimator itself is handed over; it must agree with the map."""
    _cmd, _b, seen = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    at = seen["inputs"]["kv_bytes_at"]
    assert callable(at)

    ctx = seen["inputs"]["n_ctx"]
    floors = seen["inputs"]["kv_bytes_floor_by_parallel"]
    assert floors, "not vacuous: there is a map to agree with"
    for slots, floor in floors.items():
        assert floor > 0 and at(ctx, slots) == floor

    assert at(ctx // 2, 1) > 0
    assert at(0, 0) == at(256, 1)

    _cmd, _b, off = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"), owns = False)
    assert off["inputs"]["kv_bytes_at"] is None


def test_a_caller_nkvo_cache_is_charged_once_on_the_plans_host_side(tmp_path, monkeypatch):
    """-nkvo puts the WHOLE cache in host RAM and the planner already charges it on the plan's
    host side, so the seam must not also take it off the pool: taken twice, a spill was refused."""
    _cmd, _b, plain = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    _cmd, _b, forced = _launch_with(
        tmp_path, monkeypatch, Plan(reason = "declined"), extra_args = ["-nkvo"]
    )
    assert forced["inputs"]["kv_cache_bytes"] > 0
    assert forced["inputs"]["host_ram_unpriced_bytes"] == plain["inputs"]["host_ram_unpriced_bytes"]


def test_a_projector_already_on_the_cpu_is_host_ram_the_planner_admits_against(
    tmp_path, monkeypatch
):
    """--no-mmproj-offload takes the projector out of model_size and the plan's host side never
    held it, yet clip.cpp keeps it resident in a CPU backend buffer."""
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    proj = tmp_path / "proj.gguf"
    proj.write_bytes(b"GGUF")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(proj))
    monkeypatch.setenv("LLAMA_ARG_NO_MMPROJ_OFFLOAD", "1")
    orig = _backend

    def hooked(*a, **k):
        backend, gguf = orig(*a, **k)
        backend._mmproj_vram_bytes = lambda path: 3 * 1024 * MIB if path else 0
        return backend, gguf

    monkeypatch.setitem(globals(), "_backend", hooked)
    _cmd, _b, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= 3 * 1024 * MIB


def test_every_cache_ram_spelling_the_child_accepts_is_priced():
    """-cram is the short form and llama.cpp folds --cache_ram to --cache-ram, so both select the
    bound in the child; read as unset, the launch priced the 8 GiB default for a disabled cache."""
    from core.inference.llama_cpp import _extra_args_cache_ram

    assert _extra_args_cache_ram(["-cram", "0"], {}) == 0
    assert _extra_args_cache_ram(["--cache_ram=4096"], {}) == 4096
    assert _extra_args_cache_ram(["--cache-ram", "1024", "-cram", "2048"], {}) == 2048
    assert _extra_args_cache_ram(["--cache-ram-x", "7"], {}) is None
    assert _extra_args_cache_ram(["-cram", "512"], {"LLAMA_ARG_CACHE_RAM": "8192"}) == 512


def test_a_typed_cache_ram_outranks_the_field_and_the_field_outranks_the_env(tmp_path, monkeypatch):
    """The field's flag is emitted before the extras and llama.cpp is last-wins, so a typed
    --cache-ram is what the child allocates; priced from the field, the RAM rule reserved 1 GiB
    for a cache the child grows to 16."""
    plan = Plan(changed = False, n_ctx = 8192)
    _cmd, _backend, seen = _launch_with(
        tmp_path, monkeypatch, plan, cache_ram = 1024, extra_args = ["--cache-ram", "16384"]
    )
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= 16384 * MIB
    monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "16384")
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, cache_ram = 1024)
    assert seen["inputs"]["host_ram_unpriced_bytes"] < 16384 * MIB


def test_a_pass_through_zero_context_pins_the_native_window_the_plan_is_priced_at(
    tmp_path, monkeypatch
):
    """ "-c 0" is Auto to the cap and the planner was asked FIT_ONLY at the context Auto wanted."""
    plan = Plan(changed = True, n_ctx = NATIVE_CTX, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, extra_args = ["-c", "0"])
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX
    assert seen["inputs"]["context_policy_fit_only"] is False
    assert _flag(cmd, "-c") == str(NATIVE_CTX)
    assert cmd[-2:] == ["-c", "0"]
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert seen["inputs"]["context_policy_fit_only"] is True


def test_the_micro_batch_map_follows_the_slots_rung_1_may_lower(tmp_path, monkeypatch):
    """The batch floor is max(slots, 2), so a batch of 1 launches at micro-batch 4 with four
    slots and 2 with one. The gate is handed the value for every count rung 1 may step to."""
    plan = Plan(reason = "declined")
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, n_batch = 1)
    ub = seen["inputs"]["n_ubatch_by_parallel"]
    assert sorted(ub) == [1, 2, 3, 4]
    assert ub[4] == seen["inputs"]["n_ubatch"]
    assert ub[1] < ub[4]


def test_a_cpu_pinned_drafter_is_priced_at_the_context_the_planner_is_asked_at(
    tmp_path, monkeypatch
):
    """Auto caps the fallback context and the fit priced the -ngld 0 drafter's host KV there,
    while the planner is asked at the context Auto wanted and launches at it."""
    seen_ctx = []
    orig = _backend

    def hooked(*args, **kwargs):
        backend, gguf = orig(*args, **kwargs)

        def draft_bytes(n_ctx, **_kw):
            seen_ctx.append(n_ctx)
            return n_ctx * 4096

        backend._cpu_resident_draft_bytes = draft_bytes
        return backend, gguf

    monkeypatch.setitem(globals(), "_backend", hooked)
    sidecar = tmp_path / "dflash-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    plan = Plan(reason = "declined")
    _cmd, _backend_, seen = _launch_with(
        tmp_path,
        monkeypatch,
        plan,
        caps = {"supports_dflash": True, "mtp_token": "draft-mtp", "supports_ngram_mod": True},
        speculative_type = "dflash",
        dflash_draft_path = str(sidecar),
        extra_args = ["--spec-draft-ngl", "0"],
    )
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX
    assert NATIVE_CTX in seen_ctx, seen_ctx
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= NATIVE_CTX * 4096


def test_the_auto_cache_ram_clamp_charges_the_host_only_allocations(tmp_path, monkeypatch):
    """The automatic --cache-ram was bounded by the GPU shortfall alone, so a CPU-pinned
    projector, a -ngld 0 drafter or the checkpoint snapshots left the ceiling at the default on a
    host they had already filled."""
    plan = Plan(reason = "declined")
    _cmd, _backend_, base = _launch_with(tmp_path, monkeypatch, plan, avail_mib = 26 * 1024)
    bound = base["inputs"]["cache_ram_default_mib"]
    assert 1024 < bound < 8192, bound
    orig = _backend

    def hooked(*args, **kwargs):
        backend, gguf = orig(*args, **kwargs)
        backend._mmproj_vram_bytes = lambda path: 1024 * MIB if path else 0
        return backend, gguf

    monkeypatch.setitem(globals(), "_backend", hooked)
    proj = tmp_path / "proj.gguf"
    proj.write_bytes(b"GGUF")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(proj))
    monkeypatch.setenv("LLAMA_ARG_NO_MMPROJ_OFFLOAD", "1")
    _cmd, _backend_, seen = _launch_with(tmp_path, monkeypatch, plan, avail_mib = 26 * 1024)
    assert seen["inputs"]["cache_ram_default_mib"] == bound - 1024


def test_the_single_slot_retry_keeps_one_slot_after_the_plan_is_revoked(tmp_path, monkeypatch):
    """A plan that lowered --parallel launches, the server refuses the unified cache above one
    sequence, and the retry is rewritten to one slot. Revoking the plan first keeps that rewrite,
    instead of restoring the slot count the server just refused."""
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
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    backend._planned_tensor_spill = lambda inputs, **_kw: Plan(
        changed = True, n_ctx = 8192, n_parallel = 2
    )

    real_popen = subprocess.Popen
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        cmds.append(list(cmd))
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": None,
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        if len(cmds) == 1:
            backend._stdout_lines = ["a unified KV cache is only supported with a single sequence"]
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                speculative_type = "off",
                n_parallel = 4,
            )
        )

    assert len(cmds) == 2, cmds
    assert _flag(cmds[0], "--parallel") == "2"
    assert "--kv-unified" in cmds[0]
    assert _flag(cmds[1], "--fit") == "on"
    assert _flag(cmds[1], "--parallel") == "1", cmds[1]
    assert "--kv-unified" not in cmds[1]
    assert backend.effective_parallel_slots == 1


def test_a_plan_below_an_explicit_context_is_not_emitted(tmp_path, monkeypatch):
    """An explicit context is the user's, so a plan that came back below it priced a cache the
    launch will not run at and is dropped rather than rewriting -c."""
    asked = 2 * NATIVE_CTX
    plan = Plan(changed = True, n_ctx = NATIVE_CTX, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan, n_ctx = asked)
    assert seen["inputs"]["n_ctx"] == asked
    assert seen["inputs"]["context_policy_fit_only"] is False
    assert _flag(cmd, "-c") == str(asked)
    assert _flag(cmd, "--fit") == "on" and "-ot" not in cmd
    assert "-c" not in backend._spill_plan_restore
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, plan, extra_args = ["-c", str(asked)])
    assert seen["inputs"]["n_ctx"] == asked
    assert cmd[-2:] == ["-c", str(asked)] and "-ot" not in cmd
    at = Plan(changed = True, n_ctx = asked, ot_patterns = ("x",), spilled_blocks = (1,))
    cmd, _backend, _ = _launch_with(tmp_path, monkeypatch, at, n_ctx = asked)
    assert "-ot" in cmd and _flag(cmd, "-c") == str(asked)


def test_a_priced_fit_above_the_auto_cap_restores_the_context_the_coarse_fit_gave_up(
    tmp_path, monkeypatch
):
    """Auto capped -c on the coarse fit before the planner was asked at the context Auto wanted."""
    fits = Plan(priced = True, n_ctx = NATIVE_CTX, reason = "the whole load fits in VRAM")
    cmd, backend, seen = _launch_with(tmp_path, monkeypatch, fits)
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX
    assert _flag(cmd, "-c") == str(NATIVE_CTX)
    assert _flag(cmd, "--fit") == "off" and _flag(cmd, "-ngl") == "-1" and "-ot" not in cmd
    assert backend._spill_plan_restore.get("-c") == "8192"
    assert backend._effective_context_length == NATIVE_CTX
    abstain = Plan(n_ctx = NATIVE_CTX, reason = "layout or device inventory incomplete")
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, abstain)
    assert _flag(cmd, "-c") == "8192" and _flag(cmd, "--fit") == "on"
    assert backend._spill_plan_flags == []
    cmd, backend, _ = _launch_with(tmp_path, monkeypatch, fits, n_ctx = NATIVE_CTX)
    assert _flag(cmd, "-c") == str(NATIVE_CTX) and _flag(cmd, "--fit") == "on"


def test_a_revoked_plan_restores_the_context_locals_with_the_argv(tmp_path, monkeypatch):
    """A plan that raised Auto's cap rewrote -c and rebound the locals the post-launch commit and
    the ceiling are read from."""
    plan = Plan(priced = True, changed = True, n_ctx = 12288)
    cmds, backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan, n_ctx = 0)
    assert len(cmds) == 2, cmds
    assert _flag(cmds[0], "-c") == "12288" and _flag(cmds[0], "--fit") == "off"
    assert _flag(cmds[1], "-c") == "8192" and _flag(cmds[1], "--fit") == "on"
    assert backend._effective_context_length == 8192
    assert backend._max_context_length == 8192


def test_a_revoked_one_slot_plan_restores_the_shared_cache_with_the_slots(tmp_path, monkeypatch):
    """The integrity flags emit --kv-unified only above one slot and run after the plan lowered
    them, so the revocation that restores --parallel 4 must restore the shared pool too."""
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, _backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan)
    assert len(cmds) == 2, cmds
    assert _flag(cmds[0], "--parallel") == "1" and "--kv-unified" not in cmds[0]
    assert _flag(cmds[1], "--parallel") == "4" and "--kv-unified" in cmds[1], cmds[1]


def test_a_per_model_loader_mode_keeps_the_planner_off_the_launch(tmp_path, monkeypatch):
    """The per-model Mmap pick wins over the plan's --load-mode none, so the launch does not hand
    the planner the pre-cap context or its rungs."""
    plan = Plan(reason = "declined")
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, load_mode = "mmap")
    assert seen["inputs"]["load_mode"] == "mmap"
    assert seen["inputs"]["context_policy_fit_only"] is False
    assert seen["inputs"]["n_ctx"] == 8192
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, load_mode = "none")
    assert seen["inputs"]["context_policy_fit_only"] is True
    assert seen["inputs"]["n_ctx"] == NATIVE_CTX


def test_an_unbounded_prompt_cache_abstains_from_the_fits_own_none_and_reaches_the_planner(
    tmp_path, monkeypatch
):
    """--cache-ram -1 was charged as the 8 GiB default in the fit's footprint, so a
    fit proved --load-mode none for a cache that llama.cpp bounds by nothing."""
    plan = Plan(reason = "declined")
    caps = {"supports_load_mode": True}
    cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, caps = caps)
    assert _flag(cmd, "--load-mode") == "none", cmd
    assert seen["inputs"]["cache_ram_unbounded"] is False
    cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan, caps = caps, cache_ram = -1)
    assert "--load-mode" not in cmd, cmd
    assert seen["inputs"]["cache_ram_unbounded"] is True


def test_the_snapshot_carries_the_windowed_half_of_the_cache(tmp_path, monkeypatch):
    """The planner reads a saturated window in full but the full-context layers only over their
    live prefix, so the snapshot has to price the two halves apart."""
    monkeypatch.setitem(DENSE, "_sliding_window", 1024)
    monkeypatch.setitem(DENSE, "_sliding_window_pattern", tuple(i % 6 != 5 for i in range(64)))
    _cmd, backend, seen = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    inputs = seen["inputs"]
    swa = inputs["kv_swa_bytes"]
    assert 0 < swa < inputs["kv_cache_bytes"], (swa, inputs["kv_cache_bytes"])

    parts = backend._estimate_kv_cache_parts(
        inputs["n_ctx"],
        inputs["cache_type_kv"],
        n_parallel = inputs["n_parallel"],
        kv_unified = inputs["kv_unified"],
        n_ubatch = inputs["n_ubatch"],
        ctx_checkpoints = 0,
    )
    # The sum pins the ARGUMENTS: a split priced at another geometry would not add back up.
    assert sum(parts) == inputs["kv_cache_bytes"]
    assert parts[1] == swa


def test_the_snapshot_carries_the_windowed_half_as_a_callable(tmp_path, monkeypatch):
    """A window is stored per stream, so rung 1 shrinks it with the slots; the scalar half is the
    launched count's and goes stale, reading most of a smaller cache as fully live."""
    monkeypatch.setitem(DENSE, "_sliding_window", 1024)
    monkeypatch.setitem(DENSE, "_sliding_window_pattern", tuple(i % 6 != 5 for i in range(64)))
    _cmd, backend, seen = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    inputs = seen["inputs"]
    at = inputs["kv_swa_bytes_at"]
    assert callable(at)
    # Agrees with the scalar where the two overlap, and moves with the slots where it does not.
    assert at(inputs["n_ctx"], inputs["n_parallel"]) == inputs["kv_swa_bytes"]
    one = at(inputs["n_ctx"], 1)
    assert 0 < one < inputs["kv_swa_bytes"]
    assert (
        one
        == backend._estimate_kv_cache_parts(
            inputs["n_ctx"],
            inputs["cache_type_kv"],
            n_parallel = 1,
            kv_unified = inputs["kv_unified"],
            n_ubatch = inputs["n_ubatch"],
            ctx_checkpoints = 0,
        )[1]
    )

    _cmd, _b, off = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"), owns = False)
    assert off["inputs"]["kv_swa_bytes_at"] is None


def test_a_rocm_host_leaves_the_link_rate_unset(tmp_path, monkeypatch):
    """nvidia-smi on a mixed host answers for the NVIDIA cards, whose indices share nothing with
    the ROCm ids the plan credits, so the rate must stay unset there and the cost model's PCIe 5
    default stands. The same launch on a CUDA host does read the link."""
    from core.inference.llama_cpp import LlamaCppBackend

    probes = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_nvidia_link_query",
        staticmethod(lambda: probes.append(1) or "0, 5, 16\n"),
    )
    monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: True))
    _cmd, _b, seen = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    assert seen["inputs"]["link_gib_s"] is None
    assert probes == []

    monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: False))
    _cmd, _b, cuda = _launch_with(tmp_path, monkeypatch, Plan(reason = "declined"))
    assert cuda["inputs"]["link_gib_s"] is not None
    assert probes == [1]


def test_a_user_split_across_a_multi_device_plan_pins_the_child_order(tmp_path, monkeypatch):
    """The plan's per-device rows were modelled against the physical-order device list, and a
    ratio the user typed reaches the child in place of the plan's own split, so the child's
    enumeration is pinned the way it is for the plan's split or for a manual ratio."""
    from core.inference.llama_cpp import LlamaCppBackend

    pins = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_pin_visible_gpu_order_for_split",
        staticmethod(lambda env: pins.append(dict(env)) or None),
    )
    plan = Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (30, 18),
    )
    cmd, _b, _s = _launch_with(tmp_path, monkeypatch, plan, extra_args = ["--tensor-split", "1,1"])
    assert cmd.count("--tensor-split") == 1, "the user's ratio reaches the child, not the plan's"
    assert len(pins) == 1

    one_device = Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (48,),
    )
    _launch_with(tmp_path, monkeypatch, one_device, extra_args = ["--tensor-split", "1,1"])
    assert len(pins) == 1, "a single-device plan has no rows to pin"


def test_a_plan_that_moves_no_weight_is_watched_for_sysmem_fallback(tmp_path, monkeypatch):
    """On Windows CUDA a knob-only plan launches -ngl -1 --fit off with every layer on the card,
    exactly the placement WDDM pages silently, but the watch was keyed on the coarse fit's flag
    alone. A plan with -ot spills stays unwatched: slow with VRAM full is what a spill is."""
    from core.inference.llama_cpp import LlamaCppBackend

    seen = []

    def spy(self, **kw):
        seen.append(bool(kw.get("fully_gpu_offloaded")))
        return None

    monkeypatch.setattr(LlamaCppBackend, "_windows_sysmem_fallback_watch", spy)
    caps = {"supports_metrics": True}
    _launch_with(tmp_path, monkeypatch, Plan(changed = True, n_ctx = 8192, n_parallel = 1), caps = caps)
    assert seen == [True], seen
    seen.clear()
    _launch_with(
        tmp_path,
        monkeypatch,
        Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,)),
        caps = caps,
    )
    assert seen == [False], seen


def test_a_launch_recovered_under_fit_on_is_not_watched_for_sysmem_fallback(tmp_path, monkeypatch):
    """A forced full-offload launch that crashed and came back under --fit on is the fitter's
    placement, which may hold layers in host RAM, so watching it would read a legitimately slow
    partial placement as the driver paging."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    seen = []

    def spy(self, **kw):
        seen.append(bool(kw.get("fully_gpu_offloaded")))
        return None

    monkeypatch.setattr(LlamaCppBackend, "_windows_sysmem_fallback_watch", spy)
    monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)

    def run(returncodes):
        # A card the 1 KiB stub fits on outright, so the launch is a proved full offload.
        backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])
        backend.probe_server_capabilities = lambda _binary = None: {"supports_metrics": True}
        launches: list = []

        class _Process:
            pid = 123
            stdout = ()

            def __init__(self, returncode):
                self.returncode = returncode

            def poll(self):
                return self.returncode

            def terminate(self):
                return None

            def wait(self, timeout = None):
                return self.returncode

            def kill(self):
                return None

        real_popen = subprocess.Popen

        def popen(cmd, **kwargs):
            # Only the server is faked; anything else the launch runs is real.
            if not cmd or str(cmd[0]) != "/fake/llama-server":
                return real_popen(cmd, **kwargs)
            launches.append(list(cmd))
            return _Process(returncodes[len(launches) - 1])

        backend._wait_for_health = lambda timeout, **_kw: returncodes[len(launches) - 1] is None
        seen.clear()
        with patch.object(subprocess, "Popen", side_effect = popen):
            assert backend.load_model(GgufLoadIntent(gguf_path = str(gguf), model_identifier = "t"))
        return launches

    clean = run([None])
    assert clean[0][clean[0].index("--fit") + 1] == "off"
    assert seen == [True], seen

    recovered = run([1, None])
    assert len(recovered) == 2
    assert recovered[1][recovered[1].index("--fit") + 1] == "on"
    assert seen == [False], seen
