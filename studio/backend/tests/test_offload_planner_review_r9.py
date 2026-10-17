# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dense fitter arm's whole-layer placement, and the spill plan both CPU replays revoke."""

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

from core.inference.offload_planner import (  # noqa: E402
    BlockLayout,
    HostProfile,
    Plan,
    _fit_fallback_placement,
    all_resident_bytes,
    plan_placement,
)
from test_offload_cost_gate import (  # noqa: E402
    GIB,
    gated,
    graded_dense_layout,
    per_matrix_moe_layout,
)

MIB = 1024 * 1024


# ------------------------------------------------ finding 1: dense --fit on moves whole layers


def _ungraded(layout):
    return replace(
        layout,
        blocks = tuple(
            BlockLayout(
                index = b.index, spillable_bytes = b.spillable_bytes, resident_bytes = b.resident_bytes
            )
            for b in layout.blocks
        ),
    )


_ARGS = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)


@pytest.mark.parametrize("deficit_mib", [1, 50, 400, 2048])
def test_the_dense_fitter_never_overflows_part_of_a_layer(deficit_mib):
    """common/fit.cpp returns before its partial-layer step when hp_nex == 0 (fit.cpp:730-733 at
    b11541, the build Studio ships), so a dense --fit on lowers n_gpu_layers and nothing else: a
    graded and an ungraded copy of one model get one placement, a whole number of layers."""
    graded = graded_dense_layout()
    resident = all_resident_bytes(
        graded, 8192, kv_quantised = False, kv_bytes_floor = 0, kv_on_host = False, n_seq = 1
    )
    budget = resident - deficit_mib * MIB
    fine = _fit_fallback_placement(graded, gated(), budget, 8192, **_ARGS)
    coarse = _fit_fallback_placement(_ungraded(graded), gated(), budget, 8192, **_ARGS)
    assert fine is not None and coarse is not None
    fine_bytes = sum(g.bytes_total for g in fine.host_groups)
    coarse_bytes = sum(g.bytes_total for g in coarse.host_groups)
    assert fine_bytes == coarse_bytes, (fine_bytes, coarse_bytes)
    assert fine.kv_host_bytes == coarse.kv_host_bytes > 0, "a moved layer takes its cache"
    per_block = graded.blocks[0].spillable_bytes + graded.blocks[0].resident_bytes
    assert fine_bytes % per_block == 0, fine_bytes


@pytest.mark.parametrize("over_mib", [1350, 1475, 1525])
def test_the_near_tie_dense_spill_is_ranked_against_whole_layers(over_mib):
    """A card a few hundred MiB short of the 64-layer graded dense fixture: priced against a
    fitter that overflowed one ffn_down, the planner's own -ot arm tied (861 ms against 861 ms)
    and was declined, though llama.cpp's real fallback moves a whole layer and its cache to the
    CPU (3296 ms)."""
    graded = graded_dense_layout()
    opts = gated(host = HostProfile(threads = 6))
    resident = all_resident_bytes(
        graded, 8192, kv_quantised = False, kv_bytes_floor = 0, kv_on_host = False, n_seq = 1
    )
    plan = plan_placement(graded, [resident + over_mib * MIB], 94 * GIB, 8192, opts = opts)
    assert plan.spills_anything and not plan.declined_by_gate, plan.reason
    assert plan.predicted_request_ms < 0.9 * plan.predicted_fit_request_ms


def test_a_dense_load_that_fits_is_still_not_spilled():
    """The control: with room to spare there is nothing to rank, and nothing moves."""
    graded = graded_dense_layout()
    resident = all_resident_bytes(
        graded, 8192, kv_quantised = False, kv_bytes_floor = 0, kv_on_host = False, n_seq = 1
    )
    plan = plan_placement(
        graded, [resident + 8 * GIB], 94 * GIB, 8192, opts = gated(host = HostProfile(threads = 6))
    )
    assert not plan.spills_anything


def test_the_moe_fitter_still_grades_its_boundary_block():
    """The control: step 4 runs for an MoE, so its boundary block still gives up only a rung."""
    layout = per_matrix_moe_layout()
    down, whole = layout.blocks[0].ffn_down_bytes, layout.blocks[0].spillable_bytes
    resident = all_resident_bytes(
        layout, 8192, kv_quantised = False, kv_bytes_floor = 0, kv_on_host = False, n_seq = 1
    )
    placement = _fit_fallback_placement(
        layout, gated(), resident - 3 * whole - down // 2, 8192, **_ARGS
    )
    assert placement is not None
    assert sum(g.bytes_total for g in placement.host_groups) == 3 * whole + down


# ------------------------------------------------ findings 2 / 3: both CPU replays revoke the plan


def _planned_vulkan_backend(tmp_path, monkeypatch, plan):
    from test_llama_cpp_placement import _vulkan_cpu_replay_backend
    from test_offload_planner_launch import DENSE

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 30.0, free_mib = 12 * 1024, avail_mib = 128 * 1024
    )

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "found": True,
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
    }
    backend._planner_may_run = lambda *a, **k: True
    backend._planned_tensor_spill = lambda inputs, *, extra_args = None, env = None: plan
    return backend, gguf


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


_PLANS = {
    "context": Plan(changed = True, n_ctx = 29184, ot_patterns = ("x",), spilled_blocks = (1,)),
    "slots": Plan(changed = True, n_ctx = 8192, n_parallel = 1),
}


def _launch_crashing(backend, gguf, crash_text, **load_kwargs):
    """Every launch that can still reach a GPU dies of SIGSEGV printing ``crash_text``; only
    the device-less replay comes up. ``crash_text=None`` is the direct replay: nothing crashes."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent
    from test_llama_cpp_placement import _REAL_POPEN

    cmds = []

    def _cpu(cmd):
        return "--device" in cmd and cmd[cmd.index("--device") + 1] == "none"

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        rc = -11 if (crash_text is not None and not _cpu(list(cmd))) else None
        cmds.append(list(cmd))
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": rc,
                "poll": lambda self: rc,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: rc,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        if crash_text is not None and not _cpu(cmds[-1] if cmds else []):
            backend._stdout_lines = [crash_text]
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **load_kwargs)
        )
    return cmds


_OOM = "ggml_vulkan: Device memory allocation failed"
# A capability crash: _spawn_and_wait's own --fit on retry does not take it, so the argv reaching
# the rungs below is the plan's.
_SPLIT_AXIS = "GGML_ASSERT(axis == GGML_BACKEND_SPLIT_AXIS_0) failed"

_PATHS = {
    # (crash text, flash-attn rung available)
    "direct": (None, True),
    "reactive-oom": (_OOM, True),
    "reactive-noflash": (_SPLIT_AXIS, True),
    "reactive-flash-already-off": (_SPLIT_AXIS, False),
}


@pytest.mark.parametrize("which", sorted(_PLANS))
@pytest.mark.parametrize("path", sorted(_PATHS))
def test_a_cpu_replay_does_not_inherit_a_plan_priced_for_gpu_memory(
    tmp_path, monkeypatch, which, path
):
    """The plan raised -c to 29184 (or cut the slots to 1) because a GPU could hold it. The direct
    replay and the crash replay converted the argv before revoking, and the no-flash rung flipped
    the plan's --fit off first, so the contiguous match failed every time and the CPU child kept
    the GPU-priced context, -ot and slot count."""
    crash_text, flash_rung = _PATHS[path]
    backend, gguf = _planned_vulkan_backend(tmp_path, monkeypatch, _PLANS[which])
    if not flash_rung:
        backend._with_flash_attn_off = lambda *_a, **_kw: None
    kwargs = dict(n_ctx = 0, n_parallel = 4)
    if crash_text is None:
        kwargs["cpu_fallback"] = True
    cmds = _launch_crashing(backend, gguf, crash_text, **kwargs)
    if crash_text is not None:
        # The fixture really planned: the first launch carries the plan.
        planned = cmds[0]
        assert _flag(planned, "--fit") == "off" and "-ngl" in planned, planned
        assert len(cmds) > 1 and backend._cpu_fallback_reason == "vulkan_startup_crash"
    replay = cmds[-1]
    assert _flag(replay, "--device") == "none", replay
    assert "-ot" not in replay and "-ngl" not in replay, replay
    assert _flag(replay, "--parallel") == "4", replay
    assert _flag(replay, "-c") == "8192", replay
    assert backend._effective_context_length == 8192
    assert replay.count("--fit") == 1 and _flag(replay, "--fit") == "off", replay


def test_a_cpu_replay_without_a_plan_is_unchanged(tmp_path, monkeypatch):
    """The control: a declined plan leaves nothing to revoke, and the replay is what it was."""
    backend, gguf = _planned_vulkan_backend(tmp_path, monkeypatch, Plan(reason = "declined"))
    cmds = _launch_crashing(backend, gguf, None, n_ctx = 0, n_parallel = 4, cpu_fallback = True)
    replay = cmds[-1]
    assert _flag(replay, "--device") == "none" and _flag(replay, "--parallel") == "4", replay
    assert "-ot" not in replay and backend._effective_context_length == 8192


# ------------------------------------------------ finding 4: an inherited order nothing verified


def _split_plan() -> Plan:
    return Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (30, 18),
    )


@pytest.mark.parametrize(
    "order, mask, picked, keeps_split",
    [
        # A user's FASTEST_FIRST under an inherited mask: the child's device 0 is the fastest
        # card, the plan's row 0 the lowest PCI index.
        ("FASTEST_FIRST", "0,1", None, False),
        # Controls. Studio's default: PCI_BUS_ID, set at import.
        ("PCI_BUS_ID", "0,1", None, True),
        # No mask: the launch emits its own and pins PCI_BUS_ID for the child.
        ("FASTEST_FIRST", None, None, True),
        # A picked set pins PCI_BUS_ID whatever was inherited.
        ("FASTEST_FIRST", "0,1", (0, 1), True),
    ],
)
def test_an_inherited_cuda_order_nothing_verified_declines_a_split_plan(
    tmp_path, monkeypatch, order, mask, picked, keeps_split
):
    from core.inference import llama_cpp as llama_mod
    from test_offload_planner_review_r2 import _two_card_launch

    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        llama_mod.LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: False)
    )
    if picked is not None:
        original = llama_mod.LlamaCppBackend.load_model

        def load(self, intent, *a, **k):
            return original(self, replace(intent, gpu_ids = picked), *a, **k)

        monkeypatch.setattr(llama_mod.LlamaCppBackend, "load_model", load)
    cmd, backend = _two_card_launch(tmp_path, monkeypatch, _split_plan())
    if keeps_split:
        assert cmd[cmd.index("--tensor-split") + 1] == "30,18", cmd
    else:
        assert "--tensor-split" not in cmd and "-ot" not in cmd, cmd
        assert not backend._spill_plan_flags
