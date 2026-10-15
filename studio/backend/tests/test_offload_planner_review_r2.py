# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lazy-mode spelling, the fitter's comparison budget, vetoes on the ladder, Vulkan lazy-mode
emission and a reordered GPU mask under a split plan."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference import llama_cpp as llama_mod  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_cost_model import HostProfile  # noqa: E402
from core.inference.offload_planner import ContextPolicy, Plan, plan_placement  # noqa: E402
from dataclasses import replace  # noqa: E402
from test_offload_cost_gate import GIB, MIB, dense_layout, gated, moe_layout  # noqa: E402

PLE = 8 * GIB


# ------------------------------------------------ findings 1 / 5: the build's lazy-mode spelling


def test_a_legacy_lazy_off_keeps_the_table_resident_in_the_planners_price():
    """A build that predates --lazy-mode reads only --tensor-read-lazy (and its env twin). Its
    explicit "off" holds the table resident, and the planner discounted it as paged anyway."""
    read = llama_mod._per_layer_embd_read_lazily
    legacy = dict(supports_lazy_mode = True, lazy_mode_flag = "--tensor-read-lazy")
    assert read("gemma4", PLE, extra_args = ["--tensor-read-lazy", "off"], env = {}, **legacy) is False
    assert read("gemma4", PLE, env = {"LLAMA_ARG_TENSOR_READ_LAZY": "off"}, **legacy) is False
    # Unset is llama.cpp's auto in either spelling, which pages a table this size.
    assert read("gemma4", PLE, env = {}, **legacy) is True
    # The current spelling is unchanged.
    assert read("gemma4", PLE, supports_lazy_mode = True, extra_args = ["-lzm", "off"], env = {}) is False


def test_the_seam_hands_the_probed_spelling_to_the_planner():
    load = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    assert '"lazy_mode_flag":server_caps.get("lazy_mode_flag")or"--lazy-mode",' in load
    seam = "".join(inspect.getsource(LlamaCppBackend._planned_tensor_spill).split())
    assert 'lazy_mode_flag=str(inputs.get("lazy_mode_flag")or"--lazy-mode"),' in seam


# ------------------------------------------------ finding 4: what the fitter keeps on the card


def test_the_fitter_arm_keeps_the_drafter_the_spill_dropped():
    """A spill that also dropped the drafter was ranked against a fitter given the drafter's
    VRAM, which the fitter keeps; the cheaper fitter then declined a spill that wins."""
    layout = dense_layout()
    card = [10128 * MIB]
    host = dict(host = HostProfile(threads = 6))
    kept = plan_placement(layout, card, 94 * GIB, 16384, opts = gated(**host))
    dropped = plan_placement(
        layout,
        card,
        94 * GIB,
        16384,
        opts = gated(**host, draft_bytes = 2 * GIB, draft_droppable = True),
    )
    assert kept.spills_anything, kept.reason
    assert dropped.draft_dropped and dropped.spills_anything, dropped.reason
    assert dropped.predicted_fit_request_ms > kept.predicted_fit_request_ms


# ------------------------------------------------ finding 11: a veto further down the ladder


def test_a_veto_further_down_the_ladder_still_ends_the_walk():
    """The first attempt is refused for host RAM, not vetoed, so FIT_ONLY walks down; every rung
    at or above 32K per slot is then the MoE long-prompt veto, and skipping those as ordinary
    declines accepted the same -ot spill just below it."""
    layout = replace(
        moe_layout(n_blocks = 24), kv_bytes_per_token_f16 = 4 * GIB / 32768, n_ctx_train = 131072
    )
    opts = gated(
        host = HostProfile(threads = 6),
        context_policy = ContextPolicy.FIT_ONLY,
        min_penalty_reduction = 0.0,
    )
    got = plan_placement(layout, [13 * GIB], 12 * GIB, 65536, opts = opts)
    assert got.veto and got.declined_by_gate and not got.spills_anything, got.reason
    assert got.n_ctx == 65536, got.reason
    assert "tokens per slot" in got.reason and "--fit on" in got.reason, got.reason
    # Without the veto the ladder may still settle lower: the walk itself is unchanged.
    walked = plan_placement(
        layout, [13 * GIB], 12 * GIB, 65536, opts = replace(opts, moe_long_prompt_ctx = 0)
    )
    assert walked.spills_anything and walked.n_ctx < 65536, walked.reason


# ------------------------------------------------ findings 9 / 10: Vulkan lazy-mode emission


def _lazy_backend(monkeypatch, rows):
    b = LlamaCppBackend()
    b._architecture = "gemma4"
    monkeypatch.setattr(
        LlamaCppBackend, "_run_vulkan_probe", staticmethod(lambda _binary = None: rows)
    )
    b._tensor_spill_layout = lambda *a, **k: type("L", (), {"per_layer_embd_bytes": PLE})()
    return b


def _mode(b, gpu_indices):
    return b._studio_lazy_mode(
        model_path = "/m.gguf",
        server_caps = {"supports_lazy_mode": True, "lazy_mode_flag": "--lazy-mode"},
        gpu_indices = gpu_indices,
        binary = "/fake/llama-server",
        is_vulkan_backend = True,
        extra_args = [],
        env = {},
    )


def test_a_default_vulkan_selection_is_left_to_auto(monkeypatch):
    """With no selection the child's default device list drops an iGPU whenever a discrete GPU is
    present, so a visible iGPU says nothing about the devices the child loads on."""
    rows = [
        {"index": 0, "is_igpu": False, "type_known": True},
        {"index": 1, "is_igpu": True, "type_known": True},
    ]
    b = _lazy_backend(monkeypatch, rows)
    assert _mode(b, None) is None
    # An explicit selection still decides.
    assert _mode(b, [0]) == "on"
    assert _mode(b, [0, 1]) == "off"


def test_an_unread_vulkan_device_type_is_not_taken_as_mmap_capable(monkeypatch):
    """The probe reports is_igpu False when the type query fails, so "not integrated" and "not
    read" share a value; "on" against what is really an iGPU is a mode the child refuses."""
    rows = [{"index": 0, "is_igpu": False, "type_known": False}]
    b = _lazy_backend(monkeypatch, rows)
    assert _mode(b, [0]) == "off"


# ------------------------------------------------ findings 6 / 7: a reordered mask under a split


def _two_card_launch(tmp_path, monkeypatch, plan):
    """A 30 GiB model on two 12 GiB cards, both selected, with the planner's answer fixed."""
    from test_llama_cpp_placement import _backend, _launch
    from test_offload_planner_launch import DENSE

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    card = 12 * 1024
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, card, card), (1, card, card)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 30 * GIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    # Both cards selected and the fit unproved: the planner's branch, with a device list.
    backend._select_gpus = lambda *a, **k: ([0, 1], True)
    backend._select_gpus_split_aware = lambda *a, **k: ([0, 1], True)
    asked = []
    backend._planned_tensor_spill = lambda inputs, **_kw: asked.append(inputs) or plan
    launched = _launch(backend, gguf, speculative_type = "off", n_ctx = 8192, n_parallel = 1)
    assert asked, "the planner was never consulted"
    return launched["cmd"], backend


def test_a_reordered_parent_mask_declines_a_split_plan(tmp_path, monkeypatch):
    """The plan models device 0 as the lowest physical id. CUDA_VISIBLE_DEVICES=1,0 is honoured
    for the child, which repointed the plan's --tensor-split onto the other card (its -ot blocks
    stayed put), and left a rewritten split the retry's revocation could not find."""
    plan = Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (30, 18),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    cmd, backend = _two_card_launch(tmp_path, monkeypatch, plan)
    assert cmd[cmd.index("--tensor-split") + 1] == "30,18", "the control must emit the split"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    cmd, backend = _two_card_launch(tmp_path, monkeypatch, plan)
    assert "--tensor-split" not in cmd, cmd
    assert "-ot" not in cmd, cmd
    assert not backend._spill_plan_flags
