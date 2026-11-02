# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto split mode (layer vs tensor) and the MTP draft depth per split mode (core/inference/split_policy.py).

Pure decision-table cells first, then the wiring through the real LlamaCppBackend.load_model against faked GPU
inventories (no GPU, no network), then the depth through _build_speculative_flags.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

# First: the real loggers package must not be shadowed by the bare stub (routes/inference.py needs it).
from test_tp_vision_regression import _load_inference_routes_module  # noqa: E402
from test_llama_cpp_mtp_detection import _MTP_MODEL, _flags_dict, _resolver_backend  # noqa: E402

# Above pid_max, so a stray os.kill / killpg fails with ESRCH.
_FAKE_SERVER_PID = 2**31 - 1
from test_llama_cpp_placement import _REAL_POPEN, _backend, _launch  # noqa: E402

from core.inference import split_policy as sp  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

GIB = 1024**3
T4 = "Tesla T4"
NO_ENV: dict = {}


def _gpus(*specs):
    """specs: (name, total_mib, free_mib) per GPU, indexed 0.., measured idle."""
    return [
        sp.GpuInfo(index = i, name = n, total_mib = t, free_mib = f, util_pct = 0.0)
        for i, (n, t, f) in enumerate(specs)
    ]


TWO_T4 = _gpus((T4, 15_360, 15_000), (T4, 15_360, 15_000))


def _decide(**kw):
    args = dict(
        is_moe = False,
        gpus = TWO_T4,
        fits_one_gpu = False,
        binary_supports_tensor = True,
        env = NO_ENV,
    )
    args.update(kw)
    return sp.choose_split_mode(**args)


# 1. Decision table


def test_dense_on_two_matching_gpus_is_tensor():
    d = _decide()
    assert (d.mode, d.auto) == ("tensor", True)


def test_moe_without_draft_stays_layer():
    d = _decide(is_moe = True)
    assert d.mode == "layer"
    assert "MoE" in d.reason


def test_moe_resident_with_draft_is_tensor():
    d = _decide(is_moe = True, draft_active = True, fits_on = lambda idx: True)
    assert (d.mode, d.auto) == ("tensor", True)
    assert d.gpu_indices == (0, 1)


def test_moe_with_draft_but_offloaded_stays_layer():
    d = _decide(is_moe = True, draft_active = True, fits_on = lambda idx: False)
    assert d.mode == "layer"


def test_moe_rule_switch(monkeypatch):
    monkeypatch.setattr(sp, "SPLIT_POLICY", sp.SplitPolicyThresholds(moe_resident_with_mtp = False))
    d = sp.choose_split_mode(
        is_moe = True,
        gpus = TWO_T4,
        fits_one_gpu = False,
        binary_supports_tensor = True,
        env = NO_ENV,
        draft_active = True,
        thresholds = sp.SPLIT_POLICY,
    )
    assert d.mode == "layer"


@pytest.mark.parametrize(
    "gpus",
    [
        _gpus((T4, 15_360, 15_000), ("NVIDIA L4", 23_034, 22_000)),
        _gpus(
            ("NVIDIA GeForce RTX 3090", 24_576, 24_000), ("NVIDIA GeForce RTX 3090", 12_288, 12_000)
        ),
        _gpus((T4, 15_360, 15_000), (None, 15_360, 15_000)),
        _gpus((T4, 15_360, 15_000), (T4, 0, 15_000)),
    ],
    ids = ["different-model", "different-total", "unknown-name", "unknown-total"],
)
def test_mismatched_or_unknown_gpus_stay_layer(gpus):
    d = _decide(gpus = gpus)
    assert (d.mode, d.auto) == ("layer", True)


def test_totals_within_two_percent_match():
    d = _decide(gpus = _gpus((T4, 15_360, 15_000), (T4, 15_100, 14_800)))
    assert d.mode == "tensor"


def test_fits_one_gpu_dense_is_widened_onto_the_idle_pair():
    assert sp.TP_WIDEN_FITS_ONE_GPU is True
    d = _decide(fits_one_gpu = True)
    assert (d.mode, d.auto, d.gpu_indices) == ("tensor", True, (0, 1))
    assert "widened" in d.reason


def test_fits_one_gpu_not_widened_when_switched_off_or_moe_or_busy():
    assert (
        _decide(fits_one_gpu = True, widen_fits_one_gpu = False).mode,
        _decide(fits_one_gpu = True, widen_fits_one_gpu = False).auto,
    ) == ("none", False)
    assert _decide(fits_one_gpu = True, is_moe = True, draft_active = True).mode == "none"
    busy = _gpus((T4, 15_360, 15_000), (T4, 15_360, 6_000))
    assert _decide(fits_one_gpu = True, gpus = busy).mode == "none"


def test_more_than_two_gpus_use_the_best_matching_idle_pair():
    gpus = _gpus(
        (T4, 15_360, 9_000),
        ("NVIDIA L4", 23_034, 23_000),
        ("NVIDIA L4", 23_034, 22_900),
        (T4, 15_360, 15_000),
    )
    d = _decide(gpus = gpus, fits_on = lambda idx: True)
    assert (d.mode, d.gpu_indices) == ("tensor", (1, 2))


def test_needs_more_than_the_pair_stays_layer():
    gpus = _gpus(*[(T4, 15_360, 15_000)] * 4)
    d = _decide(gpus = gpus, fits_on = lambda idx: False)
    assert d.mode == "layer" and "more than 2 GPUs" in d.reason


def test_one_gpu_never_splits():
    d = _decide(gpus = TWO_T4[:1])
    assert (d.mode, d.auto) == ("none", False)


def test_user_toggle_wins():
    d = _decide(user_tensor_parallel = True, is_moe = True)
    assert (d.mode, d.auto) == ("tensor", False)


@pytest.mark.parametrize("override", ["layer", "row", "none", "tensor"])
def test_user_extras_split_mode_wins(override):
    d = _decide(split_mode_override = override)
    assert (d.mode, d.auto) == (override, False)


def test_user_tensor_split_ratio_keeps_layer():
    d = _decide(user_tensor_split = True)
    assert (d.mode, d.auto) == ("layer", False)


@pytest.mark.parametrize(
    "env, mode",
    [({"LLAMA_ARG_SPLIT_MODE": "layer"}, "layer"), ({"LLAMA_ARG_TENSOR_SPLIT": "1,1"}, "layer")],
)
def test_inherited_env_wins(env, mode):
    d = _decide(env = env)
    assert (d.mode, d.auto) == (mode, False)


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_kill_switch_keeps_todays_behaviour(value):
    d = _decide(env = {sp.AUTO_TENSOR_SPLIT_ENV: value})
    assert (d.mode, d.auto) == ("layer", False)


def test_default_constant_controls_policy(monkeypatch):
    monkeypatch.setattr(sp, "AUTO_TENSOR_SPLIT_DEFAULT", False)
    assert _decide().mode == "layer"
    assert _decide(env = {sp.AUTO_TENSOR_SPLIT_ENV: "1"}).mode == "tensor"


def test_binary_without_tensor_split_stays_layer():
    d = _decide(binary_supports_tensor = False)
    assert (d.mode, d.auto) == ("layer", True)


def test_p2p_veto_does_not_block_tensor_but_is_recorded():
    d = _decide(p2p_veto = "no NVLink-capable part in the selection")
    assert d.mode == "tensor"
    assert "no NVLink-capable part" in d.reason and "host memory" in d.reason


def test_busy_or_shared_gpus_stay_layer():
    busy = _gpus((T4, 15_360, 15_000), (T4, 15_360, 6_000))
    assert _decide(gpus = busy).mode == "layer"
    assert _decide(shared_gpus = frozenset({1})).mode == "layer"


def test_draft_active_is_reported():
    d = _decide(draft_active = True)
    assert d.mode == "tensor" and f"MTP depth {sp.MTP_DRAFT_DEPTH_BY_SPLIT['tensor']}" in d.reason


@pytest.mark.parametrize(
    "caps, ok",
    [
        ({"flags": {"--split-mode": "how to split the model: none, layer, row, tensor"}}, True),
        ({"flags": {"--split-mode": "how to split the model: none, layer, row"}}, False),
        ({"flags": {}}, False),
        ({}, False),
        (None, False),
    ],
)
def test_binary_capability_probe(caps, ok):
    assert sp.binary_supports_tensor_split(caps) is ok


# 2. MTP depth table


def test_depth_table():
    assert sp.mtp_draft_depth_for_split("tensor", 2) == sp.MTP_DRAFT_DEPTH_BY_SPLIT["tensor"] == 2
    assert sp.mtp_draft_depth_for_split("layer", 2) == 2
    # "none" / unknown / None keep the caller's platform default (2 GPU, 3 CPU).
    assert sp.mtp_draft_depth_for_split("none", 3) == 3
    assert sp.mtp_draft_depth_for_split(None, 2) == 2
    assert sp.mtp_draft_depth_for_split("row", 2) == 2


def _mtp_flags(
    monkeypatch,
    *,
    split_mode,
    spec_draft_n_max = None,
    gpus = True,
):
    backend = _resolver_backend(monkeypatch)
    backend._nextn_predict_layers = 1
    return _flags_dict(
        backend._build_speculative_flags(
            speculative_type = "mtp",
            spec_draft_n_max = spec_draft_n_max,
            extra_args = None,
            model_identifier = _MTP_MODEL,
            model_path = None,
            gpus = gpus,
            binary = "/fake/llama-server",
            split_mode = split_mode,
        )
    )


def test_tensor_split_picks_its_depth(monkeypatch):
    assert _mtp_flags(monkeypatch, split_mode = "tensor")["--spec-draft-n-max"] == str(
        sp.MTP_DRAFT_DEPTH_BY_SPLIT["tensor"]
    )
    monkeypatch.setitem(sp.MTP_DRAFT_DEPTH_BY_SPLIT, "tensor", 4)
    assert _mtp_flags(monkeypatch, split_mode = "tensor")["--spec-draft-n-max"] == "4"


def test_layer_and_single_gpu_keep_todays_depth(monkeypatch):
    assert _mtp_flags(monkeypatch, split_mode = None)["--spec-draft-n-max"] == "2"
    assert _mtp_flags(monkeypatch, split_mode = "layer")["--spec-draft-n-max"] == "2"


def test_cpu_keeps_todays_depth(monkeypatch):
    assert _mtp_flags(monkeypatch, split_mode = "tensor", gpus = False)["--spec-draft-n-max"] == "3"


def test_user_depth_wins_over_the_table(monkeypatch):
    assert (
        _mtp_flags(monkeypatch, split_mode = "tensor", spec_draft_n_max = 3)["--spec-draft-n-max"]
        == "3"
    )


# 3. Wiring through load_model

_TP_HELP = {
    "--split-mode": "how to split the model across multiple GPUs, one of: none, layer, row, tensor"
}


def _wired_backend(
    tmp_path,
    monkeypatch,
    *,
    memory,
    model_gib,
    names,
    tensor_help = True,
):
    import weakref

    import core.inference.llama_cpp as llama_cpp_module

    monkeypatch.setattr(llama_cpp_module, "_owned_backends", weakref.WeakSet())
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._get_gguf_size_bytes = lambda _path: int(model_gib * GIB)
    backend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB = 256
    backend._gpu_names_by_physical_id = lambda: dict(names)
    backend._auto_tensor_platform_reason = lambda: None
    # Measured idle on every card, unless a cell says otherwise.
    backend._gpu_utilization_by_physical_id = lambda: dict.fromkeys(range(16), 0.0)
    real_probe = LlamaCppBackend.probe_server_capabilities

    def probe(binary = None):
        # A no-argument probe must not read this host's real llama-server.
        caps = dict(real_probe(binary or "/fake/llama-server"))
        caps["flags"] = dict(caps.get("flags") or {}, **(_TP_HELP if tensor_help else {}))
        return caps

    backend.probe_server_capabilities = probe
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_TENSOR_SPLIT", raising = False)
    monkeypatch.delenv(sp.AUTO_TENSOR_SPLIT_ENV, raising = False)
    # Stand-in pid: stopping it must never walk the real process table (pid 1 once SIGTERMed everything).
    backend._collect_descendants = lambda _pid: ([], True)
    backend._leading_process_group = lambda _pid: None
    backend._tree_kill_surviving_process = lambda _pid: True
    return backend, gguf


def _split_flag(cmd):
    for i, tok in enumerate(cmd):
        if tok in ("--split-mode", "-sm"):
            return cmd[i + 1]
    return None


TWO_T4_MEM = [(0, 15_360, 15_360), (1, 15_360, 15_360)]
TWO_T4_NAMES = {0: T4, 1: T4}


def test_wired_dense_two_gpus_launches_tensor(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) == "tensor"
    assert backend._auto_split_decision.mode == "tensor" and backend._auto_split_decision.auto


def test_wired_auto_tensor_publishes_the_vram_fit_context(tmp_path, monkeypatch):
    # The layer load it replaces publishes the fitted context; the auto tensor plan prices one too.
    backends = []
    off, on, on_backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 6,
        names = TWO_T4_NAMES,
        setup = lambda backend: (backends.append(backend), _native_ctx(backend, 32768)),
        gpu_memory_mode = "auto",
        n_ctx = 0,
    )
    assert _split_flag(off["cmd"]) != "tensor" and _split_flag(on["cmd"]) == "tensor"
    assert backends[0].vram_fit_context_length == backends[0].max_context_length > 0
    assert on_backend._auto_tensor_live
    assert on_backend.vram_fit_context_length == on_backend.max_context_length > 0


def _per_token_kv(
    backend,
    kv_per_tok,
    cc_per_tok = 0,
):
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * kv_per_tok
    backend._compute_buffer_ctx_bytes = lambda ctx, *a, **k: int(ctx) * cc_per_tok


def test_wired_auto_tensor_publishes_no_unpriced_ceiling(tmp_path, monkeypatch):
    # 1024 tokens fit the pair, the planner's 2048 floor does not: no ceiling is vouched for.
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 28, names = TWO_T4_NAMES
    )
    _native_ctx(backend, 32768)
    _per_token_kv(backend, 256 * 1024)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 1024)["cmd"]
    assert _split_flag(cmd) == "tensor"
    assert backend.vram_fit_context_length is None


def test_wired_auto_tensor_never_launches_a_plan_over_a_gpu_budget(tmp_path, monkeypatch):
    # GPU 1 cannot hold its replicated scratch at the planned context, so the planner weights it 0.
    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(0, 24_576, 24_576), (1, 19_700, 24_576)],
        model_gib = 2,
        names = TWO_T4_NAMES,
    )
    _native_ctx(backend, 32768)
    _per_token_kv(backend, 64 * 1024, 768 * 1024)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 0)["cmd"]
    assert _split_flag(cmd) != "tensor"


@pytest.mark.parametrize(
    "extra_args, env",
    [
        pytest.param(["--device", "CUDA0"], {}, id = "extras-device"),
        pytest.param(None, {"LLAMA_ARG_DEVICE": "CUDA0"}, id = "inherited-device"),
    ],
)
def test_wired_gpu_pick_owns_the_device_flags_it_strips(tmp_path, monkeypatch, extra_args, env):
    # The GPU pick strips --device / LLAMA_ARG_DEVICE from the child, so they do not own placement.
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    launched = _launch(
        backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, gpu_ids = [0, 1], extra_args = extra_args
    )
    assert _split_flag(launched["cmd"]) == "tensor"
    assert "--device" not in launched["cmd"] and "LLAMA_ARG_DEVICE" not in launched["env"]


def test_wired_widen_never_trades_the_bigger_gpu_for_a_smaller_pair(tmp_path, monkeypatch):
    # A100 + 2x T4: the model fits the idle A100 alone, so it is not moved onto the T4 pair.
    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(0, 81_920, 81_920), (1, 15_360, 15_360), (2, 15_360, 15_360)],
        model_gib = 6,
        names = {0: "NVIDIA A100-SXM4-80GB", 1: T4, 2: T4},
    )
    _native_ctx(backend, 131072)
    _per_token_kv(backend, 512 * 1024)
    launched = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 0)
    assert _split_flag(launched["cmd"]) != "tensor"
    assert launched["env"].get("CUDA_VISIBLE_DEVICES") == "0"
    assert backend.max_context_length == 131072


def test_widen_keeps_the_single_gpu_pick():
    three = [
        sp.GpuInfo(
            index = 0, name = "NVIDIA A100-SXM4-80GB", total_mib = 81_920, free_mib = 81_920, util_pct = 0.0
        ),
        sp.GpuInfo(index = 1, name = T4, total_mib = 15_360, free_mib = 15_360, util_pct = 0.0),
        sp.GpuInfo(index = 2, name = T4, total_mib = 15_360, free_mib = 15_360, util_pct = 0.0),
    ]
    d = _decide(gpus = three, fits_one_gpu = True, single_gpu_index = 0)
    assert (d.mode, d.auto) == ("none", False) and "0" in d.reason
    # The pick inside the pair still widens, and a caller naming no pick keeps today's rule.
    d = _decide(gpus = three, fits_one_gpu = True, single_gpu_index = 1)
    assert (d.mode, d.auto, tuple(d.gpu_indices)) == ("tensor", True, (1, 2))
    d = _decide(gpus = three, fits_one_gpu = True)
    assert (d.mode, d.auto) == ("tensor", True)


@pytest.mark.parametrize(
    "single_util, shared",
    [
        pytest.param(0.0, frozenset(), id = "matching-idle-pick-ranked-elsewhere"),
        pytest.param(90.0, frozenset(), id = "busy-pick"),
        pytest.param(None, frozenset(), id = "unread-pick"),
        pytest.param(0.0, frozenset({0}), id = "held-pick"),
    ],
)
def test_widen_vetoed_only_by_an_idle_card_of_another_class(single_util, shared):
    # Three T4s: the one-GPU pick (0) sits outside the idle pair, but it is no bigger card to keep.
    three = [
        sp.GpuInfo(index = 0, name = T4, total_mib = 15_360, free_mib = 15_000, util_pct = single_util),
        sp.GpuInfo(index = 1, name = T4, total_mib = 15_360, free_mib = 15_360, util_pct = 0.0),
        sp.GpuInfo(index = 2, name = T4, total_mib = 15_360, free_mib = 15_360, util_pct = 0.0),
    ]
    d = _decide(gpus = three, fits_one_gpu = True, single_gpu_index = 0, shared_gpus = shared)
    assert (d.mode, d.auto, tuple(d.gpu_indices)) == ("tensor", True, (1, 2))


def test_wired_kill_switch(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setenv(sp.AUTO_TENSOR_SPLIT_ENV, "0")
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_user_layer_extras_win(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    cmd = _launch(
        backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, extra_args = ["--split-mode", "layer"]
    )["cmd"]
    assert _split_flag(cmd) == "layer"


def test_wired_moe_without_draft_stays_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_fits_one_gpu_widened_to_tensor(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) == "tensor"
    assert backend._auto_split_decision.mode == "tensor"


def test_wired_fits_one_gpu_unchanged_when_widening_is_off(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(sp, "TP_WIDEN_FITS_ONE_GPU", False)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"
    assert backend._auto_split_decision.mode == "none"


def test_wired_four_gpus_tensor_runs_on_the_pair(tmp_path, monkeypatch):
    mem = [(i, 15_360, 15_360) for i in range(4)]
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = mem, model_gib = 20, names = {i: T4 for i in range(4)}
    )
    got = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert _split_flag(got["cmd"]) == "tensor"
    assert len(backend._auto_split_decision.gpu_indices) == 2
    mask = got["env"].get("CUDA_VISIBLE_DEVICES", "")
    assert len([x for x in mask.split(",") if x]) == 2, mask


def test_wired_four_gpus_too_big_for_a_pair_stays_layer(tmp_path, monkeypatch):
    mem = [(i, 15_360, 15_360) for i in range(4)]
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = mem, model_gib = 40, names = {i: T4 for i in range(4)}
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_mismatched_gpus_stay_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = {0: T4, 1: "NVIDIA L4"}
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_binary_without_tensor_stays_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 20,
        names = TWO_T4_NAMES,
        tensor_help = False,
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_moe_with_mtp_head_resident_is_tensor(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    _with_mtp_caps(backend)
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    cmd = _launch(backend, gguf, model_identifier = _MTP_MODEL, gpu_memory_mode = "auto", n_ctx = 4096)[
        "cmd"
    ]
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert backend._auto_split_decision.mode == "tensor"
    assert "MoE" in backend._auto_split_decision.reason


# 4. Risk gates


def test_busy_by_utilisation_blocks_the_pair():
    gpus = [
        sp.GpuInfo(0, T4, 15_360, 15_000, util_pct = 0.0),
        sp.GpuInfo(1, T4, 15_360, 15_000, util_pct = 85.0),
    ]
    d = _decide(gpus = gpus)
    assert d.mode == "layer" and "busy" in d.reason


@pytest.mark.parametrize("unknown", [(0, 1), (1,)])
def test_unknown_utilisation_is_not_idle(unknown):
    # Free memory does not prove a card is idle: a failed / N/A reading keeps layer split.
    gpus = [
        sp.GpuInfo(i, T4, 15_360, 15_000, util_pct = None if i in unknown else 0.0) for i in range(2)
    ]
    d = _decide(gpus = gpus)
    assert d.mode == "layer" and "utilisation unknown" in d.reason


def test_unknown_utilisation_on_a_third_gpu_keeps_the_verified_pair():
    gpus = [sp.GpuInfo(i, T4, 15_360, 15_000, util_pct = 0.0) for i in range(2)]
    gpus.append(sp.GpuInfo(2, T4, 15_360, 15_300, util_pct = None))
    d = _decide(gpus = gpus)
    assert d.mode == "tensor" and d.gpu_indices == (0, 1)


def test_unknown_utilisation_allowed_when_the_check_is_off():
    gpus = [sp.GpuInfo(i, T4, 15_360, 15_000, util_pct = None) for i in range(2)]
    th = sp.SplitPolicyThresholds(max_util_pct = None)
    assert _decide(gpus = gpus, thresholds = th).mode == "tensor"


def test_utilisation_check_can_be_switched_off():
    gpus = [sp.GpuInfo(i, T4, 15_360, 15_000, util_pct = 90.0) for i in range(2)]
    th = sp.SplitPolicyThresholds(max_util_pct = None)
    assert _decide(gpus = gpus, thresholds = th).mode == "tensor"


def test_auto_never_asks_for_more_than_two_gpus_even_if_thresholds_say_so():
    gpus = _gpus(*[(T4, 15_360, 15_000)] * 4)
    d = _decide(gpus = gpus, thresholds = sp.SplitPolicyThresholds(max_gpus = 4))
    assert d.mode == "tensor" and len(d.gpu_indices) == 2
    d = _decide(gpus = gpus, thresholds = sp.SplitPolicyThresholds(min_gpus = 3, max_gpus = 4))
    assert d.mode != "tensor"


@pytest.mark.parametrize(
    "desc, ok",
    [
        ("how to split the model across multiple GPUs, one of: none, layer, row", False),
        ("one of:\n- none: one GPU\n- layer (default)\n- row: rows, see --tensor-split", False),
        ("one of: none, layer, row, tensor", True),
        ("- tensor: split tensors across GPUs", True),
    ],
)
def test_capability_probe_only_matches_a_listed_tensor_value(desc, ok):
    assert sp.binary_supports_tensor_split({"flags": {"--split-mode": desc}}) is ok


def test_platform_outside_the_measured_one_keeps_todays_behaviour():
    d = _decide(platform_reason = "measured on NVIDIA CUDA only (this is a ROCm / HIP build)")
    assert d.mode == "layer" and not d.auto and "ROCm" in d.reason


def test_platform_gate_does_not_override_the_user_toggle():
    d = _decide(platform_reason = "measured on Linux only", user_tensor_parallel = True)
    assert d.mode == "tensor"


def test_platform_reason_helper(monkeypatch):
    import sys as _sys

    import torch

    monkeypatch.setattr(_sys, "platform", "win32")
    assert "Linux" in LlamaCppBackend._auto_tensor_platform_reason()
    monkeypatch.setattr(_sys, "platform", "linux")
    monkeypatch.setattr(torch.version, "hip", "6.2", raising = False)
    assert "HIP" in LlamaCppBackend._auto_tensor_platform_reason()
    # AMD SDK wheels leave version.hip unset and only carry "rocm" in __version__.
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+rocm7.0")
    assert "HIP" in LlamaCppBackend._auto_tensor_platform_reason()
    monkeypatch.setattr(torch, "__version__", "2.9.0+cu128")
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising = False)
    assert LlamaCppBackend._auto_tensor_platform_reason() is None


def test_utilisation_helper_needs_pci_indices(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", None)
    assert LlamaCppBackend._gpu_utilization_by_physical_id() == {}


def test_utilisation_helper_takes_the_max_sample(monkeypatch):
    from types import SimpleNamespace

    from utils.hardware import gpu_query

    outs = iter(["0, 0\n1, 3\n", "0, 2\n1, 77\n"])
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(
        gpu_query,
        "run_nvidia_smi",
        lambda *a, **k: SimpleNamespace(returncode = 0, stdout = next(outs)),
    )
    got = LlamaCppBackend._gpu_utilization_by_physical_id(samples = 2, interval_s = 0)
    assert got == {0: 2.0, 1: 77.0}


@pytest.mark.parametrize("second", ["exit", "raise"])
def test_utilisation_helper_keeps_samples_read_before_a_failed_one(monkeypatch, second):
    from types import SimpleNamespace

    from utils.hardware import gpu_query

    calls = []

    def run(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(returncode = 0, stdout = "0, 0\n1, 95\n")
        if second == "raise":
            raise subprocess.TimeoutExpired("nvidia-smi", 5)
        return SimpleNamespace(returncode = 9, stdout = "")

    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", run)
    got = LlamaCppBackend._gpu_utilization_by_physical_id(samples = 2, interval_s = 0)
    assert got == {0: 0.0, 1: 95.0}


def test_wired_other_studio_model_on_gpu1_blocks_the_pair(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    backend._other_planned_vram_mib = lambda: {1: 100}
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"
    assert backend._auto_split_decision.mode != "tensor"
    assert "busy" in backend._auto_split_decision.reason


def test_wired_busy_gpu1_by_utilisation_blocks_the_pair(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    backend._gpu_utilization_by_physical_id = lambda: {0: 0.0, 1: 95.0}
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


@pytest.mark.parametrize("util", [{}, {0: 0.0}])
def test_wired_unread_utilisation_blocks_the_pair(tmp_path, monkeypatch, util):
    # nvidia-smi failed, timed out, or answered N/A for a card: not proven idle.
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    backend._gpu_utilization_by_physical_id = lambda: dict(util)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"
    assert "utilisation unknown" in backend._auto_split_decision.reason


def test_wired_rocm_or_windows_keeps_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    backend._auto_tensor_platform_reason = lambda: "measured on NVIDIA CUDA only (ROCm / HIP)"
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"
    assert not backend._auto_split_decision.auto


@pytest.mark.parametrize("veto, p2p", [("no NVLink between GPUs 0 and 1", None), (None, "1")])
def test_wired_p2p_veto_still_gates_ggml_cuda_p2p_on_an_auto_tensor_pair(
    tmp_path, monkeypatch, veto, p2p
):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    for var in ("GGML_CUDA_P2P", "UNSLOTH_DISABLE_DC_TUNING", "UNSLOTH_DISABLE_DC_P2P"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "_is_datacenter_gpu", staticmethod(lambda *_a, **_k: True))
    seen = []

    def fake_veto(
        cls,
        gpu_indices = None,
        *_a,
        **_k,
    ):
        seen.append(sorted(gpu_indices or []))
        return veto

    monkeypatch.setattr(LlamaCppBackend, "_p2p_veto_reason", classmethod(fake_veto))
    got = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert _split_flag(got["cmd"]) == "tensor"
    assert got["env"].get("GGML_CUDA_P2P") == p2p
    assert [0, 1] in seen


# 5. Auto only engages or leaves the load as it was


def _with_mtp_caps(backend):
    """A binary that runs draft-mtp, so an embedded head really engages."""
    probe = backend.probe_server_capabilities
    backend.probe_server_capabilities = lambda binary = None: dict(
        probe(binary), mtp_token = "draft-mtp", supports_mtp = True, mtp_probe_inconclusive = False
    )
    backend._probe_mtp_decode = lambda *a, **k: True
    return backend


def _launch_off_vs_on(
    tmp_path,
    monkeypatch,
    *,
    memory,
    model_gib,
    names,
    setup = None,
    **load_kwargs,
):
    """The same load with the auto policy off (today's behaviour) and on: (off, on, on_backend)."""
    got = []
    for value in ("0", None):
        sub = tmp_path / f"auto_{value}"
        sub.mkdir()
        backend, gguf = _wired_backend(
            sub, monkeypatch, memory = memory, model_gib = model_gib, names = names
        )
        if value is None:
            monkeypatch.delenv(sp.AUTO_TENSOR_SPLIT_ENV, raising = False)
        else:
            monkeypatch.setenv(sp.AUTO_TENSOR_SPLIT_ENV, value)
        if setup is not None:
            setup(backend)
        launched = _launch(backend, gguf, **load_kwargs)
        got.append((launched, backend))
    monkeypatch.delenv(sp.AUTO_TENSOR_SPLIT_ENV, raising = False)
    (off, _), (on, on_backend) = got
    return off, on, on_backend


def _norm_cmd(launched):
    """The argv with the per-run tmp dir and port masked, so two launches compare equal."""
    out, skip = [], False
    for tok in launched["cmd"]:
        if skip:
            out.append("<port>")
            skip = False
            continue
        out.append(str(tok).replace("auto_None", "auto_X").replace("auto_0", "auto_X"))
        skip = tok in ("--port", "--api-key-file")
    return out


def _placement(launched):
    cmd = launched["cmd"]
    fit = cmd[cmd.index("--fit") + 1] if "--fit" in cmd else None
    return {
        "split": _split_flag(cmd),
        "mask": launched["env"].get("CUDA_VISIBLE_DEVICES"),
        "fit": fit,
        "ngl": cmd[cmd.index("--n-gpu-layers") + 1] if "--n-gpu-layers" in cmd else None,
    }


@pytest.mark.parametrize(
    "latch",
    ["_tensor_quant_kv_unsupported_binary", "_tensor_split_aborts"],
)
def test_wired_known_tensor_refusal_leaves_the_load_as_it_was(tmp_path, monkeypatch, latch):
    def setup(backend):
        setattr(backend, latch, lambda *a, **k: True)

    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 6,
        names = TWO_T4_NAMES,
        setup = setup,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
    )
    assert _placement(on) == _placement(off)
    assert _split_flag(on["cmd"]) != "tensor"
    assert backend._layer_preserves_tensor_intent is False


def test_wired_latch_tripping_after_the_decision_restores_the_pool(tmp_path, monkeypatch):
    def setup(backend):
        calls = []

        def aborts(*_a, **_k):
            calls.append(1)
            return len(calls) > 1

        backend._tensor_split_aborts = aborts

    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 6,
        names = TWO_T4_NAMES,
        setup = setup,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
    )
    assert _placement(on) == _placement(off)
    assert backend._layer_preserves_tensor_intent is False
    assert backend._auto_split_decision.mode == "layer"
    assert not backend._auto_split_decision.auto


def test_wired_tensor_buffer_reserve_too_big_leaves_the_load_as_it_was(tmp_path, monkeypatch):
    def setup(backend):
        real = backend._estimate_compute_buffer_bytes

        def compute(
            *a,
            per_device_tensor = False,
            **k,
        ):
            if per_device_tensor:
                return 14 * GIB
            return real(*a, per_device_tensor = per_device_tensor, **k)

        backend._estimate_compute_buffer_bytes = compute

    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 6,
        names = TWO_T4_NAMES,
        setup = setup,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
    )
    assert _placement(on) == _placement(off)
    assert backend._layer_preserves_tensor_intent is False


@pytest.mark.parametrize("model_gib", [26.0, 27.0, 28.4])
def test_wired_four_gpus_never_lose_the_cards_a_layer_split_used(tmp_path, monkeypatch, model_gib):
    mem = [(i, 15_360, 15_360) for i in range(4)]
    off, on, _backend_on = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = mem,
        model_gib = model_gib,
        names = {i: T4 for i in range(4)},
        gpu_memory_mode = "auto",
        n_ctx = 4096,
    )
    if _split_flag(on["cmd"]) != "tensor":
        assert _placement(on) == _placement(off)
    else:
        assert on["env"].get("CUDA_VISIBLE_DEVICES", "").count(",") == 1


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--tensor-split", "3,1"],
        ["-ts", "1,1"],
        ["--tensor-split=1,1"],
        ["--device", "CUDA0"],
        ["-ngl", "20"],
        ["--fit", "on"],
        ["-ot", "blk\\.1\\.=CPU"],
        ["--cpu-moe"],
        ["--n-cpu-moe", "4"],
        ["--mmproj-device", "CUDA1"],
        ["--device-draft", "CUDA1"],
        ["--moe-cache-mib", "2048"],
    ],
)
def test_wired_user_placement_in_extras_keeps_todays_behaviour(tmp_path, monkeypatch, extra_args):
    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 20,
        names = TWO_T4_NAMES,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        extra_args = list(extra_args),
    )
    assert _split_flag(on["cmd"]) != "tensor"
    assert _norm_cmd(on) == _norm_cmd(off)
    assert on["env"].get("CUDA_VISIBLE_DEVICES") == off["env"].get("CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize(
    "var, value",
    [
        ("LLAMA_ARG_N_GPU_LAYERS", "20"),
        ("LLAMA_ARG_DEVICE", "CUDA0"),
        ("MTMD_BACKEND_DEVICE", "CUDA1"),
        ("LLAMA_ARG_MOE_CACHE_MIB", "2048"),
    ],
)
def test_wired_user_placement_in_env_keeps_layer(tmp_path, monkeypatch, var, value):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setenv(var, value)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_extras_dflash_keeps_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    drafter = tmp_path / "dflash.gguf"
    drafter.write_bytes(b"GGUF")
    backend._get_gguf_size_bytes = lambda path: GIB // 4 if "dflash" in str(path) else 20 * GIB
    cmd = _launch(
        backend,
        gguf,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        extra_args = ["--model-draft", str(drafter), "--spec-type", "draft-dflash"],
    )["cmd"]
    assert "draft-dflash" in cmd
    assert _split_flag(cmd) != "tensor"
    assert backend._auto_split_decision.mode != "tensor"


@pytest.mark.parametrize(
    "load_kwargs",
    [
        {"speculative_type": "ngram"},
        {"speculative_type": "off"},
        {"extra_args": ["--spec-type", "ngram-mod"]},
    ],
)
def test_wired_moe_with_an_mtp_head_that_does_not_run_stays_layer(
    tmp_path, monkeypatch, load_kwargs
):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    _with_mtp_caps(backend)
    backend._nextn_predict_layers = 1
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    cmd = _launch(
        backend,
        gguf,
        model_identifier = _MTP_MODEL,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        **load_kwargs,
    )["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_unloaded_nextn_weights_are_not_priced(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 16, names = TWO_T4_NAMES
    )
    backend._unloaded_nextn_bytes = lambda _path: 8 * GIB
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, speculative_type = "off")
    assert backend._auto_split_decision.mode == "tensor"
    assert "fits one GPU" in backend._auto_split_decision.reason


def _host_lookup_tables(monkeypatch, backend, gib):
    import core.inference.llama_cpp as llama_cpp_module
    backend._gguf_tensor_scan = lambda _path: ("gemma3n", {}, 0)
    monkeypatch.setattr(
        llama_cpp_module, "_host_pinned_input_bytes", lambda *_a, **_k: int(gib * GIB)
    )


def test_wired_host_lookup_tables_are_not_priced_on_one_gpu(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 16, names = TWO_T4_NAMES
    )
    _host_lookup_tables(monkeypatch, backend, 8)
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, speculative_type = "off")
    assert backend._auto_split_decision.mode == "tensor"
    assert "fits one GPU" in backend._auto_split_decision.reason


def test_wired_host_lookup_tables_are_not_priced_on_the_pair(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 34, names = TWO_T4_NAMES
    )
    _host_lookup_tables(monkeypatch, backend, 10)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, speculative_type = "off")["cmd"]
    assert backend._auto_split_decision.mode == "tensor"
    assert _split_flag(cmd) == "tensor"


def test_wired_host_lookup_tables_are_credited_on_a_discrete_pair_beside_a_unified_gpu(
    tmp_path, monkeypatch
):
    # Unified GPU 0 voids the whole-host credit, but the auto pair is discrete and settled.
    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(0, 12_288, 65_536), (1, 15_360, 15_360), (2, 15_360, 15_360)],
        model_gib = 34,
        names = {0: "NVIDIA GB10", 1: T4, 2: T4},
    )
    backend._rocm_unified_memory_gpu_ids = lambda: {0}
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
    _host_lookup_tables(monkeypatch, backend, 10)
    launched = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, speculative_type = "off")
    assert _split_flag(launched["cmd"]) == "tensor"
    assert tuple(backend._auto_split_decision.gpu_indices) == (1, 2)


def test_wired_integrated_cuda_ordinal_off_the_pci_order_voids_the_pair_credit(
    tmp_path, monkeypatch
):
    # CUDA ordinal 2 is integrated, but ordinals and PCI ids disagree: GPU 2 may be either T4.
    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(0, 15_360, 15_360), (1, 15_360, 15_360), (2, 12_288, 65_536)],
        model_gib = 34,
        names = {0: T4, 1: T4, 2: "NVIDIA GB10"},
    )
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_INTEGRATED_CUDA_IDS",
        {LlamaCppBackend._integrated_cuda_mask_key(): (2,)},
    )
    _host_lookup_tables(monkeypatch, backend, 10)
    launched = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, speculative_type = "off")
    assert _split_flag(launched["cmd"]) != "tensor"


def test_wired_p2p_diagnostic_reads_the_pair_that_runs(tmp_path, monkeypatch):
    mem = [(0, 13_000, 15_360), (1, 13_000, 15_360), (2, 15_360, 15_360), (3, 15_360, 15_360)]
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = mem, model_gib = 20, names = {i: T4 for i in range(4)}
    )
    seen = []

    def fake_veto(
        cls,
        gpu_indices = None,
        *_a,
        **_k,
    ):
        seen.append(sorted(gpu_indices or []))
        return "no NVLink"

    monkeypatch.setattr(LlamaCppBackend, "_p2p_veto_reason", classmethod(fake_veto))
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_split_decision.gpu_indices == (2, 3)
    assert seen and all(s == [2, 3] for s in seen), seen
    assert "no NVLink" in backend._auto_split_decision.reason


def test_wired_widened_model_p2p_diagnostic_is_recorded(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_p2p_veto_reason", classmethod(lambda cls, *_a, **_k: "no NVLink")
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_split_decision.mode == "tensor"
    assert "no NVLink" in backend._auto_split_decision.reason


@pytest.mark.parametrize(
    "setup",
    [
        lambda mp: mp.setenv(sp.AUTO_TENSOR_SPLIT_ENV, "0"),
        lambda mp: mp.setenv("LLAMA_ARG_SPLIT_MODE", "layer"),
    ],
)
def test_wired_no_utilisation_probe_when_auto_cannot_choose_tensor(tmp_path, monkeypatch, setup):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    probed = []
    backend._gpu_utilization_by_physical_id = lambda: probed.append("util") or {}
    backend._gpu_names_by_physical_id = lambda: probed.append("names") or dict(TWO_T4_NAMES)
    setup(monkeypatch)
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert probed == []


def test_wired_identical_repeat_request_reuses_an_auto_tensor_server(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend.tensor_parallel
    intent = backend._last_load_intent
    assert backend._runtime_matches_intent(intent, intent.extra_args)
    assert not backend._runtime_matches_intent(
        dataclasses.replace(intent, extra_args = ["--split-mode", "layer"]), ["--split-mode", "layer"]
    )


def test_status_and_load_report_an_auto_tensor_split_as_toggle_off(tmp_path, monkeypatch):
    _llama_runtime_fields = _load_inference_routes_module()._llama_runtime_fields

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend.tensor_parallel
    assert _llama_runtime_fields(backend)["tensor_parallel"] is False

    (tmp_path / "toggled").mkdir()
    toggled, gguf2 = _wired_backend(
        tmp_path / "toggled", monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    _launch(toggled, gguf2, gpu_memory_mode = "auto", n_ctx = 4096, tensor_parallel = True)
    assert _llama_runtime_fields(toggled)["tensor_parallel"] is True


def test_wired_auto_tensor_start_failure_retries_with_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    cmds = []

    def wait(timeout, **_kw):
        if _split_flag(cmds[-1]["cmd"]) == "tensor":
            raise RuntimeError(
                "llama-server aborted on --split-mode tensor (split-axis geometry); "
                "retrying with layer split."
            )
        return True

    backend._wait_for_health = wait

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        cmds.append({"cmd": list(cmd), "env": kwargs.get("env") or dict(os.environ)})
        return type(
            "Process",
            (),
            {
                "pid": _FAKE_SERVER_PID,
                "stdout": (),
                "returncode": None,
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        ok = backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf), model_identifier = "test", gpu_memory_mode = "auto", n_ctx = 4096
            )
        )
    assert ok
    assert [_split_flag(c["cmd"]) for c in cmds][0] == "tensor"
    assert _split_flag(cmds[-1]["cmd"]) != "tensor"
    assert not backend.tensor_parallel
    assert backend._auto_split_attempt is None
    assert not backend._auto_tensor_live


def test_wired_unload_clears_the_auto_decision(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_split_decision is not None and backend._auto_tensor_live
    backend.unload_model()
    # Nothing to reset: the live view is gated on the tensor state every reset clears.
    assert backend._auto_tensor_live is False
    assert backend._auto_split_decision is None


def test_wired_unload_of_a_fake_server_signals_nothing(tmp_path, monkeypatch):
    import utils.process_lifetime as lifetime

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    sent = []
    monkeypatch.setattr(lifetime, "terminate_pid", lambda pid, **_k: sent.append(("pid", pid)))
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append(("pg", pgid)), raising = False)
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._process.pid == _FAKE_SERVER_PID
    backend.unload_model()
    assert sent == []


def test_gpu_names_use_nvidia_smi_indices_when_the_planner_does(monkeypatch):
    # Planner ids are nvidia-smi (PCI) indices; torch FASTEST_FIRST ordinals would swap the names.
    from utils.hardware import gpu_query

    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    out = types.SimpleNamespace(returncode = 0, stdout = "0, NVIDIA L4\n1, Tesla T4\n2, Tesla T4\n")
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", lambda *a, **k: out)
    fake_torch = types.SimpleNamespace(
        version = types.SimpleNamespace(hip = None, cuda = "12.4"),
        cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: 3,
            get_device_properties = lambda i: types.SimpleNamespace(
                name = ["Tesla T4", "Tesla T4", "NVIDIA L4"][i]
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    assert LlamaCppBackend._gpu_names_by_physical_id() == {
        0: "NVIDIA L4",
        1: "Tesla T4",
        2: "Tesla T4",
    }


# 6. Every fallback / retry / recovery runs with the policy off


def _launch_scripted(backend, gguf, behave, **load_kwargs):
    """load_model with a fake child per spawn. ``behave(cmd)`` -> (healthy, stdout lines, rc).
    Returns (result or the exception, [cmd, ...])."""
    import time as _time

    spawns = []

    class _Proc:
        def __init__(self, cmd):
            self.healthy, lines, self.rc = behave(cmd)
            self.stdout = list(lines)
            self.pid = _FAKE_SERVER_PID
            self.returncode = None if self.healthy else self.rc

        def poll(self):
            return None if self.healthy else self.rc

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout = None):
            return 0

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        spawns.append(list(cmd))
        return _Proc(cmd)

    def wait(timeout, **_kw):
        proc = backend._process
        if proc is not None and not proc.healthy:
            deadline = _time.monotonic() + 2.0
            while len(backend._stdout_lines) < len(proc.stdout) and _time.monotonic() < deadline:
                _time.sleep(0.01)
            return False
        return True

    backend._wait_for_health = wait
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        try:
            result = backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **load_kwargs)
            )
        except Exception as exc:  # noqa: BLE001 - the cell asserts on it
            result = exc
    return result, spawns


def _tensor_crashes(lines, rc = 134):
    return lambda cmd: (False, lines, rc) if _split_flag(cmd) == "tensor" else (True, [], 0)


def _record_warnings(monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    seen = []
    real = llama_cpp_module.logger.warning
    monkeypatch.setattr(
        llama_cpp_module.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)) or real(msg, *a, **k),
    )
    return seen


def _assert_no_split_state(backend):
    assert not backend.tensor_parallel
    assert not backend._auto_tensor_live
    assert backend._auto_split_decision is None


def test_wired_auto_tensor_post_health_failure_is_not_retried(tmp_path, monkeypatch):
    # A healthy tensor server whose load then fails is not a tensor failure.
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    backend._apply_detected_audio = lambda *a, **k: False
    result, spawns = _launch_scripted(
        backend, gguf, lambda cmd: (True, [], 0), gpu_memory_mode = "auto", n_ctx = 4096
    )
    assert result is False
    assert [_split_flag(c) for c in spawns] == ["tensor"]


@pytest.mark.parametrize("flag", ["_health_wait_cancelled", "_shutting_down"])
def test_wired_auto_tensor_torn_down_mid_load_is_not_retried(tmp_path, monkeypatch, flag):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    warnings = _record_warnings(monkeypatch)

    def behave(cmd):
        setattr(backend, flag, True)
        return False, [], 1

    try:
        result, spawns = _launch_scripted(backend, gguf, behave, gpu_memory_mode = "auto", n_ctx = 4096)
    finally:
        backend._shutting_down = False
    assert result is False
    assert [_split_flag(c) for c in spawns] == ["tensor"]
    assert not [w for w in warnings if "loading without it" in w]
    _assert_no_split_state(backend)


def test_wired_abandoned_auto_tensor_warns_once(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    warnings = _record_warnings(monkeypatch)
    result, spawns = _launch_scripted(
        backend, gguf, _tensor_crashes(["boom"], rc = 1), gpu_memory_mode = "auto", n_ctx = 4096
    )
    assert result is True
    assert [_split_flag(c) for c in spawns][0] == "tensor"
    assert len([w for w in warnings if "loading without it" in w]) == 1
    assert not [w for w in warnings if "--fit on" in w]


def test_wired_scheduler_abort_on_auto_tensor_does_not_block_the_layer_load(tmp_path, monkeypatch):
    # The scheduler-abort memo keyed the unchanged request and refused the layer reload.
    monkeypatch.setattr(LlamaCppBackend, "_sched_reserve_abort_keys", set())
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    result, spawns = _launch_scripted(
        backend,
        gguf,
        _tensor_crashes(
            ["GGML_ASSERT(*cur_backend_id != -1) failed", "ggml_backend_sched_split_graph"]
        ),
        gpu_memory_mode = "auto",
        n_ctx = 4096,
    )
    assert result is True, result
    assert _split_flag(spawns[0]) == "tensor"
    assert [s for s in spawns[1:] if _split_flag(s) == "tensor"] == []
    assert len(spawns) == 2
    assert not LlamaCppBackend._sched_reserve_abort_keys


def test_wired_auto_tensor_kernel_crash_reloads_without_the_policy(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    (tmp_path / "auto_None").mkdir()
    (tmp_path / "auto_0").mkdir()
    backend, gguf = _wired_backend(
        tmp_path / "auto_None", monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    off_backend, off_gguf = _wired_backend(
        tmp_path / "auto_0", monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    monkeypatch.setenv(sp.AUTO_TENSOR_SPLIT_ENV, "0")
    off = _launch(off_backend, off_gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    monkeypatch.delenv(sp.AUTO_TENSOR_SPLIT_ENV)
    llama_cpp_module._owned_backends.discard(off_backend)
    result, spawns = _launch_scripted(
        backend,
        gguf,
        _tensor_crashes(["CUDA error: device kernel image is invalid"], rc = 1),
        gpu_memory_mode = "auto",
        n_ctx = 4096,
    )
    assert result is True, result
    assert _split_flag(spawns[0]) == "tensor" and len(spawns) == 2
    assert _norm_cmd({"cmd": spawns[-1]}) == _norm_cmd(off)
    assert backend._arch_gate_dropped_tensor_parallel is False
    assert not backend._auto_tensor_live
    assert not backend._auto_split_decision.auto


def test_wired_auto_tensor_mtp_decode_failure_keeps_mtp_on_layer(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    _with_mtp_caps(backend)
    backend._probe_mtp_decode = lambda *a, **k: False
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    spawns = []
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        spawns.append(list(cmd))
        return type(
            "P",
            (),
            {
                "pid": _FAKE_SERVER_PID,
                "stdout": (),
                "returncode": None,
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf), model_identifier = _MTP_MODEL, gpu_memory_mode = "auto", n_ctx = 4096
            )
        )
    final = spawns[-1]
    assert _split_flag(spawns[0]) == "tensor"
    assert _split_flag(final) != "tensor"
    assert final[final.index("--spec-type") + 1] == "draft-mtp"
    assert not backend._auto_tensor_live


def test_wired_concurrent_load_cannot_clear_the_running_attempt(tmp_path, monkeypatch):
    import threading
    import time as _time

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    in_tensor = threading.Event()
    last = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        last["cmd"] = list(cmd)
        return type(
            "P",
            (),
            {
                "pid": _FAKE_SERVER_PID,
                "stdout": (),
                "returncode": None,
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def wait(timeout, **_kw):
        if _split_flag(last["cmd"]) == "tensor" and not in_tensor.is_set():
            in_tensor.set()
            _time.sleep(0.5)  # the queued load reaches load_model meanwhile
            raise RuntimeError("tensor start failed")
        return True

    backend._wait_for_health = wait
    results = {}

    def run(name, **kw):
        try:
            results[name] = backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096, **kw)
            )
        except Exception as exc:  # noqa: BLE001
            results[name] = exc

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        a = threading.Thread(target = run, args = ("A",), kwargs = {"gpu_memory_mode": "auto"})
        a.start()
        assert in_tensor.wait(10)
        b = threading.Thread(
            target = run, args = ("B",), kwargs = {"gpu_memory_mode": "auto", "n_parallel": 2}
        )
        b.start()
        a.join(30)
        b.join(30)
    assert results.get("A") is True, results
    assert results.get("B") is True, results


def _kv_per_token(backend, kib):
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * kib * 1024


@pytest.mark.parametrize(
    "load_kwargs",
    [{"n_ctx": 65536}, {"n_ctx": 0, "extra_args": ["--ctx-size", "65536"]}],
)
def test_wired_pair_must_hold_the_requested_context(tmp_path, monkeypatch, load_kwargs):
    # Weights fit a pair; the requested 65536 context needs a third card, which layer split adds.
    mem = [(i, 15_360, 15_360) for i in range(4)]
    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = mem,
        model_gib = 20,
        names = {i: T4 for i in range(4)},
        setup = lambda b: _kv_per_token(b, 256),
        gpu_memory_mode = "auto",
        **load_kwargs,
    )
    assert _split_flag(on["cmd"]) != "tensor"
    assert _norm_cmd(on) == _norm_cmd(off)
    assert on["env"].get("CUDA_VISIBLE_DEVICES") == off["env"].get("CUDA_VISIBLE_DEVICES")


def test_wired_pair_holding_the_requested_context_still_goes_tensor(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _kv_per_token(backend, 64)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 32768)["cmd"]
    assert _split_flag(cmd) == "tensor"


def test_wired_explicit_tensor_toggle_after_an_auto_load_reloads(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_tensor_live
    intent = backend._last_load_intent
    toggled = dataclasses.replace(intent, tensor_parallel = True)
    assert not backend._runtime_matches_intent(toggled, toggled.extra_args)
    tensor_extras = ["--split-mode", "tensor"]
    assert not backend._runtime_matches_intent(
        dataclasses.replace(intent, extra_args = tensor_extras), tensor_extras
    )


def test_live_auto_state_follows_the_tensor_state_every_reset_clears(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_tensor_live
    backend._tensor_parallel = False
    assert not backend._auto_tensor_live
    assert backend._auto_split_attempt is None


def test_gpu_names_fall_back_to_the_nvml_inventory(monkeypatch):
    from utils.hardware import gpu_query

    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(
        gpu_query, "run_nvidia_smi", lambda *a, **k: types.SimpleNamespace(returncode = 9, stdout = "")
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_NVML_ROWS",
        [
            {"index": "0", "name": "Tesla T4"},
            {"index": "1", "name": "Tesla T4"},
            {"index": "1", "name": "Tesla T4 MIG", "mig": "1"},
        ],
    )
    assert LlamaCppBackend._gpu_names_by_physical_id() == {0: "Tesla T4", 1: "Tesla T4"}


# 7. Admission prices what the tensor planner charges


def _native_ctx(backend, n):
    read = backend._read_gguf_metadata

    def read_metadata(path):
        read(path)
        backend._context_length = n

    backend._read_gguf_metadata = read_metadata


def _mtp_reserve(
    backend,
    per_ctx = None,
    flat = 0,
    kv_sized = True,
):
    _with_mtp_caps(backend)
    backend._mtp_draft_kv_bytes = lambda ctx, *a, **k: (
        (int(ctx) * per_ctx if per_ctx else 0) if kv_sized else None
    )
    backend._estimate_mtp_overhead_bytes = lambda ctx, *a, **k: flat + (
        int(ctx) * per_ctx if per_ctx else 0
    )


def _assert_same_load(off, on):
    assert _split_flag(on["cmd"]) != "tensor"
    assert _norm_cmd(on) == _norm_cmd(off)
    assert on["env"].get("CUDA_VISIBLE_DEVICES") == off["env"].get("CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize(
    "load_kwargs",
    [{"n_ctx": 16384}, {"n_ctx": 0, "extra_args": ["--ctx-size", "16384"]}],
)
def test_wired_requested_context_past_native_is_not_shortened(tmp_path, monkeypatch, load_kwargs):
    def setup(b):
        _kv_per_token(b, 16)
        _native_ctx(b, 8192)

    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 6,
        names = TWO_T4_NAMES,
        setup = setup,
        gpu_memory_mode = "auto",
        **load_kwargs,
    )
    _assert_same_load(off, on)
    # Refused up front: the planner caps an explicit context at native.
    assert "abandoned" not in backend._auto_split_decision.reason


def test_wired_unpriceable_kv_with_auto_context_keeps_todays_load(tmp_path, monkeypatch):
    def setup(b):
        b._can_estimate_kv = lambda: False
        _native_ctx(b, 32768)

    off, on, _ = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = [(0, 24_000, 24_000), (1, 24_000, 24_000)],
        model_gib = 8,
        names = TWO_T4_NAMES,
        setup = setup,
        gpu_memory_mode = "auto",
        n_ctx = 0,
    )
    _assert_same_load(off, on)


@pytest.mark.parametrize("model_gib", [18.5, 19.5])
def test_wired_unsized_drafter_kv_is_priced_like_the_planner(tmp_path, monkeypatch, model_gib):
    # The planner adds a 2 GiB cushion ON TOP of the drafter weights it can size.
    def setup(b):
        _kv_per_token(b, 200)
        _mtp_reserve(b, flat = int(1.5 * GIB), kv_sized = False)

    off, on, backend = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = [(i, 15_360, 15_360) for i in range(4)],
        model_gib = model_gib,
        names = {i: T4 for i in range(4)},
        setup = setup,
        model_identifier = _MTP_MODEL,
        gpu_memory_mode = "auto",
        n_ctx = 32768,
        speculative_type = "mtp",
    )
    _assert_same_load(off, on)
    assert "abandoned" not in backend._auto_split_decision.reason


def test_wired_measured_mtp_reserve_is_not_overcharged(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 26.2, names = TWO_T4_NAMES
    )
    _mtp_reserve(backend, flat = 256 * 1024**2)
    cmd = _launch(
        backend,
        gguf,
        model_identifier = _MTP_MODEL,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        speculative_type = "mtp",
    )["cmd"]
    assert _split_flag(cmd) == "tensor"


def test_wired_cpu_drafter_is_not_priced_on_the_gpu(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 25.5, names = TWO_T4_NAMES
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._get_gguf_size_bytes = (
        lambda path: 4 * GIB if str(path) == str(sidecar) else int(25.5 * GIB)
    )
    _mtp_reserve(backend, flat = 4 * GIB)
    probe = backend.probe_server_capabilities
    backend.probe_server_capabilities = lambda binary = None: dict(
        probe(binary), supports_dspark = True, spec_draft_n_max_flag = "--spec-draft-n-max"
    )
    cmd = _launch(
        backend,
        gguf,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        dspark_draft_path = str(sidecar),
        extra_args = ("--spec-draft-ngl", "0"),
    )["cmd"]
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert _split_flag(cmd) == "tensor"


def test_wired_auto_context_keeps_the_auto_drafter_drop(tmp_path, monkeypatch):
    # main drops an Auto drafter rather than shorten the context; tensor kept it at 24857.
    def setup(b):
        _kv_per_token(b, 128)
        _mtp_reserve(b, per_ctx = 128 * 1024)
        _native_ctx(b, 32768)

    off, on, _ = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 22,
        names = TWO_T4_NAMES,
        setup = setup,
        model_identifier = _MTP_MODEL,
        gpu_memory_mode = "auto",
        n_ctx = 0,
        speculative_type = "auto",
    )
    _assert_same_load(off, on)
    assert "--spec-type" not in on["cmd"]


def test_wired_auto_context_mtp_short_of_native_runs_mains_load(tmp_path, monkeypatch):
    def setup(b):
        _kv_per_token(b, 2048)
        _mtp_reserve(b, flat = 64 * 1024**2)
        _native_ctx(b, 131072)

    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    off, on, _ = _launch_off_vs_on(
        tmp_path,
        monkeypatch,
        memory = TWO_T4_MEM,
        model_gib = 18,
        names = TWO_T4_NAMES,
        setup = setup,
        model_identifier = _MTP_MODEL,
        gpu_memory_mode = "auto",
        n_ctx = 0,
        speculative_type = "auto",
    )
    _assert_same_load(off, on)
    assert on["cmd"][on["cmd"].index("--spec-type") + 1] == "draft-mtp"


def test_wired_advisor_load_avoids_a_serving_models_gpu(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    # Unregistered helper: the pair must still avoid the serving model's card.
    serving = types.SimpleNamespace(
        is_active = True, _planned_vram_mib = {1: 1024}, _child_gpu_physical_ids = (1,)
    )
    monkeypatch.setattr(llama_cpp_module, "_serving_backends_snapshot", lambda: [serving])
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_advisor_load_avoids_a_resident_media_models_gpu(tmp_path, monkeypatch):
    import core.inference.gpu_residency as gpu_residency
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(llama_cpp_module, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(gpu_residency, "media_held_mib", lambda: {1: 2048})
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


def test_wired_moe_without_draft_runs_no_gpu_probe(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    probed = []
    backend._gpu_utilization_by_physical_id = lambda: probed.append("util") or {}
    backend._gpu_names_by_physical_id = lambda: probed.append("names") or dict(TWO_T4_NAMES)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"
    assert probed == []
    assert "MoE" in backend._auto_split_decision.reason


def test_wired_abandoned_auto_tensor_cancelled_after_its_kill_keeps_no_state(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    kill = backend._kill_process

    def kill_then_shut_down(*a, **k):
        proc = backend._process
        kill(*a, **k)
        if proc is not None and not getattr(proc, "healthy", True):
            backend._shutting_down = True

    backend._kill_process = kill_then_shut_down
    try:
        result, spawns = _launch_scripted(
            backend, gguf, _tensor_crashes(["boom"], rc = 1), gpu_memory_mode = "auto", n_ctx = 4096
        )
    finally:
        backend._shutting_down = False
    assert result is False
    assert [_split_flag(c) for c in spawns] == ["tensor"]
    _assert_no_split_state(backend)


def test_wired_reload_cancelled_after_the_old_server_dies_keeps_no_state(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_tensor_live
    kill = backend._kill_process

    def kill_then_cancel(*a, **k):
        kill(*a, **k)
        backend._cancel_event.set()

    backend._kill_process = kill_then_cancel
    with patch.object(subprocess, "Popen", side_effect = AssertionError("no spawn")):
        ok = backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf), model_identifier = "test", gpu_memory_mode = "auto", n_ctx = 8192
            )
        )
    assert ok is False
    _assert_no_split_state(backend)


def test_diffusion_start_drops_the_chat_split_decision(monkeypatch, tmp_path):
    import test_diffusion_auto_fit_ngl as diffusion

    from core.inference.llama_cpp import _AutoSplitAttempt

    fit_host = diffusion._fit_host

    def host_after_an_auto_tensor_chat(*a, **k):
        backend = fit_host(*a, **k)
        backend._tensor_parallel = True
        backend._auto_split = _AutoSplitAttempt(
            decision = sp.SplitDecision("tensor", True, "x"), engaged = True
        )
        return backend

    monkeypatch.setattr(diffusion, "_fit_host", host_after_an_auto_tensor_chat)
    _cmd, backend = diffusion._launch(monkeypatch, tmp_path, [(0, 7_800, 8_188)])
    _assert_no_split_state(backend)


# 8. A load with no healthy server leaves no split state


def _frames(exc):
    names, tb = [], getattr(exc, "__traceback__", None)
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return names


def _always_crashes(rc = 1):
    return lambda cmd: (False, ["boom"], rc)


def test_failed_explicit_tensor_load_keeps_no_split_state(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    result, spawns = _launch_scripted(
        backend,
        gguf,
        _always_crashes(),
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        tensor_parallel = True,
    )
    assert result is not True and spawns
    assert _split_flag(spawns[0]) == "tensor"
    _assert_no_split_state(backend)


def test_failed_layer_retry_after_an_abandoned_auto_tensor_keeps_no_split_state(
    tmp_path, monkeypatch
):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    result, spawns = _launch_scripted(
        backend, gguf, _always_crashes(), gpu_memory_mode = "auto", n_ctx = 4096
    )
    assert result is not True
    assert _split_flag(spawns[0]) == "tensor" and _split_flag(spawns[-1]) != "tensor"
    _assert_no_split_state(backend)


def test_terminal_load_failure_keeps_no_split_state(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    result, _spawns = _launch_scripted(
        backend, gguf, _always_crashes(), gpu_memory_mode = "auto", n_ctx = 4096
    )
    assert "_raise_terminal_load_failure" in _frames(result), result
    _assert_no_split_state(backend)


def test_failed_cpu_replay_keeps_no_split_state(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    backend._auto_vulkan_cpu_fallback_eligible = lambda *a, **k: True
    backend._prepare_cpu_fallback_launch = lambda _b, cmd, *a, **k: (
        list(cmd) + ["--device", "none"],
        None,
        None,
    )
    result, spawns = _launch_scripted(
        backend, gguf, _always_crashes(rc = -11), gpu_memory_mode = "auto", n_ctx = 4096
    )
    assert "_try_auto_vulkan_cpu_fallback" in _frames(result), result
    assert any("none" in c for c in spawns)
    _assert_no_split_state(backend)


# 9. Unload before the retry, Manual residents, flash attention off


def test_unload_before_the_layer_retry_cancels_it(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    real = llama_cpp_module.logger.warning

    def warn(msg, *a, **k):
        if "Auto tensor split abandoned" in str(msg):
            backend.unload_model()
        return real(msg, *a, **k)

    monkeypatch.setattr(llama_cpp_module.logger, "warning", warn)
    result, spawns = _launch_scripted(
        backend, gguf, _tensor_crashes(["boom"], rc = 1), gpu_memory_mode = "auto", n_ctx = 4096
    )
    assert result is False
    assert [_split_flag(c) for c in spawns] == ["tensor"]
    assert not backend.is_loaded
    # Seen as a cancelled load by the route, not a failure.
    assert backend._cancel_event.is_set()


def test_wired_auto_pair_avoids_a_manual_residents_gpu(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    resident = types.SimpleNamespace(
        is_active = True, _planned_vram_mib = {}, _child_gpu_physical_ids = (1,)
    )
    monkeypatch.setattr(llama_cpp_module, "_serving_backends_snapshot", lambda: [resident])
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"


@pytest.mark.parametrize("flag", ["--flash-attn", "-fa"])
def test_wired_flash_attention_off_keeps_layer_up_front(tmp_path, monkeypatch, flag):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, extra_args = [flag, "off"])[
        "cmd"
    ]
    assert _split_flag(cmd) != "tensor"
    reason = backend._auto_split_decision.reason
    assert "flash attention" in reason and "abandoned" not in reason


def test_wired_auto_pair_avoids_a_residents_companion_gpu(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(i, 15_360, 15_360) for i in range(4)],
        model_gib = 6,
        names = {i: T4 for i in range(4)},
    )
    resident = types.SimpleNamespace(
        is_active = True, _planned_vram_mib = {2: 1024}, _child_gpu_physical_ids = (2, 0)
    )
    monkeypatch.setattr(llama_cpp_module, "_serving_backends_snapshot", lambda: [resident])
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    picked = set(backend._auto_split_decision.gpu_indices or ())
    assert not picked & {0, 2}


def test_wired_resident_with_unknown_gpus_keeps_layer(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    # An unpinned Manual (or Vulkan) resident: it may hold any card.
    resident = types.SimpleNamespace(
        is_active = True, _planned_vram_mib = {}, _child_gpu_physical_ids = None
    )
    monkeypatch.setattr(llama_cpp_module, "_serving_backends_snapshot", lambda: [resident])
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) != "tensor"
    assert "unknown" in backend._auto_split_decision.reason


@pytest.mark.parametrize(
    "sidecar, tensor",
    [
        (dict(loaded_model = "small", device = "whisper.cpp", _forced_cpu = False), False),
        (dict(loaded_model = None, device = None, loading = True), False),
        (dict(loaded_model = "small", device = "whisper.cpp", _forced_cpu = True), True),
        (dict(loaded_model = "small", device = "cpu"), True),
        (dict(loaded_model = "small", device = "llama.cpp", _gpu_disabled = True), True),
    ],
)
def test_wired_resident_dictation_gpu_keeps_layer(tmp_path, monkeypatch, sidecar, tensor):
    import core.inference.stt_registry as stt_registry

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    loading = sidecar.pop("loading", False)
    fake = types.SimpleNamespace(is_loading = lambda: loading, **sidecar)
    monkeypatch.setattr(
        stt_registry,
        "sidecar_for",
        lambda engine: fake
        if engine == "gguf"
        else types.SimpleNamespace(is_loading = lambda: False, loaded_model = None, device = None),
    )
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert (_split_flag(cmd) == "tensor") is tensor
    if not tensor:
        assert "unknown" in backend._auto_split_decision.reason


def test_wired_fits_one_probe_prefers_discrete_gpus_like_the_placement(tmp_path, monkeypatch):
    # Unified-memory GPU 0 fits the MoE alone, but placement runs it on the discrete pair.
    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(0, 65_536, 65_536), (1, 15_360, 15_360), (2, 15_360, 15_360)],
        model_gib = 20,
        names = {0: "NVIDIA GB10", 1: T4, 2: T4},
    )
    backend._rocm_unified_memory_gpu_ids = lambda: {0}
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
    _with_mtp_caps(backend)
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    cmd = _launch(backend, gguf, model_identifier = _MTP_MODEL, gpu_memory_mode = "auto", n_ctx = 4096)[
        "cmd"
    ]
    assert _split_flag(cmd) == "tensor"
    assert tuple(backend._auto_split_decision.gpu_indices) == (1, 2)


def _orchestrator(
    active = None,
    loading = (),
    alive = False,
):
    return types.SimpleNamespace(
        active_model_name = active, loading_models = set(loading), is_worker_alive = lambda: alive
    )


@pytest.mark.parametrize(
    "where, orchestrator, tensor",
    [
        ("primary", _orchestrator(active = "org/small"), False),
        ("primary", _orchestrator(loading = ["org/small"]), False),
        ("slot", _orchestrator(active = "org/small"), False),
        ("slot", _orchestrator(loading = ["org/small"]), False),
        ("filling", _orchestrator(loading = ["org/small"]), False),
        ("stuck", _orchestrator(alive = True), False),
        ("primary", _orchestrator(), True),
        ("slot", _orchestrator(), True),
    ],
)
def test_wired_transformers_model_keeps_layer(tmp_path, monkeypatch, where, orchestrator, tensor):
    from core.inference import model_slots
    import core.inference.orchestrator as orchestrator_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    slot = types.SimpleNamespace(orchestrator = orchestrator)
    monkeypatch.setattr(
        orchestrator_module, "_inference_backend", orchestrator if where == "primary" else None
    )
    monkeypatch.setattr(model_slots, "slots", [slot] if where == "slot" else [])
    monkeypatch.setattr(model_slots, "loading", (slot, "model") if where == "filling" else None)
    monkeypatch.setattr(model_slots, "stuck", [slot] if where == "stuck" else [])
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert (_split_flag(cmd) == "tensor") is tensor
    if not tensor:
        assert "unknown" in backend._auto_split_decision.reason


def test_module_keeps_the_real_loggers_package_without_structlog():
    # Without structlog, this module's imports must leave a loggers package in place.
    code = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path = None, target = None):\n"
        "        if name.split('.')[0] == 'structlog':\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {_TESTS_DIR!r})\n"
        "import test_split_policy\n"
        "import loggers.media_progress\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, timeout = 300
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_wired_fits_one_probe_prices_the_runtime_footprint(tmp_path, monkeypatch):
    # Fits one T4 bare, but not with CUDA context and drafter reserves.
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 13.2, names = TWO_T4_NAMES
    )
    _with_mtp_caps(backend)
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    cmd = _launch(backend, gguf, model_identifier = _MTP_MODEL, gpu_memory_mode = "auto", n_ctx = 4096)[
        "cmd"
    ]
    assert "fits one GPU" not in backend._auto_split_decision.reason
    assert _split_flag(cmd) == "tensor"


def _arbiter(residents):
    def setup(monkeypatch):
        from core.inference import gpu_arbiter
        monkeypatch.setattr(gpu_arbiter, "residents", lambda: dict(residents))

    return setup


def _primary(**kw):
    def setup(monkeypatch):
        import core.inference.orchestrator as orchestrator_module

        fake = _orchestrator(alive = kw.pop("alive", False))
        for key, value in kw.items():
            setattr(fake, key, value)
        monkeypatch.setattr(orchestrator_module, "_inference_backend", fake)

    return setup


def _slot_worker(monkeypatch):
    from core.inference import model_slots
    monkeypatch.setattr(
        model_slots, "slots", [types.SimpleNamespace(orchestrator = _orchestrator(alive = True))]
    )


def _stuck_gguf_slot(monkeypatch):
    # unload_model raised and _kill_process dropped the handle: the server may still be on a GPU.
    from core.inference import model_slots
    stuck = types.SimpleNamespace(
        llama = types.SimpleNamespace(is_active = False), orchestrator = _orchestrator()
    )
    monkeypatch.setattr(model_slots, "stuck", [stuck])


def _st_embedder(device):
    def setup(monkeypatch):
        import core.rag.embeddings as embeddings
        monkeypatch.setattr(embeddings, "_model", types.SimpleNamespace(device = device))

    return setup


def _gguf_embedder(on_gpu, *, starting = False):
    def setup(monkeypatch):
        import threading

        import core.rag.embeddings as embeddings
        from core.rag.embed_llama_server import LlamaServerBackend

        backend = object.__new__(LlamaServerBackend)
        backend._lifecycle_lock = threading.Lock()
        if starting:
            backend._lifecycle_lock.acquire()
        backend._process = types.SimpleNamespace(poll = lambda: None, pid = _FAKE_SERVER_PID)
        backend._on_gpu = on_gpu
        monkeypatch.setattr(embeddings, "_backend", backend)

    return setup


def _embedder_loading(monkeypatch):
    import threading

    import core.rag.embeddings as embeddings

    lock = threading.Lock()
    lock.acquire()
    monkeypatch.setattr(embeddings, "_lock", lock)


def _decision_worker(device):
    def setup(monkeypatch):
        import core.systemone.laya_runtime as laya
        monkeypatch.setattr(laya, "_device_name", device)

    return setup


def _segmenter(holds_vram, call_device = None):
    def setup(monkeypatch):
        import core.inference.diffusion_segment as diffusion_segment
        fake = types.SimpleNamespace(holds_vram = lambda: holds_vram, _call_device = call_device)
        monkeypatch.setattr(diffusion_segment, "_segmenter", fake)

    return setup


def _prompt_upsampler(loading, holds_vram):
    def setup(monkeypatch):
        import core.inference.flux3_prompt_upsampler as upsampler
        fake = types.SimpleNamespace(is_loading = lambda: loading, holds_vram = lambda: holds_vram)
        monkeypatch.setattr(upsampler, "_sidecar", fake)

    return setup


def _tts_codec(device):
    def setup(monkeypatch):
        fake = types.SimpleNamespace(_codec_devices = {"snac": device})
        monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", fake)

    return setup


def _worker(module, attr, **state):
    """A training / export singleton in ``module`` reporting ``state`` (method name -> result)."""

    def setup(monkeypatch):
        import importlib
        fake = types.SimpleNamespace(**{name: (lambda v = v: v) for name, v in state.items()})
        monkeypatch.setattr(importlib.import_module(module), attr, fake)

    return setup


@pytest.mark.parametrize(
    "setup, tensor",
    [
        # Arbiter residents: unknown ids may be either card; known ones are off limits even unpriced.
        pytest.param(_arbiter({"diffusion": None}), False, id = "media-unknown-gpus"),
        pytest.param(_arbiter({"decisions": None}), False, id = "decisions-unknown-gpus"),
        pytest.param(_arbiter({"video": frozenset({1})}), False, id = "media-on-gpu1"),
        pytest.param(_arbiter({"chat": None}), True, id = "chat-claim-alone"),
        pytest.param(_primary(_managed_engine = object()), False, id = "managed-engine"),
        pytest.param(_primary(alive = True), False, id = "primary-worker-alive"),
        pytest.param(_slot_worker, False, id = "kept-slot-worker-alive"),
        pytest.param(_stuck_gguf_slot, False, id = "stuck-gguf-slot"),
        pytest.param(_st_embedder("cuda:0"), False, id = "st-embedder-gpu"),
        pytest.param(_st_embedder("cpu"), True, id = "st-embedder-cpu"),
        pytest.param(_gguf_embedder(True), False, id = "gguf-embedder-gpu"),
        pytest.param(_gguf_embedder(False, starting = True), False, id = "gguf-embedder-starting"),
        pytest.param(_gguf_embedder(False), True, id = "gguf-embedder-cpu"),
        pytest.param(_embedder_loading, False, id = "embedder-loading"),
        pytest.param(_decision_worker("cuda"), False, id = "decision-worker-gpu"),
        pytest.param(_decision_worker("cpu"), True, id = "decision-worker-cpu"),
        pytest.param(_segmenter(True), False, id = "segmenter-gpu"),
        pytest.param(_segmenter(False, "cuda:0"), False, id = "segmenter-call-on-gpu"),
        pytest.param(_segmenter(False, "cpu"), True, id = "segmenter-cpu"),
        pytest.param(_prompt_upsampler(False, True), False, id = "prompt-upsampler-gpu"),
        pytest.param(_prompt_upsampler(True, False), False, id = "prompt-upsampler-loading"),
        pytest.param(_prompt_upsampler(False, False), True, id = "prompt-upsampler-cpu"),
        pytest.param(_tts_codec("cuda"), False, id = "tts-codec-gpu"),
        pytest.param(_tts_codec("cpu"), True, id = "tts-codec-cpu"),
        pytest.param(
            _worker("core.training.training", "_training_backend", is_training_active = True),
            False,
            id = "llm-training",
        ),
        pytest.param(
            _worker("core.training.training", "_training_backend", is_training_active = False),
            True,
            id = "llm-training-idle",
        ),
        pytest.param(
            _worker("core.training.diffusion_training_service", "_service", is_active = True),
            False,
            id = "diffusion-training",
        ),
        pytest.param(
            _worker(
                "core.export.orchestrator",
                "_export_backend",
                is_export_active = True,
                is_worker_alive = True,
            ),
            False,
            id = "export-running",
        ),
        pytest.param(
            _worker(
                "core.export.orchestrator",
                "_export_backend",
                is_export_active = False,
                is_worker_alive = True,
            ),
            False,
            id = "export-worker-alive",
        ),
        pytest.param(
            _worker(
                "core.export.orchestrator",
                "_export_backend",
                is_export_active = False,
                is_worker_alive = False,
            ),
            True,
            id = "export-idle",
        ),
    ],
)
def test_wired_any_resident_without_gpu_ids_keeps_layer(tmp_path, monkeypatch, setup, tensor):
    from core.inference import model_slots
    import core.inference.orchestrator as orchestrator_module

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(orchestrator_module, "_inference_backend", None)
    monkeypatch.setattr(model_slots, "slots", [])
    setup(monkeypatch)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert (_split_flag(cmd) == "tensor") is tensor


@pytest.mark.parametrize(
    "lines, rc, when",
    [
        (["GGML_ASSERT(split_axis == GGML_BACKEND_SPLIT_AXIS_0) failed"], -6, "kill"),
        (["split_mode_tensor and kv cache quantization not implemented"], 1, "kill"),
        (["boom"], 1, "health"),
    ],
    ids = ["split-axis-abort", "quantized-kv-refusal", "unhealthy-start"],
)
def test_wired_auto_tensor_exit_cancelled_reports_a_cancelled_load(
    tmp_path, monkeypatch, lines, rc, when
):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 20, names = TWO_T4_NAMES
    )
    monkeypatch.setattr(LlamaCppBackend, "_tensor_split_abort_keys", set())
    monkeypatch.setattr(LlamaCppBackend, "_tensor_quant_kv_unsupported_binaries", set())
    kill = backend._kill_process

    def kill_then_cancel(*a, **k):
        proc = backend._process
        kill(*a, **k)
        if proc is not None and not getattr(proc, "healthy", True):
            backend._cancel_event.set()

    if when == "kill":
        backend._kill_process = kill_then_cancel
        behave = _tensor_crashes(lines, rc = rc)
    else:
        crash = _tensor_crashes(lines, rc = rc)

        def behave(cmd):
            backend._cancel_event.set()
            return crash(cmd)

    result, spawns = _launch_scripted(backend, gguf, behave, gpu_memory_mode = "auto", n_ctx = 4096)
    # The route answers a False as a cancelled load; a raise would be a 500.
    assert result is False
    assert [_split_flag(c) for c in spawns] == ["tensor"]
    _assert_no_split_state(backend)


@pytest.mark.parametrize(
    "blocked",
    [
        ("core.inference.model_slots",),
        ("core.inference.model_slots", "core.inference.orchestrator"),
    ],
)
def test_wired_resident_source_that_cannot_import_holds_nothing(tmp_path, monkeypatch, blocked):
    # Minimal install without FastAPI: model_slots cannot import, so nothing it tracks is loaded.
    import core.inference as inference_package

    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delattr(inference_package, name.rsplit(".", 1)[1], raising = False)
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    assert _split_flag(cmd) == "tensor"


@pytest.mark.parametrize("child", [(0,), (1,), None])
def test_wired_unregistered_helper_backend_keeps_its_gpus(tmp_path, monkeypatch, child):
    import core.inference.llama_cpp as llama_cpp_module

    backend, gguf = _wired_backend(
        tmp_path,
        monkeypatch,
        memory = [(i, 15_360, 15_360) for i in range(3)],
        model_gib = 6,
        names = {i: T4 for i in range(3)},
    )
    assert backend in llama_cpp_module._owned_backends_snapshot()
    helper = types.SimpleNamespace(
        is_active = True, _planned_vram_mib = {}, _child_gpu_physical_ids = child
    )
    monkeypatch.setattr(llama_cpp_module, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(llama_cpp_module, "_owned_backends_snapshot", lambda: [backend, helper])
    cmd = _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)["cmd"]
    decision = backend._auto_split_decision
    if child is None:
        assert _split_flag(cmd) != "tensor" and "unknown" in decision.reason
    else:
        # The helper's card is never the pair, nor the one-GPU pick that keeps a widen off.
        expected = tuple(i for i in range(3) if i not in child)
        assert _split_flag(cmd) == "tensor" and tuple(decision.gpu_indices) == expected
