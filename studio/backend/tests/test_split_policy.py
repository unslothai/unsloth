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

from test_llama_cpp_mtp_detection import _MTP_MODEL, _flags_dict, _resolver_backend  # noqa: E402
from test_llama_cpp_placement import _REAL_POPEN, _backend, _launch  # noqa: E402

from core.inference import split_policy as sp  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

GIB = 1024**3
T4 = "Tesla T4"
NO_ENV: dict = {}


def _gpus(*specs):
    """specs: (name, total_mib, free_mib) per GPU, indexed 0.."""
    return [
        sp.GpuInfo(index = i, name = n, total_mib = t, free_mib = f) for i, (n, t, f) in enumerate(specs)
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


# --------------------------------------------------------------------------
# 1. Decision table
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# 2. MTP depth table
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# 3. Wiring through load_model (faked GPUs, fake llama-server)
# --------------------------------------------------------------------------

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
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._get_gguf_size_bytes = lambda _path: int(model_gib * GIB)
    backend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB = 256
    backend._gpu_names_by_physical_id = lambda: dict(names)
    # The measured platform (Linux + NVIDIA CUDA) and no utilisation reading, unless a cell says otherwise.
    backend._auto_tensor_platform_reason = lambda: None
    backend._gpu_utilization_by_physical_id = lambda: {}
    real_probe = LlamaCppBackend.probe_server_capabilities

    def probe(binary = None):
        caps = dict(real_probe(binary))
        caps["flags"] = dict(caps.get("flags") or {}, **(_TP_HELP if tensor_help else {}))
        return caps

    backend.probe_server_capabilities = probe
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_TENSOR_SPLIT", raising = False)
    monkeypatch.delenv(sp.AUTO_TENSOR_SPLIT_ENV, raising = False)
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
    # The head must really run: a binary with draft-mtp, so the launch emits it.
    _with_mtp_caps(backend)
    monkeypatch.setattr(LlamaCppBackend, "n_moe_layers", property(lambda self: 40))
    cmd = _launch(backend, gguf, model_identifier = _MTP_MODEL, gpu_memory_mode = "auto", n_ctx = 4096)[
        "cmd"
    ]
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert backend._auto_split_decision.mode == "tensor"
    assert "MoE" in backend._auto_split_decision.reason


# --------------------------------------------------------------------------
# 4. Risk gates: idle measurement, the 2-GPU cap, capability probe, platform, P2P
# --------------------------------------------------------------------------


def test_busy_by_utilisation_blocks_the_pair():
    gpus = [
        sp.GpuInfo(0, T4, 15_360, 15_000, util_pct = 0.0),
        sp.GpuInfo(1, T4, 15_360, 15_000, util_pct = 85.0),
    ]
    d = _decide(gpus = gpus)
    assert d.mode == "layer" and "busy" in d.reason


def test_unknown_utilisation_falls_back_to_the_memory_check():
    gpus = [sp.GpuInfo(i, T4, 15_360, 15_000, util_pct = None) for i in range(2)]
    assert _decide(gpus = gpus).mode == "tensor"


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
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
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


def test_wired_other_studio_model_on_gpu1_blocks_the_pair(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    # A small model another Studio backend serves on GPU 1: below the free-memory bar on its own.
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
    # The launch gate ran on the auto pair itself.
    assert [0, 1] in seen


# --------------------------------------------------------------------------
# 5. Auto only engages or leaves the load as it was: review regressions
# --------------------------------------------------------------------------


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
    # A model that fits one T4: a recorded tensor refusal / abort must not force it onto two GPUs.
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
    # Defence in depth: a downgrade that fires after auto chose tensor puts the pool back.
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
    # Pricing the pair below the tensor budget must not leave a 3-GPU layer fit on 2 GPUs + CPU.
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
    "var, value", [("LLAMA_ARG_N_GPU_LAYERS", "20"), ("LLAMA_ARG_DEVICE", "CUDA0")]
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
    # A small drafter, so the pooled tensor budget is not what keeps the layer split.
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
    # 16 GiB on disk, 8 GiB of it NextN blocks llama.cpp skips without draft-mtp: fits one T4.
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 16, names = TWO_T4_NAMES
    )
    backend._unloaded_nextn_bytes = lambda _path: 8 * GIB
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096, speculative_type = "off")
    assert backend._auto_split_decision.mode == "tensor"
    assert "fits one GPU" in backend._auto_split_decision.reason


def test_wired_p2p_diagnostic_reads_the_pair_that_runs(tmp_path, monkeypatch):
    # GPUs 2 and 3 have the most free memory: the diagnostic asks about them, not all four.
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
    # A request that now asks for something else is still a different load.
    assert not backend._runtime_matches_intent(
        dataclasses.replace(intent, extra_args = ["--split-mode", "layer"]), ["--split-mode", "layer"]
    )


def test_status_and_load_report_an_auto_tensor_split_as_toggle_off(tmp_path, monkeypatch):
    from routes.inference import _llama_runtime_fields

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
                "pid": 123,
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
    # The retry is the load without the policy: same as the kill switch would launch.
    assert not backend._auto_tensor_suppressed


def test_wired_unload_clears_the_auto_decision(tmp_path, monkeypatch):
    backend, gguf = _wired_backend(
        tmp_path, monkeypatch, memory = TWO_T4_MEM, model_gib = 6, names = TWO_T4_NAMES
    )
    _launch(backend, gguf, gpu_memory_mode = "auto", n_ctx = 4096)
    assert backend._auto_split_decision is not None and backend._auto_tensor_engaged
    backend.unload_model()
    assert backend._auto_split_decision is None
    assert backend._auto_tensor_engaged is False


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
