# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data-center llama.cpp env tuning.

FP32 cuBLAS accumulation (GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F) plus, for
multi-GPU, peer-to-peer (GGML_CUDA_P2P) and larger launch queues
(CUDA_SCALE_LAUNCH_QUEUES) must be set only for datacenter/professional NVIDIA
GPUs (A100/H100/H200/B200/GB200/GB300/L40/RTX PRO 6000 ...), never for consumer
GeForce, AMD/ROCm, CPU or macOS, where FP32-accum carries a real throughput
cost. A user-supplied value must always win, and UNSLOTH_DISABLE_DC_TUNING=1
must turn the whole thing off.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(
    names,
    *,
    hip = None,
    cuda_ok = True,
):
    """A torch stub exposing only what _is_datacenter_gpu / _effective_gpu_count
    touch: torch.version.hip, torch.cuda.is_available/device_count and
    get_device_properties(i).name."""
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_ok,
        device_count = lambda: len(names),
        get_device_properties = lambda i: types.SimpleNamespace(name = names[i]),
    )
    return t


# ---------------------------------------------------------------------------
# _is_datacenter_gpu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names,expected",
    [
        # Datacenter / professional parts (real torch device names, lower/mixed case).
        (["NVIDIA A100-SXM4-80GB"], True),
        (["NVIDIA A30"], True),
        (["NVIDIA H100 80GB HBM3"], True),
        (["NVIDIA H200"], True),
        (["NVIDIA H800"], True),
        (["NVIDIA GH200 480GB"], True),
        (["NVIDIA B200"], True),
        (["NVIDIA GB200"], True),
        (["NVIDIA L40S"], True),
        (["NVIDIA L4"], True),
        (["NVIDIA RTX PRO 6000 Blackwell Server Edition"], True),
        (["NVIDIA RTX 6000 Ada Generation"], True),
        # Consumer GeForce: never.
        (["NVIDIA GeForce RTX 4090"], False),
        (["NVIDIA GeForce RTX 5090"], False),
        (["NVIDIA GeForce RTX 3090"], False),
        (["NVIDIA GeForce RTX 2080 Ti"], False),
        (["NVIDIA GeForce GTX 1080"], False),
        # Homogeneous multi-DC: all must match.
        (["NVIDIA B200", "NVIDIA B200"], True),
        (["NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3"], True),
        # Mixed DC + consumer: treated as non-DC so tuning never lands on GeForce.
        (["NVIDIA B200", "NVIDIA GeForce RTX 4090"], False),
        (["NVIDIA GeForce RTX 4090", "NVIDIA B200"], False),
    ],
)
def test_is_datacenter_gpu(monkeypatch, names, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(names))
    assert LlamaCppBackend._is_datacenter_gpu() is expected


def test_is_datacenter_gpu_respects_selection(monkeypatch):
    # A mixed box where only the DC GPU is selected -> True; only consumer -> False.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(["NVIDIA B200", "NVIDIA GeForce RTX 4090"]),
    )
    assert LlamaCppBackend._is_datacenter_gpu([0]) is True
    assert LlamaCppBackend._is_datacenter_gpu([1]) is False
    assert LlamaCppBackend._is_datacenter_gpu([0, 1]) is False


def test_is_datacenter_gpu_out_of_range_indices_skipped(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"]))
    # Out-of-range / negative indices are skipped; the one valid DC GPU still wins.
    assert LlamaCppBackend._is_datacenter_gpu([0, 5, -1]) is True
    # Only invalid indices -> nothing seen -> False (fail closed for the flag).
    assert LlamaCppBackend._is_datacenter_gpu([5, 9]) is False


def test_is_datacenter_gpu_rocm_is_false(monkeypatch):
    # ROCm reuses torch.cuda.*; even an MI300X-named part must not qualify here.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(["AMD Instinct MI300X"], hip = "6.2.0"),
    )
    assert LlamaCppBackend._is_datacenter_gpu() is False


def test_is_datacenter_gpu_no_cuda_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], cuda_ok = False))
    assert LlamaCppBackend._is_datacenter_gpu() is False


def test_is_datacenter_gpu_missing_torch_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._is_datacenter_gpu() is False


# ---------------------------------------------------------------------------
# _effective_gpu_count
# ---------------------------------------------------------------------------


def test_effective_gpu_count_explicit_selection(monkeypatch):
    # Explicit selection: length of the list, regardless of how many are visible.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    assert LlamaCppBackend._effective_gpu_count([0]) == 1
    assert LlamaCppBackend._effective_gpu_count([0, 1, 2]) == 3


def test_effective_gpu_count_none_uses_visible(monkeypatch):
    # None == "use every visible GPU" -> visible device count.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    assert LlamaCppBackend._effective_gpu_count(None) == 4


def test_effective_gpu_count_no_cuda_is_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], cuda_ok = False))
    assert LlamaCppBackend._effective_gpu_count(None) == 0


def test_effective_gpu_count_missing_torch_is_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._effective_gpu_count(None) == 0


# ---------------------------------------------------------------------------
# _apply_datacenter_env (the env-injection decision)
# ---------------------------------------------------------------------------


def test_apply_env_single_dc_gpu_sets_only_fp32(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"]))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0]) is True
    assert env == {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"}
    # No multi-GPU flags on a single GPU.
    assert "GGML_CUDA_P2P" not in env
    assert "CUDA_SCALE_LAUNCH_QUEUES" not in env


def test_apply_env_multi_dc_gpu_sets_all(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"
    assert env["GGML_CUDA_P2P"] == "1"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"


def test_apply_env_none_indices_uses_visible_count(monkeypatch):
    # gpu_indices None on a 2x DC box -> multi-GPU flags applied.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100", "NVIDIA H100"]))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, None) is True
    assert env["GGML_CUDA_P2P"] == "1"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"


def test_apply_env_consumer_gpu_is_noop(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA GeForce RTX 4090"] * 2))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is False
    assert env == {}


def test_apply_env_user_value_wins(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    env = {
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "0",  # user explicitly disabled
        "CUDA_SCALE_LAUNCH_QUEUES": "8x",  # user override
    }
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    # setdefault must not clobber user-provided values.
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "0"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "8x"
    # The one the user didn't set still gets the default.
    assert env["GGML_CUDA_P2P"] == "1"


def test_apply_env_disable_flag_respected(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_TUNING", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is False
    assert env == {}


def test_apply_env_fail_open_on_detection_error(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", None)  # detection raises -> False
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0]) is False
    assert env == {}
