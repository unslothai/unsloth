# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 runs on CUDA / ROCm or the CPU only: a model on XPU (or another non-CUDA
accelerator), an explicit non-CUDA device, or a CPU model on a host whose accelerator is not CUDA
(where the Trainer would run the batches on it) is refused before any process group or mesh."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_pg_tp_media_gloo import (  # noqa: E402,F401
    Params4bit,
    _FakeModel,
    _T,
    native_env,
)
from test_fsdp2_native_guards_gloo import _shard_device_seen, _tiny_llama  # noqa: E402


def test_xpu_model_is_refused(native_env):
    native = native_env
    with pytest.raises(RuntimeError, match = "CUDA / ROCm.*uses xpu"):
        native._shard_device(_FakeModel([_T("xpu:0"), _T("xpu:0")]))
    with pytest.raises(RuntimeError, match = "uses xpu"):
        native._shard_device(_FakeModel([_T("cpu")]), "xpu")
    with pytest.raises(RuntimeError, match = "uses hpu"):
        native._shard_device(_FakeModel([_T("cpu"), _T("hpu")]))


def test_cpu_model_on_a_non_cuda_accelerator_host_is_refused(native_env, monkeypatch):
    pytest.importorskip("transformers")
    native = native_env
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(native, "_non_cuda_accelerator", lambda: "xpu", raising = False)
    with pytest.raises(RuntimeError, match = "uses xpu"):
        native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))
    # The CPU asked for explicitly (TrainingArguments(use_cpu = True)) still runs.
    assert native._shard_device(_FakeModel([_T("cpu")]), "cpu").type == "cpu"
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    assert native._shard_device(_FakeModel([_T("cpu")])).type == "cpu"


def test_cpu_only_host_keeps_the_cpu_mesh(native_env, monkeypatch):
    pytest.importorskip("transformers")
    native = native_env
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(native, "_non_cuda_accelerator", lambda: None, raising = False)
    assert _shard_device_seen(native, monkeypatch, _tiny_llama()).type == "cpu"


def test_cpu_mesh_refuses_gpu_resident_quantized_weights(native_env, monkeypatch):
    """use_cpu = True with a 4-bit model already on the GPU: the CPU mesh does not replicate frozen
    weights, so fully_shard would move packed storage away from its quantization state."""
    native = native_env
    model = _FakeModel([_T("cuda:0"), Params4bit("cuda:0")])
    with pytest.raises(RuntimeError, match = "CPU mesh .*quantized weights are on the GPU"):
        native._shard_device(model, "cpu")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    with pytest.raises(RuntimeError, match = "CPU mesh .*quantized weights are on the GPU"):
        native._shard_device(model)
    # Unquantized GPU weights on a CPU mesh still move as before.
    assert native._shard_device(_FakeModel([_T("cuda:0")]), "cpu").type == "cpu"
