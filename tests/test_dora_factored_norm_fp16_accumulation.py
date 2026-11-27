# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DoRA factored norm (unsloth/kernels/dora.py) under torch.backends.cuda.matmul.allow_fp16_accumulation.

cuBLAS refuses float16 inputs with a float32 output when that flag is on (CUDABlas.cpp gemm_internal<at::Half,
float>: "gemm input type at::Half and output type float is not supported with allowFP16AccumulationCuBLAS"), so
the float16 cross term through torch.mm(..., out_dtype = torch.float32) crashed every float16 DoRA forward. With
the flag on, a float16 weight must take the chunked float32 GEMM; everything else keeps mm.dtype. CPU only: the
CUDA route is selected with a patched is_cuda and a spy on torch.mm."""

import importlib.util
import os
import sys

import pytest
import torch

pytest.importorskip("peft")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "kernels", "dora.py")

if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
    pytest.skip("torch has no allow_fp16_accumulation", allow_module_level = True)


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("_unsloth_kernels_dora_fp16acc", _PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    print("dora module:", m.__file__)
    return m


@pytest.fixture
def cuda_route(mod, monkeypatch):
    monkeypatch.setattr(mod, "_MM_OUT_DTYPE", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True), raising = False)
    calls = []
    real_mm = torch.mm

    def spy_mm(*args, **kwargs):
        if "out_dtype" in kwargs:
            calls.append(args[0].dtype)
            return real_mm(args[0].float(), args[1].float())
        return real_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", spy_mm)
    return calls


@pytest.mark.parametrize("w_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp16_accumulation", [False, True])
def test_mm_out_dtype_only_where_cublas_accepts_it(
    mod, cuda_route, monkeypatch, w_dtype, fp16_accumulation
):
    # Saved and restored by hand: monkeypatch's undo deletes this property on some pytest versions (9.2).
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = fp16_accumulation
    try:
        _check_mm_out_dtype(mod, cuda_route, w_dtype, fp16_accumulation)
    finally:
        matmul.allow_fp16_accumulation = previous


def _check_mm_out_dtype(mod, cuda_route, w_dtype, fp16_accumulation):
    torch.manual_seed(0)
    W = (torch.randn(64, 128) * 0.02).to(w_dtype)
    A, B = torch.randn(8, 128) / 128**0.5, torch.randn(64, 8) * 1e-2
    norm = mod.factored_dora_weight_norm(W, A, B, 2.0, torch.float32)
    fallback = w_dtype == torch.float16 and fp16_accumulation
    assert cuda_route == ([] if fallback else [w_dtype])
    ref = torch.linalg.vector_norm(W.double() + 2.0 * (B.double() @ A.double()), dim = 1)
    assert ((norm.double() - ref).abs() / ref).max().item() < 1e-5


def test_flag_read_is_compile_clean(mod):
    def f(x):
        return x * 2 if mod._fp16_accumulation() else x + 1

    torch._dynamo.reset()
    out = torch.compile(f, fullgraph = True, backend = "eager")(torch.full((2,), 3.0))
    assert out.tolist() == [4.0, 4.0]  # flag off by default
