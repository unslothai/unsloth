# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""With torchao selected, fp16 small-M decode keeps the fused kernel; bf16 goes to torchao (CPU recorders)."""

import types

import pytest
import torch

from unsloth.kernels import fp8 as F8


class _CudaLike(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _operands(
    M,
    dtype,
    N = 256,
    K = 512,
):
    X = torch.randn(M, K).to(dtype).as_subclass(_CudaLike)
    weight = torch.zeros(N, K, dtype = torch.float8_e4m3fn)
    scale = torch.ones(N // 128, K // 128, dtype = torch.float32)
    return X, weight, scale


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def small_m(X, weight, weight_scale):
        seen.append("small_m")
        return torch.zeros(X.shape[:-1] + (weight.shape[0],), dtype = X.dtype)

    def act_quant(X, block):
        seen.append("act_quant")
        x = torch.Tensor.as_subclass(X, torch.Tensor)
        return x.to(torch.float8_e4m3fn), torch.ones(x.shape[0], x.shape[-1] // block)

    def gemm(name):
        def run(
            A,
            B,
            As,
            Bs,
            *args,
            output_dtype = torch.bfloat16,
            **kwargs,
        ):
            seen.append(name)
            return torch.zeros(A.shape[:-1] + (B.shape[0],), dtype = output_dtype)

        return run

    triton_gemm = gemm("triton_gemm")
    monkeypatch.setattr(F8, "_FP8_BLOCK_SMALL_M", True)
    monkeypatch.setattr(F8, "fp8_block_small_m_linear", small_m)
    monkeypatch.setattr(F8, "act_quant", act_quant)
    monkeypatch.setattr(F8, "_fp8_kernel_unsupported", lambda *a, **k: False)
    monkeypatch.setattr(F8, "_fp8_triton_device_context", lambda *a, **k: _NullContext())
    monkeypatch.setattr(F8, "_prefer_triton_block_matmul_at", lambda device, M: False)
    monkeypatch.setattr(F8, "w8a8_block_fp8_matmul_triton", triton_gemm)
    monkeypatch.setattr(
        F8,
        "torchao_blockwise_gemm",
        lambda a, a_s, b, b_s, block_size: gemm("torchao")(a, b, a_s, b_s),
    )
    return seen


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _select(monkeypatch, backend):
    target = F8.torchao_block_matmul if backend == "torchao" else F8.w8a8_block_fp8_matmul_triton
    monkeypatch.setattr(F8, "fp8_block_matmul", target)


def _forward(X, weight, scale):
    ctx = types.SimpleNamespace()
    return F8.FP8BlockQuantLinear.forward(ctx, X, weight, scale)


@pytest.mark.parametrize("M", [1, 4, 16])
def test_fp16_decode_stays_fused_with_torchao(monkeypatch, calls, M):
    _select(monkeypatch, "torchao")
    X, weight, scale = _operands(M, torch.float16)
    assert F8.can_use_fp8_block_small_m(X, weight, scale, [128, 128])
    out = _forward(X, weight, scale)
    assert calls == ["small_m"]
    assert out.dtype == torch.float16 and out.shape == (M, weight.shape[0])


@pytest.mark.parametrize("M", [1, 4, 16])
def test_bf16_decode_still_goes_to_torchao(monkeypatch, calls, M):
    _select(monkeypatch, "torchao")
    X, weight, scale = _operands(M, torch.bfloat16)
    assert not F8.can_use_fp8_block_small_m(X, weight, scale, [128, 128])
    _forward(X, weight, scale)
    assert calls == ["act_quant", "torchao"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_selected_uses_fused_for_both(monkeypatch, calls, dtype):
    _select(monkeypatch, "triton")
    X, weight, scale = _operands(2, dtype)
    _forward(X, weight, scale)
    assert calls == ["small_m"]


@pytest.mark.parametrize("backend", ["torchao", "triton"])
def test_fp16_large_m_and_kill_switch_keep_two_launches(monkeypatch, calls, backend):
    _select(monkeypatch, backend)
    X, weight, scale = _operands(F8._SMALL_M_MAX + 1, torch.float16)
    _forward(X, weight, scale)
    assert calls == ["act_quant", "triton_gemm"]
    calls.clear()
    monkeypatch.setattr(F8, "_FP8_BLOCK_SMALL_M", False)
    X, weight, scale = _operands(1, torch.float16)
    _forward(X, weight, scale)
    assert calls == ["act_quant", "triton_gemm"]


def test_fp16_decode_stays_fused_in_small_m_first_route(monkeypatch, calls):
    _select(monkeypatch, "torchao")
    monkeypatch.setattr(F8, "can_use_fp8_block_gemv", lambda *a: False)
    monkeypatch.setattr(F8, "fp8_torch_block_quant_forward", lambda *a: calls.append("autograd"))
    X, weight, scale = _operands(1, torch.float16)
    with torch.no_grad():
        F8._fp8_block_small_m_first(X, weight, scale)
    assert calls == ["small_m"]
    calls.clear()
    X, weight, scale = _operands(1, torch.bfloat16)
    with torch.no_grad():
        F8._fp8_block_small_m_first(X, weight, scale)
    assert calls == ["autograd"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemv_keeps_its_calls_when_torchao_is_selected(monkeypatch, calls, dtype):
    _select(monkeypatch, "torchao")
    monkeypatch.setattr(F8, "can_use_fp8_block_gemv", lambda *a: True)
    monkeypatch.setattr(F8, "fp8_block_gemv", lambda *a: calls.append("gemv"))
    X, weight, scale = _operands(4, dtype)
    with torch.no_grad():
        F8._fp8_block_small_m_first(X, weight, scale)
    assert calls == ["gemv"]


def test_triton_selected_keeps_fused_before_gemv(monkeypatch, calls):
    _select(monkeypatch, "triton")
    monkeypatch.setattr(F8, "can_use_fp8_block_gemv", lambda *a: True)
    monkeypatch.setattr(F8, "fp8_block_gemv", lambda *a: calls.append("gemv"))
    X, weight, scale = _operands(4, torch.float16)
    with torch.no_grad():
        F8._fp8_block_small_m_first(X, weight, scale)
    assert calls == ["small_m"]
