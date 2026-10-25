# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block FP8 GEMM: the TMA descriptor kernel (datacenter Blackwell) must match the pointer kernel bit for bit."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("needs an NVIDIA CUDA device", allow_module_level = True)

from unsloth.kernels import fp8 as fp8_mod


def _inputs(
    M,
    N,
    K,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    A = torch.randn(M, K, device = "cuda", generator = g, dtype = torch.float32).to(torch.bfloat16)
    W = torch.randn(N, K, device = "cuda", generator = g) * 0.05
    nb, kb = -(-N // 128), -(-K // 128)
    Wp = torch.zeros(nb * 128, kb * 128, device = "cuda")
    Wp[:N, :K] = W
    s = Wp.view(nb, 128, kb, 128).abs().amax(dim = (1, 3)).clamp(min = 1e-12) / 448.0
    Wq = (Wp.view(nb, 128, kb, 128) / s[:, None, :, None]).view(nb * 128, kb * 128)[:N, :K]
    Wq = Wq.contiguous().to(torch.float8_e4m3fn)
    qa, sa = fp8_mod.act_quant(A, 128) if K % 128 == 0 else (None, None)
    return A, Wq, s.contiguous(), qa, sa


def _pointer(qa, Wq, sa, Ws, dtype, monkeypatch):
    monkeypatch.setattr(fp8_mod, "_FP8_TMA_GEMM", False)
    out = fp8_mod.w8a8_block_fp8_matmul_triton(qa, Wq, sa, Ws, [128, 128], output_dtype = dtype)
    monkeypatch.undo()
    return out


def _blackwell():
    return torch.cuda.get_device_capability()[0] == 10


@pytest.mark.skipif(not _blackwell(), reason = "TMA block FP8 GEMM only routes on sm_100")
@pytest.mark.parametrize(
    "M, N, K",
    [
        (8192, 4096, 4096),
        (200, 256, 512),
        (1000, 6144, 4096),
        (4096, 4160, 1024),
        (300, 1024, 14336),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tma_matches_pointer_kernel_bitwise(M, N, K, dtype, monkeypatch):
    _, Wq, Ws, qa, sa = _inputs(M, N, K)
    calls = []
    real = fp8_mod._launch_fp8_tma_gemm
    monkeypatch.setattr(
        fp8_mod, "_launch_fp8_tma_gemm", lambda *a: (calls.append(a[5]), real(*a))[1]
    )
    out = fp8_mod.w8a8_block_fp8_matmul_triton(qa, Wq, sa, Ws, [128, 128], output_dtype = dtype)
    monkeypatch.undo()
    assert (
        M in calls
    ), "TMA path did not run for this GEMM"  # a[5] is M; the self check uses M = 200
    assert fp8_mod._FP8_TMA_GEMM, "self check disabled the TMA path"
    ref = _pointer(qa, Wq, sa, Ws, dtype, monkeypatch)
    bits = {2: torch.int16, 4: torch.int32}[out.element_size()]
    assert torch.equal(out.view(bits), ref.view(bits))


@pytest.mark.skipif(not _blackwell(), reason = "TMA block FP8 GEMM only routes on sm_100")
def test_kill_switch_and_small_m_use_the_pointer_kernel(monkeypatch):
    _, Wq, Ws, qa, sa = _inputs(256, 256, 256)
    calls = []
    monkeypatch.setattr(fp8_mod, "_launch_fp8_tma_gemm", lambda *a: calls.append(1))
    monkeypatch.setattr(fp8_mod, "_FP8_TMA_GEMM", False)
    fp8_mod.w8a8_block_fp8_matmul_triton(qa, Wq, sa, Ws, [128, 128], output_dtype = torch.bfloat16)
    monkeypatch.setattr(fp8_mod, "_FP8_TMA_GEMM", True)
    _, Wq, Ws, qa, sa = _inputs(64, 256, 256)  # M < 128 -> 64-row tiles, pointer kernel
    fp8_mod.w8a8_block_fp8_matmul_triton(qa, Wq, sa, Ws, [128, 128], output_dtype = torch.bfloat16)
    assert calls == []


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 9), (9, 0), (12, 0)])
def test_other_archs_never_route_to_tma(capability, monkeypatch):
    fp8_mod._fp8_tma_gemm_arch_at.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capability)
    try:
        assert fp8_mod._fp8_tma_gemm_arch_at(0) is False
    finally:
        fp8_mod._fp8_tma_gemm_arch_at.cache_clear()
