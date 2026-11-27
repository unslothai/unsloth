# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-channel FP8 GEMV for 2..16 decode rows: routing, accuracy against fp64, split-K, kill switch."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

CUDA = torch.cuda.is_available()
FP8_GPU = CUDA and torch.version.hip is None and torch.cuda.get_device_capability() >= (8, 9)
needs_fp8_gpu = pytest.mark.skipif(not FP8_GPU, reason = "needs an sm89+ NVIDIA GPU")


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


def _weights(
    N,
    K,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(N, K, device = "cuda", generator = g) * 0.02
    scale = (W.abs().amax(1, keepdim = True) / 448.0).float()
    return (W / scale).to(torch.float8_e4m3fn), scale


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize(
    "N, K, expected",
    [(4096, 4096, False), (2**31 // 4096, 4096, False), (2**31 // 4096 + 1, 4096, True)],
)
def test_rows_launch_widens_past_int32(F, monkeypatch, N, K, expected):
    rec = _Recorder()
    monkeypatch.setattr(F, "_fp8_rowwise_gemv_rows_kernel", rec)
    monkeypatch.setattr(F, "_fp8_triton_device_context", lambda *a, **k: nullcontext())
    monkeypatch.setattr(F, "_FP8_ROWS_CONFIG_OVERRIDE", (32, 128, 1, 4, 4))
    X = torch.empty(4, 1, K, device = "meta", dtype = torch.bfloat16)
    W = torch.empty(N, K, device = "meta", dtype = torch.float8_e4m3fn)
    S = torch.empty(N, 1, device = "meta", dtype = torch.float32)
    F.fp8_rowwise_gemv(X, W, S)
    assert len(rec.calls) == 1
    assert rec.calls[0][2]["LONG_INDEXING"] is expected


@pytest.mark.parametrize(
    "dtype, N, K, split",
    [
        (torch.bfloat16, 4096, 12288, 8),  # long K (down_proj)
        (torch.float16, 1024, 4096, 8),  # narrow N (kv_proj)
        (torch.bfloat16, 4096, 4096, 1),
        (torch.float16, 12288, 4096, 1),
        (torch.bfloat16, 512, 2048, 1),
    ],
)
def test_split_k_rule(F, dtype, N, K, split):
    assert F._fp8_rows_config(N, K, dtype)[2] == split


@needs_fp8_gpu
def test_routing(F, monkeypatch):
    Wq, scale = _weights(1024, 512)
    for M, ok in ((1, True), (2, True), (16, True), (17, False)):
        X = torch.randn(M, 1, 512, device = "cuda", dtype = torch.bfloat16)
        assert F.can_use_fp8_rowwise_gemv(X, Wq, scale) is ok, M
    X = torch.randn(4, 1, 512, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    assert not F.can_use_fp8_rowwise_gemv(X, Wq, scale)
    blocks = torch.ones(8, 4, device = "cuda")
    assert not F.can_use_fp8_rowwise_gemv(X.detach(), Wq, blocks)
    monkeypatch.setattr(F, "_FP8_GEMV_ROWS", False)
    assert not F.can_use_fp8_rowwise_gemv(X.detach(), Wq, scale)
    assert F.can_use_fp8_rowwise_gemv(X.detach()[:1], Wq, scale)


@needs_fp8_gpu
def test_one_row_keeps_the_single_row_kernel(F, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("rows kernel used for one row")

    monkeypatch.setattr(F, "_fp8_rowwise_gemv_rows", boom)
    Wq, scale = _weights(512, 256)
    X = torch.randn(1, 1, 256, device = "cuda", dtype = torch.bfloat16)
    F.fp8_rowwise_gemv(X, Wq, scale)


@needs_fp8_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [2, 3, 8, 16])
@pytest.mark.parametrize("N, K", [(1024, 4096), (4096, 4096), (200, 320)])
@pytest.mark.parametrize("cfg", [None, (16, 128, 8, 4, 4), (32, 256, 1, 4, 3), (64, 128, 2, 4, 4)])
def test_rows_accuracy(F, monkeypatch, dtype, M, N, K, cfg):
    monkeypatch.setattr(F, "_FP8_ROWS_CONFIG_OVERRIDE", cfg)
    Wq, scale = _weights(N, K)
    X = torch.randn(M, 1, K, device = "cuda", dtype = dtype)
    Wref = Wq.double() * scale.double()
    ref = (X.double().reshape(M, K) @ Wref.t()).reshape(M, 1, N)
    y = F.fp8_rowwise_gemv(X, Wq, scale)
    assert y.shape == (M, 1, N) and y.dtype == dtype
    # The dequantize-then-matmul path rounds W to 16 bits first; ours reads the exact fp8 values.
    dq = X @ (Wq.float() * scale).to(dtype).t()
    err, err_dq = (y.double() - ref).abs(), (dq.double() - ref).abs()
    ulp = torch.finfo(dtype).eps * ref.abs().max().item()
    assert err.max().item() <= err_dq.max().item() + ulp
    assert err.mean().item() <= err_dq.mean().item() * 1.05
    # Fixed reduction order: repeat calls are bitwise equal (split-K sums partials, no atomics).
    assert torch.equal(y, F.fp8_rowwise_gemv(X, Wq, scale))


@needs_fp8_gpu
def test_rows_through_fp8_linear_callers(F):
    from unsloth.kernels.utils import fast_linear_forward

    class P(torch.nn.Module):
        def __init__(self, Wq, scale):
            super().__init__()
            self.weight = torch.nn.Parameter(Wq, requires_grad = False)
            self.weight_scale = scale
            self.bias = None

        def forward(self, x):
            raise AssertionError("not routed")

    Wq, scale = _weights(1024, 512)
    X = torch.randn(4, 1, 512, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        out = fast_linear_forward(P(Wq, scale), X)
    ref = X.double() @ (Wq.double() * scale.double()).t()
    assert (out.double() - ref).abs().max().item() < 0.05 * ref.abs().max().item()


@needs_fp8_gpu
def test_rows_off_where_triton_miscompiles_the_gemv(F, monkeypatch):
    # Blackwell + Triton < 3.4: only the exact one-row BLOCK_N 1 launch is trusted there.
    monkeypatch.setattr(F, "_fp8_gemv_miscompiles", lambda device: True)
    monkeypatch.setattr(F, "_triton_builds_blackwell", lambda: True)
    Wq, scale = _weights(1024, 512)
    assert not F.can_use_fp8_rowwise_gemv(
        torch.randn(4, 1, 512, device = "cuda", dtype = torch.bfloat16), Wq, scale
    )
    assert F.can_use_fp8_rowwise_gemv(
        torch.randn(1, 1, 512, device = "cuda", dtype = torch.bfloat16), Wq, scale
    )
