# SPDX-License-Identifier: AGPL-3.0-only
# Rowwise FP8 (FbgemmFp8Linear) must run where FBGEMM has no kernel (sm120 raises "cutlass cannot initialize").
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason = "FP8 GEMMs need CUDA sm89+",
)


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


def _rowwise_weight(N, K):
    w = torch.randn(N, K, device = "cuda") * 0.02
    scale = (w.abs().amax(1, keepdim = True) / 448).float()
    return (w / scale).to(torch.float8_e4m3fn), scale


def _backends(F):
    out = ["dequant", "scaled_mm"]
    if hasattr(torch.ops.fbgemm, "f8f8bf16_rowwise"):
        out.append("fbgemm")
    return out


def test_backend_is_one_that_runs_here(F):
    backend = F._fp8_rowwise_backend(torch.device("cuda", torch.cuda.current_device()))
    assert backend in ("fbgemm", "scaled_mm", "dequant")
    X = torch.randn(64, 256, device = "cuda", dtype = torch.bfloat16)
    w, s = _rowwise_weight(512, 256)
    y = F.fp8_linear(X, w, s)
    ref = X.float() @ (w.float() * s).t()
    assert torch.isfinite(y).all()
    assert ((y.float() - ref).norm() / ref.norm()) < 0.05


@pytest.mark.parametrize("backend", ["dequant", "scaled_mm", "fbgemm"])
def test_every_backend_matches_reference(F, backend):
    if backend not in _backends(F):
        pytest.skip("FBGEMM not installed")
    if (
        backend != "dequant"
        and F._fp8_rowwise_backend(torch.device("cuda", torch.cuda.current_device())) == "dequant"
    ):
        pytest.skip("no FP8 GEMM on this GPU")
    if (
        backend == "fbgemm"
        and F._fp8_rowwise_backend(torch.device("cuda", torch.cuda.current_device())) != "fbgemm"
    ):
        pytest.skip("FBGEMM has no kernel for this GPU")
    torch.manual_seed(0)
    w, s = _rowwise_weight(768, 512)
    X = torch.randn(2, 48, 512, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    bias = torch.randn(768, device = "cuda", dtype = torch.bfloat16)
    y = F.FbgemmFp8Linear_matmul.apply(X, w, s, bias, backend)
    W = w.float() * s
    ref = X.detach().float() @ W.t() + bias.float()
    assert y.shape == (2, 48, 768) and y.dtype == torch.bfloat16
    assert ((y.float() - ref).norm() / ref.norm()) < 0.05
    y.float().sum().backward()
    ref_dx = torch.ones(2, 48, 768, device = "cuda") @ W
    assert ((X.grad.float() - ref_dx).norm() / ref_dx.norm()) < 0.01


def test_quantize_matches_fbgemm(F):
    if not hasattr(torch.ops.fbgemm, "quantize_fp8_per_row"):
        pytest.skip("FBGEMM not installed")
    torch.manual_seed(0)
    x = torch.randn(256, 1024, device = "cuda", dtype = torch.bfloat16)
    x[3] *= 1000
    x[5] = 0
    for scale_ub in (None, torch.tensor([30.0], device = "cuda")):
        q_ref, s_ref = torch.ops.fbgemm.quantize_fp8_per_row(x, scale_ub = scale_ub)
        q, s = F._quantize_fp8_per_row(x, scale_ub)
        # FBGEMM's fast-math division can land 1 ulp lower, flipping exact rounding ties by one FP8 step.
        torch.testing.assert_close(s.view(-1), s_ref, rtol = 1e-6, atol = 0)
        assert (q.float() != q_ref.float()).float().mean() < 1e-3
        steps = (q.float() - q_ref.float()).abs() / q_ref.float().abs().clamp(min = 2**-9)
        assert steps.max() <= 0.125
