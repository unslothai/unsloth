# SPDX-License-Identifier: AGPL-3.0-only
# Rowwise FP8 must run where FBGEMM has no kernel (sm120 raises "cutlass cannot initialize").
import os

import pytest

torch = pytest.importorskip("torch")

cuda_available = torch.cuda.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
dev = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

pytestmark = pytest.mark.skipif(
    not ((cuda_available and torch.cuda.get_device_capability() >= (8, 9)) or xpu_available),
    reason = "FP8 GEMMs need CUDA sm89+ or XPU",
)


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


def _rowwise_weight(N, K):
    w = torch.randn(N, K, device = dev) * 0.02
    scale = (w.abs().amax(1, keepdim = True) / 448).float()
    return (w / scale).to(torch.float8_e4m3fn), scale


def _backends(F):
    device = torch.device(dev, 0)
    ones = lambda n: torch.ones(n, dtype = torch.float32, device = device)
    out = ["dequant"]
    if F._rowwise_gemm_works(
        lambda x: torch._scaled_mm(
            x, x.t(), scale_a = ones((128, 1)), scale_b = ones((1, 128)), out_dtype = torch.bfloat16
        ),
        device,
    ):
        out.append("scaled_mm")
    if F._fp8_rowwise_backend(device) == "fbgemm":
        out.append("fbgemm")
    return out


def test_backend_is_one_that_runs_here(F):
    backend = F._fp8_rowwise_backend(torch.device(dev, 0))
    assert backend in ("fbgemm", "scaled_mm", "dequant")
    X = torch.randn(64, 256, device = dev, dtype = torch.bfloat16)
    w, s = _rowwise_weight(512, 256)
    y = F.fp8_linear(X, w, s)
    ref = X.float() @ (w.float() * s).t()
    assert torch.isfinite(y).all()
    assert ((y.float() - ref).norm() / ref.norm()) < 0.05


@pytest.mark.parametrize("backend", ["dequant", "scaled_mm", "fbgemm"])
def test_every_backend_matches_reference(F, backend):
    if backend not in _backends(F):
        pytest.skip(f"{backend} is not usable on this GPU")
    torch.manual_seed(0)
    w, s = _rowwise_weight(768, 512)
    X = torch.randn(2, 48, 512, device = dev, dtype = torch.bfloat16, requires_grad = True)
    # FbgemmFp8Linear stores its bias in float32.
    bias = torch.randn(768, device = dev, dtype = torch.float32)
    y = F.FbgemmFp8Linear_matmul.apply(X, w, s, bias, backend)
    W = w.float() * s
    ref = X.detach().float() @ W.t() + bias.float()
    assert y.shape == (2, 48, 768) and y.dtype == torch.bfloat16
    assert ((y.float() - ref).norm() / ref.norm()) < 0.05
    y.float().sum().backward()
    ref_dx = torch.ones(2, 48, 768, device = dev) @ W
    assert ((X.grad.float() - ref_dx).norm() / ref_dx.norm()) < 0.01
    assert (
        F.FbgemmFp8Linear_matmul.apply(X[:1, :1].detach(), w, s, bias, backend).dtype
        == torch.bfloat16
    )


def test_quantize_matches_fbgemm(F):
    if not hasattr(torch.ops.fbgemm, "quantize_fp8_per_row"):
        pytest.skip("FBGEMM not installed")
    torch.manual_seed(0)
    x = torch.randn(256, 1024, device = dev, dtype = torch.bfloat16)
    x[3] *= 1000
    x[5] = 0
    for scale_ub in (None, torch.tensor([30.0], device = dev)):
        q_ref, s_ref = torch.ops.fbgemm.quantize_fp8_per_row(x, scale_ub = scale_ub)
        q, s = F._quantize_fp8_per_row(x, scale_ub)
        # FBGEMM's fast-math division can land 1 ulp lower, flipping exact rounding ties by one FP8 step.
        torch.testing.assert_close(s.view(-1), s_ref, rtol = 1e-6, atol = 0)
        assert (q.float() != q_ref.float()).float().mean() < 1e-3
        steps = (q.float() - q_ref.float()).abs() / q_ref.float().abs().clamp(min = 2**-9)
        assert steps.max() <= 0.125
