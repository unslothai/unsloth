"""GPU correctness tests for unsloth.kernels.geglu's Triton kernels.

Compares the exact and tanh-approx GeGLU forward/backward Triton kernels
against a plain PyTorch reference implementation.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "GeGLU Triton kernels require a CUDA GPU"
)


def _reference_exact_forward(gate, up):
    f = 0.5 * gate.float() * (1.0 + torch.erf(gate.float() * (1.0 / math.sqrt(2.0))))
    return f.to(up.dtype) * up


def _reference_approx_forward(gate, up):
    x = gate.float()
    s = math.sqrt(2.0 / math.pi)
    f = 0.5 * x * (1.0 + torch.tanh(s * x * (1.0 + 0.044715 * x * x)))
    return f.to(up.dtype) * up


@pytest.fixture(params = [torch.float16, torch.bfloat16])
def dtype(request):
    return request.param


@pytest.mark.parametrize("shape", [(2, 8, 64), (4, 17, 256), (1, 1, 1024)])
def test_geglu_exact_forward_matches_reference(shape, dtype):
    from unsloth.kernels.geglu import geglu_exact_forward_kernel

    torch.manual_seed(0)
    gate = torch.randn(shape, dtype = dtype, device = "cuda")
    up = torch.randn(shape, dtype = dtype, device = "cuda")

    out = geglu_exact_forward_kernel(gate, up)
    ref = _reference_exact_forward(gate, up)

    assert out.shape == gate.shape
    assert out.dtype == gate.dtype
    torch.testing.assert_close(out, ref, atol = 1e-2, rtol = 1e-2)


@pytest.mark.parametrize("shape", [(2, 8, 64), (4, 17, 256)])
def test_geglu_approx_forward_matches_reference(shape, dtype):
    from unsloth.kernels.geglu import geglu_approx_forward_kernel

    torch.manual_seed(1)
    gate = torch.randn(shape, dtype = dtype, device = "cuda")
    up = torch.randn(shape, dtype = dtype, device = "cuda")

    out = geglu_approx_forward_kernel(gate, up)
    ref = _reference_approx_forward(gate, up)

    assert out.shape == gate.shape
    assert out.dtype == gate.dtype
    torch.testing.assert_close(out, ref, atol = 1e-2, rtol = 1e-2)


def test_geglu_exact_backward_matches_autograd(dtype):
    from unsloth.kernels.geglu import geglu_exact_backward_kernel

    torch.manual_seed(2)
    bsz, hd = 16, 128
    e = torch.randn((bsz, hd), dtype = dtype, device = "cuda")
    g = torch.randn((bsz, hd), dtype = dtype, device = "cuda")
    dw = torch.randn((bsz, hd), dtype = dtype, device = "cuda")

    e_ref = e.clone().requires_grad_(True)
    g_ref = g.clone().requires_grad_(True)
    h_ref = _reference_exact_forward(e_ref, g_ref)
    h_ref.backward(dw)

    # The kernel reuses the e/g buffers to return the *other* tensor's
    # gradient: dL/dg lands where e was, dL/de lands where g was.
    h_out, grad_g, grad_e = geglu_exact_backward_kernel(dw.clone(), e.clone(), g.clone())

    torch.testing.assert_close(h_out, h_ref.detach(), atol = 1e-2, rtol = 1e-2)
    torch.testing.assert_close(grad_e, e_ref.grad, atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(grad_g, g_ref.grad, atol = 2e-2, rtol = 2e-2)


def test_geglu_approx_backward_matches_autograd(dtype):
    from unsloth.kernels.geglu import geglu_approx_backward_kernel

    torch.manual_seed(3)
    bsz, hd = 16, 128
    e = torch.randn((bsz, hd), dtype = dtype, device = "cuda")
    g = torch.randn((bsz, hd), dtype = dtype, device = "cuda")
    dw = torch.randn((bsz, hd), dtype = dtype, device = "cuda")

    e_ref = e.clone().requires_grad_(True)
    g_ref = g.clone().requires_grad_(True)
    h_ref = _reference_approx_forward(e_ref, g_ref)
    h_ref.backward(dw)

    h_out, grad_g, grad_e = geglu_approx_backward_kernel(dw.clone(), e.clone(), g.clone())

    torch.testing.assert_close(h_out, h_ref.detach(), atol = 1e-2, rtol = 1e-2)
    torch.testing.assert_close(grad_e, e_ref.grad, atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(grad_g, g_ref.grad, atol = 2e-2, rtol = 2e-2)
