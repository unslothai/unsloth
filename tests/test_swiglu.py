"""GPU correctness tests for unsloth.kernels.swiglu's Triton kernels.

Compares the SwiGLU forward/backward Triton kernels against a plain
PyTorch reference implementation (e * sigmoid(e) * g).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "SwiGLU Triton kernels require a CUDA GPU"
)


def _reference_forward(e, g):
    se = torch.sigmoid(e.float())
    f = (se * e.float()).to(g.dtype)
    return f * g


@pytest.fixture(params = [torch.float16, torch.bfloat16])
def dtype(request):
    return request.param


@pytest.mark.parametrize("shape", [(2, 8, 64), (4, 17, 256), (1, 1, 1024)])
def test_swiglu_forward_matches_reference(shape, dtype):
    from unsloth.kernels.swiglu import swiglu_fg_kernel

    torch.manual_seed(0)
    e = torch.randn(shape, dtype = dtype, device = "cuda")
    g = torch.randn(shape, dtype = dtype, device = "cuda")

    out = swiglu_fg_kernel(e, g)
    ref = _reference_forward(e, g)

    assert out.shape == e.shape
    assert out.dtype == e.dtype
    torch.testing.assert_close(out, ref, atol = 1e-2, rtol = 1e-2)


def test_swiglu_backward_matches_autograd(dtype):
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel

    torch.manual_seed(1)
    bsz, hd = 16, 128
    e = torch.randn((bsz, hd), dtype = dtype, device = "cuda")
    g = torch.randn((bsz, hd), dtype = dtype, device = "cuda")
    dw = torch.randn((bsz, hd), dtype = dtype, device = "cuda")

    e_ref = e.clone().requires_grad_(True)
    g_ref = g.clone().requires_grad_(True)
    h_ref = _reference_forward(e_ref, g_ref)
    h_ref.backward(dw)

    # The kernel reuses the e/g buffers to return the *other* tensor's
    # gradient: dL/dg lands where e was, dL/de lands where g was.
    h_out, grad_g, grad_e = swiglu_DWf_DW_dfg_kernel(dw.clone(), e.clone(), g.clone())

    torch.testing.assert_close(h_out, h_ref.detach(), atol = 1e-2, rtol = 1e-2)
    torch.testing.assert_close(grad_e, e_ref.grad, atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(grad_g, g_ref.grad, atol = 2e-2, rtol = 2e-2)
