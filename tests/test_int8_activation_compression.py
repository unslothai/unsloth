"""CPU tests for INT8 LoRA activation compression (UNSLOTH_QUANTIZE_ACTIVATIONS).

Covers the building blocks that don't need a GPU:
  * per-row quant/dequant round-trip error stays within the INT8 step
  * _Int8LoRALinear forward is exact (only the *stored* activation is quantised)
  * _Int8LoRALinear gradients match a plain Linear reference within tolerance
"""

import pytest

import unsloth  # noqa: F401  # runs unsloth/import_fixes.py

torch = pytest.importorskip("torch")
ku = pytest.importorskip("unsloth.kernels.utils")


def test_quant_dequant_roundtrip_error_bounded():
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype = torch.float32)
    x_q, scale = ku.quant_act(x)

    assert x_q.dtype == torch.int8
    assert scale.shape == (32, 1)  # one scale per row

    recovered = ku.dequant_act(x_q, scale)
    # Symmetric per-row INT8 => max error is half a step = amax / 127 / 2.
    step = x.abs().amax(dim = -1, keepdim = True) / 127.0
    assert torch.all((x - recovered).abs() <= step + 1e-6)


def test_int8_linear_forward_is_exact():
    torch.manual_seed(1)
    x = torch.randn(8, 16, dtype = torch.float32, requires_grad = True)
    weight = torch.randn(32, 16, dtype = torch.float32, requires_grad = True)

    out = ku._Int8LoRALinear.apply(x, weight)
    ref = x @ weight.T
    # Forward uses full-precision x (quantisation only affects the saved tensor).
    assert torch.allclose(out, ref, atol = 1e-5, rtol = 1e-5)


def test_int8_linear_backward_matches_reference():
    torch.manual_seed(2)
    x = torch.randn(8, 16, dtype = torch.float32)
    weight = torch.randn(32, 16, dtype = torch.float32)

    x_a = x.clone().requires_grad_(True)
    w_a = weight.clone().requires_grad_(True)
    x_b = x.clone().requires_grad_(True)
    w_b = weight.clone().requires_grad_(True)

    grad_out = torch.randn(8, 32, dtype = torch.float32)

    # Reference: plain linear (no quantisation).
    (x_b @ w_b.T).backward(grad_out)
    # INT8 path.
    ku._Int8LoRALinear.apply(x_a, w_a).backward(grad_out)

    # grad_x uses the exact weight, so it should match closely.
    assert torch.allclose(x_a.grad, x_b.grad, atol = 1e-4, rtol = 1e-4)
    # grad_weight uses the dequantised activation, so allow the INT8 error band.
    assert torch.allclose(w_a.grad, w_b.grad, atol = 1e-1, rtol = 5e-2)
