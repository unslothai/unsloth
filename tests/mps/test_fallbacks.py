import sys
import types
from importlib.machinery import ModuleSpec


# 1. Perfect Mocks to isolate from Triton/BNB on Windows
def perfect_mock(name):
    m = types.ModuleType(name)
    m.__spec__ = ModuleSpec(name, None)
    m.__file__ = f"{name}.py"
    m.__path__ = []
    # If it's bitsandbytes, add some expected attributes
    if name == "bitsandbytes":
        m.__version__ = "0.42.0"
        m.functional = types.ModuleType("bitsandbytes.functional")
        m.functional.__spec__ = ModuleSpec("bitsandbytes.functional", None)
    return m


sys.modules["triton"] = perfect_mock("triton")
sys.modules["triton.language"] = perfect_mock("triton.language")
sys.modules["triton.jit"] = perfect_mock("triton.jit")
sys.modules["triton.runtime"] = perfect_mock("triton.runtime")
sys.modules["triton.runtime.jit"] = perfect_mock("triton.runtime.jit")
sys.modules["bitsandbytes"] = perfect_mock("bitsandbytes")
sys.modules["bitsandbytes.functional"] = sys.modules["bitsandbytes"].functional

# 2. Mock unsloth_zoo and its device_type
unsloth_zoo = perfect_mock("unsloth_zoo")
unsloth_zoo.utils = perfect_mock("unsloth_zoo.utils")
unsloth_zoo.utils.Version = lambda x: x
unsloth_zoo.device_type = perfect_mock("unsloth_zoo.device_type")
unsloth_zoo.device_type.DEVICE_TYPE = "mps"
unsloth_zoo.device_type.DEVICE_TYPE_TORCH = "mps"
unsloth_zoo.device_type.is_hip = lambda: False
unsloth_zoo.device_type.get_device_type = lambda: "mps"
unsloth_zoo.device_type.DEVICE_COUNT = 1
unsloth_zoo.device_type.ALLOW_PREQUANTIZED_MODELS = False
sys.modules["unsloth_zoo"] = unsloth_zoo
sys.modules["unsloth_zoo.utils"] = unsloth_zoo.utils
sys.modules["unsloth_zoo.device_type"] = unsloth_zoo.device_type

import torch
import torch.nn.functional as F
import pytest
import math

# Import fallbacks directly from the local files to avoid full unsloth init package reload if possible
# But we already mocked bitsandbytes so it should be fine.
from unsloth.kernels.mps.rms_layernorm import mps_rms_layernorm
from unsloth.kernels.mps.layernorm import mps_layernorm
from unsloth.kernels.mps.rope_embedding import mps_rope_embedding, mps_rope_embedding_qk
from unsloth.kernels.mps.cross_entropy_loss import mps_cross_entropy_loss
from unsloth.kernels.mps.swiglu import mps_swiglu_forward, mps_swiglu_backward
from unsloth.kernels.mps.geglu import (
    mps_geglu_exact_forward,
    mps_geglu_exact_backward,
    mps_geglu_approx_forward,
    mps_geglu_approx_backward,
)
from unsloth.kernels.mps.linear import mps_gemv, mps_linear_forward


def test_rms_layernorm_parity():
    print("Testing RMS LayerNorm Parity...")
    X = torch.randn(2, 16, 32, requires_grad=True)
    W = torch.randn(32, requires_grad=True)
    eps = 1e-5

    # Reference (Manual implementation for RMSNorm)
    X_f32 = X.to(torch.float32)
    variance = X_f32.pow(2).mean(-1, keepdim=True)
    rms_inv = torch.rsqrt(variance + eps)
    Y_ref = (X_f32 * rms_inv).to(X.dtype) * W

    # Fallback
    Y_mps = mps_rms_layernorm(X, W, eps)

    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    # Gradient check
    Y_ref.sum().backward()
    grad_W_ref = W.grad.clone()
    grad_X_ref = X.grad.clone()

    W.grad.zero_()
    X.grad.zero_()

    Y_mps.sum().backward()
    assert torch.allclose(grad_W_ref, W.grad, atol=1e-5)
    assert torch.allclose(grad_X_ref, X.grad, atol=1e-5)
    print("✅ RMS LayerNorm Parity Passed")


def test_layernorm_parity():
    print("Testing LayerNorm Parity...")
    X = torch.randn(2, 16, 32, requires_grad=True)
    W = torch.randn(32, requires_grad=True)
    b = torch.randn(32, requires_grad=True)
    eps = 1e-5

    # Reference
    Y_ref = F.layer_norm(X, (32,), W, b, eps)

    # Fallback
    Y_mps = mps_layernorm(X, W, b, eps)

    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    # Gradient check
    Y_ref.sum().backward()
    grad_W_ref = W.grad.clone()
    grad_b_ref = b.grad.clone()
    grad_X_ref = X.grad.clone()

    W.grad.zero_()
    b.grad.zero_()
    X.grad.zero_()

    Y_mps.sum().backward()
    assert torch.allclose(grad_W_ref, W.grad, atol=1e-5)
    assert torch.allclose(grad_b_ref, b.grad, atol=1e-5)
    assert torch.allclose(grad_X_ref, X.grad, atol=1e-5)
    print("✅ LayerNorm Parity Passed")


def test_swiglu_parity():
    print("Testing SwiGLU Parity...")
    # SwiGLU forward: silu(e) * g
    e = torch.randn(2, 16, 32, requires_grad=True)
    g = torch.randn(2, 16, 32, requires_grad=True)

    # Forward
    Y_ref = F.silu(e) * g
    Y_mps = mps_swiglu_forward(e, g)
    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    # Backward
    dw = torch.randn_like(Y_ref)

    # Reference backward via autograd
    loss = (Y_ref * dw).sum()
    loss.backward()
    ge_ref = e.grad.clone()
    gg_ref = g.grad.clone()

    # Fallback backward
    h_mps, de_mps, dg_mps = mps_swiglu_backward(dw, e, g)

    assert torch.allclose(ge_ref, de_mps, atol=1e-5)
    assert torch.allclose(gg_ref, dg_mps, atol=1e-5)
    print("✅ SwiGLU Parity Passed")


def test_cross_entropy_parity():
    print("Testing Cross-Entropy Parity...")
    logits = torch.randn(4, 32, 128, requires_grad=True)
    labels = torch.randint(0, 128, (4, 32))
    labels[0, 5] = -100  # ignore index

    # Reference
    loss_ref = F.cross_entropy(
        logits.view(-1, 128), labels.view(-1), ignore_index=-100, reduction="mean"
    )

    # Fallback
    loss_mps = mps_cross_entropy_loss(logits, labels)

    assert torch.allclose(loss_ref, loss_mps, atol=1e-5)

    # Gradient check
    loss_ref.backward()
    grad_logits_ref = logits.grad.clone()

    logits.grad.zero_()
    loss_mps.backward()
    assert torch.allclose(grad_logits_ref, logits.grad, atol=1e-5)
    print("✅ Cross-Entropy Parity Passed")


def test_rope_embedding_parity():
    print("Testing RoPE Embedding Parity...")
    batch, n_heads, seq_len, head_dim = 2, 4, 16, 32
    X = torch.randn(batch, n_heads, seq_len, head_dim, requires_grad=True)
    cos = torch.randn(seq_len, head_dim // 2)
    sin = torch.randn(seq_len, head_dim // 2)

    def rotate_half(x):
        shape = x.shape
        half = shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    X_f32 = X.to(torch.float32)
    cos_f32 = cos.to(torch.float32)
    sin_f32 = sin.to(torch.float32)
    cos_full = torch.cat((cos_f32, cos_f32), dim=-1)
    sin_full = torch.cat((sin_f32, sin_f32), dim=-1)
    X_rotated = rotate_half(X_f32)
    Y_ref = (X_f32 * cos_full) + (X_rotated * sin_full)

    Y_mps = mps_rope_embedding(X, cos, sin)

    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    Y_ref.sum().backward()
    grad_X_ref = X.grad.clone()
    X.grad.zero_()
    Y_mps.sum().backward()
    assert torch.allclose(grad_X_ref, X.grad, atol=1e-4)
    print("✅ RoPE Embedding Parity Passed")


def test_rope_embedding_qk_parity():
    print("Testing RoPE Embedding QK Parity...")
    batch, n_heads, seq_len, head_dim = 2, 4, 16, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim, requires_grad=True)
    K = torch.randn(batch, n_heads, seq_len, head_dim, requires_grad=True)
    cos = torch.randn(seq_len, head_dim // 2)
    sin = torch.randn(seq_len, head_dim // 2)

    def rotate_half(x):
        shape = x.shape
        half = shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    Q_f32 = Q.to(torch.float32)
    K_f32 = K.to(torch.float32)
    cos_f32 = cos.to(torch.float32)
    sin_f32 = sin.to(torch.float32)
    cos_full = torch.cat((cos_f32, cos_f32), dim=-1)
    sin_full = torch.cat((sin_f32, sin_f32), dim=-1)
    Q_rotated = rotate_half(Q_f32)
    K_rotated = rotate_half(K_f32)
    Q_out_ref = (Q_f32 * cos_full) + (Q_rotated * sin_full)
    K_out_ref = (K_f32 * cos_full) + (K_rotated * sin_full)

    Q_out_mps, K_out_mps = mps_rope_embedding_qk(Q, K, cos, sin)

    assert torch.allclose(Q_out_ref, Q_out_mps, atol=1e-5)
    assert torch.allclose(K_out_ref, K_out_mps, atol=1e-5)

    Q_out_ref.sum().backward()
    K_out_ref.sum().backward()
    grad_Q_ref = Q.grad.clone()
    grad_K_ref = K.grad.clone()
    Q.grad.zero_()
    K.grad.zero_()
    Q_out_mps.sum().backward()
    K_out_mps.sum().backward()
    assert torch.allclose(grad_Q_ref, Q.grad, atol=1e-4)
    assert torch.allclose(grad_K_ref, K.grad, atol=1e-4)
    print("✅ RoPE Embedding QK Parity Passed")


def test_geglu_exact_parity():
    print("Testing GEGLU Exact Parity...")
    gate = torch.randn(2, 16, 64, requires_grad=True)
    up = torch.randn(2, 16, 64, requires_grad=True)

    Y_ref = F.gelu(gate, approximate="none") * up
    Y_mps = mps_geglu_exact_forward(gate, up)
    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    dw = torch.randn_like(Y_ref)
    loss_ref = (Y_ref * dw).sum()
    loss_ref.backward()
    grad_gate_ref = gate.grad.clone()
    grad_up_ref = up.grad.clone()

    gate.grad.zero_()
    up.grad.zero_()
    h_mps, de_mps, dg_mps = mps_geglu_exact_backward(dw, gate, up)
    assert torch.allclose(grad_gate_ref, de_mps, atol=1e-4)
    assert torch.allclose(grad_up_ref, dg_mps, atol=1e-4)
    print("✅ GEGLU Exact Parity Passed")


def test_geglu_approx_parity():
    print("Testing GEGLU Approx Parity...")
    gate = torch.randn(2, 16, 64, requires_grad=True)
    up = torch.randn(2, 16, 64, requires_grad=True)

    Y_ref = F.gelu(gate, approximate="tanh") * up
    Y_mps = mps_geglu_approx_forward(gate, up)
    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    dw = torch.randn_like(Y_ref)
    loss_ref = (Y_ref * dw).sum()
    loss_ref.backward()
    grad_gate_ref = gate.grad.clone()
    grad_up_ref = up.grad.clone()

    gate.grad.zero_()
    up.grad.zero_()
    h_mps, de_mps, dg_mps = mps_geglu_approx_backward(dw, gate, up)
    assert torch.allclose(grad_gate_ref, de_mps, atol=1e-4)
    assert torch.allclose(grad_up_ref, dg_mps, atol=1e-4)
    print("✅ GEGLU Approx Parity Passed")


def test_gemv_parity():
    print("Testing GEMV Parity...")
    # Test 2D input
    X = torch.randn(16, 32, requires_grad=True)
    W = torch.randn(64, 32, requires_grad=True)

    Y_ref = torch.matmul(X, W.t())
    Y_mps = mps_gemv(X, W)
    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    # Test with output buffer
    out = torch.empty(16, 64)
    Y_mps_out = mps_gemv(X, W, out=out)
    assert torch.allclose(Y_ref, Y_mps_out, atol=1e-5)

    # Test gradient
    Y_ref.sum().backward()
    grad_X_ref = X.grad.clone()
    grad_W_ref = W.grad.clone()

    X.grad.zero_()
    W.grad.zero_()
    Y_mps.sum().backward()
    assert torch.allclose(grad_X_ref, X.grad, atol=1e-5)
    assert torch.allclose(grad_W_ref, W.grad, atol=1e-5)
    print("✅ GEMV Parity Passed")


def test_linear_forward_parity():
    print("Testing Linear Forward Parity...")
    X = torch.randn(2, 16, 32, requires_grad=True)
    W = torch.randn(64, 32, requires_grad=True)
    bias = torch.randn(64, requires_grad=True)

    # With bias
    Y_ref = F.linear(X, W, bias)
    Y_mps = mps_linear_forward(X, W, bias)
    assert torch.allclose(Y_ref, Y_mps, atol=1e-5)

    # Without bias
    Y_ref_nb = F.linear(X, W)
    Y_mps_nb = mps_linear_forward(X, W)
    assert torch.allclose(Y_ref_nb, Y_mps_nb, atol=1e-5)

    # Gradient check with bias
    Y_ref.sum().backward()
    grad_X_ref = X.grad.clone()
    grad_W_ref = W.grad.clone()
    grad_b_ref = bias.grad.clone()

    X.grad.zero_()
    W.grad.zero_()
    bias.grad.zero_()
    Y_mps.sum().backward()
    assert torch.allclose(grad_X_ref, X.grad, atol=1e-5)
    assert torch.allclose(grad_W_ref, W.grad, atol=1e-5)
    assert torch.allclose(grad_b_ref, bias.grad, atol=1e-5)
    print("✅ Linear Forward Parity Passed")


if __name__ == "__main__":
    try:
        test_rms_layernorm_parity()
        test_layernorm_parity()
        test_swiglu_parity()
        test_cross_entropy_parity()
        test_rope_embedding_parity()
        test_rope_embedding_qk_parity()
        test_geglu_exact_parity()
        test_geglu_approx_parity()
        test_gemv_parity()
        test_linear_forward_parity()
        print("\n🚀 ALL MPS FALLBACK PARITY TESTS PASSED (CPU VERIFIED)")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
