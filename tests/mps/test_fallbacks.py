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


if __name__ == "__main__":
    try:
        test_rms_layernorm_parity()
        test_layernorm_parity()
        test_swiglu_parity()
        test_cross_entropy_parity()
        print("\n🚀 ALL MPS FALLBACK PARITY TESTS PASSED (CPU VERIFIED)")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
