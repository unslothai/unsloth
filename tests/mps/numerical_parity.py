import torch
import torch.nn.functional as F
import os
import sys
import importlib.util


# 1. Standalone Loading
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load kernels directly
fast_lora_path = os.path.join(ROOT, "unsloth/kernels/mps/fast_lora.py")
fast_lora_mod = load_module_from_path("unsloth.kernels.mps.fast_lora", fast_lora_path)
mps_matmul_lora = fast_lora_mod.mps_matmul_lora
mps_apply_lora_mlp_swiglu = fast_lora_mod.mps_apply_lora_mlp_swiglu
mps_apply_lora_qkv = fast_lora_mod.mps_apply_lora_qkv
mps_apply_lora_o = fast_lora_mod.mps_apply_lora_o
mps_apply_lora_mlp_geglu_exact = fast_lora_mod.mps_apply_lora_mlp_geglu_exact
mps_apply_lora_mlp_geglu_approx = fast_lora_mod.mps_apply_lora_mlp_geglu_approx

lora_pytorch_path = os.path.join(ROOT, "unsloth/kernels/mps/lora_pytorch.py")
lora_pytorch_mod = load_module_from_path("unsloth.kernels.mps.lora_pytorch", lora_pytorch_path)
pytorch_rope_embedding_qk = lora_pytorch_mod.pytorch_rope_embedding_qk


def test_matmul_lora_parity():
    print("Testing Matmul LoRA Parity...")
    X = torch.randn(2, 16, 32)
    W = torch.randn(64, 32)
    A = torch.randn(8, 32)
    B = torch.randn(64, 8)
    s = 2.0

    # Reference
    Y_ref = torch.matmul(X, W.t()) + torch.matmul(torch.matmul(X, A.t()), B.t()) * s

    # Implementation
    Y_impl = mps_matmul_lora(X, W, None, A, B, s)

    assert torch.allclose(Y_ref, Y_impl, atol=1e-5)
    print("✅ Matmul LoRA Passed")


def test_lora_mlp_parity():
    print("Testing LoRA MLP SwiGLU Parity...")
    X = torch.randn(2, 4, 8)
    gateW, upW, downW = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    gateA, gateB = torch.randn(4, 8), torch.randn(16, 4)
    upA, upB = torch.randn(4, 8), torch.randn(16, 4)
    downA, downB = torch.randn(4, 16), torch.randn(8, 4)
    s = 1.5

    # Reference (Manual composition)
    e = (
        torch.matmul(X, gateW.t())
        + torch.matmul(torch.matmul(X, gateA.t()), gateB.t()) * s
    )
    g = torch.matmul(X, upW.t()) + torch.matmul(torch.matmul(X, upA.t()), upB.t()) * s
    h = F.silu(e) * g
    Y_ref = (
        torch.matmul(h, downW.t())
        + torch.matmul(torch.matmul(h, downA.t()), downB.t()) * s
    )

    # Implementation
    Y_impl = mps_apply_lora_mlp_swiglu(
        X,
        gateW,
        None,
        gateA,
        gateB,
        s,
        upW,
        None,
        upA,
        upB,
        s,
        downW,
        None,
        downA,
        downB,
        s,
    )

    assert torch.allclose(Y_ref, Y_impl, atol=1e-5)
    print("✅ LoRA MLP SwiGLU Passed")


def test_lora_qkv_parity():
    print("Testing LoRA QKV Parity...")
    X = torch.randn(2, 4, 8)
    qw, kw, vw = torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 8)
    qa, qb = torch.randn(4, 8), torch.randn(16, 4)
    ka, kb = torch.randn(4, 8), torch.randn(16, 4)
    va, vb = torch.randn(4, 8), torch.randn(16, 4)
    qs, ks, vs = 1.5, 1.5, 1.5

    Q_ref = torch.matmul(X, qw.t()) + torch.matmul(torch.matmul(X, qa.t()), qb.t()) * qs
    K_ref = torch.matmul(X, kw.t()) + torch.matmul(torch.matmul(X, ka.t()), kb.t()) * ks
    V_ref = torch.matmul(X, vw.t()) + torch.matmul(torch.matmul(X, va.t()), vb.t()) * vs

    Q_impl, K_impl, V_impl = mps_apply_lora_qkv(
        X, qw, None, qa, qb, qs, kw, None, ka, kb, ks, vw, None, va, vb, vs
    )

    assert torch.allclose(Q_ref, Q_impl, atol=1e-5)
    assert torch.allclose(K_ref, K_impl, atol=1e-5)
    assert torch.allclose(V_ref, V_impl, atol=1e-5)
    print("✅ LoRA QKV Passed")


def test_lora_o_parity():
    print("Testing LoRA O Parity...")
    X = torch.randn(2, 4, 16)
    ow = torch.randn(16, 16)
    oa, ob = torch.randn(4, 16), torch.randn(16, 4)
    os = 2.0

    Y_ref = torch.matmul(X, ow.t()) + torch.matmul(torch.matmul(X, oa.t()), ob.t()) * os
    Y_impl = mps_apply_lora_o(X, ow, None, oa, ob, os)

    assert torch.allclose(Y_ref, Y_impl, atol=1e-5)
    print("✅ LoRA O Passed")


def test_lora_mlp_geglu_exact_parity():
    print("Testing LoRA MLP GEGLU Exact Parity...")
    X = torch.randn(2, 4, 8)
    gateW, upW, downW = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    gateA, gateB = torch.randn(4, 8), torch.randn(16, 4)
    upA, upB = torch.randn(4, 8), torch.randn(16, 4)
    downA, downB = torch.randn(4, 16), torch.randn(8, 4)
    s = 1.5

    e = torch.matmul(X, gateW.t()) + torch.matmul(torch.matmul(X, gateA.t()), gateB.t()) * s
    g = torch.matmul(X, upW.t()) + torch.matmul(torch.matmul(X, upA.t()), upB.t()) * s
    h = F.gelu(e, approximate="none") * g
    Y_ref = torch.matmul(h, downW.t()) + torch.matmul(torch.matmul(h, downA.t()), downB.t()) * s

    Y_impl = mps_apply_lora_mlp_geglu_exact(
        X, gateW, None, gateA, gateB, s, upW, None, upA, upB, s, downW, None, downA, downB, s
    )

    assert torch.allclose(Y_ref, Y_impl, atol=1e-5)
    print("✅ LoRA MLP GEGLU Exact Passed")


def test_lora_mlp_geglu_approx_parity():
    print("Testing LoRA MLP GEGLU Approx Parity...")
    X = torch.randn(2, 4, 8)
    gateW, upW, downW = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    gateA, gateB = torch.randn(4, 8), torch.randn(16, 4)
    upA, upB = torch.randn(4, 8), torch.randn(16, 4)
    downA, downB = torch.randn(4, 16), torch.randn(8, 4)
    s = 1.5

    e = torch.matmul(X, gateW.t()) + torch.matmul(torch.matmul(X, gateA.t()), gateB.t()) * s
    g = torch.matmul(X, upW.t()) + torch.matmul(torch.matmul(X, upA.t()), upB.t()) * s
    h = F.gelu(e, approximate="tanh") * g
    Y_ref = torch.matmul(h, downW.t()) + torch.matmul(torch.matmul(h, downA.t()), downB.t()) * s

    Y_impl = mps_apply_lora_mlp_geglu_approx(
        X, gateW, None, gateA, gateB, s, upW, None, upA, upB, s, downW, None, downA, downB, s
    )

    assert torch.allclose(Y_ref, Y_impl, atol=1e-5)
    print("✅ LoRA MLP GEGLU Approx Passed")


def test_pytorch_rope_embedding_qk_parity():
    print("Testing PyTorch RoPE Embedding QK Parity...")
    batch, n_heads, seq_len, head_dim = 2, 4, 16, 32
    Q = torch.randn(batch, n_heads, seq_len, head_dim)
    K = torch.randn(batch, n_heads, seq_len, head_dim)
    cos = torch.randn(seq_len, head_dim // 2)
    sin = torch.randn(seq_len, head_dim // 2)

    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = torch.cat((sin, sin), dim=-1)
    Q_rotated = rotate_half(Q)
    K_rotated = rotate_half(K)
    Q_ref = (Q * cos_full) + (Q_rotated * sin_full)
    K_ref = (K * cos_full) + (K_rotated * sin_full)

    Q_impl, K_impl = pytorch_rope_embedding_qk(Q, K, cos, sin)

    assert torch.allclose(Q_ref, Q_impl, atol=1e-5)
    assert torch.allclose(K_ref, K_impl, atol=1e-5)
    print("✅ PyTorch RoPE Embedding QK Passed")


if __name__ == "__main__":
    test_matmul_lora_parity()
    test_lora_mlp_parity()
    test_lora_qkv_parity()
    test_lora_o_parity()
    test_lora_mlp_geglu_exact_parity()
    test_lora_mlp_geglu_approx_parity()
    test_pytorch_rope_embedding_qk_parity()
    print("\n🚀 ALL CHUNK 3 KERNELS VERIFIED NUMERICALLY.")
