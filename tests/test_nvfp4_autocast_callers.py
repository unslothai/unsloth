# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""NVFP4 decode callers must get X.dtype back under mixed autocast (in-place / out= ops do not autocast)."""

import os
import types

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA"),
    pytest.mark.skipif(torch.version.hip is not None, reason = "the fused path is CUDA only"),
    pytest.mark.skipif(
        not has_real_cuda() or torch.cuda.get_device_capability() < (8, 0), reason = "needs bf16"
    ),
]

_AMP_CASES = [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)]
N_HEADS, N_KV, HEAD_DIM, HIDDEN = 4, 2, 32, 128


@pytest.fixture(scope = "module")
def mods():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import nvfp4, utils
    return nvfp4, utils


@pytest.fixture(autouse = True)
def fused(mods, monkeypatch):
    nvfp4, _ = mods
    monkeypatch.setattr(nvfp4, "_fused_env", nvfp4._read_fused_env)
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", "1")


def _packed(out_features, in_features, seed, dtype):
    from unsloth.kernels import nvfp4

    g = torch.Generator(device = "cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype = torch.uint8, generator = g)
    scale = (torch.rand(out_features, in_features // 16, generator = g) * 0.5 + 0.01).to(
        torch.float8_e4m3fn
    )
    packed, scale = packed.cuda(), scale.cuda()
    # Decoded weights stay under 0.15 (scale / global_scale times E2M1), so fp16 never overflows.
    gs = torch.tensor([20.0], device = "cuda")
    packed.quant_state = nvfp4.NVFP4QuantState(scale, gs, (out_features, in_features), dtype)
    return packed


def _nvfp4_linear_module(in_features, out_features, seed, dtype):
    layer = torch.nn.Linear(in_features, out_features, bias = False, device = "cuda")
    del layer.weight
    layer.weight = _packed(out_features, in_features, seed, dtype)
    return layer


def _dense_weight(layer, dtype):
    from unsloth.kernels import nvfp4
    qs = layer.weight.quant_state
    return nvfp4.nvfp4_dequantize(layer.weight, qs.scale, qs.global_scale, dtype)


def _fwd_calls(nvfp4):
    return nvfp4._FUSED_CALLS["fwd"]


@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("dtype, amp", _AMP_CASES)
def test_fast_linear_forward_lora_decode_under_mixed_autocast(mods, bsz, dtype, amp):
    nvfp4, utils = mods
    from peft import LoraConfig, get_peft_model

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Linear(256, 128, bias = False)

    torch.manual_seed(0)
    model = get_peft_model(
        Block().cuda().to(dtype),
        LoraConfig(r = 8, lora_alpha = 16, lora_dropout = 0.0, target_modules = ["o_proj"]),
    ).base_model.model
    proj = model.o_proj
    del proj.base_layer.weight
    proj.base_layer.weight = _packed(128, 256, 3, dtype)
    torch.nn.init.normal_(proj.lora_B["default"].weight, std = 0.02)

    X = torch.randn(bsz, 1, 256, device = "cuda", dtype = dtype)
    W = _dense_weight(proj.base_layer, torch.float64)
    A = proj.lora_A["default"].weight.double()
    B = proj.lora_B["default"].weight.double()
    Xd = X.double()
    ref = Xd @ W.t() + proj.scaling["default"] * (Xd @ A.t()) @ B.t()

    # Without an X.dtype LoRA scratch, torch.mv / torch.mm autocast the LoRA product (separate, older limitation).
    temp_lora = torch.empty((8,) if bsz == 1 else (bsz, 8), device = "cuda", dtype = dtype)
    before = _fwd_calls(nvfp4)
    with torch.no_grad(), torch.autocast("cuda", dtype = amp):
        out = utils.fast_linear_forward(proj, X, temp_lora = temp_lora)
    assert _fwd_calls(nvfp4) == before + 1, "fused forward did not run"
    assert out.dtype == dtype and out.shape == (bsz, 1, 128)
    torch.testing.assert_close(out.double(), ref, atol = 5e-2, rtol = 5e-2)


def _attention(dtype):
    config = types.SimpleNamespace(
        num_attention_heads = N_HEADS,
        num_key_value_heads = N_KV,
        hidden_size = HIDDEN,
        sliding_window = None,
    )
    # cos = 1 and sin = 0 make RoPE the identity, so the reference needs no rotation.
    rotary = types.SimpleNamespace(
        extend_rope_embedding = lambda x, n: None,
        get_cached = lambda n, i: (
            torch.ones(4096, HEAD_DIM, device = "cuda"),
            torch.zeros(4096, HEAD_DIM, device = "cuda"),
        ),
    )
    return types.SimpleNamespace(
        config = config,
        num_key_value_groups = N_HEADS // N_KV,
        head_dim = HEAD_DIM,
        q_proj = _nvfp4_linear_module(HIDDEN, N_HEADS * HEAD_DIM, 10, dtype),
        k_proj = _nvfp4_linear_module(HIDDEN, N_KV * HEAD_DIM, 11, dtype),
        v_proj = _nvfp4_linear_module(HIDDEN, N_KV * HEAD_DIM, 12, dtype),
        o_proj = _nvfp4_linear_module(N_HEADS * HEAD_DIM, HIDDEN, 13, dtype),
        rotary_emb = rotary,
    )


def _reference(attn, X, K, V):
    bsz = X.shape[0]
    Xd = X.double()
    Q = (Xd @ _dense_weight(attn.q_proj, torch.float64).t()).view(bsz, N_HEADS, 1, HEAD_DIM)
    K = K.double().repeat_interleave(N_HEADS // N_KV, dim = 1)
    V = V.double().repeat_interleave(N_HEADS // N_KV, dim = 1)
    A = torch.softmax((Q / HEAD_DIM**0.5) @ K.transpose(2, 3), dim = -1) @ V
    O = A.transpose(1, 2).reshape(bsz, 1, -1)
    return O @ _dense_weight(attn.o_proj, torch.float64).t()


@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("dtype, amp", _AMP_CASES)
def test_llama_attention_decode_under_mixed_autocast(mods, bsz, dtype, amp):
    nvfp4, _ = mods
    import unsloth  # noqa: F401
    from unsloth.models import llama

    torch.manual_seed(0)
    attn = _attention(dtype)
    cached = 5
    past = (
        torch.randn(bsz, N_KV, cached, HEAD_DIM, device = "cuda", dtype = dtype),
        torch.randn(bsz, N_KV, cached, HEAD_DIM, device = "cuda", dtype = dtype),
    )
    K, V = past
    for step, do_prefill in enumerate((True, False)):
        X = torch.randn(bsz, 1, HIDDEN, device = "cuda", dtype = dtype)
        before = _fwd_calls(nvfp4)
        with torch.no_grad(), torch.autocast("cuda", dtype = amp):
            out, (K, V) = llama.LlamaAttention_fast_forward_inference(
                attn,
                X,
                (K, V),
                torch.full((bsz, 1), cached + step, device = "cuda"),
                attention_mask = None,
                do_prefill = do_prefill,
            )
        assert _fwd_calls(nvfp4) == before + 4, "fused forward did not run for every projection"
        # bsz 1 runs in place in the cache dtype; bsz > 1 goes through SDPA, which autocasts.
        assert out.dtype == (dtype if bsz == 1 else amp)
        assert K.dtype == dtype and V.dtype == dtype
        assert K.shape == (bsz, N_KV, cached + step + 1, HEAD_DIM)
        torch.testing.assert_close(out.double(), _reference(attn, X, K, V), atol = 5e-2, rtol = 5e-2)


@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
def test_bias_keeps_the_autocast_dtype(mods, bias_dtype):
    nvfp4, _ = mods
    # fp16 1000s against all ones weights: 128000 is finite in bf16, inf in fp16.
    packed = torch.full((128, 64), 0x22, device = "cuda", dtype = torch.uint8)
    scale = torch.ones(128, 8, device = "cuda")
    gs = torch.ones(1, device = "cuda")
    W = nvfp4.nvfp4_dequantize(packed, scale, gs, torch.float16)
    bias = torch.zeros(128, device = "cuda", dtype = bias_dtype, requires_grad = True)

    def run(fn):
        X = torch.full((4, 128), 1000.0, device = "cuda", dtype = torch.float16, requires_grad = True)
        bias.grad = None
        with torch.autocast("cuda", dtype = torch.bfloat16):
            Y = fn(X)
            loss = Y.float().square().mean()
        loss.backward()
        return Y, X.grad, bias.grad.clone()

    before = _fwd_calls(nvfp4)
    Yf, gf, bf = run(lambda X: nvfp4.nvfp4_linear(X, packed, scale, gs, bias))
    assert _fwd_calls(nvfp4) == before + 1, "fused forward did not run"
    Yd, gd, bd = run(lambda X: torch.nn.functional.linear(X, W, bias))
    assert Yf.dtype == Yd.dtype == torch.bfloat16
    assert Yf.isfinite().all() and gf.isfinite().all() and bf.isfinite().all()
    assert gf.dtype == torch.float16 and bf.dtype == bias_dtype
    torch.testing.assert_close(Yf, Yd)
    torch.testing.assert_close(gf, gd)
    torch.testing.assert_close(bf, bd)


def test_no_autocast_keeps_x_dtype_with_bias(mods):
    nvfp4, _ = mods
    packed = torch.full((128, 64), 0x22, device = "cuda", dtype = torch.uint8)
    scale = torch.ones(128, 8, device = "cuda")
    gs = torch.ones(1, device = "cuda")
    X = torch.randn(4, 128, device = "cuda", dtype = torch.bfloat16)
    for bias_dtype in (torch.bfloat16, torch.float32):
        bias = torch.randn(128, device = "cuda", dtype = bias_dtype)
        assert nvfp4.nvfp4_linear(X, packed, scale, gs, bias).dtype == torch.bfloat16
