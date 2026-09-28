# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
# Packed NVFP4 bases (plain `weight` attribute = weight_packed with an NVFP4QuantState) through Unsloth's fused LoRA kernels.
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


@pytest.fixture(scope = "module")
def mods():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fast_lora, nvfp4, utils
    return fast_lora, nvfp4, utils


def _packed(out_features, in_features, seed):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    packed = torch.randint(
        0, 256, (out_features, in_features // 2), generator = g, device = "cuda", dtype = torch.uint8
    )
    # Asymmetric group scales so a wrong group axis or orientation shows up.
    scale = torch.rand(out_features, in_features // 16, generator = g, device = "cuda") * 3 + 0.25
    scale = scale * torch.linspace(0.5, 2.0, in_features // 16, device = "cuda")
    return packed, scale.to(torch.float8_e4m3fn), torch.tensor([37.5], device = "cuda")


def _lora_model(
    mods,
    names_shapes,
    r = 8,
    seed = 0,
):
    """PEFT LoRA over nn.Linears whose base weights are swapped for packed NVFP4; returns (nvfp4 model, dense twin)."""
    from peft import LoraConfig, get_peft_model

    _, nvfp4, _ = mods

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, (o, i) in names_shapes.items():
                setattr(self, name, torch.nn.Linear(i, o, bias = False))

    cfg = LoraConfig(r = r, lora_alpha = 16, lora_dropout = 0.0, target_modules = list(names_shapes))
    torch.manual_seed(seed)
    dense = get_peft_model(Block().cuda().to(torch.bfloat16), cfg)
    torch.manual_seed(seed)
    quant = get_peft_model(Block().cuda().to(torch.bfloat16), cfg)
    for k, (name, (o, i)) in enumerate(names_shapes.items()):
        packed, scale, gs = _packed(o, i, seed + k)
        W = nvfp4.nvfp4_dequantize(packed, scale, gs, torch.bfloat16)
        d_layer = getattr(dense.base_model.model, name)
        q_layer = getattr(quant.base_model.model, name)
        d_layer.base_layer.weight.data.copy_(W)
        del q_layer.base_layer.weight
        packed.quant_state = nvfp4.NVFP4QuantState(scale, gs, (o, i), torch.bfloat16)
        q_layer.base_layer.weight = packed
        # Non-zero B: zero-initialised B hides dA bugs.
        for layer in (d_layer, q_layer):
            torch.manual_seed(100 + k)
            torch.nn.init.normal_(layer.lora_B["default"].weight, std = 0.02)
            torch.manual_seed(200 + k)
            torch.nn.init.normal_(layer.lora_A["default"].weight, std = 0.02)
    return quant.base_model.model, dense.base_model.model


def _grads(model, names):
    return [getattr(model, n).lora_A["default"].weight.grad.float() for n in names] + [
        getattr(model, n).lora_B["default"].weight.grad.float() for n in names
    ]


def _close(a, b, tol):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min = 1e-12)) < tol


def _run(mods, names_shapes, fused, reference):
    q, d = _lora_model(mods, names_shapes)
    in_dim = next(iter(names_shapes.values()))[1]
    X = torch.randn(2, 24, in_dim, device = "cuda", dtype = torch.bfloat16)
    Xq, Xd = X.clone().requires_grad_(True), X.clone().requires_grad_(True)
    outs_q, outs_d = fused(q, Xq), reference(d, Xd)
    if not isinstance(outs_q, (tuple, list)):
        outs_q, outs_d = (outs_q,), (outs_d,)
    sum(o.float().square().mean() for o in outs_q).backward()
    sum(o.float().square().mean() for o in outs_d).backward()
    for oq, od in zip(outs_q, outs_d):
        assert oq.dtype == torch.bfloat16
        assert _close(oq, od, 1e-2)
    assert _close(Xq.grad, Xd.grad, 2e-2)
    for gq, gd in zip(_grads(q, names_shapes), _grads(d, names_shapes)):
        assert _close(gq, gd, 2e-2)


def test_dequantize_matches_compressed_tensors(mods):
    ct = pytest.importorskip("compressed_tensors.compressors.nvfp4.base")
    _, nvfp4, _ = mods
    packed, scale, gs = _packed(96, 160, 1)
    ref = ct.NVFP4PackedCompressor.decompress(
        {"weight_packed": packed, "weight_scale": scale, "weight_global_scale": gs}, None
    )["weight"]
    ours = nvfp4.nvfp4_dequantize(packed, scale, gs, ref.dtype)
    assert torch.equal(ours, ref)
    assert torch.equal(nvfp4.nvfp4_dequantize(packed.t(), scale, gs, ref.dtype), ref.t())


def test_get_lora_parameters_returns_the_nvfp4_state(mods):
    _, nvfp4, utils = mods
    q, _ = _lora_model(mods, {"o_proj": (64, 96)})
    W, W_quant, A, B, S = utils.get_lora_parameters(q.o_proj)
    assert W.dtype == torch.uint8 and type(W_quant) is nvfp4.NVFP4QuantState
    assert A is not None and B is not None


def test_lora_w_matches_dense(mods):
    fast_lora, _, _ = mods
    _run(
        mods,
        {"o_proj": (80, 96)},
        lambda m, X: fast_lora.apply_lora_o(m, X),
        lambda m, X: m.o_proj(X),
    )


def test_lora_qkv_matches_dense(mods):
    fast_lora, _, _ = mods
    shapes = {"q_proj": (128, 96), "k_proj": (32, 96), "v_proj": (32, 96)}
    _run(
        mods,
        shapes,
        lambda m, X: fast_lora.apply_lora_qkv(m, X, inplace = False),
        lambda m, X: (m.q_proj(X), m.k_proj(X), m.v_proj(X)),
    )


def test_lora_mlp_matches_dense(mods):
    fast_lora, _, _ = mods
    shapes = {"gate_proj": (160, 96), "up_proj": (160, 96), "down_proj": (96, 160)}

    def reference(m, X):
        return m.down_proj(torch.nn.functional.silu(m.gate_proj(X)) * m.up_proj(X))

    _run(mods, shapes, lambda m, X: fast_lora.apply_lora_mlp_swiglu(m, X, inplace = False), reference)


def test_fast_linear_forward_decode(mods):
    _, _, utils = mods
    q, d = _lora_model(mods, {"o_proj": (80, 96)})
    X = torch.randn(1, 1, 96, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        assert _close(utils.fast_linear_forward(q.o_proj, X), d.o_proj(X), 1e-2)
        X2 = torch.randn(3, 1, 96, device = "cuda", dtype = torch.bfloat16)
        assert _close(utils.fast_linear_forward(q.o_proj, X2), d.o_proj(X2), 1e-2)


def test_bnb_4bit_path_is_unchanged(mods):
    bnb = pytest.importorskip("bitsandbytes")
    _, _, utils = mods
    layer = bnb.nn.Linear4bit(128, 64, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4")
    dense = (torch.randn(64, 128) * 0.05).to(torch.bfloat16)
    layer.weight = bnb.nn.Params4bit(dense, requires_grad = False, quant_type = "nf4")
    layer = layer.cuda()
    W = layer.weight
    out = utils.fast_dequantize(W, W.quant_state)
    ref = bnb.functional.dequantize_4bit(W.data, W.quant_state)
    assert torch.equal(out, ref)
