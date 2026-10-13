# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen3.5 fused decoder layer (unsloth/models/_qwen3_5_layer.py).

CPU: the version probes resolve on the installed transformers (< 5.15 `layer_type` and a bound
`chunk_gated_delta_rule`, >= 5.15 `block_type` and a module-level kernel), and every fallback
returns the stock forward's output with the caller's kwargs.
GPU: the fused path matches the stock path to bf16 rounding, LoRA on every projection included.
"""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

_PATH = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models", "_qwen3_5_layer.py")


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location("_u35_layer", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _config():
    cfg = Qwen3_5TextConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 32,
        linear_value_head_dim = 32,
        layer_types = ["linear_attention", "full_attention"],
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def _layers(dtype = torch.float32, device = "cpu"):
    torch.manual_seed(0)
    cfg = _config()
    layers = [modeling.Qwen3_5DecoderLayer(cfg, layer_idx = i).to(device, dtype) for i in range(2)]
    with torch.no_grad():
        for layer in layers:
            for p in layer.parameters():
                p.add_(torch.randn_like(p) * 0.02)
    rotary = modeling.Qwen3_5TextRotaryEmbedding(cfg).to(device)
    return cfg, layers, rotary


def _inputs(
    rotary,
    dtype,
    device,
    L = 24,
):
    x = torch.randn(2, L, 64, device = device, dtype = dtype)
    pos = torch.arange(L, device = device)[None, None].expand(3, 2, L)
    return x, pos, rotary(x, pos)


def _restore():
    cls = modeling.Qwen3_5DecoderLayer
    orig = cls.__dict__.get("_unsloth_original_forward")
    if orig is not None:
        cls.forward = orig
        del cls._unsloth_original_forward


def _run(layer, x, pe, pos, **kw):
    out = layer(x, position_embeddings = pe, position_ids = pos, **kw)
    return out[0] if isinstance(out, tuple) else out


def test_version_probes_resolve():
    M = _load()
    _, layers, _ = _layers()
    assert M._block_type(layers[0]) == "linear_attention"
    assert M._block_type(layers[1]) == "full_attention"
    assert callable(M._chunk_kernel(layers[0].linear_attn))
    assert M._masks_any_batch(layers[0].linear_attn) in (True, False)
    assert M._layer_static_ok(layers[0]) and M._layer_static_ok(layers[1])


@pytest.mark.parametrize(
    "kw",
    [
        {"output_attentions": True},
        {"cu_seq_lens_q": "packed"},
        {"seq_idx": "packed"},
    ],
)
def test_fallback_matches_stock(kw):
    M = _load()
    # With a GPU visible the stock GDN dispatches to a Triton kernel, which needs CUDA tensors.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, layers, rotary = _layers(device = device)
    x, pos, pe = _inputs(rotary, torch.float32, device)
    with torch.no_grad():
        ref = [_run(layer, x, pe, pos) for layer in layers]
    calls = []
    try:
        M.patch_qwen3_5_decoder_layers(torch.nn.ModuleList(layers))
        orig = modeling.Qwen3_5DecoderLayer._unsloth_original_forward

        def spy(self, *a, **k):
            calls.append(k)
            k = {
                n: v
                for n, v in k.items()
                if n not in ("cu_seq_lens_q", "seq_idx", "output_attentions")
            }
            return orig(self, *a, **k)

        modeling.Qwen3_5DecoderLayer._unsloth_original_forward = spy
        with torch.no_grad():
            out = [_run(layer, x, pe, pos, use_cache = False, **kw) for layer in layers]
    finally:
        _restore()
    assert len(calls) == 2 and all(set(kw) <= set(c) and c["use_cache"] is False for c in calls)
    for o, r in zip(out, ref):
        torch.testing.assert_close(o, r)


def test_quantized_or_biased_projection_falls_back():
    M = _load()
    assert M._lora_params(torch.nn.Linear(4, 4, bias = True)) is None
    assert M._lora_params(torch.nn.Linear(4, 4, bias = False))[1] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "compiled regions need a GPU")
@pytest.mark.parametrize("padded", [False, True])
def test_fused_matches_stock_with_lora_everywhere(padded):
    peft = pytest.importorskip("peft")
    M = _load()
    cfg, layers, rotary = _layers(torch.bfloat16, "cuda")
    model = torch.nn.ModuleList(layers)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    targets += ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
    peft.inject_adapter_in_model(peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = targets), model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.05)
    x, pos, pe = _inputs(rotary, torch.bfloat16, "cuda")
    mask = None
    if padded:
        mask = torch.ones(2, x.shape[1], dtype = torch.long, device = "cuda")
        mask[1, :5] = 0

    def step():
        xi = x.clone().requires_grad_()
        out = _run(layers[0], xi, pe, pos, attention_mask = mask)
        out = _run(layers[1], out, pe, pos, attention_mask = None)
        out.float().pow(2).mean().backward()
        grads = {
            n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad(set_to_none = True)
        return out.detach().float(), grads

    ref_out, ref_g = step()
    try:
        assert M.patch_qwen3_5_decoder_layers(model) == 2
        out, g = step()
    finally:
        _restore()
    assert set(g) == set(ref_g) and any("in_proj_qkv.lora_B" in n for n in g)
    rel = (out - ref_out).norm() / ref_out.norm()
    assert rel < 2e-2, rel
    for n in ref_g:
        rel = (g[n] - ref_g[n]).norm() / ref_g[n].norm().clamp_min(1e-12)
        assert rel < 5e-2, (n, float(rel))
