# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Under block compile (torch >= 2.11, the default) a gradient checkpoint replay of a Llama-style decoder
layer still skips the fused LoRA MLP's down projection.

The compiled layer cannot read the eager thread-local token, so the block's wrapper passes a call-scoped
argument into the replay call and the traced LoRA_MLP drops the down GEMM in that graph. The checkpointed forward's
no-grad call trials the same skip graph, so a step still compiles two graphs per layer kind (no grad, grad)
and later steps compile nothing. The skipped output is NaN-filled: the replay output is NaN, while every
LoRA gradient and the input gradient are bitwise equal to the same compiled run with the skip off.
"""

from __future__ import annotations

import types

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import fast_lora
from unsloth.kernels.swiglu import _DWf_DW_dfg_traced

if not fast_lora.TRACE_LORA_FUNCTIONS:
    pytest.skip(
        "block compile traces the fused LoRA MLP only on torch >= 2.11", allow_module_level = True
    )

zoo_gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
if not hasattr(zoo_gc, "gradient_checkpoint_replay_module"):
    pytest.skip("unsloth_zoo without gradient_checkpoint_replay_module", allow_module_level = True)

from unsloth.models import _block_compile
from unsloth.models.llama import LlamaDecoderLayer_fast_forward

H, I, N_HEADS, N_LAYERS, SEQ = 64, 96, 4, 3, 8


def _swiglu_fg_reference(e, g):
    return torch.nn.functional.silu(e.float()).to(e.dtype) * g


def _aot_eager_compile(fn = None, **kwargs):
    # Inductor on CPU would take most of the test; aot_eager keeps dynamo's guards and AOTAutograd's split.
    kwargs.pop("options", None)
    kwargs.pop("backend", None)
    return _real_compile(fn, backend = "aot_eager", **kwargs)


_real_compile = torch.compile


class _Attention(torch.nn.Module):
    """transformers' attention, returning the 3-tuple the fast decoder layer unpacks."""

    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(
        self,
        hidden_states,
        position_embeddings = None,
        **kwargs,
    ):
        out, _ = self.attn(hidden_states, position_embeddings, None)
        return out, None, None


@pytest.fixture
def block(monkeypatch):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    import unsloth.models.llama as llama

    monkeypatch.setattr(fast_lora, "swiglu_fg_kernel", _swiglu_fg_reference)
    monkeypatch.setattr(fast_lora, "swiglu_DWf_DW_dfg_kernel", _DWf_DW_dfg_traced)
    monkeypatch.setattr(fast_lora, "_SKIPPED_FILL", float("nan"))
    monkeypatch.setattr(fast_lora, "_SKIP_DOWN_ENABLED", True)
    monkeypatch.setattr(llama, "fast_rms_layernorm", lambda norm, X, **kwargs: norm(X))
    monkeypatch.setattr(Qwen3DecoderLayer, "forward", LlamaDecoderLayer_fast_forward)
    monkeypatch.setattr(torch, "compile", _aot_eager_compile)
    # Block compile's per-layer wrapper, as patch_block_compile installs it on CUDA.
    _block_compile._wrap(Qwen3DecoderLayer, dict(_block_compile._COMPILE_OPTIONS))
    monkeypatch.setattr(torch, "compile", _real_compile)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    fast_lora.MLP_DOWN_SKIP_STATS.clear()
    yield
    fast_lora.MLP_DOWN_SKIP_STATS.clear()
    torch._dynamo.reset()


def _model(lora_dropout = 0.0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(3407)
    config = Qwen3Config(
        hidden_size = H,
        intermediate_size = I,
        num_attention_heads = N_HEADS,
        num_key_value_heads = 2,
        head_dim = H // N_HEADS,
        num_hidden_layers = N_LAYERS,
        vocab_size = 128,
        attn_implementation = "eager",
    )
    model = Qwen3ForCausalLM(config).float()
    cfg = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = lora_dropout,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        init_lora_weights = False,
    )
    model = get_peft_model(model, cfg)
    inner = model.base_model.model.model
    for layer in inner.layers:
        layer.self_attn = _Attention(layer.self_attn)
        layer.mlp.forward = types.MethodType(fast_lora.apply_lora_mlp_swiglu, layer.mlp)
    model.train()
    return model, inner


def _step(
    model,
    inner,
    skip,
    replay_outputs = None,
    seed = 0,
):
    """One step through Unsloth's reentrant checkpoint, each layer its own checkpoint (LlamaModel_fast_forward)."""
    fast_lora._SKIP_DOWN_ENABLED = skip
    model.zero_grad(set_to_none = True)
    torch.manual_seed(seed)
    x = torch.randn(2, SEQ, H).requires_grad_(True)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0).expand(2, -1))
    hidden = x
    for layer in inner.layers:

        def custom_forward(*inputs, layer = layer):
            out = layer(*inputs, position_embeddings = pe)
            if replay_outputs is not None and torch.is_grad_enabled():
                replay_outputs.append(out[0].detach().clone())
            return out

        custom_forward._unsloth_checkpoint_module = layer
        hidden = zoo_gc.unsloth_gradient_checkpoint(custom_forward, hidden, use_reentrant = True)[0]
    weight = torch.randn(hidden.shape, generator = torch.Generator().manual_seed(1))
    (hidden * weight).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    grads["x"] = x.grad.clone()
    return grads


def _graphs():
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def _assert_grads_match(a, b):
    # Two differently compiled graphs: Inductor's CPU codegen may round a few ulps apart on some CPUs
    # (seen on AMD runners with torch 2.11). The NaN-filled replay outputs already prove nothing reads them.
    assert a.keys() == b.keys()
    for name in a:
        assert torch.isfinite(a[name]).all(), name
        torch.testing.assert_close(a[name], b[name], rtol = 1e-5, atol = 1e-7, msg = name)


def test_compiled_replay_skips_down_projection(block):
    model, inner = _model()
    replays = []
    head = _step(model, inner, skip = True, replay_outputs = replays)
    # Engaged: counted once per layer, and every replayed layer output is the unfilled (NaN) down output.
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS
    assert len(replays) == N_LAYERS and all(torch.isnan(r).all() for r in replays)
    graphs = _graphs()
    assert graphs >= 2  # the layer ran compiled: a no-grad and a grad-mode (replay) graph
    # Steady state: more steps add no graph and no recompile.
    for seed in (1, 2):
        _step(model, inner, skip = True, seed = seed)
    assert _graphs() == graphs
    assert not torch._dynamo.utils.counters["recompiles"]
    assert not torch._dynamo.utils.counters["graph_break"]
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 3 * N_LAYERS

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    for layer in inner.layers:
        layer.__dict__.pop("_unsloth_block_checked", None)
    fast_lora.MLP_DOWN_SKIP_STATS.clear()
    replays.clear()
    base = _step(model, inner, skip = False, replay_outputs = replays)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    assert all(torch.isfinite(r).all() for r in replays)
    # The skip trades the replay graph for its skip variant: the same number of graphs.
    assert _graphs() == graphs
    _assert_grads_match(base, head)

    # Same gradients as the uncompiled, unskipped layer without checkpointing.
    import unsloth.models.llama as llama

    model.zero_grad(set_to_none = True)
    torch.manual_seed(0)
    x = torch.randn(2, SEQ, H).requires_grad_(True)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0).expand(2, -1))
    hidden = x
    for layer in inner.layers:
        hidden = llama.LlamaDecoderLayer_fast_forward(layer, hidden, position_embeddings = pe)[0]
    weight = torch.randn(hidden.shape, generator = torch.Generator().manual_seed(1))
    (hidden * weight).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, head[name], rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(x.grad, head["x"], rtol = 1e-4, atol = 1e-5)


def test_compiled_replay_with_forward_hook_keeps_down_projection(block):
    model, inner = _model()
    seen = []
    handle = inner.layers[1].register_forward_hook(lambda m, a, out: seen.append(out[0].detach()))
    try:
        _step(model, inner, skip = True)
    finally:
        handle.remove()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS - 1
    assert all(torch.isfinite(v).all() for v in seen)


def test_compiled_lora_dropout_keeps_down_projection(block):
    model, inner = _model(lora_dropout = 0.1)
    replays = []
    _step(model, inner, skip = True, replay_outputs = replays)
    assert all(torch.isfinite(r).all() for r in replays)


def test_compiled_calls_outside_a_replay_never_skip(block):
    model, inner = _model()
    torch.manual_seed(0)
    x = torch.randn(2, SEQ, H, requires_grad = True)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0))
    with torch.no_grad():
        out = inner.layers[0](x, position_embeddings = pe)[0]
    assert torch.isfinite(out).all()
    out = inner.layers[0](x, position_embeddings = pe)[0]
    assert torch.isfinite(out).all()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    assert getattr(fast_lora._SKIP_DOWN_TLS, "armed", None) is None


def test_only_the_tagged_forward_is_eligible():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    assert LlamaDecoderLayer_fast_forward._unsloth_mlp_output_unread is True
    assert not getattr(Qwen3DecoderLayer.forward, "_unsloth_mlp_output_unread", False) or (
        Qwen3DecoderLayer.forward is LlamaDecoderLayer_fast_forward
    )
    tagged = []
    import inspect

    for name in (
        "llama",
        "gemma",
        "gemma2",
        "cohere",
        "granite",
        "qwen3_moe",
        "falcon_h1",
        "mistral",
    ):
        module = __import__(f"unsloth.models.{name}", fromlist = ["_"])
        for fn_name, fn in vars(module).items():
            if inspect.isfunction(fn) and getattr(fn, "_unsloth_mlp_output_unread", False):
                tagged.append(fn.__name__)
    assert set(tagged) == {"LlamaDecoderLayer_fast_forward", "GemmaDecoderLayer_fast_forward"}
