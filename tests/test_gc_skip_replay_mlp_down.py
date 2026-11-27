# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A gradient checkpoint replay of a Llama-style decoder layer skips the fused LoRA MLP's down projection.

The replay only rebuilds what LoRA_MLP saves (X, e, g). The down output feeds residual + i, the layer's
output, which is the checkpoint's output: its nested backward reads only the incoming gradient, never the
value. So the replay may return an unfilled tensor there. The tests fill it with NaN and check every LoRA
gradient and the input gradient are bitwise equal to the replay that runs the down projection, with the
skip counted once per layer per backward. Anything that could read the value turns the skip off: a
checkpoint wrapping more than one layer, forward hooks, LoRA dropout / bias, or a traced block.
"""

from __future__ import annotations

import functools
import types

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import fast_lora
from unsloth.kernels.swiglu import _DWf_DW_dfg_traced

zoo_gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
if not hasattr(zoo_gc, "gradient_checkpoint_replay_module"):
    pytest.skip("unsloth_zoo without gradient_checkpoint_replay_module", allow_module_level = True)

from unsloth.models.llama import LlamaDecoderLayer_fast_forward

H, I, N_HEADS, N_LAYERS, SEQ = 64, 96, 4, 3, 7


def _swiglu_fg_reference(e, g):
    return torch.nn.functional.silu(e.float()).to(e.dtype) * g


@pytest.fixture(autouse = True)
def _torch_swiglu(monkeypatch):
    # The SwiGLU kernels are Triton; the traced reference is the same math on any device.
    monkeypatch.setattr(fast_lora, "swiglu_fg_kernel", _swiglu_fg_reference)
    monkeypatch.setattr(fast_lora, "swiglu_DWf_DW_dfg_kernel", _DWf_DW_dfg_traced)
    monkeypatch.setattr(fast_lora, "_SKIPPED_FILL", float("nan"))
    monkeypatch.setattr(fast_lora, "_SKIP_DOWN_ENABLED", True)
    # The fast decoder layer Unsloth installs on Llama / Qwen2 / Qwen3 / Mistral.
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    import unsloth.models.llama as llama

    # The RMSNorm kernel is Triton too; the module's own forward is the same math.
    monkeypatch.setattr(llama, "fast_rms_layernorm", lambda norm, X, **kwargs: norm(X))
    monkeypatch.setattr(LlamaDecoderLayer, "forward", LlamaDecoderLayer_fast_forward)
    monkeypatch.setattr(Qwen3DecoderLayer, "forward", LlamaDecoderLayer_fast_forward)
    fast_lora.MLP_DOWN_SKIP_STATS.clear()
    yield
    fast_lora.MLP_DOWN_SKIP_STATS.clear()


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


def _model(
    arch,
    lora_dropout = 0.0,
    device = "cpu",
):
    from peft import LoraConfig, get_peft_model

    if arch == "llama":
        from transformers import LlamaConfig as Config, LlamaForCausalLM as Model
    else:
        from transformers import Qwen3Config as Config, Qwen3ForCausalLM as Model
    torch.manual_seed(3407)
    config = Config(
        hidden_size = H,
        intermediate_size = I,
        num_attention_heads = N_HEADS,
        num_key_value_heads = 2,
        head_dim = H // N_HEADS,
        num_hidden_layers = N_LAYERS,
        vocab_size = 128,
        attn_implementation = "eager",
    )
    model = Model(config).float()
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
    model = get_peft_model(model, cfg).to(device)
    inner = model.base_model.model.model
    for layer in inner.layers:
        layer.self_attn = _Attention(layer.self_attn)
        layer.mlp.forward = types.MethodType(fast_lora.apply_lora_mlp_swiglu, layer.mlp)
    model.train()
    return model, inner


def _hf_checkpoint(layer, hidden, pe, checkpoint):
    # transformers' GradientCheckpointingLayer: partial(super().__call__, **kwargs).
    return checkpoint(
        functools.partial(torch.nn.Module.__call__.__get__(layer), position_embeddings = pe), hidden
    )


def _forward_partial(layer, hidden, pe, checkpoint):
    # transformers >= 5.23 / Unsloth block compile: partial(forward, self, **kwargs).
    return checkpoint(functools.partial(type(layer).forward, layer, position_embeddings = pe), hidden)


def _custom_forward(layer, hidden, pe, checkpoint):
    # LlamaModel_fast_forward on older transformers: a closure tagged with its module.
    def custom_forward(*inputs):
        return layer(*inputs, position_embeddings = pe)

    custom_forward._unsloth_checkpoint_module = layer
    return checkpoint(custom_forward, hidden)


def _two_layer_segment(layers, hidden, pe, checkpoint):
    # One checkpoint over two layers: the first layer's output is read by the second, so no skip.
    def both(x):
        x = layers[0](x, position_embeddings = pe)[0]
        return layers[1](x, position_embeddings = pe)

    return checkpoint(both, hidden)


def _unsloth_checkpoint(function, *args):
    return zoo_gc.unsloth_gradient_checkpoint(function, *args, use_reentrant = True)


def _offloaded_checkpoint(function, *args):
    return zoo_gc.Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)


def _smart_checkpoint(function, *args):
    # use_gradient_checkpointing = "unsloth": the offloading CheckpointFunction.
    zoo_gc.initialize_unsloth_gradient_checkpointing(torch.float32)
    return zoo_gc.UnslothCheckpointFunction.apply(function, True, *args)


CHECKPOINTS = {
    "unsloth": _unsloth_checkpoint,
    "offloaded": _offloaded_checkpoint,
    "smart": _smart_checkpoint,
}
# The offloading checkpointer stages through CUDA buffers; on CPU it runs only where CUDA exists.
CHECKPOINT_PARAMS = [
    pytest.param(name, marks = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA"))
    if name == "smart"
    else name
    for name in CHECKPOINTS
]
CALLS = {
    "hf_call": _hf_checkpoint,
    "forward_partial": _forward_partial,
    "custom_forward": _custom_forward,
}


def _run(
    model,
    inner,
    call,
    checkpoint,
    skip,
    segment = False,
):
    fast_lora._SKIP_DOWN_ENABLED = skip
    model.zero_grad(set_to_none = True)
    torch.manual_seed(0)
    device = next(model.parameters()).device
    x = torch.randn(2, SEQ, H).to(device).requires_grad_(True)
    position_ids = torch.arange(SEQ, device = device).unsqueeze(0).expand(2, -1)
    pe = inner.rotary_emb(x, position_ids)
    hidden = x
    if segment:
        hidden = _two_layer_segment(inner.layers, hidden, pe, checkpoint)[0]
        hidden = inner.layers[2](hidden, position_embeddings = pe)[0]
    else:
        for layer in inner.layers:
            out = call(layer, hidden, pe, checkpoint)
            hidden = out[0] if isinstance(out, tuple) else out
    weight = torch.randn(hidden.shape, generator = torch.Generator().manual_seed(1)).to(device)
    (hidden * weight).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    grads["x"] = x.grad.clone()
    return grads


def _assert_bitwise(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert torch.isfinite(a[name]).all(), name
        assert torch.equal(a[name], b[name]), name


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
@pytest.mark.parametrize("checkpoint", CHECKPOINT_PARAMS)
@pytest.mark.parametrize("call", list(CALLS))
def test_replay_skips_down_projection_bitwise(arch, checkpoint, call):
    model, inner = _model(arch, device = "cuda" if checkpoint == "smart" else "cpu")
    base = _run(model, inner, CALLS[call], CHECKPOINTS[checkpoint], skip = False)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    head = _run(model, inner, CALLS[call], CHECKPOINTS[checkpoint], skip = True)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS
    _assert_bitwise(base, head)
    # Same gradients as no checkpointing at all (the replay is the forward).
    model.zero_grad(set_to_none = True)
    plain = _run(
        model, inner, lambda layer, h, pe, _: layer(h, position_embeddings = pe), None, skip = True
    )
    for name in plain:
        torch.testing.assert_close(plain[name], head[name], rtol = 1e-5, atol = 1e-6)


def test_multi_layer_segment_never_skips():
    model, inner = _model("llama")
    base = _run(model, inner, None, _unsloth_checkpoint, skip = False, segment = True)
    head = _run(model, inner, None, _unsloth_checkpoint, skip = True, segment = True)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    _assert_bitwise(base, head)


def test_forward_hook_turns_skip_off():
    model, inner = _model("llama")
    seen = []
    handle = inner.layers[1].register_forward_hook(
        lambda m, a, out: seen.append(out[0].sum().item())
    )
    try:
        base = _run(model, inner, _hf_checkpoint, _unsloth_checkpoint, skip = False)
        head = _run(model, inner, _hf_checkpoint, _unsloth_checkpoint, skip = True)
    finally:
        handle.remove()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS - 1
    assert all(v == v for v in seen)  # the hooked layer's replay output is real, never NaN
    _assert_bitwise(base, head)


def test_lora_dropout_runs_down_projection():
    model, inner = _model("llama", lora_dropout = 0.1)
    _run(model, inner, _hf_checkpoint, _unsloth_checkpoint, skip = True)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0


def test_no_skip_outside_a_replay():
    model, inner = _model("qwen3")
    torch.manual_seed(0)
    x = torch.randn(2, SEQ, H)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0))
    with torch.no_grad():
        out = inner.layers[0](x, position_embeddings = pe)[0]
    out = inner.layers[0](x, position_embeddings = pe)[0]
    assert torch.isfinite(out).all()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    assert getattr(fast_lora._SKIP_DOWN_TLS, "armed", None) is None


def test_traced_layer_has_no_graph_break():
    # Block compile traces the decoder layer and (torch >= 2.11) the fused LoRA MLP: the skip folds away.
    model, inner = _model("qwen3")
    layer = inner.layers[0]
    torch.manual_seed(0)
    x = torch.randn(2, SEQ, H)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0))
    compiled = torch.compile(
        lambda h: LlamaDecoderLayer_fast_forward(layer, h, position_embeddings = pe)[0],
        # Before torch 2.11 the fused LoRA functions are dynamo-disabled, so the block splits there anyway.
        fullgraph = fast_lora.TRACE_LORA_FUNCTIONS,
        backend = "aot_eager",
    )
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), layer(x, position_embeddings = pe)[0])
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0


def test_only_the_post_norm_free_layer_arms():
    # Llama and Gemma arm the skip: their MLP output feeds only the layer output; Gemma 2 / 3
    # (post-FFN norm reads it), Cohere / Granite, Falcon-H1 (no fused LoRA MLP) and MoE never arm.
    import inspect
    import unsloth.models as models

    arming = []
    for name in (
        "llama",
        "gemma",
        "gemma2",
        "cohere",
        "granite",
        "qwen3_moe",
        "falcon_h1",
        "mistral",
        "qwen3",
    ):
        module = __import__(f"unsloth.models.{name}", fromlist = ["_"])
        for fn_name, fn in vars(module).items():
            # torch.compile parks graph-break continuations (`__resume_at_<offset>_<n>`) in the compiled function's
            # module globals, and their source is that function's, so a compile earlier in the worker adds a copy.
            if (
                inspect.isfunction(fn)
                and fn.__module__ == module.__name__
                and fn.__code__.co_name == fn_name
            ):
                if "arm_mlp_output_unread(" in inspect.getsource(fn):
                    arming.append(fn_name)
    assert sorted(arming) == ["GemmaDecoderLayer_fast_forward", "LlamaDecoderLayer_fast_forward"]
    source = inspect.getsource(LlamaDecoderLayer_fast_forward)
    tail = source[source.index("armed = arm_mlp_output_unread") :]
    assert "hidden_states = residual + hidden_states\n\n    outputs = (hidden_states,)" in tail
    del models
