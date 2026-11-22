# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The checkpoint replay down projection skip on Gemma (GEGLU) decoder layers.

Gemma's layer feeds its MLP output only into `residual + mlp(x)`, the layer output, so a replay of the layer
never needs the down output. The tests NaN-fill the skipped output and check every LoRA gradient and the
input gradient are bitwise equal to the replay that runs the down projection, eager and under block
compile, for both GEGLU forms, with the skip counted once per layer. Gemma 2 is not covered: its post-MLP
norm reads the projection. Cohere and Granite load through FastModel (no fused LoRA MLP) and Falcon-H1
keeps PEFT's MLP, so none of them arms.

The observer guards are shared with the Llama-style skip and run for both: no skip when an accelerate hook
wraps the MLP or the decoder layer, or an MLP pre-hook can run the MLP itself; a compiled replay never
reaches another thread's call; and the compile warm-up never authorises another call to skip.
"""

from __future__ import annotations

import math

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import fast_lora

zoo_gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
if not hasattr(zoo_gc, "gradient_checkpoint_replay_module"):
    pytest.skip("unsloth_zoo without gradient_checkpoint_replay_module", allow_module_level = True)

import unsloth.models.gemma as gemma
import unsloth.models.llama as llama
from unsloth.models import _block_compile

H, I, N_HEADS, N_LAYERS, SEQ = 64, 96, 4, 3, 8


def _geglu_forward(approximate):
    def forward(e, g):
        return torch.nn.functional.gelu(e.float(), approximate = approximate).to(e.dtype) * g

    return forward


def _geglu_backward(approximate):
    # The Triton kernels' contract: (DW, e, g) -> (h, df, de), the same math in torch.
    def backward(DW, e, g):
        x = e.float()
        if approximate == "tanh":
            s = math.sqrt(2.0 / math.pi)
            t = torch.tanh(s * (x + 0.044715 * x * x * x))
            f = 0.5 * x * (1.0 + t)
            df_de = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * s * (1.0 + 3.0 * 0.044715 * x * x)
        else:
            f = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
            df_de = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + x * torch.exp(
                -0.5 * x * x
            ) / math.sqrt(2.0 * math.pi)
        f = f.to(DW.dtype)
        h = f * g
        df = DW * f
        de = ((DW * g).float() * df_de).to(DW.dtype)
        return h, df, de

    return backward


def _family(name):
    """(config, model class, unsloth layer forward, fused MLP kernel)."""
    from transformers import GemmaConfig, GemmaForCausalLM

    if name == "llama":  # the Llama-style SwiGLU skip shares every guard below
        from transformers import LlamaConfig, LlamaForCausalLM
        return (
            LlamaConfig(),
            LlamaForCausalLM,
            llama.LlamaDecoderLayer_fast_forward,
            fast_lora.apply_lora_mlp_swiglu,
        )

    kernel = (
        fast_lora.apply_lora_mlp_geglu_approx
        if name == "gemma_approx"
        else fast_lora.apply_lora_mlp_geglu_exact
    )
    config = GemmaConfig(hidden_activation = "gelu_pytorch_tanh")
    return config, GemmaForCausalLM, gemma.GemmaDecoderLayer_fast_forward, kernel


FAMILIES = ["gemma_approx", "gemma_exact"]
# Observer guards (accelerate hooks, pre-hooks, threads, warm-up) are shared with the Llama-style skip.
GUARD_FAMILIES = FAMILIES + ["llama"]


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
        out = self.attn(hidden_states, position_embeddings, None)[0]
        return out, None, None


@pytest.fixture(autouse = True)
def _torch_kernels(monkeypatch):
    # The GEGLU and norm kernels are Triton; these are the same math on any device.
    monkeypatch.setattr(fast_lora, "geglu_approx_forward_kernel", _geglu_forward("tanh"))
    monkeypatch.setattr(fast_lora, "geglu_approx_backward_kernel", _geglu_backward("tanh"))
    monkeypatch.setattr(fast_lora, "geglu_exact_forward_kernel", _geglu_forward("none"))
    monkeypatch.setattr(fast_lora, "geglu_exact_backward_kernel", _geglu_backward("none"))
    monkeypatch.setattr(fast_lora, "_SKIPPED_FILL", float("nan"))
    monkeypatch.setattr(fast_lora, "_SKIP_DOWN_ENABLED", True)
    from unsloth.kernels.swiglu import _DWf_DW_dfg_traced

    monkeypatch.setattr(
        fast_lora,
        "swiglu_fg_kernel",
        lambda e, g: torch.nn.functional.silu(e.float()).to(e.dtype) * g,
    )
    monkeypatch.setattr(fast_lora, "swiglu_DWf_DW_dfg_kernel", _DWf_DW_dfg_traced)
    monkeypatch.setattr(gemma, "fast_rms_layernorm", lambda norm, X, **kwargs: norm(X))
    monkeypatch.setattr(llama, "fast_rms_layernorm", lambda norm, X, **kwargs: norm(X))
    fast_lora.MLP_DOWN_SKIP_STATS.clear()
    yield
    fast_lora.MLP_DOWN_SKIP_STATS.clear()


def _model(
    family,
    monkeypatch,
    lora_dropout = 0.0,
):
    from peft import LoraConfig, get_peft_model

    config, model_cls, layer_forward, kernel = _family(family)
    config.hidden_size = H
    config.intermediate_size = I
    config.num_attention_heads = N_HEADS
    config.num_key_value_heads = 2
    config.head_dim = H // N_HEADS
    config.num_hidden_layers = N_LAYERS
    config.vocab_size = 128
    config._attn_implementation = "eager"
    torch.manual_seed(3407)
    model = model_cls(config).float()
    layer_cls = type(model.model.layers[0])
    monkeypatch.setattr(layer_cls, "forward", layer_forward)
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
        llama._install_fused_lora_mlp(layer.mlp, kernel)
    model.train()
    return model, inner, layer_cls


def _step(
    model,
    inner,
    skip,
    replay_outputs = None,
    checkpoint = True,
    seed = 0,
):
    """Each layer its own Unsloth reentrant checkpoint, as LlamaModel_fast_forward runs them."""
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

        if checkpoint:
            custom_forward._unsloth_checkpoint_module = layer
            hidden = zoo_gc.unsloth_gradient_checkpoint(custom_forward, hidden, use_reentrant = True)[
                0
            ]
        else:
            hidden = custom_forward(hidden)[0]
    weight = torch.randn(hidden.shape, generator = torch.Generator().manual_seed(1))
    (hidden * weight).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    grads["x"] = x.grad.clone()
    return grads


def _assert_bitwise(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert torch.isfinite(a[name]).all(), name
        assert torch.equal(a[name], b[name]), name


@pytest.mark.parametrize("family", FAMILIES)
def test_replay_skips_down_projection_bitwise(family, monkeypatch):
    model, inner, _ = _model(family, monkeypatch)
    base_replays, head_replays = [], []
    base = _step(model, inner, skip = False, replay_outputs = base_replays)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    assert all(torch.isfinite(r).all() for r in base_replays)
    head = _step(model, inner, skip = True, replay_outputs = head_replays)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS
    # The replayed layer outputs carry the NaN fill, yet no gradient does.
    assert len(head_replays) == N_LAYERS and all(torch.isnan(r).all() for r in head_replays)
    _assert_bitwise(base, head)
    plain = _step(model, inner, skip = True, checkpoint = False)
    for name in plain:
        torch.testing.assert_close(plain[name], head[name], rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_lora_dropout_and_hooks_keep_down_projection(family, monkeypatch):
    model, inner, _ = _model(family, monkeypatch, lora_dropout = 0.1)
    _step(model, inner, skip = True)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    model, inner, _ = _model(family, monkeypatch)
    handle = inner.layers[1].register_forward_hook(lambda m, a, out: None)
    try:
        _step(model, inner, skip = True)
    finally:
        handle.remove()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS - 1


@pytest.mark.parametrize("family", FAMILIES)
def test_no_skip_outside_a_replay(family, monkeypatch):
    model, inner, _ = _model(family, monkeypatch)
    x = torch.randn(2, SEQ, H)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0))
    with torch.no_grad():
        assert torch.isfinite(inner.layers[0](x, position_embeddings = pe)[0]).all()
    assert torch.isfinite(inner.layers[0](x, position_embeddings = pe)[0]).all()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
    assert getattr(fast_lora._SKIP_DOWN_TLS, "armed", None) is None


_real_compile = torch.compile


def _aot_eager_compile(fn = None, **kwargs):
    kwargs.pop("options", None)
    kwargs.pop("backend", None)
    return _real_compile(fn, backend = "aot_eager", **kwargs)


@pytest.mark.skipif(
    not fast_lora.TRACE_LORA_FUNCTIONS,
    reason = "block compile traces the fused LoRA MLP on torch >= 2.11",
)
@pytest.mark.parametrize("family", FAMILIES)
def test_compiled_replay_skips_down_projection(family, monkeypatch):
    model, inner, layer_cls = _model(family, monkeypatch)
    assert fast_lora.block_skip_down_eligible(inner.layers[0], layer_cls.forward)
    monkeypatch.setattr(torch, "compile", _aot_eager_compile)
    _block_compile._wrap(layer_cls, dict(_block_compile._COMPILE_OPTIONS))
    monkeypatch.setattr(torch, "compile", _real_compile)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        replays = []
        head = _step(model, inner, skip = True, replay_outputs = replays)
        assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS
        assert len(replays) == N_LAYERS and all(torch.isnan(r).all() for r in replays)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        assert graphs >= 2  # the layer ran compiled: a no-grad and a grad-mode (replay) graph
        _step(model, inner, skip = True, seed = 1)
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs
        assert not torch._dynamo.utils.counters["recompiles"]
        assert not torch._dynamo.utils.counters["graph_break"]

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        for layer in inner.layers:
            layer.__dict__.pop("_unsloth_block_checked", None)
        fast_lora.MLP_DOWN_SKIP_STATS.clear()
        replays.clear()
        base = _step(model, inner, skip = False, replay_outputs = replays)
        assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
        assert all(torch.isfinite(r).all() for r in replays)
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs
        assert not torch._dynamo.utils.counters["graph_break"]
        for name in base:
            assert torch.isfinite(head[name]).all(), name
            torch.testing.assert_close(base[name], head[name], rtol = 1e-5, atol = 1e-7, msg = name)
    finally:
        torch._dynamo.reset()


def test_post_norm_and_unfused_layers_stay_off():
    import unsloth.models.cohere as cohere
    import unsloth.models.falcon_h1 as falcon_h1
    import unsloth.models.gemma2 as gemma2
    import unsloth.models.granite as granite

    assert gemma.GemmaDecoderLayer_fast_forward._unsloth_mlp_output_unread is True
    for fn in (
        gemma2.Gemma2DecoderLayer_fast_forward,
        falcon_h1.FalconH1DecoderLayer_fast_forward,
        cohere.CohereDecoderLayer_fast_forward,
        granite.GraniteDecoderLayer_fast_forward,
    ):
        assert not getattr(fn, "_unsloth_mlp_output_unread", False), fn.__name__


def _accelerate_observer(target, kind):
    """An observer outside the layer body: an accelerate hook whose post_forward reads the output (as a
    placement or offload hook may), or an MLP forward pre-hook that runs the MLP's forward itself."""
    hooks = pytest.importorskip("accelerate.hooks")

    if kind == "mlp_pre_hook":
        handle = target.register_forward_pre_hook(
            lambda module, args: (args[0] + module.forward(args[0]).tanh(),)
        )
        return handle.remove

    class _Square(hooks.ModelHook):
        def post_forward(self, module, output):
            if isinstance(output, tuple):
                return (output[0].square(), *output[1:])
            return output.square()

    hooks.add_hook_to_module(target, _Square())
    return lambda: hooks.remove_hook_from_module(target)


OBSERVERS = ["mlp_accelerate", "decoder_accelerate", "mlp_pre_hook"]


def _observed(inner, kind):
    layer = inner.layers[1]
    return _accelerate_observer(layer if kind == "decoder_accelerate" else layer.mlp, kind)


@pytest.mark.parametrize("observer", OBSERVERS)
@pytest.mark.parametrize("family", GUARD_FAMILIES)
def test_observed_mlp_output_keeps_down_projection(family, observer, monkeypatch):
    """An accelerate-wrapped MLP or decoder, or an MLP pre-hook, can read the MLP output: no skip there."""
    model, inner, _ = _model(family, monkeypatch)
    remove = _observed(inner, observer)
    try:
        base = _step(model, inner, skip = False)
        head = _step(model, inner, skip = True)
    finally:
        remove()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS - 1
    _assert_bitwise(base, head)


@pytest.mark.skipif(
    not fast_lora.TRACE_LORA_FUNCTIONS,
    reason = "block compile traces the fused LoRA MLP on torch >= 2.11",
)
@pytest.mark.parametrize("observer", OBSERVERS)
@pytest.mark.parametrize("family", GUARD_FAMILIES)
def test_observed_mlp_output_is_not_compiled_eligible(family, observer, monkeypatch):
    model, inner, layer_cls = _model(family, monkeypatch)
    layer = inner.layers[1]
    assert fast_lora.block_skip_down_eligible(layer, layer_cls.forward)
    remove = _observed(inner, observer)
    try:
        assert not fast_lora.block_skip_down_eligible(layer, layer_cls.forward)
    finally:
        remove()
    assert fast_lora.block_skip_down_eligible(layer, layer_cls.forward)


def _in_thread(fn):
    import threading

    box = {}

    def target():
        try:
            box["out"] = fn()
        except BaseException as e:  # re-raised in the caller
            box["error"] = e

    thread = threading.Thread(target = target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["out"]


def _compiled_block(layer_cls, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(torch, "compile", _aot_eager_compile)
        _block_compile._wrap(layer_cls, dict(_block_compile._COMPILE_OPTIONS))
    torch._dynamo.reset()


@pytest.mark.skipif(
    not fast_lora.TRACE_LORA_FUNCTIONS,
    reason = "block compile traces the fused LoRA MLP on torch >= 2.11",
)
@pytest.mark.parametrize("family", GUARD_FAMILIES)
def test_compiled_replay_never_reaches_another_threads_call(family, monkeypatch):
    """While one thread runs a compiled block replay that skips, another thread's ordinary compiled MLP
    call still runs its down projection: the skip is an argument of the replay call, not process state."""
    model, inner, layer_cls = _model(family, monkeypatch)
    _compiled_block(layer_cls, monkeypatch)
    mlp = inner.layers[2].mlp
    torch.manual_seed(5)
    x = torch.randn(2, SEQ, H)
    expected = mlp(x).detach()
    other = torch.compile(lambda t: mlp(t), backend = "aot_eager", fullgraph = True)
    seen = []
    real = _block_compile._no_ddp_graph_split

    def overlapping():
        if (
            not seen
            and torch.is_grad_enabled()
            and zoo_gc.gradient_checkpoint_replay_module() is not None
        ):
            seen.append(_in_thread(lambda: other(x).detach()))
        return real()

    monkeypatch.setattr(_block_compile, "_no_ddp_graph_split", overlapping)
    try:
        replays = []
        _step(model, inner, skip = True, replay_outputs = replays)
    finally:
        torch._dynamo.reset()
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == N_LAYERS  # the replays still skipped
    assert all(torch.isnan(r).all() for r in replays)
    assert len(seen) == 1
    assert torch.isfinite(seen[0]).all()
    torch.testing.assert_close(seen[0], expected, rtol = 1e-5, atol = 1e-6)


@pytest.mark.skipif(
    not fast_lora.TRACE_LORA_FUNCTIONS,
    reason = "block compile traces the fused LoRA MLP on torch >= 2.11",
)
@pytest.mark.parametrize("family", GUARD_FAMILIES)
def test_compile_warm_up_never_authorises_another_call(family, monkeypatch):
    """The grad-mode trial before a block's first no-grad call may compile the skip graph, but a grad-mode
    call of another layer from another thread while it runs is not a replay and keeps the projection."""
    model, inner, layer_cls = _model(family, monkeypatch)
    _compiled_block(layer_cls, monkeypatch)
    torch.manual_seed(5)
    x = torch.randn(2, SEQ, H)
    pe = inner.rotary_emb(x, torch.arange(SEQ).unsqueeze(0).expand(2, -1))
    expected = inner.layers[1](x, position_embeddings = pe)[0].detach()
    for layer in inner.layers:
        layer.__dict__.pop("_unsloth_block_checked", None)
    torch._dynamo.reset()
    seen = []
    real = _block_compile._skip_down_eligible

    def during_trial(layer, original):
        if not seen and layer is inner.layers[0] and torch.is_grad_enabled():
            seen.append(None)  # inside layer 0's trial: run layer 1 from another thread
            seen[0] = _in_thread(lambda: inner.layers[1](x, position_embeddings = pe)[0].detach())
        return real(layer, original)

    monkeypatch.setattr(_block_compile, "_skip_down_eligible", during_trial)
    try:
        with torch.no_grad():
            first = inner.layers[0](x, position_embeddings = pe)[0]
    finally:
        torch._dynamo.reset()
    assert torch.isfinite(first).all()
    assert len(seen) == 1
    assert torch.isfinite(seen[0]).all()
    torch.testing.assert_close(seen[0], expected, rtol = 1e-5, atol = 1e-6)
    assert fast_lora.MLP_DOWN_SKIP_STATS["skipped"] == 0
