# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One compiled region per decoder layer (unsloth/models/_block_compile.py).

CPU: sequence dims (not widths) are marked dynamic, rebound globals resolve, and a block that
fails to compile retries with flex_attention outside, then keeps the stock forward.
GPU: compiled Qwen3 and Qwen3.5 layers match the stock forward and gradients with no graph break
(Qwen3.5's fla kernel runs as the opaque op inside the block).
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


def _load(name = "_block_compile"):
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(f"_u{name}", os.path.join(_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mark_seq_dynamic_skips_widths(monkeypatch):
    M = _load()
    marked = []
    monkeypatch.setattr(
        torch._dynamo, "maybe_mark_dynamic", lambda t, d: marked.append((t.dim(), d))
    )
    L = 16
    hidden = torch.zeros(2, L, L)  # hidden size equal to the sequence length
    mask4d = torch.zeros(2, 1, L, L)
    mask2d = torch.zeros(2, L)
    mrope_ids = torch.zeros(3, 2, L, dtype = torch.long)
    M._mark_seq_dynamic(L, (hidden, {"m": mask4d}, [mask2d, "x", None], mrope_ids))
    assert sorted(marked) == [(2, 1), (3, 1), (3, 2), (4, 2), (4, 3)]


def test_rebind_globals_resolves():
    M = _load()
    ns = {"__name__": "not_a_registered_module", "OFFSET": 3}
    exec("def f(x):\n    return x + OFFSET\n", ns)
    f = M._rebind_globals(ns["f"])
    assert f is not ns["f"] and f(1) == 4
    assert sys.modules[f.__module__].__dict__ is f.__globals__
    g = M._rebind_globals(f)
    assert g is f  # already owned by a registered module
    exec("def h():\n    global OFFSET\n    OFFSET += 1\n", ns)
    assert M._rebind_globals(ns["h"]) is ns["h"]  # a copied namespace would miss the write


class _Boom(torch._dynamo.exc.TorchDynamoException):
    pass


def _toy_class():
    class ToyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            return self.proj(hidden_states) + hidden_states

    return ToyDecoderLayer


@pytest.mark.parametrize("fails_in", [("full",), ("full", "split")])
def test_compile_failure_falls_back(monkeypatch, fails_in):
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            mode = {M._FULL: "full", M._SPLIT: "split"}[M._IN_BLOCK[0]]
            calls.append(mode)
            if mode in fails_in:
                raise _Boom("lowering failed")
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    original = cls.forward
    M._wrap(cls, {})
    layer = cls()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        want = original(layer, x)
        got = layer(x)
        torch.testing.assert_close(got, want)
        layer(x)
    if fails_in == ("full",):
        assert layer._unsloth_block_mode == M._SPLIT
        assert calls == ["full", "split", "split"]
    else:
        assert layer._unsloth_block_mode == M._OFF
        assert calls == ["full", "split"]  # then the stock forward, never compiled again
    # The mode is per layer: a sibling of the same class still starts compiled.
    assert "_unsloth_block_mode" not in cls().__dict__
    # Eval keeps the stock forward (logits equal transformers bit for bit).
    calls.clear()
    sibling = cls().eval()
    sibling(x)
    assert calls == []


def test_gemma4_band_fast_path_in_blocks(monkeypatch):
    M = _load()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    scans, routed, plain_calls = [], [], []
    band = types.ModuleType("fake_banded_attention")
    band._mask_is_plain_band = lambda mask, S, w: scans.append((S, w)) or True
    zoo = types.ModuleType("fake_flash_sliding")

    def plain(module, q, k, v, mask, **kw):
        plain_calls.append(mask)
        return "plain", None

    def router(module, q, k, v, mask, **kw):
        routed.append(mask)
        return "router", None

    router._unsloth_gemma4_flash = True
    router.__module__ = zoo.__name__
    zoo._ORIG_SDPA = [plain]
    monkeypatch.setitem(sys.modules, band.__name__, band)
    monkeypatch.setitem(sys.modules, zoo.__name__, zoo)
    monkeypatch.setattr(M, "_BAND_MODULE", band.__name__)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", router)
    M._gemma4_router_hook()
    wrapped = ALL_ATTENTION_FUNCTIONS["sdpa"]
    assert wrapped is not router
    # Carries zoo's marker, so zoo's patch on a later model load does not wrap it again
    # with it as the fallback (that loop recursed forever).
    assert wrapped._unsloth_gemma4_flash
    M._gemma4_router_hook()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is wrapped

    class Gemma4TextAttention(torch.nn.Module):
        is_sliding, sliding_window, is_causal = True, 64, True

    layer = torch.nn.Module()
    layer.self_attn = attn = Gemma4TextAttention()
    short, long = torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 96, 96)
    M._precheck_band(layer, 32, (), {"attention_mask": short})
    M._precheck_band(layer, 96, (long,), {})
    assert scans == [(32, 64)]  # longer than the window: never scanned
    assert (short._unsloth_block_band, long._unsloth_block_band) == (True, False)

    q = torch.randn(1, 2, 32, 16)
    assert wrapped(attn, q, q, q, short)[0] == "router"  # eager: the stock router
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    M._IN_BLOCK[0] = M._FULL
    try:
        out, _ = wrapped(attn, q, q, q, short, scaling = 0.25)
        want = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal = True, scale = 0.25)
        torch.testing.assert_close(out, want.transpose(1, 2))
        ql = torch.randn(1, 2, 96, 16)
        assert wrapped(attn, ql, ql, ql, long)[0] == "plain"
    finally:
        M._IN_BLOCK[0] = M._OFF
    assert routed == [short] and plain_calls == [long]


def test_kill_switch(monkeypatch):
    M = _load()
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE", "0")
    assert M.patch_block_compile(torch.nn.Module()) == []
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE", "1")
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    assert not M._enabled()
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    assert M._enabled()
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    assert not M._enabled()


def test_sm_count_is_read_outside_the_trace(monkeypatch):
    # Stands in for Triton's hip_utils.get_device_properties, which Dynamo cannot trace.
    M = _load()
    fla = types.ModuleType("fake.fla.modules.fused_norm_gate")
    calls = []

    @torch._dynamo.disable
    def get_multiprocessor_count(tensor_idx = 0):
        calls.append(tensor_idx)
        return 7

    fla.get_multiprocessor_count = get_multiprocessor_count
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    M._route_sm_count({fla.__name__: fla})
    out = torch.compile(
        lambda x: x * fla.get_multiprocessor_count(0), fullgraph = True, backend = "eager"
    )(torch.ones(2))
    assert out.tolist() == [7.0, 7.0] and calls == [0]


def _grads_after_step(layers, run, x):
    for layer in layers:
        layer.zero_grad(set_to_none = True)
    out = x
    for layer in layers:
        out = run(layer, out)
    loss = out.float().pow(2).mean()
    loss.backward()
    return loss.detach(), {
        n: p.grad.clone() for layer in layers for n, p in layer.named_parameters()
    }


def _check_block(layers, run, x, cls):
    """Stock forward + backward vs the compiled block, two sequence lengths, no graph break."""
    from torch._dynamo.utils import counters

    import unsloth.models._block_compile as M

    want = [_grads_after_step(layers, run, xi) for xi in x]
    original = cls.forward
    try:
        M._wrap(M._rebind_class(cls) or cls, M._COMPILE_OPTIONS)
        counters.clear()
        got = [_grads_after_step(layers, run, xi) for xi in x]
        assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    finally:
        cls.forward = original
        torch._dynamo.reset()
    for (lw, gw), (lg, gg) in zip(want, got):
        torch.testing.assert_close(lg, lw, rtol = 1e-4, atol = 1e-5)
        for k in gw:
            torch.testing.assert_close(gg[k], gw[k], rtol = 1e-3, atol = 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "compiled block needs CUDA")
def test_qwen3_block_matches_stock():
    pytest.importorskip("unsloth")
    from transformers.models.qwen3 import modeling_qwen3 as modeling
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    cfg = Qwen3Config(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layers = [modeling.Qwen3DecoderLayer(cfg, layer_idx = i).cuda() for i in range(2)]
    rotary = modeling.Qwen3RotaryEmbedding(cfg).cuda()

    def run(layer, h):
        pos = torch.arange(h.shape[1], device = h.device)[None].expand(h.shape[0], -1)
        out = layer(h, position_embeddings = rotary(h, pos), position_ids = pos)
        return out[0] if isinstance(out, tuple) else out

    x = [torch.randn(2, L, 64, device = "cuda", requires_grad = True) for L in (24, 40)]
    _check_block(layers, run, x, modeling.Qwen3DecoderLayer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fla kernels need CUDA")
def test_qwen3_5_gdn_block_has_no_break():
    pytest.importorskip("unsloth")
    pytest.importorskip("fla.ops.gated_delta_rule")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    import unsloth.models._block_compile as M

    cfg = Qwen3_5TextConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        linear_num_key_heads = 4,
        linear_num_value_heads = 16,
        linear_key_head_dim = 32,
        linear_value_head_dim = 32,
        layer_types = ["linear_attention", "full_attention"],
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layers = [
        modeling.Qwen3_5DecoderLayer(cfg, layer_idx = i).cuda().to(torch.bfloat16) for i in range(2)
    ]
    rotary = modeling.Qwen3_5TextRotaryEmbedding(cfg).cuda()
    x = [torch.randn(2, L, 64, device = "cuda", dtype = torch.bfloat16) for L in (64, 128)]
    M._prepare_gdn(layers[0], x[0])
    if not M._GDN_OP[0]:
        pytest.skip("fla's opaque op failed its self-check on this GPU")

    def run(layer, h):
        pos = torch.arange(h.shape[1], device = h.device)[None, None].expand(3, h.shape[0], -1)
        out = layer(h, position_embeddings = rotary(h, pos), position_ids = pos)
        return out[0] if isinstance(out, tuple) else out

    from torch._dynamo.utils import counters

    cls = modeling.Qwen3_5DecoderLayer
    original = cls.forward
    want = [_grads_after_step(layers, run, xi.clone().requires_grad_()) for xi in x]
    try:
        M._wrap(cls, M._COMPILE_OPTIONS)
        counters.clear()
        got = [_grads_after_step(layers, run, xi.clone().requires_grad_()) for xi in x]
        assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    finally:
        cls.forward = original
        torch._dynamo.reset()
    for (lw, gw), (lg, gg) in zip(want, got):
        torch.testing.assert_close(lg, lw, rtol = 2e-2, atol = 1e-3)
        for k in gw:
            num = (gg[k].float() - gw[k].float()).norm()
            assert num <= 0.05 * gw[k].float().norm() + 1e-4, (
                k,
                float(num),
                float(gw[k].float().norm()),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fla kernels need a GPU")
def test_opaque_fla_op_matches_fla_function():
    pytest.importorskip("fla.ops.gated_delta_rule.chunk")
    M = _load("_gdn_op")
    gen = torch.Generator(device = "cuda").manual_seed(0)
    B, T, H, D = 2, 150, 4, 128
    mk = lambda *s, dt = torch.bfloat16: torch.randn(*s, device = "cuda", dtype = dt, generator = gen)
    q, k, v = mk(B, T, H, D), mk(B, T, H, D), mk(B, T, H, D)
    g = -mk(B, T, H, dt = torch.float32).abs() * 0.1
    beta = mk(B, T, H).sigmoid()
    assert M._fla_ops(lambda: (q, k, v, g, beta))
    # A wrong fake meta or a numeric mismatch must keep the op out of the block.
    M._FLA = None
    orig = M._chunk_meta
    M._chunk_meta = lambda *a: orig(*a)[:-1] + (((1,), torch.bfloat16),)
    try:
        assert M._fla_ops(lambda: (q, k, v, g, beta)) is False
    finally:
        M._chunk_meta = orig


def test_causal_depthwise_conv_matches_conv1d():
    M = _load("_gdn_op")
    torch.manual_seed(0)
    x, w = torch.randn(2, 6, 37), torch.randn(6, 4)
    want = torch.nn.functional.conv1d(x, w[:, None], groups = 6, padding = 3)[..., :37]
    torch.testing.assert_close(M._causal_depthwise_conv(x, w), want, rtol = 1e-5, atol = 1e-5)
