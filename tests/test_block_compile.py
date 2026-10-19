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
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    # Other test modules set these process-wide (e.g. UNSLOTH_COMPILE_DISABLE=1 at import).
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)


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
        # Grad-mode trial (full fails, split), then the no-grad call twice.
        assert calls == ["full", "split", "split", "split"]
    else:
        assert layer._unsloth_block_mode == M._OFF
        assert calls == ["full", "split"]  # then the stock forward, never compiled again
    # A sibling of the same kind starts where this layer ended, without retrying what failed.
    calls.clear()
    with torch.no_grad():
        cls()(x)
    assert calls == (["split", "split"] if fails_in == ("full",) else [])
    # Eval keeps the stock forward (logits equal transformers bit for bit).
    calls.clear()
    sibling = cls().eval()
    sibling(x)
    assert calls == []


def test_cache_calls_keep_the_stock_forward(monkeypatch):
    from transformers.cache_utils import DynamicCache

    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    class CacheLayerDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            past_key_value = None,
            **kwargs,
        ):
            return self.proj(hidden_states)

    M._wrap(CacheLayerDecoderLayer, {})
    layer = CacheLayerDecoderLayer()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        layer(x, past_key_value = DynamicCache())
        layer(x, DynamicCache())
        layer(x, past_key_values = DynamicCache())
    assert calls == []
    with torch.no_grad():
        layer(x)
    assert calls  # no cache: compiled


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
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    assert not M._torch_supported()
    monkeypatch.setattr(torch, "__version__", "2.7.1+cu126")
    assert M._torch_supported()


def test_forced_float32_is_decided_per_model(monkeypatch):
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    cls = _toy_class()

    def model_of(forced):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([cls(), cls()])
        if forced is not None:
            model._unsloth_forced_float32 = forced
        return model

    # Forced by this load's env (the stamp lands after the first post_patch_model).
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    forced = model_of(None)
    assert M.patch_block_compile(forced) == []
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    plain = model_of(False)
    assert M.patch_block_compile(plain) == [cls.__name__]
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        forced.layers[0](x)
    assert calls == []
    with torch.no_grad():
        plain.layers[0](x)
    assert len(calls) == 2  # grad-mode trial + the call
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    with torch.no_grad():
        plain.layers[1](x)
    assert len(calls) == 4
    # get_peft_model re-patches a wrapper after another load reset the env: the stamp holds.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")

    class Delegating(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    forced._unsloth_forced_float32 = True
    wrapper = torch.nn.Module()
    wrapper.base = forced
    M.patch_block_compile(wrapper)
    M.patch_block_compile(Delegating(forced))
    calls.clear()
    with torch.no_grad():
        forced.layers[0](x)
    assert calls == []


def test_checkpoint_recompute_keeps_the_no_grad_mode(monkeypatch):
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            mode = {M._FULL: "full", M._SPLIT: "split"}[M._IN_BLOCK[0]]
            grad = torch.is_grad_enabled()
            calls.append((mode, grad))
            if grad:
                torch.rand(3)
                if mode == "full":
                    raise _Boom("grad-mode lowering failed")
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()
    x = torch.randn(2, 4, 8)
    rng = torch.get_rng_state()
    with torch.no_grad():
        layer(x)
    # The grad-mode trial demoted first, so the no-grad half already ran in SPLIT.
    assert calls == [("full", True), ("split", True), ("split", False)]
    assert torch.equal(torch.get_rng_state(), rng)
    calls.clear()
    layer(x.requires_grad_())
    assert calls == [("split", True)]


def test_gdn_self_check_leaves_default_rng_alone():
    # Checkpoint recompute replays the default generator for LoRA dropout masks.
    pytest.importorskip("fla.ops.gated_delta_rule.chunk")
    if not has_real_cuda():
        pytest.skip("needs a real CUDA device")
    G = _load("_gdn_op")
    gen = torch.Generator(device = "cuda").manual_seed(0)
    B, T, H, D = 1, 70, 2, 128
    mk = lambda *s, dt = torch.bfloat16: torch.randn(*s, device = "cuda", dtype = dt, generator = gen)
    q, k, v = mk(B, T, H, D), mk(B, T, H, D), mk(B, T, H, D)
    g = -mk(B, T, H, dt = torch.float32).abs() * 0.1
    beta = mk(B, T, H).sigmoid()
    state = torch.cuda.get_rng_state()
    assert G._fla_ops(lambda: (q, k, v, g, beta))
    assert torch.equal(torch.cuda.get_rng_state(), state)


def test_gdn_self_check_stays_out_of_a_non_reentrant_checkpoint(monkeypatch):
    # #14063: the check's autograd.grad must not recompute the checkpointed layer mid-forward.
    from torch.utils.checkpoint import checkpoint

    G = _load("_gdn_op")
    chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    for name in (
        "chunk_gated_delta_rule_fwd",
        "chunk_gated_delta_rule_bwd",
        "l2norm_fwd",
        "l2norm_bwd",
    ):
        setattr(chunk, name, lambda *a, **k: None)
    for name in ("fla", "fla.ops", "fla.ops.gated_delta_rule"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule.chunk", chunk)

    def self_check(q, k, v, g, beta):
        ins = [t.requires_grad_() for t in (q, k, v)]
        out = (ins[0] * ins[1]).sigmoid() * ins[2]
        return all(gr is not None for gr in torch.autograd.grad(out.sum(), ins))

    monkeypatch.setattr(G, "_self_check", self_check)
    calls = []
    w = torch.randn(8, 8, requires_grad = True)

    def layer(x):
        calls.append(1)
        h = (x @ w).sigmoid()
        G._fla_ops(lambda: (h, h, h, h, h))
        return (h @ w).sigmoid()

    checkpoint(layer, torch.randn(4, 8), use_reentrant = False).sum().backward()
    assert G._FLA
    assert len(calls) == 2  # forward, then one recompute in backward
    assert w.grad is not None


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


@pytest.mark.skipif(not has_real_cuda(), reason = "compiled block needs CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "fla kernels need CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "fla kernels need a GPU")
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


def test_gdn_adapter_routes_only_the_self_checked_call(monkeypatch):
    M = _load()
    seen = []
    monkeypatch.setattr(M, "_GDN_OP", [True])
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch.ops,
        "unsloth",
        types.SimpleNamespace(qwen3_5_gdn_chunk = lambda *a: seen.append("op") or ("out",)),
        raising = False,
    )
    adapter = M._gdn_chunk_adapter(lambda *a, **kw: seen.append("fla") or ("fla", None))
    t = torch.zeros(1, 4, 2, 8)
    gb = torch.zeros(1, 4, 2)
    stock = dict(
        use_qk_l2norm_in_kernel = True, initial_state = None, output_final_state = False, cu_seqlens = None
    )
    adapter(t, t, t, g = gb, beta = gb, **stock)
    for extra in (
        {"head_first": True},
        {"use_beta_sigmoid_in_kernel": True},
        {"scale": 0.1},
        {"output_final_state": True},
        {"cu_seqlens": torch.tensor([0, 4])},
        {"use_qk_l2norm_in_kernel": False},
    ):
        adapter(t, t, t, g = gb, beta = gb, **dict(stock, **extra))
    assert seen == ["op"] + ["fla"] * 6


@pytest.mark.parametrize("linear_env", ["auto", "0"])
def test_nf4_block_route_matches_both_checkpoint_halves(monkeypatch, linear_env):
    # Package context so `from ..kernels import bnb_override` resolves to the fakes below.
    pkg = types.ModuleType("_ufake")
    pkg.__path__ = []
    models = types.ModuleType("_ufake.models")
    models.__path__ = []
    kernels = types.ModuleType("_ufake.kernels")
    kernels.__path__ = []
    O = types.ModuleType("_ufake.kernels.bnb_override")
    O._TRACE, O._TRACE_BF16, O._LINEAR = True, True, linear_env
    O._eligible = lambda self, x, weight, quant_state: True
    O._linear = lambda x, weight, quant_state, bias: torch.full_like(x, 7.0)
    kernels.bnb_override = O

    class Linear4bit(torch.nn.Module):
        def __init__(self, dtype, bias):
            super().__init__()
            self.weight = types.SimpleNamespace(quant_state = types.SimpleNamespace(dtype = dtype))
            self.bias = torch.nn.Parameter(torch.zeros(8)) if bias else None

        def forward(self, x):
            return torch.zeros_like(x)

    bnb = types.ModuleType("bitsandbytes")
    bnb.nn = types.SimpleNamespace(Linear4bit = Linear4bit)
    for name, mod in (
        ("_ufake", pkg),
        ("_ufake.models", models),
        ("_ufake.kernels", kernels),
        ("_ufake.kernels.bnb_override", O),
        ("bitsandbytes", bnb),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location(
        "_ufake.models._block_compile", os.path.join(_DIR, "_block_compile.py")
    )
    M = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, M)
    spec.loader.exec_module(M)
    M._nf4_in_blocks()
    if linear_env == "0":
        assert not getattr(Linear4bit.forward, "_unsloth_block_nf4", False)
        return
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(M, "_IN_BLOCK", [M._FULL])

    def routed(
        dtype,
        rows,
        bias = False,
    ):
        return bool(Linear4bit(dtype, bias)(torch.ones(rows, 8, dtype = dtype)).eq(7).all())

    assert routed(torch.bfloat16, 16)
    assert routed(torch.float32, 8)
    assert not routed(torch.bfloat16, 1)  # one row: fast_gemv only without grad
    assert not routed(torch.float32, 4)  # bitsandbytes' fp32 small-row kernel
    assert not routed(torch.bfloat16, 16, bias = True)  # trainable bias


def test_causal_depthwise_conv_matches_conv1d():
    M = _load("_gdn_op")
    torch.manual_seed(0)
    x, w = torch.randn(2, 6, 37), torch.randn(6, 4)
    want = torch.nn.functional.conv1d(x, w[:, None], groups = 6, padding = 3)[..., :37]
    torch.testing.assert_close(M._causal_depthwise_conv(x, w), want, rtol = 1e-5, atol = 1e-5)


def test_disable_unless_decode_keeps_original_traceable_on_old_torch(monkeypatch):
    # torch < 2.8: compiler.disable(fn, recursive = False) marks fn itself and returns it.
    import unsloth.import_fixes as F
    import unsloth_zoo.temporary_patches.utils as zoo_utils

    marked = []

    def old_disable(
        fn = None,
        recursive = True,
        **kwargs,
    ):
        fn._old_disable_marked = True
        marked.append(fn)
        return fn

    monkeypatch.setattr(torch.compiler, "disable", old_disable)
    monkeypatch.setattr(
        zoo_utils,
        "torch_compiler_disable_unless_decode",
        zoo_utils.torch_compiler_disable_unless_decode,
    )
    F.fix_torch_compiler_disable_unless_decode()
    fixed = zoo_utils.torch_compiler_disable_unless_decode
    assert fixed._unsloth_disable_copy

    def attention(x):
        return x + 1

    wrapped = fixed(attention)
    assert wrapped._unsloth_undisabled is attention
    assert not getattr(attention, "_old_disable_marked", False)
    disabled_copy = marked[-1]
    assert disabled_copy is not attention and disabled_copy.__code__ is not attention.__code__
    assert disabled_copy(1) == attention(1) == 2
    assert torch.compiler.disable is old_disable


def test_moe_experts_run_outside_the_block(monkeypatch):
    M = _load()

    class ToyExperts(torch.nn.Module):
        def forward(self, x):
            return x * 2

    class LoraWrapper(torch.nn.Module):  # PEFT-style wrapper around the experts
        def __init__(self):
            super().__init__()
            self.base_layer = ToyExperts()

        def forward(self, x):
            return self.base_layer(x) + 1

    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = LoraWrapper()
    layer.self_attn = torch.nn.Linear(4, 4)
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: True)
    M._experts_outside(layer)
    outer = layer.mlp.experts
    assert (
        outer.forward.__func__ is not LoraWrapper.forward
        if hasattr(outer.forward, "__func__")
        else True
    )
    assert outer._unsloth_block_outside and outer.base_layer._unsloth_block_outside
    assert not getattr(layer.self_attn, "_unsloth_block_outside", False)
    from torch._dynamo.utils import counters

    counters.clear()
    out = torch.compile(lambda x: layer.mlp.experts(x) * 3, backend = "eager")(torch.ones(2))
    assert out.tolist() == [9.0, 9.0] and sum(counters["graph_break"].values()) >= 1
    # A stash-reading experts forward (no fold) stays inside; so does everything under "inside".
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    other = torch.nn.Module()
    other.experts = LoraWrapper()
    M._experts_outside(other)
    assert not getattr(other.experts, "_unsloth_block_outside", False)
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: True)
    monkeypatch.setenv("UNSLOTH_BLOCK_MOE", "inside")
    third = torch.nn.Module()
    third.experts = LoraWrapper()
    M._experts_outside(third)
    assert not getattr(third.experts, "_unsloth_block_outside", False)


def test_selective_dequant_keeps_bnb4bit_experts_outside_the_block(monkeypatch):
    # The zoo plan reads token counts on the host and declines in a trace; opted-in experts stay eager.
    M = _load()

    class Params4bit(torch.nn.Parameter):
        pass

    class ToyExperts(torch.nn.Module):
        def __init__(self, quantized):
            super().__init__()
            weight = torch.zeros(2, 4, 4)
            self.gate_up_proj = (
                Params4bit(weight, requires_grad = False) if quantized else torch.nn.Parameter(weight)
            )

        def forward(self, x):
            return x * 2

    def layer(quantized):
        out = torch.nn.Module()
        out.mlp = torch.nn.Module()
        out.mlp.experts = ToyExperts(quantized)
        return out

    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    for key in (
        "UNSLOTH_BLOCK_MOE",
        "UNSLOTH_MOE_SELECTIVE_DEQUANT",
        "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS",
    ):
        monkeypatch.delenv(key, raising = False)
    default = layer(True)
    M._experts_outside(default)
    assert not getattr(default.mlp.experts, "_unsloth_block_outside", False)
    for key, value in (
        ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1"),
        ("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "32"),
    ):
        monkeypatch.setenv(key, value)
        quantized, plain = layer(True), layer(False)
        M._experts_outside(quantized)
        M._experts_outside(plain)
        assert quantized.mlp.experts._unsloth_block_outside
        assert not getattr(plain.mlp.experts, "_unsloth_block_outside", False)
        monkeypatch.delenv(key)
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "abc")
    bad = layer(True)
    M._experts_outside(bad)
    assert not getattr(bad.mlp.experts, "_unsloth_block_outside", False)


def test_selective_dequant_off_never_reads_a_packed_expert_stack(monkeypatch):
    # MXFP4 experts expose gate_up_proj as a property that dequantizes the whole stack.
    M = _load()
    reads = []

    class PackedExperts(torch.nn.Module):
        @property
        def gate_up_proj(self):
            reads.append(1)
            return torch.zeros(1)

        def forward(self, x):
            return x

    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    for key in (
        "UNSLOTH_BLOCK_MOE",
        "UNSLOTH_MOE_SELECTIVE_DEQUANT",
        "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS",
    ):
        monkeypatch.delenv(key, raising = False)
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = PackedExperts()
    M._experts_outside(layer)
    assert reads == []
    # Opted in, a packed stack is still not a bnb 4-bit one: never read, never moved out.
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1")
    layer.__dict__.pop("_unsloth_block_moe_done", None)
    M._experts_outside(layer)
    assert reads == []
    assert not getattr(layer.mlp.experts, "_unsloth_block_outside", False)
