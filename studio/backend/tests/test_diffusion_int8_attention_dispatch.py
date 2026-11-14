# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 attention route through diffusers' attention dispatch (``diffusion_int8_attention_dispatch``): which
calls reach the kernel, which keep the incumbent, the switches, the card check and race, the install hook in
``apply_attention_backend``, and that a compiled block traces it with no graph break. CPU: the card is faked and the
op's implementation is replaced by a recorder; the CUDA tests at the end run the real kernel."""

from __future__ import annotations

import copy
import pickle
import types

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_dispatch as D

attention_dispatch = pytest.importorskip("diffusers.models.attention_dispatch")
AttentionBackendName = attention_dispatch.AttentionBackendName


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    monkeypatch.delenv(D.DISPATCH_ENV, raising = False)
    stats = dict(I.STATS)
    D.ROUTE_STATS.clear()
    yield
    I.STATS.clear()
    I.STATS.update(stats)
    D.ROUTE_STATS.clear()


def _route(
    min_tokens = 0,
    dtype = torch.bfloat16,
    head_dim = 64,
    incumbent = None,
):
    incumbent = AttentionBackendName.NATIVE if incumbent is None else incumbent
    return D.Int8Route(incumbent, min_tokens, dtype, 0, head_dim, "cuda")


def _qkv(
    sq = 32,
    sk = None,
    heads = 2,
    head_dim = 64,
    dtype = torch.bfloat16,
    grad = False,
):
    g = torch.Generator().manual_seed(0)
    sk = sq if sk is None else sk
    q = torch.randn(1, sq, heads, head_dim, generator = g).to(dtype)
    k = torch.randn(1, sk, heads, head_dim, generator = g).to(dtype)
    v = torch.randn(1, sk, heads, head_dim, generator = g).to(dtype)
    if grad:
        q.requires_grad_(True)
    return q, k, v


@pytest.fixture
def card(monkeypatch):
    """CPU tensors count as the route's card; the op records its calls and returns half the native result."""
    calls = []
    monkeypatch.setattr(D, "_on_card", lambda x, route: True)

    def impl(q, k, v):
        calls.append((tuple(q.shape), tuple(k.shape)))
        return I.native_attention(q, k, v) * 0.5

    monkeypatch.setattr(I, "int8_attention_impl", impl)
    I.int8_attention_op()
    return calls


def _fake_cuda(
    monkeypatch,
    found = None,
    capability = (8, 9),
    impl = "cuda",
):
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: impl)
    seen = []

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        seen.append(
            {
                "heads": heads,
                "head_dim": head_dim,
                "name": incumbent_name,
                "timed": timed,
                "incumbent": incumbent,
            }
        )
        out = found(head_dim) if callable(found) else found
        return {"impl": impl, **(out or {"ok": True, "min_tokens": 0})}

    monkeypatch.setattr(I, "verdict", verdict)
    return seen


# ----------------------------------------------------------------------------------------------- the route marker


def test_route_reads_as_its_incumbent():
    route = _route(incumbent = AttentionBackendName._NATIVE_CUDNN)
    assert route == "_native_cudnn" and route.value == "_native_cudnn"
    assert AttentionBackendName(route) is AttentionBackendName._NATIVE_CUDNN
    assert str(getattr(route, "value", route)) == "_native_cudnn"
    assert I._sdpa_incumbent(route)[0] == "_native_cudnn"
    assert D._backend_name(None) == "native" and _route(incumbent = "sage") == "sage"
    # a route over a route keeps the first incumbent
    assert (
        D.Int8Route(route, 0, torch.float16, 0, 64).incumbent is AttentionBackendName._NATIVE_CUDNN
    )


def test_route_survives_copies():
    route = _route(min_tokens = 2560, dtype = torch.float16, head_dim = 128)
    for clone in (copy.deepcopy(route), copy.copy(route), pickle.loads(pickle.dumps(route))):
        assert isinstance(clone, D.Int8Route) and clone == route
        assert (clone.min_tokens, clone.dtype, clone.head_dim, clone.index, clone.impl) == (
            2560,
            torch.float16,
            128,
            0,
            "cuda",
        )
        assert clone.incumbent is AttentionBackendName.NATIVE


# ----------------------------------------------------------------------------------------------- which calls route


def _call(route, *args, **kwargs):
    return D._reason(route, args, {"backend": route, **kwargs})


def test_clean_self_and_cross_attention_route(card):
    q, k, v = _qkv()
    assert _call(_route(), q, k, v) is None
    assert _call(_route(), q, k, v, attn_mask = None, dropout_p = 0.0, is_causal = False) is None
    assert _call(_route(), q, k, v, scale = 64**-0.5) is None
    q, k, v = _qkv(sq = 40, sk = 24)
    assert _call(_route(), q, k, v) is None


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"attn_mask": torch.ones(1, 1, 1, 32, dtype = torch.bool)}, "mask"),
        ({"is_causal": True}, "causal"),
        ({"dropout_p": 0.1}, "dropout"),
        ({"enable_gqa": True}, "gqa"),
        ({"scale": 0.5}, "scale"),
        ({"attention_kwargs": {"x": 1}}, "attention kwargs"),
        ({"parallel_config": object()}, "context parallel"),
        ({"return_lse": True}, "unknown arguments"),
    ],
)
def test_calls_the_kernel_cannot_take_keep_the_incumbent(card, kwargs, reason):
    q, k, v = _qkv()
    assert _call(_route(), q, k, v, **kwargs) == reason


def test_tensor_checks(card, monkeypatch):
    q, k, v = _qkv()
    assert _call(_route(dtype = torch.float16), q, k, v) == "dtype"
    assert _call(_route(head_dim = 128), q, k, v) == "shapes"
    assert _call(_route(min_tokens = 33), q, k, v) == "short"
    q2, k2, v2 = _qkv(sq = 4096, sk = 512)
    assert (
        _call(_route(min_tokens = 1024), q2, k2, v2) == "short"
    )  # short text keys of a cross-attention
    assert _call(_route(), q, k, v[:, :16]) == "not [B, S, H, D]"
    assert _call(_route(), q[0], k, v) == "not [B, S, H, D]"
    qg, kg, vg = _qkv(grad = True)
    assert _call(_route(), qg, kg, vg) == "needs gradients"
    with torch.no_grad():
        assert _call(_route(), qg, kg, vg) is None
    monkeypatch.setattr(D, "_on_card", lambda x, route: False)
    assert _call(_route(), q, k, v) == "device"


def test_real_device_check_on_cpu():
    q, k, v = _qkv()
    assert _call(_route(), q, k, v) == "device"


def test_wrapper_routes_and_passes_the_incumbent(card, monkeypatch):
    seen = []

    def original(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return I.native_attention(args[0], args[1], args[2])

    monkeypatch.setitem(D._STATE, "original", original)
    q, k, v = _qkv()
    route = _route(incumbent = AttentionBackendName._NATIVE_CUDNN)
    out = D.routed_dispatch_attention_fn(q, k, v, backend = route)
    assert card == [((1, 32, 2, 64), (1, 32, 2, 64))] and not seen
    assert torch.allclose(out.float(), I.native_attention(q, k, v).float() * 0.5, atol = 1e-2)
    mask = torch.ones(1, 1, 1, 32, dtype = torch.bool)
    D.routed_dispatch_attention_fn(q, k, v, attn_mask = mask, backend = route)
    # the incumbent gets its own backend, never the marker
    assert seen == [AttentionBackendName._NATIVE_CUDNN] and type(seen[0]) is AttentionBackendName
    D.routed_dispatch_attention_fn(q, k, v, backend = AttentionBackendName.NATIVE)
    assert seen[-1] is AttentionBackendName.NATIVE and len(card) == 1
    assert D.ROUTE_STATS == {"int8": 1, "mask": 1}


def test_wrapper_patches_diffusers_modules():
    from diffusers.models.transformers import transformer_wan

    original = D.original_dispatch()
    assert original is not None and not getattr(original, "_unsloth_int8_dispatch", False)
    assert D.install_dispatch_route() >= 2
    assert transformer_wan.dispatch_attention_fn is D.routed_dispatch_attention_fn
    assert attention_dispatch.dispatch_attention_fn is D.routed_dispatch_attention_fn
    assert D.original_dispatch() is original
    # a plain backend passes through unchanged
    q, k, v = _qkv(dtype = torch.float32)
    ref = original(q, k, v, backend = AttentionBackendName.NATIVE)
    out = D.routed_dispatch_attention_fn(q, k, v, backend = AttentionBackendName.NATIVE)
    assert torch.equal(out, ref)


# ----------------------------------------------------------------------------------------------- real processors


def _wan_dit():
    from diffusers import WanTransformer3DModel
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 64,
        in_channels = 4,
        out_channels = 4,
        text_dim = 32,
        freq_dim = 32,
        ffn_dim = 64,
        num_layers = 2,
        rope_max_seq_len = 32,
    ).to(torch.bfloat16)


def _wan_inputs(
    frames = 2,
    size = 8,
    text = 6,
):
    g = torch.Generator().manual_seed(1)
    return dict(
        hidden_states = torch.randn(1, 4, frames, size, size, generator = g).to(torch.bfloat16),
        timestep = torch.tensor([500]),
        encoder_hidden_states = torch.randn(1, text, 32, generator = g).to(torch.bfloat16),
        return_dict = False,
    )


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def test_wan_routes_self_and_cross_attention_unmasked(card, monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    dit = _wan_dit()
    dit.set_attention_backend("native")
    pipe = _Pipe(dit)
    assert D.install(pipe, _TARGET) == 4 and D.routed_count(pipe) == 4
    with torch.no_grad():
        dit(**_wan_inputs())
    # 2 blocks x (self, cross): 32 video tokens; the 6 text tokens are taken too (forced: no token floor)
    assert sorted(card) == sorted(
        [((1, 32, 2, 64), (1, 32, 2, 64))] * 2 + [((1, 32, 2, 64), (1, 6, 2, 64))] * 2
    )
    assert D.ROUTE_STATS == {"int8": 4}
    assert D.remove(pipe) == 4 and D.routed_count(pipe) == 0
    assert all(
        type(m.processor._attention_backend) is AttentionBackendName
        for m in D.routable_modules(dit)
    )


def test_wan_auto_race_floor_keeps_short_cross_attention(card, monkeypatch):
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    monkeypatch.setattr(I, "RACE_TOKENS", (64, 16))
    dit = _wan_dit()
    dit.set_attention_backend("native")
    pipe = _Pipe(dit)
    assert D.install(pipe, _TARGET) == 4
    assert all(s["timed"] and s["name"] == "dispatch:native" for s in seen)
    routes = {m.processor._attention_backend.min_tokens for m in D.routable_modules(dit)}
    assert routes == {16}
    with torch.no_grad():
        dit(**_wan_inputs())
    assert len(card) == 2 and D.ROUTE_STATS == {"int8": 2, "short": 2}


def test_race_incumbent_is_the_processor_backend(monkeypatch):
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": None})
    calls = []
    original = D.original_dispatch()

    def spy(*args, **kwargs):
        calls.append(kwargs.get("backend"))
        return original(*args, **kwargs)

    dit = _wan_dit()
    dit.set_attention_backend("native")
    D.install_dispatch_route()
    monkeypatch.setitem(D._STATE, "original", spy)
    monkeypatch.setattr(D, "long_verdict", lambda *a, **k: None)
    assert D.install(_Pipe(dit), _TARGET) == 0  # lost both races: nothing routes
    q, k, v = _qkv(dtype = torch.float32)
    seen[0]["incumbent"](q, k, v)
    assert calls == [AttentionBackendName.NATIVE]


def test_long_race_routes_long_calls_only(card, monkeypatch):
    """A card where the kernel loses at the race's sizes but wins at video lengths routes from that length on."""
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": None})
    asked = []
    monkeypatch.setattr(D, "long_verdict", lambda *a, **k: asked.append(a[2:5]) or 24)
    dit = _wan_dit()
    dit.set_attention_backend("native")
    pipe = _Pipe(dit)
    assert D.install(pipe, _TARGET) == 4 and asked
    assert {m.processor._attention_backend.min_tokens for m in D.routable_modules(dit)} == {24}
    with torch.no_grad():
        dit(**_wan_inputs())
    assert D.ROUTE_STATS == {"int8": 2, "short": 2}
    # forced never needs it
    asked.clear()
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert D.install(pipe, _TARGET) == 4 and not asked


def test_long_verdict_thresholds(monkeypatch):
    times = {}

    def median_ms(fns, rounds = 5):
        return dict(times[current[0]])

    current = [None]

    def probe(
        device,
        dtype,
        tokens,
        heads,
        head_dim,
        seed = 0,
    ):
        current[0] = tokens
        return (None, None, None)

    monkeypatch.setattr(I, "_median_ms", median_ms)
    monkeypatch.setattr(I, "probe_inputs", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    large, small = D.LONG_RACE_TOKENS
    cases = [
        ({large: (10.0, 9.0), small: (5.0, 4.0)}, (small + I.RACE_TOKENS[0]) // 2),
        ({large: (10.0, 9.0), small: (5.0, 4.95)}, (large + small) // 2),
        ({large: (10.0, 9.9), small: (5.0, 4.0)}, None),
    ]
    for n, (rows, want) in enumerate(cases):
        times.clear()
        times.update({t: {"incumbent": a, "int8": b} for t, (a, b) in rows.items()})
        assert D.long_verdict("cuda:0", torch.bfloat16, 2, 64, None, f"case{n}") == want
    # cached per key
    times.clear()
    assert D.long_verdict("cuda:0", torch.bfloat16, 2, 64, None, "case0") == cases[0][1]


def test_masked_processors_keep_the_incumbent(card, monkeypatch):
    """Z-Image's padded batch and LTX-2's text cross-attention pass a mask: those calls never reach the kernel."""
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

    _fake_cuda(monkeypatch)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    attn = Attention(
        query_dim = 128,
        dim_head = 64,
        heads = 2,
        qk_norm = "rms_norm",
        eps = 1e-5,
        bias = False,
        out_bias = False,
        processor = ZSingleStreamAttnProcessor(),
    ).to(torch.bfloat16)
    dit = torch.nn.Module()
    dit.attention = attn
    D.install(_Pipe(dit), _TARGET)
    assert isinstance(attn.processor._attention_backend, D.Int8Route)
    x = torch.randn(2, 12, 128).to(torch.bfloat16)
    mask = torch.ones(2, 12, dtype = torch.bool)
    mask[1, 9:] = False
    with torch.no_grad():
        attn(x, attention_mask = mask)
        attn(x)
    assert D.ROUTE_STATS == {"mask": 1, "int8": 1} and len(card) == 1


def test_ltx2_text_cross_attention_masked(card, monkeypatch):
    ltx2 = pytest.importorskip("diffusers.models.transformers.transformer_ltx2")
    _fake_cuda(monkeypatch)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    attn = ltx2.LTX2Attention(query_dim = 128, heads = 2, kv_heads = 2, dim_head = 64).to(torch.bfloat16)
    dit = torch.nn.Module()
    dit.attn = attn
    assert D.install(_Pipe(dit), _TARGET) == 1
    x = torch.randn(1, 16, 128).to(torch.bfloat16)
    text = torch.randn(1, 8, 128).to(torch.bfloat16)
    bias = torch.zeros(1, 1, 8).to(torch.bfloat16)
    bias[..., 6:] = -10000.0
    with torch.no_grad():
        attn(x)
        attn(x, encoder_hidden_states = text, attention_mask = bias)
    assert D.ROUTE_STATS == {"int8": 1, "mask": 1}


def test_strided_and_non_dispatch_processors_are_left_alone(monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")

    class Strided:
        _unsloth_int8 = False
        _attention_backend = None
        _parallel_config = None

        def __call__(self, attn, x):
            from diffusers.models.attention_dispatch import dispatch_attention_fn
            return dispatch_attention_fn(x, x, x)

    class Plain:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, attn, x):
            return F.scaled_dot_product_attention(x, x, x)

    class Parallel:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, attn, x):
            from diffusers.models.attention_dispatch import dispatch_attention_fn
            return dispatch_attention_fn(x, x, x)

    dit = torch.nn.Module()
    for name, proc in (("a", Strided()), ("b", Plain()), ("c", Parallel())):
        m = torch.nn.Module()
        m.heads = 2
        m.head_dim = 64
        m.processor = proc
        setattr(dit, name, m)
    dit.c.processor._parallel_config = object()
    assert D.routable_modules(dit) == []
    dit.c.processor._parallel_config = None
    assert D.routable_modules(dit) == [dit.c]


# ----------------------------------------------------------------------------------------------- switches and gate


def test_kill_switches_and_off_card(monkeypatch):
    dit = _wan_dit()
    pipe = _Pipe(dit)
    assert D.install(pipe, types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16)) == 0
    seen = _fake_cuda(monkeypatch)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert D.install(pipe, _TARGET, requested = "unsloth_int8") == 0
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    monkeypatch.setenv(D.DISPATCH_ENV, "0")
    assert D.install(pipe, _TARGET) == 0 and not seen
    monkeypatch.delenv(D.DISPATCH_ENV)
    assert D.install(pipe, _TARGET) == 4
    # the switch turned off later: the next install drops the routes
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert D.install(pipe, _TARGET) == 0 and D.routed_count(pipe) == 0


def test_auto_only_races_listed_families(monkeypatch):
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit = _wan_dit()
    pipe = _Pipe(dit)
    monkeypatch.setattr(D, "AUTO_DITS", frozenset())
    assert D.install(pipe, _TARGET) == 0 and not seen
    # by name it routes any family, untimed
    assert D.install(pipe, _TARGET, requested = "unsloth_int8") == 4
    assert seen and not any(s["timed"] for s in seen)
    routes = {m.processor._attention_backend.min_tokens for m in D.routable_modules(dit)}
    assert routes == {0}


def test_gate_failure_keeps_native_even_when_forced(monkeypatch):
    _fake_cuda(monkeypatch, found = {"ok": False, "min_tokens": None})
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    pipe = _Pipe(_wan_dit())
    assert D.install(pipe, _TARGET) == 0 and D.routed_count(pipe) == 0


def test_no_kernel_for_the_card(monkeypatch):
    seen = _fake_cuda(monkeypatch, impl = None)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert D.install(_Pipe(_wan_dit()), _TARGET) == 0 and not seen


def test_triton_stand_in_races_even_by_name(monkeypatch):
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 2560}, impl = "triton")
    pipe = _Pipe(_wan_dit())
    assert D.install(pipe, _TARGET, requested = "unsloth_int8") == 4
    assert [s["timed"] for s in seen[:2]] == [False, True]
    assert {
        m.processor._attention_backend.min_tokens for m in D.routable_modules(pipe.transformer)
    } == {2560}


def test_rocm_auto_stays_off(monkeypatch):
    from core.inference import rocm_bf16

    seen = _fake_cuda(monkeypatch)
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: True)
    assert D.install(_Pipe(_wan_dit()), _TARGET) == 0 and not seen


def test_install_never_raises(monkeypatch):
    _fake_cuda(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(I, "verdict", boom)
    pipe = _Pipe(_wan_dit())
    assert D.install(pipe, _TARGET) == 0 and D.routed_count(pipe) == 0


def test_status_reports_routes(card, monkeypatch):
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 2560})
    pipe = _Pipe(_wan_dit())
    D.install(pipe, _TARGET)
    found = D.status(pipe)
    assert found["modules"] == 4 and found["enabled"]
    assert list(found["routes"].values()) == [4]


# ----------------------------------------------------------------------------------------------- the install hook


def test_apply_attention_backend_installs_the_route(monkeypatch):
    from core.inference import diffusion_attention as A

    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: "_native_cudnn")
    monkeypatch.setattr(I, "install_requested", lambda *a, **k: 0)
    got = []
    monkeypatch.setattr(
        D, "install", lambda pipe, target, requested = None, logger = None: got.append(requested) or 3
    )
    assert (
        A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, requested = "auto")
        == I.BACKEND
    )
    monkeypatch.setattr(D, "install", lambda *a, **k: 0)
    assert A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET) == "_native_cudnn"
    assert got == ["auto"]
    # no target (a status-only call): never probes
    monkeypatch.setattr(D, "install", lambda *a, **k: pytest.fail("probed without a target"))
    assert A.apply_attention_backend(object(), None) == "_native_cudnn"


def test_video_load_passes_the_request():
    import inspect

    from core.inference import video

    source = inspect.getsource(video)
    assert "requested = attention_backend," in source


# ----------------------------------------------------------------------------------------------- torch.compile


def test_compiled_wan_block_has_no_graph_break(card):
    dit = _wan_dit()
    dit.set_attention_backend("native")
    assert D.install_dispatch_route()
    for m in D.routable_modules(dit):
        m.processor._attention_backend = _route()
    block = dit.blocks[0]
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 32, 128, generator = g).to(torch.bfloat16)
    text = torch.randn(1, 6, 128, generator = g).to(torch.bfloat16)
    temb = torch.randn(1, 6, 128, generator = g).to(torch.bfloat16)
    rope = dit.rope(torch.zeros(1, 4, 2, 8, 8))
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        eager = block(x, text, temb, rope)
        n_eager = len(card)
        compiled = torch.compile(block, backend = backend, fullgraph = True)
        out = compiled(x, text, temb, rope)
    assert len(graphs) == 1
    op = torch.ops.unsloth_studio.int8_attention_nhd
    ops = [n.target for n in graphs[0].graph.nodes if n.op == "call_function"]
    assert sum(1 for t in ops if t is op or t is op.default) == 2
    assert len(card) == 2 * n_eager and torch.allclose(out.float(), eager.float(), atol = 1e-2)
    torch._dynamo.reset()


def test_compiled_masked_call_stays_in_graph(card, monkeypatch):
    seen = []

    def original(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return F.scaled_dot_product_attention(
            args[0].transpose(1, 2),
            args[1].transpose(1, 2),
            args[2].transpose(1, 2),
            attn_mask = kwargs.get("attn_mask"),
        ).transpose(1, 2)

    monkeypatch.setitem(D._STATE, "original", original)
    route = _route(min_tokens = 16)

    def fn(q, k, v, mask):
        a = D.routed_dispatch_attention_fn(q, k, v, attn_mask = mask, backend = route)
        b = D.routed_dispatch_attention_fn(q, k, v, backend = route)
        return a + b

    q, k, v = _qkv()
    mask = torch.ones(1, 1, 1, 32, dtype = torch.bool)
    torch._dynamo.reset()
    with torch.no_grad():
        explain = torch._dynamo.explain(fn)(q, k, v, mask)
    assert explain.graph_break_count == 0 and explain.graph_count == 1
    torch._dynamo.reset()


# ----------------------------------------------------------------------------------------------- CUDA (real kernel)


def _card_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16


needs_kernel = pytest.mark.skipif(
    not torch.cuda.is_available()
    or I.implementation(torch.cuda.current_device(), 64) is None
    or I.implementation(torch.cuda.current_device(), 128) is None,
    reason = "needs a CUDA card with an INT8 attention kernel",
)


@pytest.mark.gpu
@needs_kernel
def test_cuda_wan_dit_routes_and_matches(monkeypatch):
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    dtype = _card_dtype()
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    dit = WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 4,
        attention_head_dim = 128,
        in_channels = 16,
        out_channels = 16,
        text_dim = 64,
        freq_dim = 64,
        ffn_dim = 256,
        num_layers = 2,
        rope_max_seq_len = 64,
    ).to("cuda", dtype)
    dit.set_attention_backend("native")
    g = torch.Generator().manual_seed(3)
    kwargs = dict(
        hidden_states = torch.randn(1, 16, 4, 48, 48, generator = g).to("cuda", dtype),
        timestep = torch.tensor([500], device = "cuda"),
        encoder_hidden_states = torch.randn(1, 64, 64, generator = g).to("cuda", dtype),
        return_dict = False,
    )
    with torch.no_grad():
        ref = dit(**kwargs)[0].float()
        target = types.SimpleNamespace(device = "cuda", dtype = dtype)
        assert D.install(_Pipe(dit), target) == 4
        before = I.STATS["int8"]
        got = dit(**kwargs)[0].float()
        assert I.STATS["int8"] - before == 4
        D.remove(_Pipe(dit))
    rel = float((got - ref).norm() / ref.norm())
    cos = float(F.cosine_similarity(got.flatten(), ref.flatten(), dim = 0))
    assert rel < 0.05 and cos > 0.998, (rel, cos)


@pytest.mark.gpu
@needs_kernel
def test_cuda_compiled_and_graphed_dispatch_call(monkeypatch):
    dtype = _card_dtype()
    I.int8_attention_op()
    index = torch.cuda.current_device()
    found = I.verdict(torch.device("cuda", index), dtype, 4, 128, None, "native", timed = False)
    assert found["ok"]
    D.install_dispatch_route()
    route = D.Int8Route(AttentionBackendName.NATIVE, 0, dtype, index, 128, found["impl"])
    q, k, v = (torch.randn(1, 2048, 4, 128, device = "cuda", dtype = dtype) for _ in range(3))

    def fn(q, k, v):
        return D.routed_dispatch_attention_fn(q, k, v, backend = route)

    torch._dynamo.reset()
    with torch.no_grad():
        ref = I.reference_attention(q, k, v)
        compiled = torch.compile(fn, fullgraph = True)
        out = compiled(q, k, v)
        assert I.accuracy(out, ref)["cos"] > 0.999
        static = [x.clone() for x in (q, k, v)]
        fn(*static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn(*static)
        for x, y in zip(static, (k, q, v)):
            x.copy_(y)
        graph.replay()
        torch.cuda.synchronize()
        assert I.accuracy(captured, I.reference_attention(k, q, v))["cos"] > 0.999
    torch._dynamo.reset()


def test_auto_list_matches_the_measured_families():
    """Qwen-Image-2.1 and Wan 2.2 race by default; Z-Image-Turbo and LTX-2 (audio output) have a per-step error with
    the kernel above their shipped INT8 weight error, so they route only on request."""
    assert D.AUTO_DITS == {"QwenImage21Transformer2DModel", "WanTransformer3DModel"}
