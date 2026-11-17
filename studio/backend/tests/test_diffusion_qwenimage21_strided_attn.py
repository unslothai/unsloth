# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 strided cuDNN attention (``diffusion_qwenimage21_attn``): the pin marker, which dispatch calls take
the strided call, the INT8 route's declined calls, the install (switch, probe, backends), a compiled call with no
graph break, and on CUDA the bitwise match with diffusers' contiguous ``_native_cudnn`` call."""

from __future__ import annotations

import copy
import pickle
import types

import pytest

torch = pytest.importorskip("torch")
import torch._dynamo  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from core.inference import diffusion_int8_attention_dispatch as D  # noqa: E402
from core.inference import diffusion_qwenimage21_attn as Q  # noqa: E402

attention_dispatch = pytest.importorskip("diffusers.models.attention_dispatch")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(Q.STRIDED_ATTN_ENV, raising = False)
    Q.STATS.clear()
    D._DECISIONS.clear()
    probed = dict(Q._PROBED)
    yield
    Q.STATS.clear()
    Q._PROBED.clear()
    Q._PROBED.update(probed)
    D._DECISIONS.clear()


def _native(q, k, v):
    return F.scaled_dot_product_attention(*(x.permute(0, 2, 1, 3) for x in (q, k, v))).permute(
        0, 2, 1, 3
    )


@pytest.fixture
def strided(monkeypatch):
    """CPU tensors count as probed-card tensors; the strided call records and returns native attention."""
    calls = []
    monkeypatch.setattr(Q, "_on_card", lambda x, backend: True)

    def fake(q, k, v):
        calls.append((tuple(q.shape), tuple(k.shape), q.stride()))
        return _native(q, k, v)

    monkeypatch.setattr(Q, "strided_cudnn_attention", fake)
    return calls


def _qkv(
    sq = 24,
    sk = None,
    heads = 2,
    head_dim = 16,
    dtype = torch.bfloat16,
):
    g = torch.Generator().manual_seed(0)
    sk = sq if sk is None else sk
    # the processor's layout: query unflattened from the projection row, key / value as the cached concatenation
    q = torch.randn(1, sq, heads * head_dim, generator = g).to(dtype).unflatten(-1, (heads, head_dim))
    k = torch.randn(1, sk, heads, head_dim, generator = g).to(dtype)
    v = torch.randn(1, sk, heads, head_dim, generator = g).to(dtype)
    return q, k, v


def test_marker_reads_as_cudnn_and_survives_copies():
    m = Q.StridedCudnn((1, 0))
    assert m == "_native_cudnn" and m.value == "_native_cudnn" and type(m.value) is str
    assert m.name == "_NATIVE_CUDNN" and m.cards == frozenset({0, 1})
    for other in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert type(other) is Q.StridedCudnn and other.cards == m.cards and other == "_native_cudnn"
    # diffusers' dispatch, which the declined calls reach, resolves it to its cuDNN backend
    assert (
        attention_dispatch.AttentionBackendName(m)
        is attention_dispatch.AttentionBackendName._NATIVE_CUDNN
    )


def test_plain_calls_only(strided):
    m = Q.StridedCudnn((0,))
    q, k, v = _qkv()
    assert Q.plain_qkv(m, (q, k, v), {"backend": m}) is not None
    assert (
        Q.plain_qkv(m, (q, k, v), {"attn_mask": None, "dropout_p": 0.0, "backend": m}) is not None
    )
    mask = torch.ones(1, 1, q.shape[1], k.shape[1], dtype = torch.bool)
    for kwargs in (
        {"attn_mask": mask},
        {"dropout_p": 0.1},
        {"is_causal": True},
        {"scale": 0.25},
        {"enable_gqa": True},
        {"attention_kwargs": {"x": 1}},
        {"parallel_config": object()},
        {"return_lse": True},
    ):
        assert Q.plain_qkv(m, (q, k, v), {"backend": m, **kwargs}) is None, kwargs
    assert Q.plain_qkv(m, (q.flatten(2, 3), k, v), {}) is None
    assert Q.plain_qkv(m, (q.float(), k.float(), v.float()), {}) is None
    assert Q.plain_qkv(m, (q, k[:, :-1], v), {}) is None
    assert Q.plain_qkv(m, (q[..., ::2], k[..., ::2], v[..., ::2]), {}) is None


def test_marker_admits_only_its_probed_dtype(strided):
    m = Q.StridedCudnn((0,), torch.bfloat16)
    q, k, v = _qkv()
    assert Q.plain_qkv(m, (q, k, v), {}) is not None
    # a bf16 probe says nothing about fp16 calls (an autocast or a later cast of the pipe)
    assert Q.plain_qkv(m, (q.half(), k.half(), v.half()), {}) is None
    for other in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert other.dtype is torch.bfloat16 and other.cards == m.cards


def test_reinstall_at_a_new_dtype_probes_it(monkeypatch):
    _tq, model = _tiny_qwen21()
    pipe = types.SimpleNamespace(transformer = model)
    seen = []
    _fake_card(monkeypatch)
    monkeypatch.setattr(
        Q, "strided_matches", lambda dev, dtype, h, d: seen.append(dtype) or dtype == torch.bfloat16
    )
    _pin(model, attention_dispatch.AttentionBackendName._NATIVE_CUDNN)
    assert Q.install(pipe, "_native_cudnn", _TARGET) == 2
    assert all(b.dtype is torch.bfloat16 for b in _backends(model))
    half = types.SimpleNamespace(device = "cuda:0", dtype = torch.float16)
    assert Q.install(pipe, "_native_cudnn", half) == 0 and torch.float16 in seen
    assert all(b.dtype is torch.bfloat16 for b in _backends(model))


def test_strided_incumbent_races_under_its_own_name():
    contiguous = attention_dispatch.AttentionBackendName._NATIVE_CUDNN
    assert D._race_name(contiguous) == "dispatch:_native_cudnn"
    assert D._race_name(Q.StridedCudnn((0,), torch.bfloat16)) == "dispatch:_native_cudnn:strided"


def test_unprobed_card_keeps_diffusers():
    q, k, v = _qkv()
    assert (
        Q.plain_qkv(Q.StridedCudnn((0,)), (q, k, v), {}) is None
    )  # CPU tensors are on no probed card


def test_dispatch_sends_plain_calls_strided(strided, monkeypatch):
    original = []
    monkeypatch.setitem(
        D._STATE, "original", lambda *a, **k: original.append(k.get("backend")) or "orig"
    )
    m = Q.StridedCudnn((0,))
    q, k, v = _qkv()
    out = D.routed_dispatch_attention_fn(q, k, v, attn_mask = None, dropout_p = 0.0, backend = m)
    assert len(strided) == 1 and torch.equal(out, _native(q, k, v))
    # the strided call saw the processor's own views, no contiguous copy
    assert strided[0][2] == q.stride()
    mask = torch.ones(1, 1, q.shape[1], k.shape[1], dtype = torch.bool)
    assert D.routed_dispatch_attention_fn(q, k, v, attn_mask = mask, backend = m) == "orig"
    # any other backend passes through untouched
    assert D.routed_dispatch_attention_fn(q, k, v, backend = "_native_cudnn") == "orig"
    assert D.routed_dispatch_attention_fn(q, k, v, backend = None) == "orig"
    assert len(strided) == 1 and original == [m, "_native_cudnn", None]
    assert Q.STATS == {"strided": 1, "stock": 1}


def test_int8_route_declined_call_runs_strided(strided, monkeypatch):
    monkeypatch.setitem(D._STATE, "original", lambda *a, **k: pytest.fail("diffusers' copies ran"))
    m = Q.StridedCudnn((0,))
    route = D.Int8Route(m, 1 << 20, torch.bfloat16, 0, 16, "cuda")
    assert route.incumbent is m and route == "_native_cudnn"
    q, k, v = _qkv()
    out = D.routed_dispatch_attention_fn(q, k, v, backend = route)
    assert len(strided) == 1 and torch.equal(out, _native(q, k, v))
    # the race times the incumbent the way the processor runs it
    assert Q.maybe_strided(route.incumbent, (q, k, v), {"backend": route.incumbent}) is not None


def test_switch_off_restores_stock(monkeypatch):
    monkeypatch.setenv(Q.STRIDED_ATTN_ENV, "0")
    assert not Q.enabled()
    assert Q.install(types.SimpleNamespace(transformer = None), "_native_cudnn", object()) == 0


def test_compiled_call_has_no_graph_break(strided):
    m = Q.StridedCudnn((0,))
    q, k, v = _qkv()
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    def fn(q, k, v):
        return D.routed_dispatch_attention_fn(q, k, v, attn_mask = None, backend = m).flatten(2, 3) * 2

    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend = backend, fullgraph = True)(q, k, v)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1 and torch.equal(out, _native(q, k, v).flatten(2, 3) * 2)


# ----------------------------------------------------------------------------------------- install on a tiny DiT


def _tiny_qwen21():
    tq = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    torch.manual_seed(0)
    model = tq.QwenImage21Transformer2DModel(
        patch_size = 1,
        in_channels = 8,
        out_channels = 8,
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        context_in_dim = 32,
        axes_dims_rope = (4, 6, 6),
    )
    return tq, model.to(torch.bfloat16).eval()


def _fake_card(monkeypatch, probe = True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(Q, "is_rocm_torch", lambda _torch: False)
    monkeypatch.setattr(Q, "_cards", lambda target, dit: [0])
    seen = []
    monkeypatch.setattr(
        Q, "strided_matches", lambda dev, dtype, h, d: seen.append((dev.index, h, d)) or probe
    )
    monkeypatch.setattr(D, "install_dispatch_route", lambda: 1)
    return seen


_TARGET = types.SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)


def _pin(model, backend):
    for m in model.modules():
        if type(getattr(m, "processor", None)).__name__ == "QwenImage21AttnProcessor":
            m.processor._attention_backend = backend


def _backends(model):
    return [
        m.processor._attention_backend
        for m in model.modules()
        if type(getattr(m, "processor", None)).__name__ == "QwenImage21AttnProcessor"
    ]


def test_install_swaps_only_cudnn_pins(monkeypatch):
    tq, model = _tiny_qwen21()
    pipe = types.SimpleNamespace(transformer = model)
    seen = _fake_card(monkeypatch)
    _pin(model, "native")
    assert Q.install(pipe, "native", _TARGET) == 0
    _pin(model, attention_dispatch.AttentionBackendName._NATIVE_CUDNN)
    assert Q.install(pipe, "_native_cudnn", _TARGET) == 2
    marks = _backends(model)
    assert all(isinstance(b, Q.StridedCudnn) and b.cards == frozenset({0}) for b in marks)
    assert seen and all(s == (0, 2, 16) for s in seen)
    # idempotent, and a re-pin through diffusers drops the marker until the next install
    assert Q.install(pipe, "_native_cudnn", _TARGET) == 2
    model.set_attention_backend("_native_cudnn")
    assert not any(isinstance(b, Q.StridedCudnn) for b in _backends(model))


def test_install_needs_the_probe(monkeypatch):
    _tq, model = _tiny_qwen21()
    _fake_card(monkeypatch, probe = False)
    _pin(model, "_native_cudnn")
    assert Q.install(types.SimpleNamespace(transformer = model), "_native_cudnn", _TARGET) == 0
    assert _backends(model) == ["_native_cudnn"] * 2 and not any(
        isinstance(b, Q.StridedCudnn) for b in _backends(model)
    )


def test_install_skips_rocm(monkeypatch):
    _tq, model = _tiny_qwen21()
    _fake_card(monkeypatch)
    monkeypatch.setattr(Q, "is_rocm_torch", lambda _torch: True)
    _pin(model, "_native_cudnn")
    assert Q.install(types.SimpleNamespace(transformer = model), "_native_cudnn", _TARGET) == 0


def test_real_processor_decode_call_takes_the_strided_path(monkeypatch, strided):
    """The stock processor's decode call (no segments, no mask) reaches the strided call with the marker, and its
    output equals the same processor on the native backend."""
    tq, model = _tiny_qwen21()
    _fake_card(monkeypatch)
    _pin(model, "_native_cudnn")
    assert (
        D.install_dispatch_route
    )  # patched to a no-op by _fake_card; point the module at the wrapper ourselves
    monkeypatch.setattr(tq, "dispatch_attention_fn", D.routed_dispatch_attention_fn)
    monkeypatch.setitem(D._STATE, "original", attention_dispatch.dispatch_attention_fn)
    Q.install(types.SimpleNamespace(transformer = model), "_native_cudnn", _TARGET)
    attn = next(m for m in model.modules() if type(m).__name__ == "QwenImage21Attention")
    x = torch.randn(1, 12, attn.inner_dim, generator = torch.Generator().manual_seed(3)).to(
        torch.bfloat16
    )
    with torch.no_grad():
        got = attn(x)
        assert len(strided) == 1
        attn.processor._attention_backend = "native"
        ref = attn(x)
    assert torch.equal(got, ref)


def test_attention_hook_installs_on_cudnn_pins():
    import inspect

    from core.inference import diffusion_attention as A

    src = inspect.getsource(A._apply_attention_backend)
    branch = src[
        src.index(
            'if backend == "_native_cudnn" and target is not None and _is_cuda_nvidia(target):'
        ) :
    ]
    assert "install_q21_strided(pipe, backend, target, logger)" in branch[:600]


# ----------------------------------------------------------------------------------------------------- CUDA


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None,
    reason = "needs an NVIDIA CUDA device",
)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("sq,sk", [(4096, 4118), (1024, 1024)])
def test_strided_equals_diffusers_cudnn_bitwise(sq, sk):
    dev = torch.device("cuda", torch.cuda.current_device())
    if not Q.strided_matches(dev, torch.bfloat16, 32, 128):
        pytest.skip("strided cuDNN differs on this card: install keeps diffusers' call")
    g = torch.Generator(device = dev).manual_seed(5)
    q = (
        torch.randn(1, sq, 32 * 128, generator = g, device = dev)
        .to(torch.bfloat16)
        .unflatten(-1, (32, 128))
    )
    k = torch.randn(1, sk, 32, 128, generator = g, device = dev).to(torch.bfloat16)
    v = torch.randn(1, sk, 32, 128, generator = g, device = dev).to(torch.bfloat16)
    ref = attention_dispatch._native_cudnn_attention(q, k, v)
    got = Q.strided_cudnn_attention(q, k, v)
    assert torch.equal(ref, got) and torch.equal(ref.flatten(2, 3), got.flatten(2, 3))
    # no output copy: a contiguous [B, S, H, D] output makes the processor's flatten a view
    flat = got.flatten(2, 3)
    assert got.is_contiguous() and flat.data_ptr() == got.data_ptr() and flat._base is not None
