# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MPS large-attention guard: SDPA calls past the score cap that the prefill kernel does not take run in
query-row chunks (torch < 2.13 MPS is wrong past about 2^29 score elements), and the split is the same computation.
CPU tensors stand in for MPS here; tests/test_mps_attention_families.py runs the real kernel on a Mac."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core import _mps_sdpa_guard as G  # noqa: E402

F = torch.nn.functional


@pytest.fixture
def as_mps(monkeypatch):
    """CPU tensors count as MPS, no prefill kernel, and a small cap so the shapes below split."""
    monkeypatch.setattr(G, "_is_mps", lambda t: isinstance(t, torch.Tensor))
    monkeypatch.setattr(G, "_prefill_serves", lambda *a: False)
    monkeypatch.setattr(G, "MPS_SCORE_ELEMENTS", 2 * 3 * 40 * 64)
    monkeypatch.setitem(G.stats, "chunked_calls", 0)
    return G.guard_sdpa(F.scaled_dot_product_attention)


def _qkv(
    b = 2,
    h = 3,
    sq = 100,
    sk = 64,
    d = 16,
    hk = None,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, sq, d, generator = g, dtype = torch.float64)
    k = torch.randn(b, hk or h, sk, d, generator = g, dtype = torch.float64)
    v = torch.randn(b, hk or h, sk, d, generator = g, dtype = torch.float64)
    return q, k, v


def test_chunk_rows_bounds_the_scores(monkeypatch):
    monkeypatch.setattr(G, "MPS_SCORE_ELEMENTS", 1000)
    q = torch.empty(1, 2, 50, 4)
    k = torch.empty(1, 2, 20, 4)
    assert G.chunk_rows(q, k) == 25  # 2 * 25 * 20 = 1000
    assert G.chunk_rows(q[:, :, :25], k) is None
    assert G.chunk_rows(q[:, :, :1], k) is None


def test_default_cap_is_studios_measured_bound():
    from core.inference.diffusion_qwenimage21_math import MPS_SCORE_ELEMENTS

    assert G.MPS_SCORE_ELEMENTS == MPS_SCORE_ELEMENTS == 2**28
    # flux.1 at 1024 px (24 heads, 4096 + 512 tokens) is past it, a 512 px call is not.
    assert G.chunk_rows(torch.empty(1, 24, 4608, 1), torch.empty(1, 24, 4608, 1)) is not None
    assert G.chunk_rows(torch.empty(1, 24, 1536, 1), torch.empty(1, 24, 1536, 1)) is None


def test_split_matches_one_call(as_mps):
    q, k, v = _qkv()
    out = as_mps(q, k, v)
    assert G.stats["chunked_calls"] == 1
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol = 1e-12)


@pytest.mark.parametrize(
    "mask_shape",
    [(2, 3, 100, 64), (2, 1, 100, 64), (1, 1, 1, 64), (2, 1, 1, 64), (100, 64), (1, 64)],
)
@pytest.mark.parametrize("kind", ["bool", "float"])
def test_masks_are_split_with_the_rows(as_mps, mask_shape, kind):
    q, k, v = _qkv()
    g = torch.Generator().manual_seed(1)
    if kind == "bool":
        mask = torch.rand(mask_shape, generator = g) > 0.3
        mask[..., 0] = True
    else:
        mask = torch.randn(mask_shape, generator = g, dtype = torch.float64)
    out = as_mps(q, k, v, attn_mask = mask)
    assert G.stats["chunked_calls"] == 1
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, attn_mask = mask), atol = 1e-12)


@pytest.mark.parametrize("sq, sk", [(100, 100), (100, 64), (64, 100)])
def test_causal_keeps_top_left_alignment(as_mps, sq, sk):
    q, k, v = _qkv(sq = sq, sk = sk)
    out = as_mps(q, k, v, is_causal = True)
    assert G.stats["chunked_calls"] == 1
    ref = F.scaled_dot_product_attention(q, k, v, is_causal = True)
    assert torch.allclose(out, ref, atol = 1e-12, equal_nan = True)


def test_positional_args_scale_and_gqa(as_mps):
    q, k, v = _qkv(h = 6, hk = 3)
    out = as_mps(q, k, v, None, 0.0, False, scale = 0.3, enable_gqa = True)
    assert G.stats["chunked_calls"] == 1
    ref = F.scaled_dot_product_attention(q, k, v, None, 0.0, False, scale = 0.3, enable_gqa = True)
    assert torch.allclose(out, ref, atol = 1e-12)


def test_three_dim_and_grad(as_mps):
    q, k, v = (t[0].clone().requires_grad_(True) for t in _qkv())
    out = as_mps(q, k, v)
    assert G.stats["chunked_calls"] == 1
    out.sum().backward()
    grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    F.scaled_dot_product_attention(q, k, v).sum().backward()
    for got, t in zip(grads, (q, k, v)):
        assert torch.allclose(got, t.grad, atol = 1e-10)


def test_small_and_prefill_calls_go_straight_through(monkeypatch, as_mps):
    calls = []

    def fake(q, k, v, *args, **kwargs):
        calls.append(tuple(q.shape))
        return F.scaled_dot_product_attention(q, k, v, *args, **kwargs)

    guarded = G.guard_sdpa(fake)
    q, k, v = _qkv()
    guarded(q[:, :, :40], k, v)  # under the cap
    monkeypatch.setattr(G, "_prefill_serves", lambda *a: True)
    guarded(q, k, v)  # the prefill kernel never builds the scores
    assert G.stats["chunked_calls"] == 0
    assert calls == [(2, 3, 40, 16), (2, 3, 100, 16)]


def test_off_mps_is_untouched(monkeypatch):
    monkeypatch.setattr(G, "MPS_SCORE_ELEMENTS", 1)
    monkeypatch.setitem(G.stats, "chunked_calls", 0)
    q, k, v = _qkv()
    out = G.guard_sdpa(F.scaled_dot_product_attention)(q, k, v)
    assert G.stats["chunked_calls"] == 0
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v))


def test_guard_is_idempotent():
    once = G.guard_sdpa(F.scaled_dot_product_attention)
    assert G.guard_sdpa(once) is once
    assert once.__wrapped__ is F.scaled_dot_product_attention


@pytest.fixture
def restore_sdpa():
    original = F.scaled_dot_product_attention
    yield
    F.scaled_dot_product_attention = original


def test_install_only_on_a_mac_with_mps(monkeypatch, restore_sdpa):
    original = F.scaled_dot_product_attention
    monkeypatch.setattr(sys, "platform", "linux")
    assert G.install_mps_large_attention_guard() is False
    assert F.scaled_dot_product_attention is original

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(torch.backends, "mps", types.SimpleNamespace(is_available = lambda: False))
    assert G.install_mps_large_attention_guard() is False
    assert F.scaled_dot_product_attention is original

    monkeypatch.setattr(torch.backends, "mps", types.SimpleNamespace(is_available = lambda: True))
    monkeypatch.setenv(G.GUARD_ENV, "0")
    assert G.install_mps_large_attention_guard() is False
    assert F.scaled_dot_product_attention is original

    monkeypatch.delenv(G.GUARD_ENV)
    assert G.install_mps_large_attention_guard() is True
    wrapped = F.scaled_dot_product_attention
    assert getattr(wrapped, G._GUARD_ATTR) and wrapped.__wrapped__ is original
    assert G.install_mps_large_attention_guard() is True
    assert F.scaled_dot_product_attention is wrapped


def test_mps_image_loads_install_the_guard(monkeypatch):
    from core.inference import diffusion_attention as A

    seen = []
    monkeypatch.setattr(
        G, "install_mps_large_attention_guard", lambda log = None: seen.append(1) or True
    )
    monkeypatch.setattr(A, "guard_rocm_fused_sdpa", lambda *a, **k: None)
    monkeypatch.setattr(A, "install_rocm_contiguous_kv", lambda *a, **k: False)
    pipe = types.SimpleNamespace(transformer = None)
    for device, want in (("cuda", 0), ("cpu", 0), ("mps", 1)):
        seen.clear()
        target = types.SimpleNamespace(device = device, backend = device, torch_device = device)
        try:
            A._apply_attention_backend_impl(pipe, None, target = target)
        except Exception:  # noqa: BLE001 -- only the install call matters here
            pass
        assert len(seen) == want, device


def test_broadcast_batch_dims_count_every_score(as_mps):
    # K / V batch dims broadcast over Q's: SDPA builds 2 * 2 score matrices, not Q's 1 * 2.
    q, k, v = (t[:, :2].clone() for t in _qkv())
    q1 = q[:1].clone()
    assert G.chunk_rows(q1, k[:1]) is None and G.chunk_rows(q1, k) is not None
    out = as_mps(q1, k, v)
    assert G.stats["chunked_calls"] == 1
    assert torch.allclose(out, F.scaled_dot_product_attention(q1, k, v), atol = 1e-12)
    # GQA: K has fewer heads than Q, and the scores still follow Q's 6 heads.
    q6, k3 = torch.empty(1, 6, 100, 4), torch.empty(1, 3, 64, 4)
    assert G.chunk_rows(q6, k3) == G.MPS_SCORE_ELEMENTS // (6 * 64)


def _compiled(fn, **kwargs):
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    return torch.compile(fn, backend = counter, **kwargs), counter


@pytest.mark.parametrize("sq", [20, 40, 100, 200])  # under the cap, at it, and past it
def test_fullgraph_traces_below_and_above_the_cap(as_mps, sq):
    q, k, v = _qkv(sq = sq)
    fn, counter = _compiled(lambda q, k, v: as_mps(q, k, v), fullgraph = True)
    out = fn(q, k, v)
    assert counter.frame_count == 1
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol = 1e-12)


def test_dynamic_fullgraph_keeps_one_graph_under_the_cap(as_mps):
    fn, counter = _compiled(lambda q, k, v: as_mps(q, k, v), fullgraph = True, dynamic = True)
    for sq in (8, 20, 30, 40):
        q, k, v = _qkv(sq = sq)
        assert torch.allclose(fn(q, k, v), F.scaled_dot_product_attention(q, k, v), atol = 1e-12)
    assert counter.frame_count == 1


def test_repeated_split_calls_never_recompile(as_mps):
    q, k, v = _qkv()
    mask = torch.randn(
        2, 1, 100, 64, generator = torch.Generator().manual_seed(1), dtype = torch.float64
    )
    fn, counter = _compiled(lambda q, k, v, m: as_mps(q, k, v, attn_mask = m))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
    with torch._dynamo.config.patch(error_on_recompile = True):
        for _ in range(12):
            assert torch.allclose(fn(q, k, v, mask), ref, atol = 1e-12)
    assert counter.frame_count == 1
    assert G.stats["chunked_calls"] == 0  # counted in eager only: a traced bump guards on the value
    as_mps(q, k, v, attn_mask = mask)
    assert G.stats["chunked_calls"] == 1


_COLD_PREFILL_TRACE = """
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from core import _mps_sdpa_guard as G
with FakeTensorMode():
    q = torch.empty(2, 3, 32, 256, device = "mps")
    fn = torch.compile(lambda q: q if G._prefill_serves(q, q, q, None) else q + 1, backend = "eager", fullgraph = True)
    fn(q)
print("TRACED")
"""


def test_cold_prefill_check_traces_under_fullgraph():
    """The real prefill predicate (torch version parse included) traces under fullgraph=True in a fresh process,
    before anything warmed the regex cache."""
    import os
    import subprocess
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH = str(backend) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("PYTORCH_MPS_DISABLE_PREFILL_ATTENTION", None)
    proc = subprocess.run(
        [sys.executable, "-c", _COLD_PREFILL_TRACE],
        cwd = str(backend),
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert proc.returncode == 0 and "TRACED" in proc.stdout, proc.stderr[-3000:]
