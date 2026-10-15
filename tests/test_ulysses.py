# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses sequence parallel exchange and KV replication, on CPU gloo ranks."""

import importlib.util
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

_MODULE = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "ulysses.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ulysses_under_test", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


U = _load()


@dataclass
class _Config:
    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict] = None
    flash_varlen_kwargs: Optional[dict] = None
    sdpa_kwargs: Optional[dict] = None
    xformers_kwargs: Optional[dict] = None


@dataclass
class _Context:
    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Any
    attention_mask: Any
    causal_mask: Any
    sliding_window: Optional[int] = None
    prefix_seg_info: Any = None
    is_causal: bool = True


def _segment_mask(lengths, total):
    lengths = [int(x) for x in lengths]
    if sum(lengths) < total:
        lengths.append(total - sum(lengths))
    doc = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    pos = torch.arange(total)
    return (doc[:, None] == doc[None, :]) & (pos[None, :] <= pos[:, None])


def _fake_run_attention(*, config, context, Q, K, V):
    assert Q.shape == (context.bsz, context.n_heads, context.q_len, context.head_dim), Q.shape
    assert K.shape == (context.bsz, config.n_kv_heads, context.kv_seq_len, context.head_dim)
    assert context.n_heads == config.n_kv_heads * config.n_groups
    K = K.repeat_interleave(config.n_groups, dim = 1)
    V = V.repeat_interleave(config.n_groups, dim = 1)
    if context.seq_info is None:
        out = F.scaled_dot_product_attention(Q, K, V, is_causal = True)
    else:
        mask = _segment_mask(context.seq_info[0].tolist(), context.q_len)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask = mask)
    return out.transpose(1, 2)


def _reference(Q, K, V, lengths = None):
    G = Q.shape[1] // K.shape[1]
    K = K.repeat_interleave(G, dim = 1)
    V = V.repeat_interleave(G, dim = 1)
    if lengths is None:
        out = F.scaled_dot_product_attention(Q, K, V, is_causal = True)
    else:
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask = _segment_mask(lengths, Q.shape[2])
        )
    return out.transpose(1, 2)


def _spawn(fn, world, *args, start_method = "spawn"):
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.start_processes(
            _entry,
            args = (world, init_file, fn, args),
            nprocs = world,
            join = True,
            start_method = start_method,
        )


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method = f"file://{init_file}", rank = rank, world_size = world
    )
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _exchange_worker(rank, world):
    torch.manual_seed(0)
    B, H, T, D = 2, 2 * world, 3 * world, 4
    full = torch.randn(B, H, T, D, dtype = torch.float64)
    S = T // world
    local = full[:, :, rank * S : (rank + 1) * S]
    heads = U.seq_to_head(local, None)
    h = H // world
    torch.testing.assert_close(heads, full[:, rank * h : (rank + 1) * h], rtol = 0, atol = 0)
    back = U.head_to_seq(heads.transpose(1, 2), None)
    torch.testing.assert_close(back, local.transpose(1, 2), rtol = 0, atol = 0)


def _attention_worker(rank, world, n_heads, n_kv, raw_len, lengths, dtype):
    torch.manual_seed(1234)
    B = 1 if lengths is not None else 2
    D = 8
    T = -(-raw_len // world) * world
    Q = torch.randn(B, n_heads, T, D, dtype = dtype)
    K = torch.randn(B, n_kv, T, D, dtype = dtype)
    V = torch.randn(B, n_kv, T, D, dtype = dtype)
    W = torch.randn(B, T, n_heads, D, dtype = dtype)
    Q[:, :, raw_len:] = 0
    K[:, :, raw_len:] = 0
    V[:, :, raw_len:] = 0

    Qr, Kr, Vr = (x.clone().requires_grad_(True) for x in (Q, K, V))
    ref = _reference(Qr, Kr, Vr, lengths)
    (ref * W).sum().backward()

    S = T // world
    sl = slice(rank * S, (rank + 1) * S)
    Ql, Kl, Vl = (x[:, :, sl].clone().requires_grad_(True) for x in (Q, K, V))
    seq_info = None
    if lengths is not None:
        lens = torch.tensor(lengths, dtype = torch.int32)
        cu = torch.zeros(len(lengths) + 1, dtype = torch.int32)
        cu[1:] = torch.cumsum(lens, 0)
        seq_info = (lens, cu, int(lens.max()))
    config = _Config("sdpa", n_kv_heads = n_kv, n_groups = n_heads // n_kv)
    context = _Context(
        bsz = B,
        q_len = S,
        kv_seq_len = S,
        n_heads = n_heads,
        head_dim = D,
        requires_grad = True,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
    )
    out = U.ulysses_attention(_fake_run_attention, config, context, Ql, Kl, Vl, None)
    assert out.shape == (B, S, n_heads, D)
    tol = dict(rtol = 1e-10, atol = 1e-10) if dtype == torch.float64 else dict(rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(out, ref[:, sl].detach(), **tol)
    (out * W[:, sl]).sum().backward()
    torch.testing.assert_close(Ql.grad, Qr.grad[:, :, sl], **tol)
    torch.testing.assert_close(Kl.grad, Kr.grad[:, :, sl], **tol)
    torch.testing.assert_close(Vl.grad, Vr.grad[:, :, sl], **tol)


_CASES = {
    2: [(4, 4, torch.float64), (4, 2, torch.float64), (4, 1, torch.float64)],
    4: [(8, 1, torch.float64), (8, 2, torch.float32), (8, 8, torch.float32)],
}
_PACKED = {2: (4, 2), 4: (8, 1)}


def _all_cases_worker(rank, world):
    _exchange_worker(rank, world)
    for n_heads, n_kv, dtype in _CASES[world]:
        _attention_worker(rank, world, n_heads, n_kv, 4 * world + 3, None, dtype)
    raw_len = 6 * world + 1
    lengths = [5, 3 * world - 2, raw_len - 3 * world - 3]
    n_heads, n_kv = _PACKED[world]
    _attention_worker(rank, world, n_heads, n_kv, raw_len, lengths, torch.float64)


@pytest.mark.parametrize("world", [2, 4])
def test_ulysses_exchange_and_attention_match_full_sequence(world):
    _spawn(_all_cases_worker, world)


def test_head_plan():
    assert U.ulysses_head_plan(8, 8, 2) == (1, 4, 4, 1)
    assert U.ulysses_head_plan(8, 2, 2) == (1, 4, 1, 4)
    assert U.ulysses_head_plan(8, 2, 4) == (2, 2, 1, 2)
    assert U.ulysses_head_plan(8, 1, 8) == (8, 1, 1, 1)
    assert U.ulysses_head_plan(32, 8, 4) == (1, 8, 2, 4)
    with pytest.raises(ValueError, match = "attention heads"):
        U.ulysses_head_plan(6, 2, 4)
    with pytest.raises(ValueError, match = "divide one another"):
        U.ulysses_head_plan(24, 6, 4)
    with pytest.raises(ValueError, match = "multiple of key/value"):
        U.ulysses_head_plan(8, 3, 2)


def test_ulysses_attention_refuses_masks_and_cache():
    config = _Config("sdpa", n_kv_heads = 2, n_groups = 1)

    def ctx(**kw):
        base = dict(
            bsz = 1,
            q_len = 4,
            kv_seq_len = 4,
            n_heads = 2,
            head_dim = 4,
            requires_grad = False,
            seq_info = None,
            attention_mask = None,
            causal_mask = None,
        )
        base.update(kw)
        return _Context(**base)

    x = torch.zeros(1, 2, 4, 4)
    for bad, match in [
        (ctx(attention_mask = torch.ones(1, 4)), "dense attention mask"),
        (ctx(causal_mask = torch.ones(4, 4)), "dense attention mask"),
        (ctx(prefix_seg_info = object()), "PrefixGrouper"),
        (ctx(kv_seq_len = 8), "KV cache"),
    ]:
        with pytest.raises(RuntimeError, match = match):
            U.ulysses_attention(_fake_run_attention, config, bad, x, x, x, None)


def _real_dispatch_worker(rank, world, run_attention, AttentionConfig, AttentionContext):
    torch.manual_seed(7)
    B, H, Hkv, T, D = 2, 4, 2, 4 * world, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, Hkv, T, D)
    V = torch.randn(B, Hkv, T, D)
    ref = _reference(Q, K, V)
    S = T // world
    sl = slice(rank * S, (rank + 1) * S)
    config = AttentionConfig(backend = "sdpa", n_kv_heads = Hkv, n_groups = H // Hkv)
    context = AttentionContext(
        bsz = B,
        q_len = S,
        kv_seq_len = S,
        n_heads = H,
        head_dim = D,
        requires_grad = False,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    out = U.ulysses_attention(run_attention, config, context, Q[:, :, sl], K[:, :, sl], V[:, :, sl], None)
    torch.testing.assert_close(out, ref[:, sl], rtol = 1e-5, atol = 1e-5)

    lengths = torch.tensor([3, T - 5], dtype = torch.int32)
    cu_seqlens = torch.zeros(3, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(lengths, 0)
    Q, K, V = Q[:1], K[:1], V[:1]
    ref = _reference(Q, K, V, lengths.tolist())
    context = AttentionContext(
        bsz = 1,
        q_len = S,
        kv_seq_len = S,
        n_heads = H,
        head_dim = D,
        requires_grad = False,
        seq_info = (lengths, cu_seqlens, int(lengths.max())),
        attention_mask = None,
        causal_mask = None,
    )
    out = U.ulysses_attention(run_attention, config, context, Q[:, :, sl], K[:, :, sl], V[:, :, sl], None)
    torch.testing.assert_close(out, ref[:, sl], rtol = 1e-5, atol = 1e-5)


def test_real_run_attention_sdpa_path():
    try:
        from unsloth.utils.attention_dispatch import (
            AttentionConfig,
            AttentionContext,
            _run_attention,
        )
    except Exception as exc:  # the unsloth package import needs an accelerator
        pytest.skip(f"unsloth.utils.attention_dispatch does not import here: {type(exc).__name__}")
    # Forked, not spawned: a fresh interpreter would re-import unsloth without the test GPU spoof.
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    _spawn(
        _real_dispatch_worker,
        2,
        _run_attention,
        AttentionConfig,
        AttentionContext,
        start_method = "fork",
    )
