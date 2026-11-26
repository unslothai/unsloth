# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The generic Ulysses whole-sequence mask: bitwise equal to the previous construction for causal,
sliding window, packed documents and their combinations, without the [T, T] int64 temporary."""

import random
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import unsloth  # noqa: F401
from unsloth.utils import ulysses_hybrid as uh


def _reference_mask(total, device, dtype, window, cu_seqlens):
    # The construction before the fix, kept verbatim as the oracle.
    idx = torch.arange(total, device = device)
    allowed = idx[None, :] <= idx[:, None]
    if window is not None:
        allowed = allowed & (idx[:, None] - idx[None, :] < window)
    if cu_seqlens is not None:
        doc = torch.bucketize(idx, cu_seqlens[1:].to(device = device, dtype = idx.dtype), right = True)
        allowed = allowed & (doc[:, None] == doc[None, :])
    mask = torch.zeros(total, total, device = device, dtype = dtype)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def _random_cu_seqlens(rng, total):
    # Documents covering a prefix of the sequence (the rest is the padding tail), or all of it.
    end = total if rng.random() < 0.5 else rng.randint(1, total)
    cuts = sorted(rng.sample(range(1, end), min(rng.randint(0, 5), end - 1))) if end > 1 else []
    return torch.tensor([0, *cuts, end], dtype = rng.choice([torch.int32, torch.int64]))


def test_mask_is_bitwise_equal_to_the_reference():
    rng = random.Random(0)
    cases = 0
    for _ in range(300):
        total = rng.randint(1, 96)
        window = rng.choice([None, 1, 2, rng.randint(1, 16), total, total + 7])
        cu_seqlens = _random_cu_seqlens(rng, total) if rng.random() < 0.6 else None
        dtype = rng.choice([torch.float16, torch.bfloat16, torch.float32])
        args = (total, torch.device("cpu"), dtype, window, cu_seqlens)
        got, want = uh._whole_sequence_mask(*args), _reference_mask(*args)
        assert got.shape == want.shape == (1, 1, total, total) and got.dtype == want.dtype
        bits = torch.int16 if dtype.itemsize == 2 else torch.int32
        assert torch.equal(got.view(bits), want.view(bits))
        cases += 1
    assert cases == 300


class _PeakBytes(TorchDispatchMode):
    """Peak bytes of live tensor storages created inside the mode (views share their base's storage)."""

    def __init__(self):
        super().__init__()
        self.live, self.current, self.peak = {}, 0, 0

    def _release(self, ptr):
        count, nbytes = self.live[ptr]
        if count == 1:
            del self.live[ptr]
            self.current -= nbytes
        else:
            self.live[ptr] = (count - 1, nbytes)

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(out)[0]:
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in self.live:
                count, nbytes = self.live[ptr]
                self.live[ptr] = (count + 1, nbytes)
            else:
                self.live[ptr] = (1, storage.nbytes())
                self.current += storage.nbytes()
                self.peak = max(self.peak, self.current)
            weakref.finalize(tensor, self._release, ptr)
        return out


def _peak_per_element(build, total, window, cu_seqlens):
    tracker = _PeakBytes()
    with tracker:
        mask = build(total, torch.device("cpu"), torch.float16, window, cu_seqlens)
    del mask
    return tracker.peak / (total * total)


@pytest.mark.parametrize(
    "window, packed", [(None, False), (4096 // 8, False), (None, True), (4096 // 8, True)]
)
def test_mask_peak_memory_has_no_int64_square(window, packed):
    total = 4096
    cu_seqlens = torch.tensor([0, 1000, 2500, total - 3]) if packed else None
    # The fp16 output (2 bytes) plus one [T, T] bool (1 byte); the old windowed build held a [T, T]
    # int64 difference beside two bools (10 bytes per element), the old unwindowed one 4.
    bound = 3.1
    assert _peak_per_element(uh._whole_sequence_mask, total, window, cu_seqlens) < bound
    reference = _peak_per_element(_reference_mask, total, window, cu_seqlens)
    assert reference > bound  # the tracker sees what it is meant to bound


@pytest.mark.parametrize("window", [5, 6, 2**31, 2**63 + 1])
def test_window_at_or_past_the_length_is_plain_causal(window):
    """A window of T or more removes nothing: the mask equals the causal one, including windows past int64."""
    total = 5
    causal = uh._whole_sequence_mask(total, "cpu", torch.float32, None, None)
    assert torch.equal(uh._whole_sequence_mask(total, "cpu", torch.float32, window, None), causal)


# Query-chunked sdpa: the long-sequence replacement for the dense mask.


def _sdpa():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward


class _Attn(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = True


def _starts(cu_seqlens):
    return None if cu_seqlens is None else [int(x) for x in cu_seqlens.tolist()]


def _qkv(
    total,
    dtype,
    heads = 4,
    kv_heads = 2,
    dim = 8,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(2, heads, total, dim, generator = g, dtype = dtype).requires_grad_()
    k = torch.randn(2, kv_heads, total, dim, generator = g, dtype = dtype).requires_grad_()
    v = torch.randn(2, kv_heads, total, dim, generator = g, dtype = dtype).requires_grad_()
    return q, k, v, torch.randn(2, total, heads, dim, generator = g, dtype = dtype)


def _dense_and_chunked(
    total,
    window,
    cu_seqlens,
    rows,
    dtype = torch.float64,
    attention_fn = None,
):
    fn = attention_fn or _sdpa()
    module = _Attn(2)
    q, k, v, g = _qkv(total, dtype)
    mask = uh._whole_sequence_mask(total, q.device, dtype, window, cu_seqlens)
    dense = fn(module, q, k, v, mask, scaling = 0.3)[0]
    dense_grads = torch.autograd.grad(dense, (q, k, v), g)
    chunked = uh._chunked_masked_attention(
        module, q, k, v, fn, {"scaling": 0.3}, window, cu_seqlens, _starts(cu_seqlens), rows
    )
    chunked_grads = torch.autograd.grad(chunked, (q, k, v), g)
    return (dense, *dense_grads), (chunked, *chunked_grads)


def test_chunked_attention_matches_the_dense_mask_in_fp64():
    """Forward and dq / dk / dv of the chunked, key-cropped path equal the dense mask's: causal, windows
    shorter and longer than a chunk, packed documents, the cp padding tail, chunk sizes that do not divide T."""
    rng = random.Random(1)
    worst, cropped = 0.0, 0
    for _ in range(24):
        total = rng.randint(40, 300)
        window = rng.choice([None, 1, rng.randint(2, 64), rng.randint(64, 400)])
        cu_seqlens = _random_cu_seqlens(rng, total) if rng.random() < 0.7 else None
        rows = rng.choice([7, 16, 32, 50, 64])
        if window is None and cu_seqlens is None:
            window = 17
        want, got = _dense_and_chunked(total, window, cu_seqlens, rows)
        for a, b in zip(want, got):
            assert a.shape == b.shape
            worst = max(worst, float((a - b).detach().abs().max()))
        cropped += total > 64 and (window or total) < total // 2
    assert worst < 1e-12, worst
    assert cropped > 0


def test_chunked_keys_are_cropped_to_what_the_chunk_can_see():
    """The key slice of a chunk starts at its first row's document or window start (rounded down to 16)."""
    seen = []

    def recording(module, q, k, v, mask, **kwargs):
        seen.append((q.shape[2], k.shape[2], tuple(mask.shape[-2:])))
        return _sdpa()(module, q, k, v, mask, **kwargs)

    cu_seqlens = torch.tensor([0, 100, 230])
    _dense_and_chunked(256, 40, cu_seqlens, 64, attention_fn = recording)
    forward, backward = seen[1:5], seen[5:]
    # Rows 0:64 see keys 0:64; 64:128 from 64 - 39 = 25 -> 16; 128:192 from document start 100 -> 96;
    # 192:256 from 192 - 39 = 153 -> 144.
    assert forward == [
        (64, 64, (64, 64)),
        (64, 112, (64, 112)),
        (64, 96, (64, 96)),
        (64, 112, (64, 112)),
    ]
    assert sorted(backward) == sorted(forward)  # checkpoint recomputes each chunk once


class _SavesMask(torch.autograd.Function):
    """Attention that keeps its mask for backward, as CUDA's efficient sdpa kernel does with attn_bias."""

    @staticmethod
    def forward(ctx, out, mask):
        ctx.save_for_backward(mask)
        return out.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


@pytest.mark.parametrize("checkpointed", [True, False])
def test_chunk_masks_do_not_live_until_backward(checkpointed, monkeypatch):
    masks = []
    build = uh._chunk_mask

    def recorded(*args):
        mask = build(*args)
        masks.append(weakref.ref(mask))
        return mask

    def saving(module, q, k, v, mask, **kwargs):
        out, _ = _sdpa()(module, q, k, v, mask, **kwargs)
        return _SavesMask.apply(out, mask), None

    monkeypatch.setattr(uh, "_chunk_mask", recorded)
    if not checkpointed:
        monkeypatch.setattr(uh, "_torch_checkpoint", lambda: lambda fn, *a, **k: fn(*a))
    q, k, v, g = _qkv(256, torch.float32)
    out = uh._chunked_masked_attention(_Attn(2), q, k, v, saving, {}, 40, None, None, 64)
    alive = sum(ref() is not None for ref in masks)
    assert len(masks) == 4
    assert alive == (0 if checkpointed else 4)
    torch.autograd.grad(out, (q, k, v), g)
    if checkpointed:
        assert len(masks) == 8  # each chunk's slice rebuilt once in backward


@pytest.mark.parametrize(
    "implementation, total, env, rows",
    [
        ("sdpa", 8192, {}, 0),  # short: the dense mask, one call
        ("sdpa", 8193, {}, 0),  # one chunk would hold the whole mask
        ("sdpa", 16384, {}, 8192),
        ("sdpa", 8192, {"UNSLOTH_CP_MASK_CHUNK_ROWS": "1024"}, 0),  # the short path stays dense
        ("sdpa", 8193, {"UNSLOTH_CP_MASK_CHUNK_ROWS": "1024"}, 1024),
        ("sdpa", 32768, {}, 4096),
        ("sdpa", 65536, {}, 2048),
        ("sdpa", 131072, {}, 1024),
        ("sdpa", 1 << 20, {}, 256),
        ("eager", 65536, {}, 0),  # eager materializes [T, T] scores anyway
        ("sdpa", 65536, {"UNSLOTH_CP_CHUNKED_MASK": "0"}, 0),
        (
            "sdpa",
            64,
            {"UNSLOTH_CP_DENSE_MASK_MAX_TOKENS": "0", "UNSLOTH_CP_MASK_CHUNK_ROWS": "16"},
            16,
        ),
        (
            "sdpa",
            64,
            {"UNSLOTH_CP_DENSE_MASK_MAX_TOKENS": "0", "UNSLOTH_CP_MASK_CHUNK_ROWS": "64"},
            0,
        ),
    ],
)
def test_chunk_rows(implementation, total, env, rows, monkeypatch):
    for name in (
        "UNSLOTH_CP_CHUNKED_MASK",
        "UNSLOTH_CP_DENSE_MASK_MAX_TOKENS",
        "UNSLOTH_CP_MASK_CHUNK_ROWS",
    ):
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert uh._mask_chunk_rows(implementation, total) == rows


def test_bf16_key_gradients_sum_over_chunks_in_float32():
    """dk / dv sum over every chunk that sees a key; summed in bf16 their error would grow with the chunk
    count (here 6.7e-3 vs the dense mask's 3.8e-3 relative error for dk). With 96 chunks over one long
    document, summed in float32 they stay at the dense mask's error against float64."""
    total, rows = 768, 8
    cu_seqlens = torch.tensor([0, total - 5])
    fn = _sdpa()
    module = _Attn(2)
    q, k, v, g = _qkv(total, torch.bfloat16, seed = 3)
    ref = [x.detach().double().requires_grad_() for x in (q, k, v)]
    mask = uh._whole_sequence_mask(total, "cpu", torch.float64, None, cu_seqlens)
    want = torch.autograd.grad(fn(module, *ref, mask, scaling = 0.3)[0], ref, g.double())
    mask = uh._whole_sequence_mask(total, "cpu", torch.bfloat16, None, cu_seqlens)
    dense = torch.autograd.grad(fn(module, q, k, v, mask, scaling = 0.3)[0], (q, k, v), g)
    out = uh._chunked_masked_attention(
        module, q, k, v, fn, {"scaling": 0.3}, None, cu_seqlens, _starts(cu_seqlens), rows
    )
    chunked = torch.autograd.grad(out, (q, k, v), g)

    def rel(a, b):
        return float((a.double() - b).norm() / b.norm())

    for name, d, c, w in zip(("dq", "dk", "dv"), dense, chunked, want):
        assert c.dtype == torch.bfloat16
        assert rel(c, w) <= 1.05 * rel(d, w), (name, rel(c, w), rel(d, w))
