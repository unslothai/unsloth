# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Host-built packed metadata must match the old device reads bit for bit, without per-step syncs."""

import math
import warnings

import pytest
import torch

import unsloth.utils.packing as packing


def _old_packed_info(seq_lengths, device):
    lengths = seq_lengths.to(device = device, dtype = torch.int32)
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype = torch.int32, device = device)
    torch.cumsum(lengths, dim = 0, dtype = torch.int32, out = cu_seqlens[1:])
    return lengths, cu_seqlens, int(lengths.max().item())


def _old_cover(seq_info, total_tokens):
    _, cu_seqlens, max_seqlen = seq_info
    padding = total_tokens - int(cu_seqlens[-1].item())
    if padding > 0:
        tail = torch.tensor([total_tokens], dtype = cu_seqlens.dtype, device = cu_seqlens.device)
        return torch.cat([cu_seqlens, tail]), max(max_seqlen, padding)
    return cu_seqlens, max_seqlen


def _random_batches(n = 200, seed = 0):
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        k = int(torch.randint(1, 33, (1,), generator = g))
        high = [2, 9, 300, 5000][i % 4]
        dtype = [torch.int32, torch.int64][i % 2]
        yield torch.randint(1, high, (k,), generator = g, dtype = dtype)


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
    assert torch.equal(a, b)


@pytest.fixture(autouse = True)
def _clear():
    packing.clear_packed_caches()
    yield
    packing.clear_packed_caches()


def test_stage_packed_metadata_matches_device_cumsum():
    for seq in _random_batches():
        staged, host = packing._stage_packed_metadata(seq)
        lengths, cu, max_seqlen = _old_packed_info(seq, "cpu")
        n = seq.numel()
        assert staged.dtype == torch.int32 and staged.numel() == 2 * n + 1
        _same(staged[:n], lengths)
        _same(staged[n:], cu)
        assert host == tuple(lengths.tolist()) and max(host) == max_seqlen
        assert all(type(x) is int for x in host)


def test_stage_packed_metadata_stays_on_host_under_a_default_device():
    # torch.set_default_device("cuda") must not move the host staging buffer (it is pinned next).
    seq = torch.tensor([3, 1, 4], dtype = torch.int32)
    torch.set_default_device("meta")
    try:
        staged, host = packing._stage_packed_metadata(seq)
    finally:
        torch.set_default_device(None)
    assert staged.device.type == "cpu" and host == (3, 1, 4)
    assert staged.tolist() == [3, 1, 4, 0, 3, 4, 8]


def test_packed_info_and_consumers_match_old_values():
    cpu = torch.device("cpu")
    for i, seq in enumerate(_random_batches()):
        packing.clear_packed_caches()
        lengths, cu, max_seqlen = packing.get_packed_info_from_kwargs(
            {"packed_seq_lengths": seq}, cpu
        )
        ref = _old_packed_info(seq, cpu)
        _same(lengths, ref[0])
        _same(cu, ref[1])
        assert max_seqlen == ref[2] and type(max_seqlen) is int
        info = (lengths, cu, max_seqlen)
        total = int(seq.sum()) + [0, 1, 7, 64][i % 4]
        new_cu, new_max = packing.cover_padded_cu_seqlens(info, total)
        old_cu, old_max = _old_cover(ref, total)
        _same(new_cu, old_cu)
        assert new_max == old_max
        segs = packing.packed_segment_lengths(info, total)
        assert segs == packing._with_padding_segment(tuple(int(x) for x in ref[0].tolist()), total)
        if seq.numel() <= 8 and int(seq.sum()) <= 64:
            mask = packing.build_sdpa_packed_attention_mask(
                info, dtype = torch.float32, device = cpu, total_tokens = total
            )
            packing.clear_packed_caches()
            # a seq_info not built by get_packed_info_from_kwargs takes the old .tolist() route
            old_mask = packing.build_sdpa_packed_attention_mask(
                (ref[0].clone(), ref[1].clone(), ref[2]),
                dtype = torch.float32,
                device = cpu,
                total_tokens = total,
            )
            _same(mask, old_mask)


def test_host_lengths_only_reused_for_the_cached_tensors():
    cpu = torch.device("cpu")
    seq = torch.tensor([3, 5, 2], dtype = torch.int32)
    lengths, cu, _ = packing.get_packed_info_from_kwargs({"packed_seq_lengths": seq}, cpu)
    assert packing._cached_host_lengths(lengths) == (3, 5, 2)
    assert packing._cached_host_lengths(lengths, cu) == (3, 5, 2)
    assert packing._cached_host_lengths(lengths.clone()) is None
    assert packing._cached_host_lengths(lengths, cu.clone()) is None
    # a foreign seq_info whose cu_seqlens is not the cumsum of these lengths keeps reading cu_seqlens[-1]
    foreign_cu = torch.tensor([0, 4, 6, 9], dtype = torch.int32)
    new_cu, new_max = packing.cover_padded_cu_seqlens((lengths, foreign_cu, 5), 12)
    assert new_cu.tolist() == [0, 4, 6, 9, 12] and new_max == 5
    packing.clear_packed_caches()
    assert packing._cached_host_lengths(lengths) is None


def test_position_id_inferred_lengths_match():
    cpu = torch.device("cpu")
    pos = torch.cat([torch.arange(n) for n in (4, 1, 9, 3)])[None]
    lengths, cu, max_seqlen = packing.get_packed_info_from_kwargs({}, cpu, pos)
    assert lengths.tolist() == [4, 1, 9, 3] and cu.tolist() == [0, 4, 5, 14, 17] and max_seqlen == 9


def test_fsdp2_clip_empty_shards_match_reference():
    from unsloth.models.fsdp2_native import native_clip_grad_norm_
    g = torch.Generator().manual_seed(0)
    for p_norm in (2.0, 1.0, math.inf, -math.inf, 3.0):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            shapes = [(3, 4), (0,), (5,), (0, 4), (7, 2), (0,)]
            params = []
            for s in shapes:
                p = torch.nn.Parameter(torch.zeros(s, dtype = dtype))
                p.grad = (torch.randn(s, generator = g) * 3).to(dtype)
                params.append(p)
            # the previous implementation: identity-filled vector, live norms scattered in by index
            grads = [p.grad for p in params]
            live = [k for k, t in enumerate(grads) if t.numel()]
            values = torch.full((len(grads),), math.inf if p_norm < 0 else 0.0, dtype = dtype)
            values[torch.tensor(live)] = torch.stack(
                list(torch._foreach_norm([grads[k] for k in live], p_norm))
            )
            expect = torch.linalg.vector_norm(torch.cat([values]), p_norm)
            got = native_clip_grad_norm_([p for p in params], max_norm = 1e9, norm_type = p_norm)
            assert torch.equal(got.to(expect.dtype), expect), (p_norm, dtype, got, expect)


def test_fused_ce_int_divisor_is_bitwise_torch_tensor():
    for v in (0, 1, 7, 2**24 + 1, 2**31 + 12345, 2**40 + 999, 3.5, True):
        a = torch.tensor(v, dtype = torch.float32)
        b = torch.full((), v, dtype = torch.float32)
        assert a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _count_syncs(fn):
    torch.cuda.synchronize()
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            out = fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return out, sum("synchronizing CUDA operation" in str(x.message) for x in w)


@_needs_cuda
def test_cuda_host_lengths_zero_syncs_and_one_read_for_device_lengths():
    dev = torch.device("cuda")
    seq = torch.tensor([5, 1, 70, 7], dtype = torch.int32)

    def per_step(s):
        info = packing.get_packed_info_from_kwargs({"packed_seq_lengths": s}, dev)
        packing.cover_padded_cu_seqlens(info, 96)
        packing.packed_segment_lengths(info, 96)
        packing.build_sdpa_packed_attention_mask(
            info, dtype = torch.float16, device = dev, total_tokens = 96
        )
        for _ in range(4):  # later layers hit the caches
            packing.get_packed_info_from_kwargs({"packed_seq_lengths": s}, dev)
            packing.cover_padded_cu_seqlens(info, 96)
        return info

    info, n = _count_syncs(lambda: per_step(seq))
    assert n == 0, f"{n} syncs for host lengths"
    ref = _old_packed_info(seq, dev)
    _same(info[0], ref[0])
    _same(info[1], ref[1])
    assert info[2] == ref[2]
    packing.clear_packed_caches()
    dev_seq = seq.to(dev)
    _, n = _count_syncs(lambda: per_step(dev_seq))
    assert n == 1, f"{n} syncs for device lengths (expected the one lengths read)"


@_needs_cuda
def test_cuda_fsdp2_clip_and_ce_divisor_no_sync():
    from unsloth.models.fsdp2_native import native_clip_grad_norm_

    params = []
    for s in [(3, 4), (0,), (5,)]:
        p = torch.nn.Parameter(torch.zeros(s, device = "cuda"))
        p.grad = torch.randn(s, device = "cuda")
        params.append(p)
    _, n = _count_syncs(lambda: native_clip_grad_norm_(params, max_norm = 1.0))
    assert n == 0, f"{n} syncs in native_clip_grad_norm_"
    _, n = _count_syncs(lambda: torch.full((), 1234, dtype = torch.float32, device = "cuda"))
    assert n == 0


def test_non_blocking_batches_only_for_pinned_cuda_loaders(monkeypatch):
    from types import SimpleNamespace

    from unsloth.models._utils import _unsloth_non_blocking_batches

    def trainer(
        pin = True,
        device = "cuda",
        non_blocking = False,
    ):
        config = SimpleNamespace(non_blocking = non_blocking)
        return SimpleNamespace(
            args = SimpleNamespace(dataloader_pin_memory = pin),
            accelerator = SimpleNamespace(dataloader_config = config, device = torch.device(device)),
        ), config

    t, config = trainer()
    assert _unsloth_non_blocking_batches(t) is True and config.non_blocking is True
    for kwargs in ({"pin": False}, {"device": "cpu"}, {"non_blocking": True}):
        t, config = trainer(**kwargs)
        before = config.non_blocking
        assert _unsloth_non_blocking_batches(t) is False and config.non_blocking == before
    monkeypatch.setenv("UNSLOTH_NON_BLOCKING_BATCHES", "0")
    t, config = trainer()
    assert _unsloth_non_blocking_batches(t) is False and config.non_blocking is False
    assert _unsloth_non_blocking_batches(SimpleNamespace()) is False


def _compile_over_segment_counts(dynamic):
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    def step(x, lengths):
        packing.clear_packed_caches()
        _, cu_seqlens, max_seqlen = packing.get_packed_info_from_kwargs(
            {"packed_seq_lengths": lengths}, x.device
        )
        return x * max_seqlen + cu_seqlens[-1]

    dynamo.reset()
    counters.clear()
    compiled = torch.compile(step, backend = "eager", dynamic = dynamic)
    for n in range(3, 15):
        lengths = torch.full((n,), 4, dtype = torch.int32)
        out = compiled(torch.ones(4), lengths)
        assert torch.equal(out, torch.ones(4) * 4 + 4 * n)
    graphs = counters["stats"]["unique_graphs"]
    dynamo.reset()
    return graphs


def test_compiled_dynamic_does_not_specialize_on_segment_count():
    # 12 segment counts: the tensor path traces a handful of graphs; a host tuple of lengths
    # would trace one per count and hit the recompile limit.
    assert _compile_over_segment_counts(dynamic = True) <= 4


def test_compiling_takes_tensor_path(monkeypatch):
    monkeypatch.setattr(packing, "_is_compiling", lambda: True)
    seq = torch.tensor([3, 5, 2], dtype = torch.int64)
    lengths, cu, max_seqlen = packing.get_packed_info_from_kwargs(
        {"packed_seq_lengths": seq}, "cpu"
    )
    old = _old_packed_info(seq, "cpu")
    _same(lengths, old[0])
    _same(cu, old[1])
    assert max_seqlen == old[2]
    # No host tuple recorded: downstream helpers take their own (traceable) route.
    assert packing._cached_host_lengths(lengths) is None


def test_pinned_to_device_falls_back_to_pageable(monkeypatch):
    t = torch.arange(5, dtype = torch.int32)

    def refuse(self, *args, **kwargs):
        raise RuntimeError("cudaHostAlloc failed: out of pinned memory")

    monkeypatch.setattr(torch.Tensor, "pin_memory", refuse)
    _same(packing._pinned_to_device(t, "cpu"), t)


class _SaveForBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, saved):
        ctx.save_for_backward(saved)
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        return grad * 2, None


def _survives_backward(saved):
    x = torch.randn(4, requires_grad = True)
    _SaveForBackward.apply(x, saved).sum().backward()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_inference_mode_metadata_is_not_served_to_a_training_step(dtype):
    """inference_mode forward then a training forward on the same batch: no inference tensors."""
    cpu = torch.device("cpu")
    kwargs = {"packed_seq_lengths": torch.tensor([3, 5], dtype = dtype)}
    with torch.inference_mode():
        info = packing.get_packed_info_from_kwargs(kwargs, cpu)
        padded = packing.cover_padded_cu_seqlens(info, 10)
        mask = packing.build_sdpa_packed_attention_mask(info, dtype = torch.float32, device = cpu)
    assert torch.is_inference(info[1])
    train = packing.get_packed_info_from_kwargs(kwargs, cpu)
    assert train is not info and not torch.is_inference(train[1])
    assert train[1].tolist() == [0, 3, 8] and train[2] == 5
    _survives_backward(train[1])
    train_padded = packing.cover_padded_cu_seqlens(train, 10)
    assert not torch.is_inference(train_padded[0]) and train_padded[0].tolist() == [0, 3, 8, 10]
    _survives_backward(train_padded[0])
    train_mask = packing.build_sdpa_packed_attention_mask(train, dtype = torch.float32, device = cpu)
    assert not torch.is_inference(train_mask)
    torch.testing.assert_close(train_mask, mask.clone())
    _survives_backward(train_mask)
    assert packing.get_packed_info_from_kwargs(kwargs, cpu) is train
    assert packing.cover_padded_cu_seqlens(train, 10) is train_padded
    assert (
        packing.build_sdpa_packed_attention_mask(train, dtype = torch.float32, device = cpu)
        is train_mask
    )


def test_xformers_mask_lru_keeps_inference_masks_out_of_training(monkeypatch):
    class _Mask:
        @classmethod
        def from_seqlens(cls, lengths):
            return cls()

    monkeypatch.setattr(packing, "_XFormersBlockMask", _Mask)
    monkeypatch.setattr(packing, "move_xformers_attention_bias", lambda mask, device: mask)
    packing._XFORMERS_MASK_CACHE.clear()
    cpu = torch.device("cpu")
    with torch.inference_mode():
        evaluated = packing._get_cached_block_mask((3, 5), None, cpu)
        assert packing._get_cached_block_mask((3, 5), None, cpu) is evaluated
    trained = packing._get_cached_block_mask((3, 5), None, cpu)
    assert trained is not evaluated
    assert packing._get_cached_block_mask((3, 5), None, cpu) is trained
    # A training-built mask serves inference_mode too: no second entry per lengths tuple.
    packing._XFORMERS_MASK_CACHE.clear()
    trained = packing._get_cached_block_mask((2, 2), None, cpu)
    with torch.inference_mode():
        assert packing._get_cached_block_mask((2, 2), None, cpu) is trained
    assert len(packing._XFORMERS_MASK_CACHE) == 1
    packing._XFORMERS_MASK_CACHE.clear()


def test_training_metadata_is_reused_under_inference_mode():
    cpu = torch.device("cpu")
    kwargs = {"packed_seq_lengths": torch.tensor([3, 5], dtype = torch.int32)}
    train = packing.get_packed_info_from_kwargs(kwargs, cpu)
    with torch.inference_mode():
        assert packing.get_packed_info_from_kwargs(kwargs, cpu) is train
        other = {"packed_seq_lengths": torch.tensor([2, 2], dtype = torch.int32)}
        info = packing.get_packed_info_from_kwargs(other, cpu)
        assert packing.get_packed_info_from_kwargs(other, cpu) is info
    assert packing.get_packed_info_from_kwargs(kwargs, cpu) is not train
    _survives_backward(packing.get_packed_info_from_kwargs(kwargs, cpu)[1])
