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
