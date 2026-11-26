# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses packed Flash attention reads no device scalar per layer: the padded cu_seqlens and the longest segment
come from the host lengths the packed metadata already holds (one build per step), with the same values as before.
CPU, a one-rank gloo group, a stub flash attention function."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.overrides import TorchFunctionMode

import unsloth  # noqa: F401
from unsloth.utils import ulysses_hybrid as uh
from unsloth.utils.packing import get_packed_info_from_kwargs

_HOST_READS = {"__int__", "__index__", "__bool__", "__float__", "item", "tolist"}


class _CountHostReads(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.reads = []

    def __torch_function__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        name = getattr(func, "__name__", "")
        if name in _HOST_READS:
            self.reads.append(name)
        return func(*args, **(kwargs or {}))


@pytest.fixture(scope = "module")
def group():
    if dist.is_initialized():
        yield dist.group.WORLD
        return
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            "gloo", init_method = "file://" + os.path.join(tmp, "store"), rank = 0, world_size = 1
        )
        try:
            yield dist.group.WORLD
        finally:
            dist.destroy_process_group()


class _Attn(torch.nn.Module):
    is_causal = True
    num_key_value_groups = 2
    sliding_window = None


@pytest.mark.parametrize(
    "lengths, total, cu, longest",
    [
        ([5, 7, 3], 16, [0, 5, 12, 15, 16], 7),  # cp padding tail: its own segment
        ([5, 7, 4], 16, [0, 5, 12, 16], 7),
        ([3, 2, 1], 16, [0, 3, 5, 6, 16], 10),  # the padding tail is the longest segment
    ],
    ids = ["cp_pad_tail", "no_tail", "long_tail"],
)
def test_packed_flash_reads_no_device_scalar_per_layer(group, lengths, total, cu, longest):
    seen = []

    def attention_fn(shard, q, k, v, mask, **kw):
        seen.append(
            (kw["cu_seq_lens_q"], kw["cu_seq_lens_k"], kw["max_length_q"], kw["max_length_k"])
        )
        return q.transpose(1, 2), None

    torch.manual_seed(0)
    q = torch.randn(1, 4, total, 8)
    k = torch.randn(1, 2, total, 8)
    packed = torch.tensor(lengths, dtype = torch.int32)
    get_packed_info_from_kwargs({"packed_seq_lengths": packed}, q.device)  # built once per step
    layers = 4
    with _CountHostReads() as mode:
        for _ in range(layers):
            out, _ = uh._ulysses_hf_attention(
                _Attn(),
                q,
                k,
                k,
                None,
                group = group,
                attention_fn = attention_fn,
                implementation = "flash_attention_2",
                packed_seq_lengths = packed,
            )
    assert mode.reads == [], mode.reads
    assert len(seen) == layers
    for cu_q, cu_k, max_q, max_k in seen:
        assert cu_q is cu_k and cu_q.dtype == torch.int32 and cu_q.tolist() == cu
        assert max_q == max_k == longest and type(max_q) is int
    torch.testing.assert_close(out, q.transpose(1, 2).reshape(1, total, 4, 8), rtol = 0, atol = 0)
