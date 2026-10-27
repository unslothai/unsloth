# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CFG-batch / CFG-parallel argument checks read one flag per device, not one per tensor (no per-argument host
sync in the guided step), with the same verdict as the per-pair ``_bitwise_equal`` form."""

from __future__ import annotations

import itertools

import pytest
import torch

from core.inference import diffusion_cfg_batch as cb
from core.inference import diffusion_multigpu as mg


def _cases():
    base = torch.arange(24, dtype = torch.float32).reshape(2, 3, 4)
    nan = torch.tensor([float("nan"), 1.0])
    neg0 = torch.tensor([-0.0, 1.0])
    return [
        (base, base),  # same object
        (base, base.clone()),  # equal bytes
        (base, base + 1),  # different values
        (base, base.reshape(4, 6)),  # different shape
        (base, base.double()),  # different dtype
        (
            base,
            base.transpose(0, 1).contiguous().transpose(0, 1),
        ),  # non-contiguous view of equal values
        (nan, nan.clone()),  # NaN: bitwise equal although nan != nan
        (neg0, torch.tensor([0.0, 1.0])),  # -0.0 vs 0.0: equal values, different bytes
        (torch.tensor(3.0).expand(5), torch.full((5,), 3.0)),  # stride-0 expand
        (torch.empty(0), torch.empty(0)),
        (base, 1.0),  # not a tensor
        (torch.ones(3, dtype = torch.bfloat16), torch.ones(3, dtype = torch.bfloat16)),
    ]


def test_bitwise_equal_all_matches_per_pair_form():
    cases = _cases()
    for a, b in cases:
        assert mg._bitwise_equal_all([(a, b)]) == mg._bitwise_equal(a, b), (a, b)
    for combo in itertools.combinations(cases, 3):
        assert mg._bitwise_equal_all(combo) == all(mg._bitwise_equal(a, b) for a, b in combo)
    assert mg._bitwise_equal_all([]) is True
    assert mg._bitwise_equal_all(iter([(cases[1][0], cases[1][1])])) is True


def test_matches_verdicts_unchanged():
    x = torch.randn(2, 8)
    t = torch.tensor([0.5])
    call = ((x,), {"timestep": t, "guidance": None, "return_dict": False})
    snap = cb._snapshot(call)
    assert cb._matches(snap, call)
    assert not cb._matches(
        snap, ((x + 1,), {"timestep": t, "guidance": None, "return_dict": False})
    )
    assert not cb._matches(snap, ((x,), {"timestep": t, "guidance": None, "return_dict": True}))
    assert not cb._matches(
        snap, ((x,), {"timestep": t + 1, "guidance": None, "return_dict": False})
    )
    assert not cb._matches(snap, ((x,), {"timestep": t, "guidance": 1.0, "return_dict": False}))
    assert not cb._matches(snap, ((x,), {"timestep": t}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_one_host_read_per_step_on_cuda(monkeypatch):
    dev = torch.device("cuda")
    tensors = [torch.randn(4, 64, device = dev) for _ in range(6)]
    call = (tuple(tensors[:2]), {f"k{i}": t for i, t in enumerate(tensors[2:])})
    snap = cb._snapshot(call)
    torch.cuda.synchronize()
    reads, equals = [0], [0]
    orig_item, orig_equal = torch.Tensor.item, torch.equal

    def item(self):
        reads[0] += 1
        return orig_item(self)

    def equal(a, b):
        equals[0] += 1
        return orig_equal(a, b)

    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch, "equal", equal)
    assert cb._matches(snap, call)
    assert (reads[0], equals[0]) == (1, 0)  # six tensor arguments on one device: one host read
    other = dict(call[1], k0 = tensors[2] + 1)
    reads[0] = 0
    assert not cb._matches(snap, (call[0], other))
    assert (reads[0], equals[0]) == (1, 0)
