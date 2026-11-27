# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch
from real_accelerator import has_real_cuda

from unsloth.models import pp_native


def _bincount_reference(logits, num_experts, top_k):
    counts = torch.zeros(num_experts, dtype = torch.float32, device = logits[0].device)
    for gate in logits:
        _, selected = torch.topk(torch.softmax(gate, dim = -1), top_k, dim = -1)
        counts = counts + torch.bincount(selected.reshape(-1), minlength = num_experts).float()
    return counts


@pytest.mark.parametrize("case", ["random", "one_expert", "unused_experts"])
def test_unmasked_counts_equal_bincount(case):
    g = torch.Generator().manual_seed(0)
    E, k = 8, 2
    logits = [torch.randn(37, E, generator = g) for _ in range(3)]
    if case == "one_expert":
        logits = [torch.full((37, E), -1e4).index_fill_(1, torch.tensor([3, 5]), 10.0)]
    elif case == "unused_experts":
        logits = [x.index_fill(1, torch.tensor([0, 6, 7]), -1e4) for x in logits]
    stats = pp_native._router_statistics(tuple(logits), E, k)
    counts = stats[:E]
    assert counts.dtype == torch.float32
    assert torch.equal(counts, _bincount_reference(logits, E, k))
    assert counts.sum().item() == sum(x.shape[0] for x in logits) * k


# Not torch.cuda.is_available(): the CUDA spoof other suites install answers True on a CPU runner.
@pytest.mark.skipif(not has_real_cuda(), reason = "needs a GPU")
def test_unmasked_statistics_do_not_sync():
    print("pp_native from", pp_native.__file__)
    g = torch.Generator(device = "cuda").manual_seed(0)
    logits = tuple(torch.randn(64, 8, device = "cuda", generator = g) for _ in range(2))
    pp_native._router_statistics(logits, 8, 2)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        stats = pp_native._router_statistics(logits, 8, 2)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(stats[:8], _bincount_reference(logits, 8, 2))
