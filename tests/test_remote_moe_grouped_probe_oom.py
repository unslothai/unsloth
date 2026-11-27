# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The remote MoE eval shim's grouped GEMM probe records no verdict when it runs out of memory: that call keeps the
per-expert loop and the next one probes again, instead of caching the loop on the block for the process."""

import types

import torch

from unsloth.models import remote_moe_shims as R


class _CudaLike:
    """A CPU weight that reports itself as CUDA, so the probe runs on CPU."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.is_cuda = True
        self.dtype = tensor.dtype
        self.device = tensor.device

    def detach(self):
        return self.tensor.detach()


def _matmul_grouped_mm(failures):
    calls = []

    def grouped_mm(inputs, weight, offsets):
        calls.append(1)
        if len(calls) <= failures:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
        rows = [inputs[:1] @ weight[0], inputs[1:] @ weight[1]]
        return torch.cat(rows)

    grouped_mm.calls = calls
    return grouped_mm


def _weights():
    g = torch.Generator().manual_seed(0)
    return tuple(_CudaLike(torch.randn(8, 16, generator = g).bfloat16()) for _ in range(2))


def test_probe_out_of_memory_is_no_verdict(monkeypatch):
    grouped_mm = _matmul_grouped_mm(failures = 1)
    other = _matmul_grouped_mm(failures = 0)
    monkeypatch.setattr(R, "_grouped_mm_candidates", lambda weight: [grouped_mm, other])
    weights = _weights()
    # The next candidate is not tried: it would be cached in place of the better one.
    assert R._grouped_mm_for(weights) is R._NO_VERDICT
    assert other.calls == []
    assert R._grouped_mm_for(weights) is grouped_mm


def test_stack_experts_does_not_cache_an_out_of_memory_probe(monkeypatch):
    experts = torch.nn.ModuleList()
    for _ in range(2):
        expert = torch.nn.Module()
        for name in R._EXPERT_PROJECTIONS:
            setattr(expert, name, torch.nn.Linear(16, 16, bias = False).bfloat16())
        expert.act_fn = torch.nn.functional.silu
        experts.append(expert)
    block = torch.nn.Module()
    block.experts = experts
    answers = [R._NO_VERDICT, R._torch_grouped_mm]
    monkeypatch.setattr(R, "_grouped_mm_for", lambda weights: answers.pop(0))
    import unsloth.models.diffusion_infer as D

    monkeypatch.setattr(D, "drop_on_move", lambda module, key: None)
    assert R._grouped_experts(block) is None
    assert "_unsloth_grouped_experts" not in block.__dict__
    stacked = R._grouped_experts(block)
    assert stacked is not None and stacked[-1] is R._torch_grouped_mm
    assert block.__dict__["_unsloth_grouped_experts"] is stacked
