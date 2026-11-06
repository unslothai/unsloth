# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The decode LoRA cast cache must follow every optimizer, including fused ones that skip _version."""

import pytest
import torch

from unsloth.kernels import utils as U


def _adapters():
    torch.manual_seed(0)
    A = torch.nn.Parameter(torch.randn(8, 16))
    B = torch.nn.Parameter(torch.randn(32, 8))
    return A, B


def _step(optimizer, params):
    for p in params:
        p.grad = torch.randn_like(p)
    optimizer.step()


@pytest.mark.parametrize(
    "make_optimizer",
    [
        pytest.param(lambda p: torch.optim.AdamW(p, lr = 0.1, fused = True), id = "adamw_fused"),
        pytest.param(lambda p: torch.optim.AdamW(p, lr = 0.1, foreach = True), id = "adamw_foreach"),
        pytest.param(lambda p: torch.optim.AdamW(p, lr = 0.1, foreach = False), id = "adamw_loop"),
        pytest.param(lambda p: torch.optim.SGD(p, lr = 0.1), id = "sgd"),
    ],
)
def test_cast_cache_follows_optimizer_steps(make_optimizer):
    A, B = _adapters()
    optimizer = make_optimizer([A, B])
    A16, B16 = U._fast_lora_weights(A, B, torch.bfloat16)
    assert torch.equal(A16, A.detach().to(torch.bfloat16))
    _step(optimizer, [A, B])
    A16, B16 = U._fast_lora_weights(A, B, torch.bfloat16)
    assert torch.equal(A16, A.detach().to(torch.bfloat16))
    assert torch.equal(B16, B.detach().to(torch.bfloat16))


def test_cast_cache_hits_without_updates():
    A, B = _adapters()
    first = U._fast_lora_weights(A, B, torch.bfloat16)
    second = U._fast_lora_weights(A, B, torch.bfloat16)
    assert first[0] is second[0] and first[1] is second[1]


def test_matching_dtype_skips_the_cache():
    A, B = _adapters()
    assert U._fast_lora_weights(A, B, torch.float32) == (A, B)
    assert "_fast_lora" not in A.__dict__
