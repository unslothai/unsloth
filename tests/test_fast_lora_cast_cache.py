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


_OPTIMIZERS = [
    pytest.param(lambda p: torch.optim.AdamW(p, lr = 0.1, fused = True), id = "adamw_fused"),
    pytest.param(lambda p: torch.optim.AdamW(p, lr = 0.1, foreach = True), id = "adamw_foreach"),
    pytest.param(lambda p: torch.optim.SGD(p, lr = 0.1), id = "sgd"),
]


@pytest.mark.parametrize("make_optimizer", _OPTIMIZERS)
@pytest.mark.parametrize(
    "first, later", [("inference", "inference"), ("inference", "no_grad"), ("no_grad", "no_grad")]
)
def test_decode_casts_are_refreshed_in_place(make_optimizer, first, later):
    # A captured decode graph reads the casts by address: an optimizer step must update the same tensors.
    modes = {"inference": torch.inference_mode, "no_grad": torch.no_grad}
    A, B = _adapters()
    optimizer = make_optimizer([A, B])
    with modes[first]():
        A16, B16 = U._fast_lora_weights(A, B, torch.bfloat16)
    assert not A16.is_inference() and not B16.is_inference()
    for _ in range(2):
        _step(optimizer, [A, B])
        with modes[later]():
            A2, B2 = U._fast_lora_weights(A, B, torch.bfloat16)
        assert A2 is A16 and B2 is B16
        assert torch.equal(A2, A.detach().to(torch.bfloat16))
        assert torch.equal(B2, B.detach().to(torch.bfloat16))


def test_decode_casts_are_remade_when_they_no_longer_fit():
    A, B = _adapters()
    with torch.no_grad():
        A16, B16 = U._fast_lora_weights(A, B, torch.bfloat16)
        A.data = torch.randn(4, 16)  # new shape (and address)
        A2, B2 = U._fast_lora_weights(A, B, torch.bfloat16)
    assert A2 is not A16 and A2.shape == (4, 16)
    assert torch.equal(A2, A.detach().to(torch.bfloat16))
    # With grad on the cast may join an autograd graph: never written in place.
    with torch.no_grad():
        A.add_(1)
    A3, _ = U._fast_lora_weights(A, B, torch.bfloat16)
    assert A3 is not A2 and torch.equal(A3, A.detach().to(torch.bfloat16))


def test_peft_projection_casts_are_refreshed_in_place():
    peft = pytest.importorskip("peft")

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 32, bias = False)

    torch.manual_seed(0)
    model = peft.get_peft_model(_Block(), peft.LoraConfig(r = 4, target_modules = ["q_proj"]))
    _, _, A, B, _, _ = U.get_lora_parameters_bias(model.base_model.model.q_proj)
    with torch.inference_mode():
        casts = U._fast_lora_weights(A, B, torch.bfloat16)
    with torch.no_grad():
        B.add_(0.5)
        again = U._fast_lora_weights(A, B, torch.bfloat16)
    assert again[0] is casts[0] and again[1] is casts[1]
    assert torch.equal(casts[1], B.detach().to(torch.bfloat16))
    with torch.no_grad():
        B.data = torch.randn(64, 4)
        remade = U._fast_lora_weights(A, B, torch.bfloat16)
    assert remade[1] is not casts[1]  # new address: the decode graphs re-capture
