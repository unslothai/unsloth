# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_EP_EXPERT_ROWS checkpoints each expert row chunk with torch's own non-reentrant checkpoint, also while
Unsloth's smart gradient checkpointing has replaced torch.utils.checkpoint.checkpoint with its forced-reentrant,
CPU-offloading version. Through the replacement each chunk ran UnslothCheckpointFunction: it advanced the
per-layer offload counters (LAST_GC_INDEX picks the last decoder layer to keep on device) and, when the received
tokens need no grad (frozen embeddings feeding the first MoE layer), gave the expert LoRA weights, which the
chunk reaches only through the experts module, no grad at all. CPU only: the offload branch is kept out of reach."""

import functools
import types

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint

import unsloth  # noqa: F401
from unsloth.models import ep_native

gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")


@pytest.fixture
def smart_gc(monkeypatch):
    """unsloth_checkpoint installed as torch.utils.checkpoint.checkpoint, as after get_peft_model; MINIMUM_SIZE
    keeps every activation on device, so no pinned host buffer (and no accelerator) is needed."""
    original = torch_checkpoint.checkpoint
    gc._capture_pristine_checkpoint_once()
    pristine = getattr(torch_checkpoint, "_unsloth_pristine_checkpoint", None) or original
    state = dict(
        LAST_GC_INDEX = 0,
        CURRENT_GC_INDEX = 0,
        FIRST_PASS = True,
        BACKWARD_PASS = False,
        MINIMUM_SIZE = 2**62,
        CPU_BUFFERS = [],
        CPU_INDEX = 0,
        USE_DOUBLE_BUFFER = False,
    )
    for name, value in state.items():
        monkeypatch.setattr(gc, name, value, raising = False)
    calls = types.SimpleNamespace(offloads = [], pristine = [])
    apply = gc.UnslothCheckpointFunction.apply

    def counted_apply(*args):
        calls.offloads.append(1)
        return apply(*args)

    @functools.wraps(pristine)
    def counted_pristine(*args, **kwargs):
        calls.pristine.append(1)
        return pristine(*args, **kwargs)

    monkeypatch.setattr(gc.UnslothCheckpointFunction, "apply", counted_apply)
    monkeypatch.setattr(
        torch_checkpoint, "_unsloth_pristine_checkpoint", counted_pristine, raising = False
    )
    monkeypatch.setattr(torch_checkpoint, "_old_checkpoint", pristine, raising = False)
    monkeypatch.setattr(torch_checkpoint, "checkpoint", gc.unsloth_checkpoint)
    calls.pristine_fn = pristine
    return calls


def _run(x_requires_grad):
    g = torch.Generator().manual_seed(0)
    lora = torch.randn(
        8, 8, generator = g
    ).requires_grad_()  # expert LoRA, reached through the module only
    other = torch.randn(8, generator = g).requires_grad_()
    got_x = torch.randn(6, 8, generator = g).requires_grad_(x_requires_grad)
    shard = types.SimpleNamespace(swaps = [], frozen = {}, per_rank = 2)

    def experts(xc, ic, wc):
        return (xc @ lora) * wc.sum(-1, keepdim = True)

    row = torch.arange(6)
    ic = torch.zeros(6, 1, dtype = torch.long)
    wc = torch.rand(6, 1, generator = g)
    y = ep_native._ep_experts_chunked(shard, experts, False, got_x, row, ic, wc, 2)
    (y.sum() + other.sum()).backward()
    assert shard.chunked_calls == 1
    return y.detach(), lora.grad, got_x.grad


def _reference(monkeypatch, x_requires_grad, pristine):
    with monkeypatch.context() as patch:
        patch.setattr(torch_checkpoint, "checkpoint", pristine)
        return _run(x_requires_grad)


@pytest.mark.parametrize("x_requires_grad", [False, True])
def test_expert_chunks_skip_the_offloading_checkpoint(smart_gc, monkeypatch, x_requires_grad):
    want_y, want_lora, want_x = _reference(monkeypatch, x_requires_grad, smart_gc.pristine_fn)
    smart_gc.pristine.clear()
    before = (gc.LAST_GC_INDEX, gc.CURRENT_GC_INDEX, gc.FIRST_PASS)
    y, lora_grad, x_grad = _run(x_requires_grad)
    assert smart_gc.offloads == [], f"{len(smart_gc.offloads)} chunks ran UnslothCheckpointFunction"
    assert len(smart_gc.pristine) == 3
    assert (gc.LAST_GC_INDEX, gc.CURRENT_GC_INDEX, gc.FIRST_PASS) == before
    torch.testing.assert_close(y, want_y, rtol = 0, atol = 0)
    assert want_lora is not None
    assert lora_grad is not None, "expert LoRA lost its grad"
    torch.testing.assert_close(lora_grad, want_lora, rtol = 0, atol = 0)
    if x_requires_grad:
        torch.testing.assert_close(x_grad, want_x, rtol = 0, atol = 0)
