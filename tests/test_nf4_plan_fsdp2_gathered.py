# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""FSDP2 QLoRA: the training forward dequantizes each all-gathered NF4 weight into the scratch
buffer through fast_dequantize's planned launch. FSDP2 frees that weight's storage after use and
re-allocates it for the next all-gather, so the same tensor comes back at a new address every
step. The plan must still be reused for it (no rebuild per call), must read the bytes now at the
new address (never stale ones), and an unmarked tensor keeps today's address-keyed rebuild."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")
bnb_functional = pytest.importorskip("bitsandbytes.functional")

_GATHERED = "_unsloth_fsdp2_gathered"


def _utils():
    from unsloth.kernels import triton_launch, utils
    if not utils._USE_NF4_KERNELS or not triton_launch._ENABLED:
        pytest.skip("fused NF4 kernels or the fast Triton relaunch are off here")
    return utils


def _quantize(seed, shape = (512, 256)):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(shape, dtype = torch.bfloat16, device = "cuda", generator = g)
    return bnb_functional.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)


def _gathered_param(q, mark):
    # What FSDP2 hands the forward for a float-packed Params4bit: a frozen parameter whose packed
    # bytes are stored as bfloat16 (fsdp2_qlora.params4bit_to_float_storage), marked when gathered.
    p = torch.nn.Parameter(
        q.reshape(-1).view(torch.bfloat16).reshape(-1, 1).clone(), requires_grad = False
    )
    if mark:
        setattr(p, _GATHERED, True)
    return p


def _reallocate(p, packed_bytes):
    """FSDP2's free_storage / alloc_storage around an all-gather, forced onto a new address, then
    the all-gather's copy-out of `packed_bytes` (uint8)."""
    storage = p.untyped_storage()
    nbytes = storage.nbytes()
    old = p.data_ptr()
    storage.resize_(0)
    # Hold freed blocks of this size until the old address is taken, so the re-allocation moves.
    blocker = []
    for _ in range(4096):
        blocker.append(torch.empty(nbytes, dtype = torch.uint8, device = "cuda"))
        if blocker[-1].data_ptr() == old:
            break
    storage.resize_(nbytes)
    p.data.view(torch.uint8).reshape(-1).copy_(packed_bytes.reshape(-1))
    return old, blocker


def _count_planned(monkeypatch, utils):
    calls = [0]
    original = utils.dequantize_nf4_planned

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(utils, "dequantize_nf4_planned", counted)
    return calls


def _equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def test_gathered_weight_reuses_its_plan_across_reallocation(monkeypatch):
    utils = _utils()
    calls = _count_planned(monkeypatch, utils)
    q, s = _quantize(0)
    q_other, _ = _quantize(1)  # another weight's packed bytes, decoded with the same QuantState
    p = _gathered_param(q, mark = True)
    for _ in range(2):
        got = utils.fast_dequantize(p, s, use_global_buffer = True)
        assert _equal(got, bnb_functional.dequantize_4bit(q, s))
    assert calls[0] == 1 and s._unsloth_nf4_plan is not None

    for step, packed in enumerate((q, q_other, q)):
        old, blocker = _reallocate(p, packed)
        assert p.data_ptr() != old
        got = utils.fast_dequantize(p, s, use_global_buffer = True)
        # The bytes at the new address, not those the plan was built from.
        assert _equal(got, bnb_functional.dequantize_4bit(packed, s)), step
        del blocker
    assert calls[0] == 1, "the planned launch was rebuilt for a re-allocated FSDP2 weight"


def test_gathered_weight_view_reuses_its_plan_across_reallocation(monkeypatch):
    """The fused LoRA MLP dequantizes `downW` in the forward and `downW.t()`, a new view each call,
    in the backward. On one GPU both hit the one plan (same address); after each FSDP2
    re-allocation they must too, and decode the bytes now at the new address."""
    utils = _utils()
    calls = _count_planned(monkeypatch, utils)
    q, s = _quantize(4)
    q_other, _ = _quantize(5)
    p = _gathered_param(q, mark = True)
    got = utils.fast_dequantize(p, s, use_global_buffer = True)
    assert _equal(got, bnb_functional.dequantize_4bit(q, s))
    for step, packed in enumerate((q_other, q, q_other)):
        old, blocker = _reallocate(p, packed)
        assert p.data_ptr() != old
        want = bnb_functional.dequantize_4bit(packed, s)
        assert _equal(utils.fast_dequantize(p, s, use_global_buffer = True), want), step
        assert _equal(utils.fast_dequantize(p.t(), s, use_global_buffer = True).t(), want), step
        assert _equal(utils.fast_dequantize(p, s, use_global_buffer = True), want), step
        del blocker
    assert calls[0] == 1, "the planned launch was rebuilt for a view of a re-allocated FSDP2 weight"


def test_gathered_weight_with_swapped_storage_rebuilds(monkeypatch):
    """A view the plan holds keeps the storage it was built on: when the parameter gets a new
    storage object instead of a resized one, the plan must not read the old bytes."""
    utils = _utils()
    calls = _count_planned(monkeypatch, utils)
    q, s = _quantize(6)
    q_other, _ = _quantize(7)
    p = _gathered_param(q, mark = True)
    utils.fast_dequantize(p.t(), s, use_global_buffer = True)
    p.data = q_other.reshape(-1).view(torch.bfloat16).reshape(-1, 1).clone()
    got = utils.fast_dequantize(p.t(), s, use_global_buffer = True).t()
    assert _equal(got, bnb_functional.dequantize_4bit(q_other, s))
    assert calls[0] == 2


def test_unmarked_weight_still_rebuilds_on_a_new_address(monkeypatch):
    utils = _utils()
    calls = _count_planned(monkeypatch, utils)
    q, s = _quantize(2)
    p = _gathered_param(q, mark = False)
    utils.fast_dequantize(p, s, use_global_buffer = True)
    assert calls[0] == 1
    _, blocker = _reallocate(p, q)
    got = utils.fast_dequantize(p, s, use_global_buffer = True)
    assert _equal(got, bnb_functional.dequantize_4bit(q, s))
    assert calls[0] == 2
    del blocker


def test_gathered_weight_plan_respects_alignment(monkeypatch):
    """A tensor that comes back off the 16 byte alignment the kernel was specialized on rebuilds."""
    utils = _utils()
    calls = _count_planned(monkeypatch, utils)
    q, s = _quantize(3)
    p = _gathered_param(q, mark = True)
    utils.fast_dequantize(p, s, use_global_buffer = True)
    plan = s._unsloth_nf4_plan
    plan[0] = plan[0] + 2  # pretend the plan was built at an address 2 bytes off this one
    got = utils.fast_dequantize(p, s, use_global_buffer = True)
    assert _equal(got, bnb_functional.dequantize_4bit(q, s))
    assert calls[0] == 2
