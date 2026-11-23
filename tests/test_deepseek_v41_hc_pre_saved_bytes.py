# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""hc_pre (model.py:959-962) must not keep an FP32 copy of the bf16 hyper-connection stream for backward:
only x (alive anyway) and pre_mix are saved, and outputs and grads stay bitwise equal to the upcast math."""

import pytest

torch = pytest.importorskip("torch")

from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _inputs(
    dtype = torch.bfloat16,
    device = "cpu",
    pre_dtype = torch.float32,
):
    torch.manual_seed(0)
    x = torch.randn(1, 64, 4, 128, dtype = dtype, device = device, requires_grad = True)
    pre = torch.rand(1, 64, 4, device = device).to(pre_dtype).requires_grad_(True)
    return x, pre


def _saved_fp32_stream_bytes(fn, x, pre):
    saved = {}

    def pack(t):
        saved[(t.data_ptr(), tuple(t.shape), t.dtype)] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(x, pre)
    out.float().sum().backward()
    return sum(
        n for (_, shape, dt), n in saved.items() if dt == torch.float32 and shape == tuple(x.shape)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_hc_pre_does_not_save_fp32_stream(device):
    x, pre = _inputs(device = device)
    assert _saved_fp32_stream_bytes(dsv41.hc_pre, x, pre) == 0
    # The upcast math does save it: the check above is not vacuous.
    x, pre = _inputs(device = device)
    assert _saved_fp32_stream_bytes(dsv41._hc_pre_upcast, x, pre) == x.numel() * 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pre_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hc_pre_bitwise_equal(dtype, pre_dtype, device):
    g = torch.randn(1, 64, 128).to(device = device, dtype = dtype)
    results = []
    for fn in (dsv41._hc_pre_upcast, dsv41.hc_pre):
        x, pre = _inputs(dtype, device, pre_dtype)
        out = fn(x, pre)
        (out.float() * g.float()).sum().backward()
        results.append((out.detach(), x.grad, pre.grad))
    for a, b in zip(*results):
        assert a.dtype == b.dtype and torch.equal(a, b)
