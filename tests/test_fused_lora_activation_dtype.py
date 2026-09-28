# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""matmul_lora must reconcile an fp32 activation against a 16-bit base weight."""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels.utils import matmul_lora

# Not module-wide: the probe test below must run on CPU / non-CUDA runners.
_gpu = pytest.mark.gpu

D_IN, D_OUT, R, ROWS, S = 32, 48, 8, 16, 2.0


def _operands(x_dtype, w_dtype, lora_dtype):
    g = torch.Generator().manual_seed(0)
    X = (torch.randn(ROWS, D_IN, generator = g) * 0.5).cuda().to(x_dtype)
    W = (torch.randn(D_OUT, D_IN, generator = g) * 0.02).cuda().to(w_dtype)
    A = (torch.randn(R, D_IN, generator = g) * 0.05).cuda().to(lora_dtype)
    B = (torch.randn(D_OUT, R, generator = g) * 0.05).cuda().to(lora_dtype)
    return X, W, A, B


def _reference(X, W, A, B):
    X32, W32, A32, B32 = (t.float() for t in (X, W, A, B))
    return X32 @ W32.t() + (X32 @ A32.t()) @ B32.t() * S


@_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a real accelerator")
@pytest.mark.parametrize("w_dtype", [torch.float16, torch.bfloat16])
def test_fp32_activation_into_16bit_weight(w_dtype):
    X, W, A, B = _operands(torch.float32, w_dtype, torch.float32)
    with torch.autocast("cuda", enabled = False):
        out = matmul_lora(X, W, None, A, B, S)
    assert out.dtype == w_dtype, "output follows the base weight compute dtype"
    ref = _reference(X, W, A, B)
    torch.testing.assert_close(out.float(), ref, rtol = 2e-2, atol = 2e-2)


@_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a real accelerator")
@pytest.mark.parametrize(
    "w_dtype,amp_dtype",
    [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)],
)
def test_autocast_dtype_differs_from_weight(w_dtype, amp_dtype):
    """In-place addmm_ is not autocast-eligible, so LoRA operands must follow `out`, not W."""
    X, W, A, B = _operands(amp_dtype, w_dtype, torch.float32)
    with torch.autocast("cuda", dtype = amp_dtype, enabled = True):
        out = matmul_lora(X, W, None, A, B, S)
    assert out.dtype == amp_dtype, "under autocast the compute dtype is the autocast dtype"
    ref = _reference(X, W, A, B)
    torch.testing.assert_close(out.float(), ref, rtol = 5e-2, atol = 5e-2)


@_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a real accelerator")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matching_dtypes_unchanged(dtype):
    X, W, A, B = _operands(dtype, dtype, dtype)
    with torch.autocast("cuda", enabled = False):
        out = matmul_lora(X, W, None, A, B, S)
    assert out.dtype == dtype
    ref = _reference(X, W, A, B)
    torch.testing.assert_close(out.float(), ref, rtol = 2e-2, atol = 2e-2)


def test_autocast_probe_uses_the_torch_device_name():
    """torch.is_autocast_enabled("hip") raises, so probe with DEVICE_TYPE_TORCH and fail open."""
    from unsloth.kernels import utils as U

    assert U._AUTOCAST_PROBE in ("device", "legacy", None)
    assert U.DEVICE_TYPE_TORCH not in ("hip", "mlx")
    assert U.torch_is_autocast_enabled() in (True, False)

    saved = U._AUTOCAST_PROBE
    try:
        U._AUTOCAST_PROBE = None
        assert U.torch_is_autocast_enabled() is False, "unresolvable device fails open"
        # torch 2.1-2.3: zero-arg form must still be asked, not assumed disabled
        U._AUTOCAST_PROBE = "legacy"
        assert U.torch_is_autocast_enabled() in (True, False)
    finally:
        U._AUTOCAST_PROBE = saved
