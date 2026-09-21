# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""matmul_lora must reconcile the activation dtype against the base weight.

The fused LoRA path multiplies the activation by the base weight directly, and
torch.matmul never promotes a mixed-precision pair, so an fp32 activation meeting
an fp16/bf16 base weight raises

    RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half

A plain nn.Linear never shows this: under autocast both operands are reconciled
for it, and outside autocast a user hitting the mismatch gets the same error from
the Linear itself. The fused path has a second, subtler exposure the Linear does
not: `out.addmm_(...)` for the LoRA term is IN-PLACE, and in-place ops are not
autocast-eligible, so its operands must follow the dtype the base matmul actually
produced (the autocast dtype when autocast is on, the weight dtype when it is
not) rather than either the activation's or the weight's dtype.

The tests below pin both halves:

  * test_fp32_activation_into_16bit_weight  -- the reported crash, autocast off.
  * test_autocast_dtype_differs_from_weight -- the in-place LoRA accumulation must
    follow `out`, not W. Pinning it to W regressed this case, which works on main.
  * test_matching_dtypes_unchanged          -- the ordinary path is untouched.

All three need a real accelerator, since matmul_lora is a CUDA/XPU/NPU path.
"""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels.utils import matmul_lora

# The three matmul tests are marked `gpu` individually rather than module-wide:
# the device-name probe test below is pure Python, and it is precisely the check
# that has to run on the CPU-only and non-CUDA runners, since the defect it pins
# (torch.is_autocast_enabled("hip")) only ever shows up off this box.
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
    """What a plain Linear plus its LoRA term would produce, in fp32."""
    X32, W32, A32, B32 = (t.float() for t in (X, W, A, B))
    return X32 @ W32.t() + (X32 @ A32.t()) @ B32.t() * S


@_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a real accelerator")
@pytest.mark.parametrize("w_dtype", [torch.float16, torch.bfloat16])
def test_fp32_activation_into_16bit_weight(w_dtype):
    """An fp32 activation must not crash against a 16-bit base weight."""
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
    """The in-place LoRA accumulation must follow `out`, not the weight.

    autocast produces the base matmul in `amp_dtype`, but `out.addmm_` is in-place
    and so is not autocast-eligible. Deriving the LoRA operand dtype from W here
    raises `self and mat2 must have the same dtype`.
    """
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
    """The ordinary path, where every operand already agrees, still works."""
    X, W, A, B = _operands(dtype, dtype, dtype)
    with torch.autocast("cuda", enabled = False):
        out = matmul_lora(X, W, None, A, B, S)
    assert out.dtype == dtype
    ref = _reference(X, W, A, B)
    torch.testing.assert_close(out.float(), ref, rtol = 2e-2, atol = 2e-2)


def test_autocast_probe_uses_the_torch_device_name():
    """The autocast query must use the TORCH device name, and fail open if rejected.

    DEVICE_TYPE is "hip" on ROCm and "mlx" on Apple Silicon, and torch's autocast
    APIs accept neither: `torch.is_autocast_enabled("hip")` raises `unknown device
    type for autocast`. Passing DEVICE_TYPE straight in would therefore turn the
    mismatched-dtype case -- the one this file exists to support -- into a crash on
    every ROCm box. DEVICE_TYPE_TORCH does that mapping; a name even it cannot
    resolve has to fall back to "no ambient autocast", so matmul_lora reconciles
    the dtypes itself rather than erroring on the probe.
    """
    from unsloth.kernels import utils as U

    assert U._AUTOCAST_DEVICE in (None, U.DEVICE_TYPE_TORCH)
    assert U._AUTOCAST_DEVICE != "hip" and U._AUTOCAST_DEVICE != "mlx"
    # never raises, whatever the device
    assert U.torch_is_autocast_enabled() in (True, False)

    saved = U._AUTOCAST_DEVICE
    try:
        U._AUTOCAST_DEVICE = None
        assert U.torch_is_autocast_enabled() is False, "unresolvable device fails open"
    finally:
        U._AUTOCAST_DEVICE = saved
