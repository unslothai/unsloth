# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/conv/_utils.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: Conv3D helpers only (Conv2D helpers dropped).
# The vendored portions remain under the MIT License below:
#
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from collections.abc import Sequence

import torch
import triton

BLOCK_K = 64


_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def _is_amd_wave32():
    """Return whether the active Triton target is an AMD wave32 device."""
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.warp_size == 32


def _validate_conv_devices(
    x,
    weight,
    bias = None,
    *,
    op: str,
):
    """Require input, weight, and optional bias on one CUDA/ROCm device."""
    if not x.is_cuda:
        message = f"{op} input must be a CUDA/ROCm tensor"
    elif weight.device != x.device or (bias is not None and bias.device != x.device):
        message = f"{op} input, weight, and bias must be on the same device"
    else:
        return
    raise ValueError(
        f"{message}, got input={x.device}, weight={weight.device}, "
        f"bias={bias.device if bias is not None else None}"
    )


def _prep_bias(bias):
    """Return a unit-stride bias without launching a dtype-conversion kernel.

    The Triton epilogues promote fp16/bf16 bias values to fp32 while loading.
    """
    if bias is None:
        return None
    return bias if bias.stride(0) == 1 else bias.contiguous()


def _ensure_layout(x, layout):
    """Materialize the physical layout expected by a convolution kernel."""
    if layout == "nhwc":
        return x.contiguous(memory_format = torch.channels_last)
    if layout == "ndhwc":
        return x.contiguous(memory_format = torch.channels_last_3d)
    if layout in ("nchw", "ncdhw"):
        return x.contiguous()
    raise ValueError(f"layout must be one of 'nchw', 'nhwc', 'ncdhw', or 'ndhwc', got {layout!r}")


def _normalize_tuple(
    value,
    dimensions: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    """Normalize an integer or fixed-length sequence to a spatial tuple."""
    if isinstance(value, int):
        result = (value,) * dimensions
    elif isinstance(value, Sequence) and len(value) == dimensions:
        result = tuple(int(v) for v in value)
    else:
        raise ValueError(f"{name} must be an int or a length-{dimensions} sequence, got {value!r}")

    minimum = 0 if allow_zero else 1
    if any(v < minimum for v in result):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {relation}, got {result}")
    return result


def _validate_padding_mode(
    padding_mode: str, spatial: tuple[int, ...], padding: tuple[int, ...]
) -> None:
    """Validate the mode and spatial limits after non-negative normalization."""
    if padding_mode not in _PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode!r}")
    if any(size <= 0 for size in spatial):
        raise ValueError(f"input spatial dimensions must be positive, got {spatial}")
    if padding_mode == "reflect" and any(
        amount >= size for amount, size in zip(padding, spatial, strict = True)
    ):
        raise ValueError(
            "reflect padding must be less than each input spatial dimension, "
            f"got padding={padding} for input {spatial}"
        )
    if padding_mode == "circular" and any(
        amount > size for amount, size in zip(padding, spatial, strict = True)
    ):
        raise ValueError(
            "circular padding must not exceed each input spatial dimension, "
            f"got padding={padding} for input {spatial}"
        )


def _padding_geometry(
    spatial: tuple[int, ...], padding: tuple[int, ...], padding_mode: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return effective dimensions and kernel padding for normalized, valid inputs."""
    if padding_mode == "zeros" or not any(padding):
        return tuple(spatial), padding
    return (
        tuple(size + 2 * amount for size, amount in zip(spatial, padding, strict = True)),
        (0,) * len(padding),
    )


def _normalize_conv3d_params(stride, padding, dilation):
    return (
        _normalize_tuple(stride, 3, "stride"),
        _normalize_tuple(padding, 3, "padding", allow_zero = True),
        _normalize_tuple(dilation, 3, "dilation"),
    )


def _out_dhw(D, H, W, T, R, S, stride, padding, dilation):
    """Return the standard Conv3D output depth, height, and width."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    OD = (D + 2 * pd - dd * (T - 1) - 1) // sd + 1
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return OD, P, Q


def _conv3d_dims(x, w_oidhw, stride, padding, dilation):
    """Validate NCDHW/OIDHW tensors and return all convolution dimensions."""
    if x.ndim != 5:
        raise ValueError(f"conv3d input must be 5-D NCDHW, got {x.ndim}-D")
    if w_oidhw.ndim != 5:
        raise ValueError(f"conv3d weight must be 5-D OIDHW, got {w_oidhw.ndim}-D")
    N, C, D, H, W_in = x.shape
    K_out, Cw, T, R, S = w_oidhw.shape
    if Cw != C:
        raise ValueError(f"weight in-channels {Cw} != input channels {C}")
    OD, P, Q = _out_dhw(D, H, W_in, T, R, S, stride, padding, dilation)
    if OD <= 0 or P <= 0 or Q <= 0:
        raise ValueError(
            "calculated conv3d output is non-positive: "
            f"({OD}, {P}, {Q}) for input {(D, H, W_in)} and kernel {(T, R, S)}"
        )
    return N, C, D, H, W_in, K_out, T, R, S, OD, P, Q


def _alloc_output_3d(N, K_out, OD, P, Q, x, layout):
    """Allocate logical NCDHW output in contiguous or channels-last-3d form."""
    return torch.empty(
        (N, K_out, OD, P, Q),
        device = x.device,
        dtype = x.dtype,
        memory_format = (torch.channels_last_3d if layout == "ndhwc" else torch.contiguous_format),
    )


def _is_1x1x1_conv(T, R, S, dilation):
    return T == 1 and R == 1 and S == 1 and dilation == (1, 1, 1)


def _is_3x3x3_conv(T, R, S):
    return T == 3 and R == 3 and S == 3


def _is_winograd_hw_3d_eligible(
    T,
    R,
    S,
    stride,
    dilation,
    C = None,
):
    if not _is_3x3x3_conv(T, R, S):
        return False
    if stride != (1, 1, 1) or dilation != (1, 1, 1):
        return False
    return not (C is not None and C < 4)


def _require_winograd_hw_3d_eligible(name, T, R, S, stride, dilation, C):
    """Raise if a shape cannot use Conv3D's H/W-only Winograd F(4,3)."""
    if not _is_winograd_hw_3d_eligible(T, R, S, stride, dilation, C):
        raise ValueError(
            f"{name} requires a 3x3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4, got {T}x{R}x{S} stride={stride} "
            f"dilation={dilation} C={C}"
        )


def _winograd_transform_storage_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use fp16 Winograd operands for bf16 input to reduce transform error."""
    return torch.float16 if dtype == torch.bfloat16 else dtype
