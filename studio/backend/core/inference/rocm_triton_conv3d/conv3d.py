# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/conv/conv3d.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: NCDHW, zeros padding, no activation (NDHWC / logger dropped); conv3d() is the upstream entry with layout fixed to ncdhw;
# the H/W Winograd routes are taken only when ALLOW_WINOGRAD is True (off by default: bf16 error 1.45-1.48x MIOpen's).
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
from enum import Enum

import torch

from ._launch import (
    _launch_1x1x1_3d,
    _launch_3x3x3_cblocked,
    _launch_general_3d,
    _launch_winograd_hw_f4x3,
    _prepare_conv_padding,
)
from ._prepack import (
    get_or_make_weight_pack_3d,
    get_or_make_weight_pack_3x3x3,
    get_or_make_winograd_hw_filter_f4x3,
    prepack_ncdhw_to_cblocked,
)
from ._utils import (
    BLOCK_K,
    _alloc_output_3d,
    _conv3d_dims,
    _ensure_layout,
    _is_1x1x1_conv,
    _is_3x3x3_conv,
    _is_amd_wave32,
    _is_winograd_hw_3d_eligible,
    _normalize_conv3d_params,
    _out_dhw,
    _prep_bias,
    _require_winograd_hw_3d_eligible,
    _validate_conv_devices,
)


# Local: the H/W Winograd routes (bf16 error 1.45-1.48x MIOpen's, direct routes 0.70x) only on opt-in; when False a
# Winograd-eligible 3x3x3 runs the direct cblocked kernel instead.
ALLOW_WINOGRAD = False


class Route3D(Enum):
    ONE_X_ONE_X_ONE = "_conv3d_1x1x1_kernel"
    WINOGRAD_HW = "_winograd_hw_f4x3_* (3 kernels)"
    WINOGRAD_HW_CBLOCKED = "_winograd_hw_f4x3_cblocked_* (3 kernels)"
    CBLOCKED_NCDHW = "_conv3d_3x3x3_cblocked_kernel"
    NDHWC_3X3X3 = "_conv3d_3x3x3_ndhwc_kernel"
    GENERAL = "_conv3d_general_kernel"


def _select_3x3x3_ncdhw_method(
    N,
    C,
    D,
    H,
    W,
    K_out,
    stride,
    padding,
    dilation,
    block_c = BLOCK_K,
):
    """Pick the NCDHW 3x3x3 method using shape and architecture heuristics.

    The wave32 policy was fitted conservatively to complete-path measurements
    on gfx1201, including activation packing. Wave64 retains its conservative
    fallback until the same sweep can be run on that hardware.
    """
    if not _is_winograd_hw_3d_eligible(3, 3, 3, stride, dilation, C):
        return "general" if C < block_c else "cblocked"

    OD, P, Q = _out_dhw(D, H, W, 3, 3, 3, stride, padding, dilation)
    tile_h = (P + 3) // 4
    tile_w = (Q + 3) // 4
    tiles = N * OD * tile_h * tile_w

    if not _is_amd_wave32():
        if C < block_c:
            return "general"
        if C >= 512 and K_out >= 512 and tiles >= 98:
            return "winograd_hw"
        return "cblocked"

    # Measured: padding C to a dot tile loses only on very narrow expanding projections.
    if C < 16 or (C < 32 and K_out >= 4 * C):
        return "general"

    # Transform/padded-GEMM-bound; the 8x cap also avoids a large Winograd output temporary.
    if C < 160 or K_out < 96 or K_out >= 8 * C or tiles < 98:
        return "cblocked"

    # OD=1: Winograd transforms 3 input planes per output plane; require a measured high-reuse shape.
    if OD == 1:
        expands_channels = K_out >= 2 * C
        balanced_channels = C >= 320 and K_out >= 320 and tiles >= 220
        compresses_channels = C >= 4 * K_out and K_out >= 96
        if not (expands_channels or balanced_channels or compresses_channels):
            return "cblocked"

    input_elements = N * C * D * H * W
    small_input = input_elements <= 64 * 1024 * 1024
    if C >= 1024 and K_out <= 128 and not small_input:
        return "cblocked"
    if (C >= 1024 and small_input) or (
        C >= 512 and small_input and (tiles <= 220 or padding != (0, 0, 0))
    ):
        return "winograd_hw_cblocked"
    return "winograd_hw"


def _resolve_route(
    T,
    R,
    S,
    stride,
    dilation,
    N,
    C,
    D,
    H,
    W,
    K_out,
    layout,
    padding = (0, 0, 0),
):
    """Single source of dispatch for conv3d.

    1x1x1 -> specialized channel-GEMM kernel. 3x3x3 -> specialized kernel picked
    by layout: NCDHW selects among cblocked and two H/W Winograd input paths;
    NDHWC uses the channels-last kernel. Everything else uses the im2col-free
    general kernel.
    """
    if _is_1x1x1_conv(T, R, S, dilation):
        return Route3D.ONE_X_ONE_X_ONE
    if _is_3x3x3_conv(T, R, S) and dilation == (1, 1, 1):
        if layout == "ndhwc":
            return Route3D.NDHWC_3X3X3
        method = _select_3x3x3_ncdhw_method(N, C, D, H, W, K_out, stride, padding, dilation)
        if method.startswith("winograd") and not ALLOW_WINOGRAD:
            method = "cblocked"
        if method == "general":
            return Route3D.GENERAL
        if method == "winograd_hw_cblocked":
            return Route3D.WINOGRAD_HW_CBLOCKED
        if method == "winograd_hw":
            return Route3D.WINOGRAD_HW
        return Route3D.CBLOCKED_NCDHW
    return Route3D.GENERAL


def conv3d_general(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
    activation = "none",
    block_k = BLOCK_K,
    layout = "ncdhw",
    padding_mode = "zeros",
):
    """conv3d using the general im2col-free kernel with K-major prepacked
    weights. Handles any kernel size / stride / padding / dilation in either
    layout. Safe to call directly: the input is forced into the physical layout
    the kernel assumes (the kernel computes strides analytically)."""
    layout = layout.lower()
    if layout not in ("ncdhw", "ndhwc"):
        raise ValueError(f"layout must be 'ncdhw' or 'ndhwc', got '{layout}'")
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    _validate_conv_devices(x, w_oidhw, bias, op = "conv3d")
    x, padding = _prepare_conv_padding(x, padding, padding_mode, layout = layout)
    x = _ensure_layout(x, layout)
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(x, w_oidhw, stride, padding, dilation)

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
    w_k, K_pad = get_or_make_weight_pack_3d(w_oidhw, block_k)
    _launch_general_3d(
        x,
        w_k,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        T,
        R,
        S,
        OD,
        P,
        Q,
        K_pad,
        stride,
        padding,
        dilation,
        block_k,
        activation,
        layout = layout,
    )
    return y


def conv3d_1x1x1(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
    activation = "none",
    block_k = BLOCK_K,
    layout = "ncdhw",
    padding_mode = "zeros",
):
    """conv3d for 1x1x1 kernels — a pure channel-reduction GEMM. Raises
    ValueError for non-1x1x1. Safe to call directly: the input is forced into
    the physical layout the kernel assumes."""
    layout = layout.lower()
    if layout not in ("ncdhw", "ndhwc"):
        raise ValueError(f"layout must be 'ncdhw' or 'ndhwc', got '{layout}'")
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    _validate_conv_devices(x, w_oidhw, bias, op = "conv3d")
    x, padding = _prepare_conv_padding(x, padding, padding_mode, layout = layout)
    x = _ensure_layout(x, layout)
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(x, w_oidhw, stride, padding, dilation)
    if not _is_1x1x1_conv(T, R, S, dilation):
        raise ValueError(
            f"conv3d_1x1x1 requires 1x1x1 kernel with dilation=1, "
            f"got {T}x{R}x{S} dilation={dilation}"
        )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
    _launch_1x1x1_3d(
        x,
        w_oidhw.contiguous(),
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        OD,
        P,
        Q,
        stride,
        padding,
        activation,
        layout = layout,
    )
    return y


def conv3d_ncdhw_cblocked(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
    activation = "none",
    block_k = BLOCK_K,
    x_blocked = None,
    padding_mode = "zeros",
):
    """NCDHW 3x3x3 conv3d with channel-blocked (NCDHWc) input packing for
    coalesced channel loads. Raises ValueError for non-3x3x3.

    x_blocked: optional pre-packed NCDHWc input (used by the benchmark to time
    the kernel without host-side packing); when None the input is packed here."""
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    _validate_conv_devices(x, w_oidhw, bias, op = "conv3d")
    x, padding = _prepare_conv_padding(
        x, padding, padding_mode, layout = "ncdhw", x_blocked = x_blocked
    )
    x = _ensure_layout(x, "ncdhw")
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(x, w_oidhw, stride, padding, dilation)
    if not _is_3x3x3_conv(T, R, S):
        raise ValueError(f"conv3d_ncdhw_cblocked requires 3x3x3 kernel, got {T}x{R}x{S}")

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ncdhw")
    bias_fp32 = _prep_bias(bias)
    w_3x3x3, C_pad = get_or_make_weight_pack_3x3x3(w_oidhw, block_k)
    if x_blocked is None:
        x_blocked, C_pad_x = prepack_ncdhw_to_cblocked(x, block_k)
    else:
        if x_blocked.ndim != 6:
            raise ValueError(
                "conv3d_ncdhw_cblocked requires a 6-D NCDHWc x_blocked "
                f"tensor, got {x_blocked.ndim}-D"
            )
        C_pad_x = x_blocked.shape[-1] * x_blocked.shape[1]
    assert C_pad_x == C_pad, f"Channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_3x3x3_cblocked(
        x_blocked,
        w_3x3x3,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        OD,
        P,
        Q,
        C_pad,
        block_k,
        stride,
        padding,
        dilation,
        activation,
    )
    return y


def conv3d_winograd_hw_f4x3(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
    activation = "none",
    block_k = BLOCK_K,
    padding_mode = "zeros",
):
    """NCDHW 3x3x3 conv3d via 2.5D Winograd (F(4x4,3x3) on H,W + direct depth).
    Raises ValueError for non-eligible convs (needs 3x3x3, stride=1, dilation=1,
    C>=4)."""
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    _validate_conv_devices(x, w_oidhw, bias, op = "conv3d")
    x, padding = _prepare_conv_padding(x, padding, padding_mode, layout = "ncdhw")
    x = _ensure_layout(x, "ncdhw")
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(x, w_oidhw, stride, padding, dilation)
    _require_winograd_hw_3d_eligible("conv3d_winograd_hw_f4x3", T, R, S, stride, dilation, C)

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ncdhw")
    bias_fp32 = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_hw_filter_f4x3(w_oidhw, block_k)
    _launch_winograd_hw_f4x3(
        x,
        U,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        OD,
        P,
        Q,
        C_pad,
        padding,
        activation,
        block_k = block_k,
    )
    return y


def conv3d_winograd_hw_f4x3_cblocked(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
    activation = "none",
    block_k = BLOCK_K,
    x_blocked = None,
    padding_mode = "zeros",
):
    """2.5D Winograd with a channel-blocked (NCDHWc) input transform for
    coalesced channel loads. Same GEMM/output transform as
    :func:`conv3d_winograd_hw_f4x3`; only the input read changes. Raises
    ValueError for non-eligible convs.

    x_blocked: optional pre-packed NCDHWc input (used by the benchmark to time
    the kernel without host-side packing); when None the input is packed here."""
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    _validate_conv_devices(x, w_oidhw, bias, op = "conv3d")
    x, padding = _prepare_conv_padding(
        x, padding, padding_mode, layout = "ncdhw", x_blocked = x_blocked
    )
    x = _ensure_layout(x, "ncdhw")
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(x, w_oidhw, stride, padding, dilation)
    _require_winograd_hw_3d_eligible(
        "conv3d_winograd_hw_f4x3_cblocked", T, R, S, stride, dilation, C
    )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ncdhw")
    bias_fp32 = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_hw_filter_f4x3(w_oidhw, block_k)
    if x_blocked is None:
        x_blocked, C_pad_x = prepack_ncdhw_to_cblocked(x, block_k)
    else:
        if x_blocked.ndim != 6:
            raise ValueError(
                "conv3d_winograd_hw_f4x3_cblocked requires a 6-D NCDHWc "
                f"x_blocked tensor, got {x_blocked.ndim}-D"
            )
        C_pad_x = x_blocked.shape[-1] * x_blocked.shape[1]
    assert C_pad_x == C_pad, f"Channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_winograd_hw_f4x3(
        x,
        U,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        OD,
        P,
        Q,
        C_pad,
        padding,
        activation,
        block_k = block_k,
        x_blocked = x_blocked,
    )
    return y


def _route_and_run(
    x,
    w_oidhw,
    bias,
    stride,
    padding,
    dilation,
    activation,
    block_k,
    layout,
    *,
    padding_mode = "zeros",
):
    """Shared dispatch body for conv3d_ncdhw / conv3d_ndhwc: resolve the route
    once and dispatch to the matching wrapper.
    """
    x, padding = _prepare_conv_padding(x, padding, padding_mode, layout = layout)
    x = _ensure_layout(x, layout)
    N, C, D, H, W = x.shape
    K_out, _C, T, R, S = w_oidhw.shape
    route = _resolve_route(
        T,
        R,
        S,
        stride,
        dilation,
        N,
        C,
        D,
        H,
        W,
        K_out,
        layout,
        padding = padding,
    )

    if route == Route3D.ONE_X_ONE_X_ONE:
        return conv3d_1x1x1(
            x,
            w_oidhw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout = layout,
        )
    if route == Route3D.WINOGRAD_HW:
        return conv3d_winograd_hw_f4x3(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route3D.WINOGRAD_HW_CBLOCKED:
        return conv3d_winograd_hw_f4x3_cblocked(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route3D.CBLOCKED_NCDHW:
        return conv3d_ncdhw_cblocked(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route3D.NDHWC_3X3X3:
        return conv3d_ndhwc_3x3x3(x, w_oidhw, bias, stride, padding, dilation, activation, block_k)
    return conv3d_general(
        x,
        w_oidhw,
        bias,
        stride,
        padding,
        dilation,
        activation,
        block_k,
        layout = layout,
    )


def conv3d_ncdhw(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
    activation = "none",
    block_k = BLOCK_K,
    padding_mode = "zeros",
):
    """NCDHW Conv3D with shape-driven specialized-kernel routing."""
    _validate_conv_devices(x, w_oidhw, bias, op = "conv3d")
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    with torch.cuda.device(x.device):
        return _route_and_run(
            x,
            w_oidhw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout = "ncdhw",
            padding_mode = padding_mode,
        )


def conv3d(
    x,
    w_oidhw,
    bias = None,
    stride = (1, 1, 1),
    padding = (0, 0, 0),
    dilation = (1, 1, 1),
):
    """Forward NCDHW 3-D conv (groups 1, zeros padding, no activation), like upstream ``conv3d(..., layout="ncdhw")``.
    fp16 / bf16 in, same dtype out, fp32 accumulation."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"conv3d only supports fp16 and bf16 inputs, got {x.dtype}")
    stride, padding, dilation = _normalize_conv3d_params(stride, padding, dilation)
    return conv3d_ncdhw(x, w_oidhw, bias, stride, padding, dilation, "none")
