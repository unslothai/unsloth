# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Vendored from ROCm/aiter (https://github.com/ROCm/aiter), commit 9e5e03626fa119f2429fa1c1578d5a715a6c2401,
# aiter/ops/triton/conv/_launch.py; gfx1151 conv3d configs from ROCm/aiter PR #6973 (065f9f99c9d8a412bd63935c2a89b196d3f110d7).
# Changes: Conv3D NCDHW launches only (Conv2D and NDHWC launches dropped).
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
import torch
import torch.nn.functional as F
import triton

from .conv3d_1x1x1 import (
    _conv3d_1x1x1_kernel,
)
from .conv3d_1x1x1 import (
    _get_config as _get_config_1x1x1_3d,
)
from .conv3d_3x3x3 import (
    _conv3d_3x3x3_cblocked_kernel,
)
from .conv3d_3x3x3 import (
    _get_config_cblocked as _get_config_3x3x3_cblocked,
)
from .conv3d_general import (
    _conv3d_general_kernel,
)
from .conv3d_general import (
    _get_config as _get_config_general_3d,
)
from .conv3d_winograd_hw_f4x3 import (
    _get_config_gemm as _get_config_wino_hw_gemm,
)
from .conv3d_winograd_hw_f4x3 import (
    _get_config_input as _get_config_wino_hw_input,
)
from .conv3d_winograd_hw_f4x3 import (
    _get_config_output as _get_config_wino_hw_output,
)
from .conv3d_winograd_hw_f4x3 import (
    _winograd_hw_f4x3_batched_gemm_kernel,
    _winograd_hw_f4x3_input_transform_kernel,
    _winograd_hw_f4x3_output_transform_kernel,
)
from .ncx_to_cblocked import (
    _get_config_3d as _get_config_prepack_3d,
)
from .ncx_to_cblocked import (
    _ncx_to_cblocked_kernel,
)
from ._utils import (
    _ensure_layout,
    _normalize_tuple,
    _padding_geometry,
    _validate_padding_mode,
    _winograd_transform_storage_dtype,
)
from ._config import (
    format_prepack_shape_key_3d,
    format_shape_key_3d,
)


def _kernel_activation(activation):
    """Map the public GELU name to the kernels' tanh approximation."""
    return "gelu_tanh" if activation == "gelu" else activation


def _make_mn_grid(M_total, K_out):
    """Grid for the GEMM-style conv kernels (1x1, 3x3 nhwc/cblocked, general):
    one program per (BLOCK_M tile of M_total) x (BLOCK_N tile of K_out)."""

    def grid(meta):
        return (triton.cdiv(M_total, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),)

    return grid


def _make_wino_input_grid(T, C_pad):
    """Grid for the Winograd F(4,3) input-transform kernels: one program per
    tile T x (BLOCK_C tile of C_pad)."""

    def grid(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    return grid


def _make_wino_gemm_grid(T, K_out):
    """Grid for the Winograd F(4,3) batched GEMM: (BLOCK_M tile of T) x
    (BLOCK_N tile of K_out) program blocks, batched over the 36 tile elements."""

    def grid(meta):
        return (
            triton.cdiv(T, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
            36,
        )

    return grid


def _make_wino_output_grid(T, K_out):
    """Grid for the Winograd F(4,3) output-transform kernels: one program per
    tile T x (BLOCK_K tile of K_out)."""

    def grid(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    return grid


def _prepare_conv_padding(
    source: torch.Tensor,
    padding,
    padding_mode: str = "zeros",
    *,
    layout: str,
    x_blocked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Prepare Conv2D or Conv3D padding before routing and activation packing.

    No-op paths return the input unchanged; callers handle their layout.
    """
    if layout in ("nchw", "nhwc"):
        rank = 4
    elif layout in ("ncdhw", "ndhwc"):
        rank = 5
    else:
        raise ValueError(
            f"layout must be one of ('nchw', 'nhwc', 'ncdhw', 'ndhwc'), got {layout!r}"
        )
    if source.ndim != rank:
        logical_layout = "NCHW" if rank == 4 else "NCDHW"
        raise ValueError(
            f"conv{rank - 2}d padding input must be {rank}-D {logical_layout}, "
            f"got {source.ndim}-D"
        )
    padding = _normalize_tuple(padding, rank - 2, "padding", allow_zero = True)
    spatial = tuple(source.shape[2:])
    _validate_padding_mode(padding_mode, spatial, padding)
    padded_spatial, conv_padding = _padding_geometry(spatial, padding, padding_mode)
    if padded_spatial == spatial:
        return source, conv_padding
    if x_blocked is not None:
        raise ValueError(
            "x_blocked cannot be used with materialized padding; "
            "omit it to pack the padded input"
        )
    pad = tuple(value for amount in reversed(padding) for value in (amount, amount))
    output = F.pad(source, pad, mode = padding_mode)
    return _ensure_layout(output, layout), conv_padding


def _launch_ncdhw_to_cblocked(x, x_blocked, N, C, D, H, W, C_pad, block_c):
    """Launch the fused NCDHW-to-NCDHWc activation pack."""
    spatial = D * H * W
    shape_key = format_prepack_shape_key_3d(N, C, D, H, W, block_c)
    config = _get_config_prepack_3d(shape_key = shape_key, M = spatial)

    def grid(meta):
        return (
            triton.cdiv(spatial, meta["BLOCK_M"]),
            triton.cdiv(C_pad, meta["BLOCK_C"]),
            N,
        )

    _ncx_to_cblocked_kernel[grid](
        x,
        x_blocked,
        C,
        spatial,
        C_PAD = C_pad,
        CB = block_c,
        **config,
    )


def _launch_general_3d(
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
    layout = "ncdhw",
):
    """Launch the general Conv3D kernel for NCDHW or NDHWC storage."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    M_total = N * OD * P * Q

    shape_key = format_shape_key_3d(
        N = N,
        C = C,
        D = D,
        H = H,
        W = W_in,
        K = K_out,
        T = T,
        R = R,
        S = S,
        sd = sd,
        sh = sh,
        sw = sw,
        pd = pd,
        ph = ph,
        pw = pw,
        dd = dd,
        dh = dh,
        dw = dw,
    )
    config = _get_config_general_3d(shape_key = shape_key, M = M_total, variants = (layout,))

    _conv3d_general_kernel[_make_mn_grid(M_total, K_out)](
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
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        M_total,
        HAS_BIAS = bias_fp32 is not None,
        ACTIVATION = _kernel_activation(activation),
        LAYOUT = layout,
        **config,
    )


def _launch_1x1x1_3d(
    x,
    w_oidhw,
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
    layout = "ncdhw",
):
    """Launch the specialized 1x1x1 Conv3D channel-reduction kernel."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    w = w_oidhw.squeeze(-1).squeeze(-1).squeeze(-1).contiguous()
    M_total = N * OD * P * Q

    shape_key = format_shape_key_3d(
        N = N,
        C = C,
        D = D,
        H = H,
        W = W_in,
        K = K_out,
        T = 1,
        R = 1,
        S = 1,
        sd = sd,
        sh = sh,
        sw = sw,
        pd = pd,
        ph = ph,
        pw = pw,
        dd = 1,
        dh = 1,
        dw = 1,
    )
    config = _get_config_1x1x1_3d(shape_key = shape_key, M = M_total, variants = (layout,))

    _conv3d_1x1x1_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w,
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
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        M_total,
        HAS_BIAS = bias_fp32 is not None,
        ACTIVATION = _kernel_activation(activation),
        LAYOUT = layout,
        **config,
    )


def _launch_3x3x3_cblocked(
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
    Cb,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch the NCDHWc-native 3x3x3 Conv3D kernel."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    M_total = N * OD * P * Q

    shape_key = format_shape_key_3d(
        N = N,
        C = C,
        D = D,
        H = H,
        W = W_in,
        K = K_out,
        T = 3,
        R = 3,
        S = 3,
        sd = sd,
        sh = sh,
        sw = sw,
        pd = pd,
        ph = ph,
        pw = pw,
        dd = dd,
        dh = dh,
        dw = dw,
    )
    config = _get_config_3x3x3_cblocked(shape_key = shape_key, M = M_total)

    _conv3d_3x3x3_cblocked_kernel[_make_mn_grid(M_total, K_out)](
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
        Cb,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        M_total,
        HAS_BIAS = bias_fp32 is not None,
        ACTIVATION = _kernel_activation(activation),
        **config,
    )


def _launch_winograd_hw_f4x3(
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
    block_k = 64,
    x_blocked = None,
):
    """Launch 2.5-D Winograd F(4x4,3x3) over H/W and direct depth."""
    pd, ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T_hw = tile_H * tile_W
    T_v = N * D * T_hw
    T_out = N * OD * T_hw

    cblocked = x_blocked is not None
    x_in = x_blocked if cblocked else x
    transform_dtype = _winograd_transform_storage_dtype(x_in.dtype)
    V = torch.empty((36, T_v, C_pad), device = x_in.device, dtype = transform_dtype)
    M = torch.empty((36, T_out, K_out), device = x_in.device, dtype = torch.float32)

    shape_key = format_shape_key_3d(
        N = N,
        C = C,
        D = D,
        H = H,
        W = W_in,
        K = K_out,
        T = 3,
        R = 3,
        S = 3,
        sd = 1,
        sh = 1,
        sw = 1,
        pd = pd,
        ph = ph,
        pw = pw,
        dd = 1,
        dh = 1,
        dw = 1,
    )
    input_variant = "cblocked" if cblocked else "ncdhw"
    input_config = _get_config_wino_hw_input(shape_key = shape_key, M = T_v, variants = (input_variant,))
    gemm_config = _get_config_wino_hw_gemm(shape_key = shape_key, M = T_out)
    output_config = _get_config_wino_hw_output(shape_key = shape_key, M = T_out)

    _winograd_hw_f4x3_input_transform_kernel[_make_wino_input_grid(T_v, C_pad)](
        x_in,
        V,
        N,
        C,
        C_pad,
        D,
        H,
        W_in,
        tile_H,
        tile_W,
        T_v,
        ph,
        pw,
        Cb = block_k,
        CBLOCKED = cblocked,
        **input_config,
    )

    _winograd_hw_f4x3_batched_gemm_kernel[_make_wino_gemm_grid(T_out, K_out)](
        V,
        U,
        M,
        N,
        D,
        OD,
        T_hw,
        T_v,
        T_out,
        K_out,
        C_pad,
        pd,
        **gemm_config,
    )

    _winograd_hw_f4x3_output_transform_kernel[_make_wino_output_grid(T_out, K_out)](
        M,
        bias_fp32,
        y,
        N,
        K_out,
        OD,
        P,
        Q,
        tile_H,
        tile_W,
        T_out,
        HAS_BIAS = bias_fp32 is not None,
        ACTIVATION = _kernel_activation(activation),
        **output_config,
    )
