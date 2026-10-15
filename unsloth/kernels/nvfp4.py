# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""W4A16 NVFP4: packed E2M1 weights (compressed-tensors `nvfp4-pack-quantized`) dequantized to 16 bit on the fly."""

import functools
import os

import torch
import triton
import triton.language as tl

from .fp8 import _fp8_triton_device_context, _is_transposed_view, _opaque_under_compile

__all__ = [
    "NVFP4_GROUP_SIZE",
    "NVFP4QuantState",
    "nvfp4_dequantize",
    "NVFP4Linear_matmul",
    "nvfp4_linear",
]

NVFP4_GROUP_SIZE = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class NVFP4QuantState:
    __slots__ = ("scale", "global_scale", "shape", "dtype")

    def __init__(
        self,
        scale,
        global_scale,
        shape,
        dtype = torch.bfloat16,
    ):
        self.scale = scale
        self.global_scale = global_scale
        self.shape = tuple(shape)
        self.dtype = dtype


def _nvfp4_dequantize_torch(packed, scale, global_scale, dtype):
    # Same arithmetic as compressed_tensors NVFP4PackedCompressor.decompress: scale / global_scale in fp32, times E2M1.
    rows, half = packed.shape
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim = -1).reshape(rows, half * 2)
    values = lut[(codes & 0x07).long()] * torch.where((codes & 0x08).bool(), -1.0, 1.0)
    group_scale = scale.to(torch.float32) / global_scale.to(torch.float32).reshape(-1)[:1]
    group_scale = group_scale.repeat_interleave(NVFP4_GROUP_SIZE, dim = 1)[:, : half * 2]
    return (values * group_scale).to(dtype)


@triton.jit
def _e2m1_to_float(code):
    # Build the fp32 bits: mag >= 2 is (1 + m / 2) * 2^(e - 1), mag 1 is 0.5, mag 0 is 0; bit 3 is the sign (code 8 = -0.0).
    mag = code & 7
    bits = tl.where(
        mag >= 2, (((mag >> 1) + 126) << 23) | ((mag & 1) << 22), tl.where(mag == 1, 126 << 23, 0)
    )
    return (bits | ((code & 8) << 28)).to(tl.float32, bitcast = True)


@triton.jit
def _e4m3_to_float(b):
    # No float8e4nv before sm_89: decode E4M3 bytes by hand (bias 7, exp 0 subnormal, 0x7F/0xFF NaN, as torch).
    exp = (b >> 3) & 15
    man = b & 7
    normal = (((exp + 120) << 23) | (man << 20)).to(tl.float32, bitcast = True)
    value = tl.where(exp == 0, man.to(tl.float32) * 0.001953125, normal)
    value = tl.where((b & 127) == 127, float("nan"), value)
    # Sign as a bit, not a negation: -x can lower to 0 - x, which turns code 0x80 (-0.0) into +0.0.
    return (value.to(tl.int32, bitcast = True) | ((b & 128) << 24)).to(tl.float32, bitcast = True)


@triton.jit
def _nvfp4_dequant_kernel(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    out_ptr,
    rows,
    half_cols,
    scale_cols,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE_E4M3_BYTES: tl.constexpr,
    INT64_OFFSETS: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    if INT64_OFFSETS:
        r = r.to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    g = tl.program_id(1) * (BLOCK_C // 8) + tl.arange(0, BLOCK_C // 8)
    row_ok = r[:, None] < rows
    byte = tl.load(
        packed_ptr + r[:, None] * half_cols + c[None, :],
        mask = row_ok & (c[None, :] < half_cols),
        other = 0,
    ).to(tl.int32)
    scale = tl.load(
        scale_ptr + r[:, None] * scale_cols + g[None, :],
        mask = row_ok & (g[None, :] < scale_cols),
        other = 0.0,
    )
    scale = _e4m3_to_float(scale.to(tl.int32)) if SCALE_E4M3_BYTES else scale.to(tl.float32)
    group_scale = tl.math.div_rn(scale, tl.load(global_scale_ptr).to(tl.float32))
    values = tl.reshape(
        tl.join(_e2m1_to_float(byte & 15), _e2m1_to_float(byte >> 4)), (BLOCK_R, BLOCK_C // 8, 16)
    )
    values = tl.reshape(values * group_scale[:, :, None], (BLOCK_R, 2 * BLOCK_C))
    oc = tl.program_id(1) * (2 * BLOCK_C) + tl.arange(0, 2 * BLOCK_C)
    tl.store(
        out_ptr + r[:, None] * (2 * half_cols) + oc[None, :],
        values.to(out_ptr.dtype.element_ty),
        mask = row_ok & (oc[None, :] < 2 * half_cols),
    )


@_opaque_under_compile(
    "nvfp4_dequantize_triton",
    lambda packed, scale, global_scale, dtype: packed.new_empty(
        (packed.shape[0], packed.shape[1] * 2), dtype = dtype
    ),
)
def _nvfp4_dequantize_triton(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    rows, half = packed.shape
    out = torch.empty((rows, half * 2), dtype = dtype, device = packed.device)
    BLOCK_R, BLOCK_C = 8, 256
    e4m3 = scale.dtype == torch.float8_e4m3fn
    if e4m3:
        scale = scale.view(torch.uint8)
    with _fp8_triton_device_context(packed):
        _nvfp4_dequant_kernel[(triton.cdiv(rows, BLOCK_R), triton.cdiv(half, BLOCK_C))](
            packed,
            scale,
            global_scale,
            out,
            rows,
            half,
            scale.shape[1],
            BLOCK_R = BLOCK_R,
            BLOCK_C = BLOCK_C,
            SCALE_E4M3_BYTES = e4m3,
            # int32 offsets wrap past 2^31 elements; int64 only then (~2% slower).
            INT64_OFFSETS = rows * half * 2 >= 2**31,
            num_warps = 4,
            enable_fp_fusion = True,
        )
    return out


def nvfp4_dequantize(
    packed,
    scale,
    global_scale,
    dtype = torch.bfloat16,
):
    """Dense (out, in) weight of a packed (out, in / 2) NVFP4 tensor; a transposed view gives the transposed weight."""
    if _is_transposed_view(packed):
        return nvfp4_dequantize(packed.t(), scale, global_scale, dtype).t()
    if packed.shape[1] * 2 > scale.shape[1] * NVFP4_GROUP_SIZE:
        raise ValueError(
            f"Unsloth: NVFP4 weight {tuple(packed.shape)} needs {packed.shape[1] * 2 // NVFP4_GROUP_SIZE} "
            f"group scales per row, got {tuple(scale.shape)}."
        )
    if packed.device.type in ("cuda", "xpu"):
        return _nvfp4_dequantize_triton(
            packed.contiguous(), scale.contiguous(), global_scale.reshape(-1)[:1], dtype
        )
    return _nvfp4_dequantize_torch(packed, scale, global_scale, dtype)


@triton.jit
def _e2m1_to_float_bits(code):
    # Magnitude bits as fp16 exp (2) + mantissa (1) = E2M1 * 2^-14 (0.5 a subnormal); upcast and rescale are exact.
    bits = ((code & 7) << 9) | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast = True).to(tl.float32) * 16384.0


@triton.jit
def _nvfp4_weight_tile(
    packed_ptr,
    scale_ptr,
    global_scale,
    rows,
    k0,
    n_rows,
    half_cols,
    scale_cols,
    stride_pr,
    stride_pc,
    stride_sr,
    stride_sc,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_E4M3_BYTES: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    INT64_OFFSETS: tl.constexpr,
    MASK_PAD: tl.constexpr,
):
    # Rounded like dense: E2M1 * (block scale / global scale) in fp32, then one cast; out-of-range decodes to 0.
    c = k0 // 2 + tl.arange(0, BLOCK_K // 2)
    g = k0 // 16 + tl.arange(0, BLOCK_K // 16)
    if INT64_OFFSETS:
        c = c.to(tl.int64)
        g = g.to(tl.int64)
    row_ok = rows[:, None] < n_rows
    byte = tl.load(
        packed_ptr + rows[:, None] * stride_pr + c[None, :] * stride_pc,
        mask = row_ok & (c[None, :] < half_cols),
        other = 0,
    ).to(tl.int32)
    scale = tl.load(
        scale_ptr + rows[:, None] * stride_sr + g[None, :] * stride_sc,
        mask = row_ok & (g[None, :] < scale_cols),
        other = 0,
    )
    scale = _e4m3_to_float(scale.to(tl.int32)) if SCALE_E4M3_BYTES else scale.to(tl.float32)
    group_scale = tl.math.div_rn(scale, global_scale)
    values = tl.reshape(
        tl.join(_e2m1_to_float_bits(byte & 15), _e2m1_to_float_bits(byte >> 4)),
        (BLOCK_R, BLOCK_K // 16, 16),
    )
    w = tl.reshape(values * group_scale[:, :, None], (BLOCK_R, BLOCK_K))
    if MASK_PAD:  # a partial tile: padding stays 0 even when scaling gives 0 * inf or 0 / 0
        w = tl.where(row_ok & ((k0 + tl.arange(0, BLOCK_K))[None, :] < half_cols * 2), w, 0.0)
    return w.to(OUT_DTYPE)


@triton.jit
def _nvfp4_matmul_fwd_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    y_ptr,
    M,
    N,
    K,
    half_cols,
    scale_cols,
    stride_xm,
    stride_xk,
    stride_pr,
    stride_pc,
    stride_sr,
    stride_sc,
    stride_ym,
    stride_yn,
    stride_ys,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SCALE_E4M3_BYTES: tl.constexpr,
    INT64_OFFSETS: tl.constexpr,
    MASK_PAD: tl.constexpr,
):
    # Split K writes fp32 partials [split, M, N], summed in a fixed order afterwards (deterministic).
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    if INT64_OFFSETS:
        split_id = split_id.to(tl.int64)
    k_start = tl.program_id(1) * k_per_split
    k_end = min(k_start + k_per_split, K)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if INT64_OFFSETS:
        rm = rm.to(tl.int64)
        rn = rn.to(tl.int64)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        if INT64_OFFSETS:
            rk = rk.to(tl.int64)
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask = (rm[:, None] < M) & (rk[None, :] < k_end),
            other = 0.0,
        )
        w = _nvfp4_weight_tile(
            packed_ptr,
            scale_ptr,
            global_scale,
            rn,
            k0,
            N,
            half_cols,
            scale_cols,
            stride_pr,
            stride_pc,
            stride_sr,
            stride_sc,
            BLOCK_N,
            BLOCK_K,
            SCALE_E4M3_BYTES,
            x_ptr.dtype.element_ty,
            INT64_OFFSETS,
            MASK_PAD,
        )
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(
        y_ptr + split_id * stride_ys + rm[:, None] * stride_ym + rn[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask = (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _nvfp4_matmul_dx_kernel(
    g_ptr,
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    dx_ptr,
    M,
    N,
    K,
    half_cols,
    scale_cols,
    stride_gm,
    stride_gn,
    stride_pr,
    stride_pc,
    stride_sr,
    stride_sc,
    stride_dm,
    stride_dk,
    stride_ds,
    n_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SCALE_E4M3_BYTES: tl.constexpr,
    INT64_OFFSETS: tl.constexpr,
    MASK_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    if INT64_OFFSETS:
        split_id = split_id.to(tl.int64)
    n_start = tl.program_id(1) * n_per_split
    n_end = min(n_start + n_per_split, N)
    num_m = tl.cdiv(M, BLOCK_M)
    num_k = tl.cdiv(K, BLOCK_K)
    group = GROUP_M * num_k
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_k = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    if INT64_OFFSETS:
        rm = rm.to(tl.int64)
        rk = rk.to(tl.int64)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype = tl.float32)
    for n0 in range(n_start, n_end, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        if INT64_OFFSETS:
            rn = rn.to(tl.int64)
        g = tl.load(
            g_ptr + rm[:, None] * stride_gm + rn[None, :] * stride_gn,
            mask = (rm[:, None] < M) & (rn[None, :] < n_end),
            other = 0.0,
        )
        w = _nvfp4_weight_tile(
            packed_ptr,
            scale_ptr,
            global_scale,
            rn,
            pid_k * BLOCK_K,
            n_end,
            half_cols,
            scale_cols,
            stride_pr,
            stride_pc,
            stride_sr,
            stride_sc,
            BLOCK_N,
            BLOCK_K,
            SCALE_E4M3_BYTES,
            g_ptr.dtype.element_ty,
            INT64_OFFSETS,
            MASK_PAD,
        )
        acc = tl.dot(g, w, acc)
    tl.store(
        dx_ptr + split_id * stride_ds + rm[:, None] * stride_dm + rk[None, :] * stride_dk,
        acc.to(dx_ptr.dtype.element_ty),
        mask = (rm[:, None] < M) & (rk[None, :] < K),
    )


_FUSED_CALLS = {"fwd": 0, "bwd": 0}


def _fused_mode():
    # 0 = always dense, 1 = always fused, unset = per-GPU row gate (UNSLOTH_NVFP4_FUSED_MAX_ROWS overrides); read per call.
    return os.environ.get("UNSLOTH_NVFP4_FUSED", "auto").strip().lower()


@functools.lru_cache(maxsize = None)
def _device_capability(index):
    return torch.cuda.get_device_capability(index)


_FUSED_OFF = ("0", "false", "off", "no")
_FUSED_FORCE = ("1", "true", "on", "yes", "force")
_IS_HIP = torch.version.hip is not None


def _fused_ok(A, packed, scale, global_scale, inner):
    # Anything unusual keeps the dense path (and its errors).
    mode = _fused_mode()
    if mode in _FUSED_OFF or _IS_HIP or not A.is_cuda:
        return False
    if A.dtype not in (torch.float16, torch.bfloat16) or A.numel() == 0 or A.shape[-1] != inner:
        return False
    index = A.get_device()
    if mode not in _FUSED_FORCE and A.numel() // inner > _fused_max_rows(index):
        return False
    if A.dtype == torch.bfloat16 and _device_capability(index) < (8, 0):
        return False
    return (
        packed.device == A.device
        and scale.device == A.device
        and global_scale.device == A.device
        and global_scale.numel() >= 1
        and packed.dtype == torch.uint8
        and packed.dim() == 2
        and packed.shape[0] > 0
        and not _is_transposed_view(packed)
        and scale.dtype in (torch.float8_e4m3fn, torch.float32, torch.float16, torch.bfloat16)
        and scale.dim() == 2
        and scale.shape[0] == packed.shape[0]
        and packed.shape[1] * 2 <= scale.shape[1] * NVFP4_GROUP_SIZE
    )


# Max activation rows where fused beats dense, per capability (measured); unmeasured GPUs stay dense.
_FUSED_MAX_ROWS = {(8, 0): 64, (8, 9): 256, (12, 0): 128}


def _fused_max_rows(index):
    override = os.environ.get("UNSLOTH_NVFP4_FUSED_MAX_ROWS")
    if override:
        return int(override)
    return _FUSED_MAX_ROWS.get(_device_capability(index), 0)


def _scale_args(scale):
    e4m3 = scale.dtype == torch.float8_e4m3fn
    return (scale.view(torch.uint8) if e4m3 else scale), e4m3


@functools.lru_cache(maxsize = None)
def _sm_count(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


def _fused_config(
    M,
    n_out,
    k_red,
    device,
    dx = False,
):
    """(BLOCK_M, BLOCK_OUT, BLOCK_RED, num_warps, num_stages, split) for an [M, k_red] @ [k_red, n_out] product."""
    index = device.index or 0
    stages = 2 if _device_capability(index) < (8, 0) else 3
    if M <= 16:
        bm, bo, br, warps = (16, 128, 64, 4) if dx else (16, 64, 128, 4)
    elif M <= 64:
        bm, bo, br, warps = (32, 128, 64, 4) if dx else (32, 64, 128, 8)
    elif M <= 256:
        bm, bo, br, warps = (64, 128, 64, 4) if dx else (128, 64, 128, 8)
    else:
        bm, bo, br, warps = 128, 64, 64, 4
    tiles = triton.cdiv(M, bm) * triton.cdiv(n_out, bo)
    # Split under four waves; fp32 partials capped at 1/8 of the 16 bit weight.
    split = 1
    while (
        tiles * split < 4 * _sm_count(index)
        and k_red >= split * 2 * 4 * br
        and 16 * split * 2 * M <= k_red
        and split < 16
    ):
        split *= 2
    return bm, bo, br, warps, stages, split


@triton.jit
def _split_sum_kernel(partial_ptr, out_ptr, numel, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    # Fixed-order sum of the split partials (determinism), one cast.
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.load(partial_ptr + offs, mask = mask, other = 0.0)
    for i in tl.static_range(1, SPLIT):
        acc += tl.load(partial_ptr + i * numel + offs, mask = mask, other = 0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask = mask)


def _split_out(M, n_out, split, dtype, device):
    out = torch.empty((M, n_out), dtype = dtype, device = device)
    if split == 1:
        return out, out, 0
    partial = torch.empty((split, M, n_out), dtype = torch.float32, device = device)
    return out, partial, partial.stride(0)


def _split_sum(partial, out):
    if partial is out:
        return out
    numel = out.numel()
    _split_sum_kernel[(triton.cdiv(numel, 1024),)](
        partial, out, numel, SPLIT = partial.shape[0], BLOCK = 1024, enable_fp_fusion = True
    )
    return out


def _needs_int64(*tensors):
    # Largest reachable element offset, not numel: a strided view can reach past 2**31 with fewer elements.
    for t in tensors:
        if t.is_contiguous():
            if t.numel() > 2**31:
                return True
        elif sum((n - 1) * abs(st) for n, st in zip(t.shape, t.stride())) >= 2**31:
            return True
    return False


def _scale_groups(packed, scale):
    # Scales past the logical width are ignored, as dense slices them off.
    return min(scale.shape[1], -(-packed.shape[1] // 8))


def _per_split(k_red, split, block):
    return triton.cdiv(triton.cdiv(k_red, split), block) * block


def _nvfp4_matmul_fwd_fused(X, packed, scale, global_scale):
    K = X.shape[-1]
    N = packed.shape[0]
    X2 = X.reshape(-1, K)
    M = X2.shape[0]
    scale, e4m3 = _scale_args(scale)
    BM, BN, BK, warps, stages, split = _CONFIG_HOOK(M, N, K, X.device, False)
    Y, buf, stride_s = _split_out(M, N, split, X.dtype, X.device)
    int64 = _needs_int64(X2, buf, packed, scale)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), split)
    with _fp8_triton_device_context(packed):
        _nvfp4_matmul_fwd_kernel[grid](
            X2, packed, scale, global_scale, buf,
            M, N, K, packed.shape[1], _scale_groups(packed, scale),
            X2.stride(0), X2.stride(1), packed.stride(0), packed.stride(1),
            scale.stride(0), scale.stride(1), buf.stride(-2), buf.stride(-1), stride_s,
            _per_split(K, split, BK),
            BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, GROUP_M = 8,
            SCALE_E4M3_BYTES = e4m3, INT64_OFFSETS = int64, MASK_PAD = N % BN != 0 or K % BK != 0,
            num_warps = warps, num_stages = stages, enable_fp_fusion = True,
        )  # fmt: skip
        Y = _split_sum(buf, Y)
    _FUSED_CALLS["fwd"] += 1
    return Y.view(*X.shape[:-1], N)


def _nvfp4_matmul_dx_fused(grad, packed, scale, global_scale):
    N = grad.shape[-1]
    K = packed.shape[1] * 2
    G = grad.reshape(-1, N)
    M = G.shape[0]
    scale, e4m3 = _scale_args(scale)
    BM, BK, BN, warps, stages, split = _CONFIG_HOOK(M, K, N, grad.device, True)
    dX, buf, stride_s = _split_out(M, K, split, grad.dtype, grad.device)
    int64 = _needs_int64(G, buf, packed, scale)
    grid = (triton.cdiv(M, BM) * triton.cdiv(K, BK), split)
    with _fp8_triton_device_context(packed):
        _nvfp4_matmul_dx_kernel[grid](
            G, packed, scale, global_scale, buf,
            M, N, K, packed.shape[1], _scale_groups(packed, scale),
            G.stride(0), G.stride(1), packed.stride(0), packed.stride(1),
            scale.stride(0), scale.stride(1), buf.stride(-2), buf.stride(-1), stride_s,
            _per_split(N, split, BN),
            BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, GROUP_M = 8,
            SCALE_E4M3_BYTES = e4m3, INT64_OFFSETS = int64, MASK_PAD = N % BN != 0 or K % BK != 0,
            num_warps = warps, num_stages = stages, enable_fp_fusion = True,
        )  # fmt: skip
        dX = _split_sum(buf, dX)
    _FUSED_CALLS["bwd"] += 1
    return dX.view(*grad.shape[:-1], K)


_CONFIG_HOOK = _fused_config


# Opaque to torch.compile, else AOTAutograd saves every layer's dequantized weight for backward.
@_opaque_under_compile(
    "nvfp4_matmul_fwd",
    lambda X, packed, scale, global_scale: X.new_empty(X.shape[:-1] + (packed.shape[0],)),
)
def _nvfp4_matmul_fwd(
    X: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    if _fused_ok(X, packed, scale, global_scale, packed.shape[1] * 2):
        return _nvfp4_matmul_fwd_fused(X, packed, scale, global_scale.reshape(-1)[:1])
    return torch.matmul(X, nvfp4_dequantize(packed, scale, global_scale, X.dtype).t())


@_opaque_under_compile(
    "nvfp4_matmul_bwd",
    lambda grad, packed, scale, global_scale: grad.new_empty(
        grad.shape[:-1] + (packed.shape[1] * 2,)
    ),
)
def _nvfp4_matmul_bwd(
    grad: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    # create_graph: the raw launch has no autograd history, so differentiate through the dense matmul.
    if not (grad.requires_grad and torch.is_grad_enabled()) and _fused_ok(
        grad, packed, scale, global_scale, packed.shape[0]
    ):
        return _nvfp4_matmul_dx_fused(grad, packed, scale, global_scale.reshape(-1)[:1])
    return torch.matmul(grad, nvfp4_dequantize(packed, scale, global_scale, grad.dtype))


class NVFP4Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        packed,
        scale,
        global_scale,
        bias = None,
    ):
        output = _nvfp4_matmul_fwd(X, packed, scale, global_scale)
        if bias is not None:
            output = output + bias
            # A float32 bias promotes; cast back only then (a no-op .to() aliases, zeroing dX under torch 2.11 compile).
            if output.dtype != X.dtype:
                output = output.to(X.dtype)
        ctx.save_for_backward(packed, scale, global_scale)
        ctx.bias_grad = bias is not None and bias.requires_grad
        ctx.dtype = X.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale, global_scale = ctx.saved_tensors
        grad_X = None
        if ctx.needs_input_grad[0]:
            grad_X = _nvfp4_matmul_bwd(grad_output, packed, scale, global_scale)
            if grad_X.dtype != ctx.dtype:
                grad_X = grad_X.to(ctx.dtype)
        grad_bias = None
        if ctx.bias_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_X, None, None, None, grad_bias


def nvfp4_linear(
    X,
    packed,
    scale,
    global_scale,
    bias = None,
):
    return NVFP4Linear_matmul.apply(X, packed, scale, global_scale, bias)
