# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Blockwise 4-bit (NF4 / FP4) and 8-bit dynamic quantization, byte-identical to the bitsandbytes
CUDA kernels, plus pure torch quantize / dequantize used where Triton is unavailable.

bitsandbytes builds its CUDA library with --use_fast_math, so kQuantizeBlockwise computes
    absmax = max(|x|) (subnormals flushed),  r = rcp.approx.ftz(absmax),  q = code(x *ftz r)
and NF4 / FP4 code the scaled value with a fixed comparison tree (`>` on the thresholds below, so
NaN, from an all zero block, codes 0). The Triton kernels issue the same PTX (rcp.approx.ftz,
mul.ftz) and read the thresholds as the exact float32 bit patterns of the compiled library, so the
packed bytes and absmax match bitsandbytes byte for byte on NVIDIA GPUs. The torch path uses an
IEEE reciprocal, which can differ from the approximate one in the last bit, so a value sitting on
a threshold can code to the neighbouring level there; dequantization is exact on both paths.
"""

import math
import os
import struct

import torch

__all__ = [
    "NF4_CODE",
    "FP4_CODE",
    "create_dynamic_map",
    "quantize_4bit_blockwise",
    "quantize_8bit_blockwise",
    "dequantize_4bit_blockwise",
    "dequantize_8bit_blockwise",
    "quantize_4bit_torch",
    "quantize_8bit_torch",
    "triton_quantize_available",
]


def _f32(bits):
    return struct.unpack(">f", struct.pack(">I", bits))[0]


# bitsandbytes nf4_dequantization_lut and the 4bit code tensor get_4bit_type("nf4") returns.
NF4_CODE = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
# get_4bit_type("fp4"): e2m1 with bias 3, normalised by 12 (the format's absmax).
FP4_CODE = tuple(
    x / 12.0
    for x in (
        0,
        0.0625,
        8.0,
        12.0,
        4.0,
        6.0,
        2.0,
        3.0,
        -0,
        -0.0625,
        -8.0,
        -12.0,
        -4.0,
        -6.0,
        -2.0,
        -3.0,
    )
)

# dQuantizeNF4 thresholds as compiled (the CUDA literals round differently from Python's parser in
# four places), ascending. A sorted comparison tree codes x as the number of thresholds below it.
_NF4_THRESHOLDS = tuple(
    _f32(b)
    for b in (
        0xBF591CD9,
        0xBF1C5270,
        0xBEEB8480,
        0xBEADEA76,
        0xBE703CEC,
        0xBE0D38BC,
        0xBD3A7871,
        0x3D22FAFF,
        0x3DF64863,
        0x3E5067E0,
        0x3E9582D4,
        0x3EC753F9,
        0x3F006D03,
        0x3F248DAF,
        0x3F5C89D9,
    )
)
# dQuantizeFP4 magnitude thresholds, ascending, and the code each magnitude bucket gets.
_FP4_THRESHOLDS = tuple(
    _f32(b)
    for b in (0x3B2AAAB9, 0x3DB00000, 0x3E555555, 0x3E955555, 0x3ED55556, 0x3F155550, 0x3F555555)
)
_FP4_BUCKET_CODE = (0, 1, 6, 7, 4, 5, 2, 3)
_FLT_MIN = 1.1754943508222875e-38
_FLT_MAX = 3.4028234663852886e38


def create_dynamic_map(
    signed = True,
    max_exponent_bits = 7,
    total_bits = 8,
):
    """The 8-bit dynamic tree code bitsandbytes quantizes double-quant absmax with (same float32s)."""
    data = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype = torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype = torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    data.append(0)
    data.append(1.0)
    assert len(data) == 2**total_bits
    data += [0] * (256 - len(data))
    data.sort()
    return torch.tensor(data, dtype = torch.float32)


# ---------------------------------------------------------------------------------------------
# Pure torch (any device). Same algorithm, IEEE reciprocal.
# ---------------------------------------------------------------------------------------------


def _ftz(x):
    return torch.where(x.abs() < _FLT_MIN, torch.zeros((), dtype = x.dtype, device = x.device), x)


def _blocks(A, blocksize):
    flat = A.reshape(-1).float()
    n = flat.numel()
    blocks = -(n // -blocksize)
    pad = blocks * blocksize - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(blocks, blocksize), n


def _absmax(x):
    # fmaxf from -FLT_MAX: NaN inputs are ignored (an all NaN block gives -FLT_MAX).
    a = x.abs()
    return torch.where(torch.isnan(a), torch.full_like(a, -_FLT_MAX), a).amax(dim = 1)


def _scaled(A, blocksize):
    x, n = _blocks(A, blocksize)
    x = _ftz(x)
    absmax = _absmax(x)
    scaled = _ftz(x * torch.reciprocal(absmax)[:, None])
    return scaled, absmax, n


def _code_4bit(scaled, quant_type):
    # A sorted comparison tree codes x as the count of thresholds below it (bucketize, left);
    # NaN compares false everywhere, so it codes as 0 there.
    nan = torch.isnan(scaled)
    if quant_type == "nf4":
        t = torch.tensor(_NF4_THRESHOLDS, dtype = torch.float32, device = scaled.device)
        q = torch.bucketize(scaled, t)
        return torch.where(nan, 0, q).to(torch.uint8)
    t = torch.tensor(_FP4_THRESHOLDS, dtype = torch.float32, device = scaled.device)
    bucket = torch.where(nan, 0, torch.bucketize(scaled.abs(), t))
    table = torch.tensor(_FP4_BUCKET_CODE, dtype = torch.uint8, device = scaled.device)
    sign = (scaled < 0).to(torch.uint8) << 3
    return table[bucket] | sign


# Blocks per chunk on the torch path, bounding its float32 temporaries (about 16M elements).
_TORCH_CHUNK = 1 << 24


def quantize_4bit_torch(
    A,
    blocksize = 64,
    quant_type = "nf4",
):
    """Returns (packed uint8 [(n + 1) // 2], fp32 absmax [blocks]); high nibble first."""
    flat = A.reshape(-1)
    n = flat.numel()
    step = max(blocksize, (_TORCH_CHUNK // blocksize) * blocksize)
    packed, absmax = [], []
    for start in range(0, max(n, 1), step):
        chunk = flat[start : start + step]
        scaled, chunk_absmax, m = _scaled(chunk, blocksize)
        # bitsandbytes codes the zero it pads the last block with (NaN in an all zero block),
        # so an odd count keeps one padded code for the low nibble of the last byte.
        q = _code_4bit(scaled.reshape(-1)[: m + (m % 2)], quant_type).view(-1, 2)
        packed.append((q[:, 0] << 4) | q[:, 1])
        absmax.append(chunk_absmax)
        if n == 0:
            break
    return torch.cat(packed), torch.cat(absmax)


def _dynamic_search(code, x):
    # dQuantize<0>: the bitsandbytes binary search over the 256-entry code, then nearest by midpoint.
    pivot = torch.full_like(x, 127, dtype = torch.long)
    upper_pivot = torch.full_like(pivot, 255)
    lower_pivot = torch.zeros_like(pivot)
    lower = torch.full_like(x, -1.0)
    upper = torch.full_like(x, 1.0)
    val = code[pivot]
    i = 64
    while i > 0:
        gt = x > val
        lower_pivot = torch.where(gt, pivot, lower_pivot)
        lower = torch.where(gt, val, lower)
        upper_pivot = torch.where(gt, upper_pivot, pivot)
        upper = torch.where(gt, upper, val)
        pivot = torch.where(gt, pivot + i, pivot - i)
        val = code[pivot]
        i >>= 1
    upper = torch.where(upper_pivot == 255, code[255], upper)
    lower = torch.where(lower_pivot == 0, code[0], lower)
    gt = x > val
    up = _ftz((upper + val) * 0.5)
    down = _ftz((lower + val) * 0.5)
    result = torch.where(
        gt,
        torch.where(x > up, upper_pivot, pivot),
        torch.where(x < down, lower_pivot, pivot),
    )
    return result.to(torch.uint8)


def quantize_8bit_torch(
    A,
    code,
    blocksize = 256,
):
    """Returns (uint8 codes shaped like A, fp32 absmax [blocks])."""
    scaled, absmax, n = _scaled(A, blocksize)
    q = _dynamic_search(code.to(device = scaled.device, dtype = torch.float32), scaled)
    return q.reshape(-1)[:n].view(A.shape), absmax


def _dequant_absmax(absmax, blocksize):
    return absmax.float().repeat_interleave(blocksize)


def dequantize_8bit_blockwise(
    A,
    absmax,
    code,
    blocksize,
    dtype = torch.float32,
):
    n = A.numel()
    values = code.to(device = A.device, dtype = torch.float32)[A.reshape(-1).long()]
    scale = _dequant_absmax(absmax, blocksize)[:n]
    return (values * scale).to(dtype).view(A.shape)


def dequantize_4bit_blockwise(
    A,
    absmax,
    blocksize,
    quant_type,
    shape,
    dtype,
    code = None,
):
    """absmax already dequantized to fp32. `code` overrides the built-in table."""
    n = math.prod(shape)
    if code is None:
        code = NF4_CODE if quant_type == "nf4" else FP4_CODE
        code = torch.tensor(code, dtype = torch.float32, device = A.device)
    else:
        code = code.to(device = A.device, dtype = torch.float32)
    packed = A.reshape(-1).view(torch.uint8)
    q = torch.stack([packed >> 4, packed & 0x0F], dim = -1).reshape(-1)[:n]
    scale = _dequant_absmax(absmax, blocksize)[:n]
    return (code[q.long()] * scale).to(dtype).view(shape)


# ---------------------------------------------------------------------------------------------
# Triton (NVIDIA: exact PTX; ROCm: same algorithm, IEEE reciprocal).
# ---------------------------------------------------------------------------------------------

# UNSLOTH_BNB_SHIM_TRITON=0 forces the torch path; read once, not inside traced code.
_SHIM_TRITON = os.environ.get("UNSLOTH_BNB_SHIM_TRITON", "1") != "0"
_KERNELS = None


_TRITON = None


def triton_quantize_available(device = None):
    """Unsloth's Triton quantize kernels (unsloth/kernels/nf4_quantize.py) are usable here. Decided
    once, eagerly: under tracing an undecided process takes the torch path rather than probing."""
    global _TRITON, _KERNELS
    if not _SHIM_TRITON:
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    if _TRITON is None:
        if torch.compiler.is_compiling():
            return False
        try:
            # Real Triton (an importable stub without a backend fails here) and Volta or newer
            # on every visible card (the PTX needs sm70); ROCm runs the IEEE variant.
            from unsloth.kernels import nf4_quantize
            _KERNELS = nf4_quantize
            _TRITON = torch.cuda.is_available() and (
                torch.version.hip is not None
                or all(
                    torch.cuda.get_device_capability(i)[0] >= 7
                    for i in range(torch.cuda.device_count())
                )
            )
        except Exception:
            _TRITON = False
    return _TRITON


def triton_failed(error):
    """A Triton compile / launch failure switches the stand-in to torch for good (not for OOM)."""
    global _TRITON
    if isinstance(error, torch.cuda.OutOfMemoryError) or torch.compiler.is_compiling():
        return False
    _TRITON = False
    print(
        f"Unsloth: bitsandbytes stand-in Triton kernels failed ({type(error).__name__}: {error}), "
        "using torch."
    )
    return True


_THRESHOLD_CACHE = {}


def _thresholds(quant_type, device):
    key = (quant_type, device)
    t = _THRESHOLD_CACHE.get(key)
    if t is None:
        values = _NF4_THRESHOLDS if quant_type == "nf4" else _FP4_THRESHOLDS
        t = _THRESHOLD_CACHE[key] = torch.tensor(values, dtype = torch.float32, device = device)
    return t


def quantize_4bit_blockwise(
    A,
    blocksize = 64,
    quant_type = "nf4",
):
    """bitsandbytes quantize_4bit core: (packed uint8 [(n + 1) // 2], fp32 absmax [blocks])."""
    if quant_type not in ("nf4", "fp4"):
        raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type!r}")
    A = A.contiguous()
    if A.is_cuda and triton_quantize_available(A.device):
        try:
            return _quantize_4bit_triton(A, blocksize, quant_type)
        except Exception as error:
            if not triton_failed(error):
                raise
    return quantize_4bit_torch(A, blocksize, quant_type)


def _quantize_4bit_triton(A, blocksize, quant_type):
    return _KERNELS.quantize_4bit_triton(
        A, _thresholds(quant_type, A.device), blocksize, quant_type == "nf4"
    )


def quantize_8bit_blockwise(
    A,
    code,
    blocksize = 256,
):
    """bitsandbytes quantize_blockwise core: (uint8 codes like A, fp32 absmax [blocks])."""
    A = A.contiguous()
    code = code.to(device = A.device, dtype = torch.float32).contiguous()
    pow2 = blocksize > 1 and blocksize & (blocksize - 1) == 0
    if A.is_cuda and pow2 and code.numel() == 256 and triton_quantize_available(A.device):
        try:
            return _quantize_8bit_triton(A, code, blocksize)
        except Exception as error:
            if not triton_failed(error):
                raise
    return quantize_8bit_torch(A, code, blocksize)


def _quantize_8bit_triton(A, code, blocksize):
    return _KERNELS.quantize_8bit_triton(A, code, blocksize)
