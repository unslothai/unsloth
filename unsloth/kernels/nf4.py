# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# NF4 dequantization in one Triton launch on torch's current stream, bit-exact with bitsandbytes
# kDequantizeBlockwise, which needs three launches for a double quantized weight:
#   absmax = fp32(code2[absmax_u8[k]] * absmax2[k // blocksize2]) + offset   (two roundings)
#   out[j] = T(code[nibble(j)] * absmax[j // blocksize])                     (high nibble first)

import contextlib
import functools
import math
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from unsloth_zoo.utils import Version

from .triton_launch import launch, tag_compile_cache

__all__ = [
    "dequantize_nf4",
]

# libdevice mul_rn is never contracted into an FMA (Inductor re-emits kernels with fp fusion on) and
# flushes subnormals like bitsandbytes. HIP / TRITON_INTERPRET=1 lack it: fusion off, opaque op.
_HAS_MUL_RN = torch.version.hip is None and os.environ.get("TRITON_INTERPRET", "0") != "1"


@triton.jit
def _mul(a, b, USE_MUL_RN: tl.constexpr):
    if USE_MUL_RN:
        return libdevice.mul_rn(a, b)
    else:
        return a * b


# Triton's JIT resolves every attribute in a kernel body, even in a dead constexpr branch, so
# tl.gather (absent on Triton 3.2, torch 2.6) lives in a helper defined only where it exists.
if hasattr(tl, "gather"):

    @triton.jit
    def _lut_gather(lut_ptr, n):
        table = tl.load(lut_ptr + tl.arange(0, 16))
        flat = tl.reshape(n, [n.numel])
        return tl.reshape(tl.gather(table, flat, 0), n.shape)

else:

    @triton.jit
    def _lut_gather(lut_ptr, n):
        return tl.load(lut_ptr + n, eviction_policy = "evict_last")


@triton.jit
def _nf4_lut(lut_ptr, n, LUT_MODE: tl.constexpr):
    # n: int32 nibbles. Every mode returns the same fp32 table entry, so the choice is speed only.
    if LUT_MODE == 0:
        # A 64-byte table gather stays in L1.
        return tl.load(lut_ptr + n, eviction_policy = "evict_last")
    elif LUT_MODE == 1:
        # Select tree on the nibble bits over table values held as scalars.
        b0 = (n & 1) != 0
        b1 = (n & 2) != 0
        b2 = (n & 4) != 0
        b3 = (n & 8) != 0
        a0 = tl.where(b0, tl.load(lut_ptr + 1), tl.load(lut_ptr + 0))
        a1 = tl.where(b0, tl.load(lut_ptr + 3), tl.load(lut_ptr + 2))
        a2 = tl.where(b0, tl.load(lut_ptr + 5), tl.load(lut_ptr + 4))
        a3 = tl.where(b0, tl.load(lut_ptr + 7), tl.load(lut_ptr + 6))
        a4 = tl.where(b0, tl.load(lut_ptr + 9), tl.load(lut_ptr + 8))
        a5 = tl.where(b0, tl.load(lut_ptr + 11), tl.load(lut_ptr + 10))
        a6 = tl.where(b0, tl.load(lut_ptr + 13), tl.load(lut_ptr + 12))
        a7 = tl.where(b0, tl.load(lut_ptr + 15), tl.load(lut_ptr + 14))
        c0 = tl.where(b1, a1, a0)
        c1 = tl.where(b1, a3, a2)
        c2 = tl.where(b1, a5, a4)
        c3 = tl.where(b1, a7, a6)
        d0 = tl.where(b2, c1, c0)
        d1 = tl.where(b2, c3, c2)
        return tl.where(b3, d1, d0)
    else:
        # Register-resident table, gathered along its only axis.
        return _lut_gather(lut_ptr, n)


@triton.jit
def _nf4_dequant_kernel(
    W_ptr,  # uint8, packed: high nibble is the even element
    absmax_ptr,  # uint8 (nested) or fp32, one per absmax block
    code2_ptr,  # fp32 [256], nested only
    absmax2_ptr,  # fp32, nested only
    offset_ptr,  # fp32 [1], nested only
    lut_ptr,  # fp32 [16], the NF4 codebook
    out_ptr,
    n_elements,
    n_bytes,
    n_blocks,
    HALF: tl.constexpr,  # packed bytes per absmax block (blocksize // 2)
    BLOCKSIZE2_SHIFT: tl.constexpr,
    NESTED: tl.constexpr,
    USE_MUL_RN: tl.constexpr,
    ROWS: tl.constexpr,  # absmax blocks per program
    WORDS: tl.constexpr,  # load the packed bytes as int32 words (needs n_bytes % 4 == 0)
    EVICT: tl.constexpr,  # "evict_first" streams the weight and the output past L2, else ""
    LUT_MODE: tl.constexpr,  # see _nf4_lut
):
    # A [ROWS, HALF] tile of packed bytes: each row is one absmax block, so its scale is
    # decoded once per row and broadcast, instead of once per byte.
    # FSDP-QLoRA packs the 4bit weight into float storage; it is bytes either way.
    W_ptr = W_ptr.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < n_blocks
    if NESTED:
        a_q = tl.load(absmax_ptr + rows, mask = row_mask, other = 0).to(tl.int32)
        c2 = tl.load(code2_ptr + a_q, mask = row_mask, other = 0.0, eviction_policy = "evict_last")
        s2 = tl.load(
            absmax2_ptr + (rows >> BLOCKSIZE2_SHIFT),
            mask = row_mask,
            other = 0.0,
            eviction_policy = "evict_last",
        )
        scale = _mul(c2, s2, USE_MUL_RN)
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + rows, mask = row_mask, other = 0.0)
    scale = scale[:, None]
    out_ty = out_ptr.dtype.element_ty

    if WORDS:
        # [ROWS, HALF // 4] int32 words; byte j of a word holds elements 2j (high) and 2j + 1.
        NW: tl.constexpr = HALF // 4
        w_offs = rows[:, None] * NW + tl.arange(0, NW)[None, :]
        w = tl.load(
            W_ptr.to(tl.pointer_type(tl.int32)) + w_offs,
            mask = w_offs < (n_bytes // 4),
            other = 0,
            eviction_policy = EVICT,
        )
        i = tl.arange(0, 8)
        shifts = (i // 2) * 8 + (1 - (i % 2)) * 4
        nib = (w[:, :, None] >> shifts[None, None, :]) & 15
        vals = _mul(
            _nf4_lut(lut_ptr, tl.reshape(nib, [ROWS, 2 * HALF]), LUT_MODE), scale, USE_MUL_RN
        )
        vals = vals.to(out_ty)
    else:
        cols = tl.arange(0, HALF)
        byte_offs = rows[:, None] * HALF + cols[None, :]
        q = tl.load(W_ptr + byte_offs, mask = byte_offs < n_bytes, other = 0, eviction_policy = EVICT)
        v_hi = _mul(_nf4_lut(lut_ptr, (q >> 4).to(tl.int32), LUT_MODE), scale, USE_MUL_RN).to(
            out_ty
        )
        v_lo = _mul(_nf4_lut(lut_ptr, (q & 15).to(tl.int32), LUT_MODE), scale, USE_MUL_RN).to(
            out_ty
        )
        vals = tl.interleave(v_hi, v_lo)  # [ROWS, 2 * HALF], high nibble first

    out_offs = rows[:, None] * (2 * HALF) + tl.arange(0, 2 * HALF)[None, :]
    tl.store(out_ptr + out_offs, vals, mask = out_offs < n_elements, eviction_policy = EVICT)


# Per compute capability (prefix match, () default) launch knobs: speed only, all bit-exact.
_CONFIGS = {
    (): (
        # (max n_bytes, target bytes per program, num_warps, words, evict, lut_mode)
        (1 << 16, 256, 2, False, False, 0),
        (1 << 20, 1024, 2, False, False, 0),
        (None, 2048, 4, False, False, 0),
    ),
    # Blackwell datacenter (B200): int32 loads + register table gather.
    (10,): (
        (1 << 16, 256, 2, True, True, 2),
        (None, 1024, 2, True, True, 2),
    ),
}
# tl.gather on a register table fails to compile on Triton 3.3; older Triton uses the L1 load.
_HAS_TL_GATHER = Version(triton.__version__) >= Version("3.6.0")
# Tests and the sweep script set this to a (target, num_warps, words, evict, lut_mode) tuple.
_CONFIG_OVERRIDE = None


@functools.lru_cache(maxsize = None)
def _capability(index: int):
    return torch.cuda.get_device_capability(index)


def _config_for(n_bytes: int, half: int, device: torch.device):
    if _CONFIG_OVERRIDE is not None:
        target, num_warps, words, evict, lut_mode = _CONFIG_OVERRIDE
    else:
        cap = _capability(device.index if device.index is not None else torch.cuda.current_device())
        table = _CONFIGS.get(cap) or _CONFIGS.get(cap[:1]) or _CONFIGS[()]
        for limit, target, num_warps, words, evict, lut_mode in table:
            if limit is None or n_bytes <= limit:
                break
    if lut_mode == 2 and not _HAS_TL_GATHER:
        lut_mode = 0
    return max(1, target // half), num_warps, words, evict, lut_mode


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _launch(kernel, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out, fp_fusion):
    nested = code2 is not None
    n_elements = out.numel()
    n_bytes = (n_elements + 1) // 2
    half = blocksize // 2
    # Integer ceil division: triton.cdiv is a JIT function on recent Triton and costs microseconds.
    n_blocks = -(-n_elements // blocksize)
    rows, num_warps, words, evict, lut_mode = _config_for(n_bytes, half, W.device)
    # int32 loads need every row to start on a word and no partial trailing word.
    words = (
        words
        and half % 4 == 0
        and n_bytes % 4 == 0
        and (W.storage_offset() * W.element_size()) % 4 == 0
    )
    # The nested-only pointers are never dereferenced for a flat state; absmax fills the slots.
    args = (
        W,
        absmax,
        code2 if nested else absmax,
        absmax2 if nested else absmax,
        offset if nested else absmax,
        code,
        out,
        n_elements,
        n_bytes,
        n_blocks,
    )
    constexprs = dict(
        HALF = half,
        BLOCKSIZE2_SHIFT = (blocksize2.bit_length() - 1) if nested else 0,
        NESTED = nested,
        USE_MUL_RN = _HAS_MUL_RN,
        ROWS = rows,
        WORDS = words,
        EVICT = "evict_first" if evict else "",
        LUT_MODE = lut_mode,
    )
    grid = (-(-n_blocks // rows),)
    options = (
        {"num_warps": num_warps}
        if fp_fusion
        else {"num_warps": num_warps, "enable_fp_fusion": False}
    )
    if kernel is _nf4_dequant_kernel:
        launch(kernel, grid, args, 7, constexprs, W.device.index, **options)
    else:
        kernel[grid](*args, **constexprs, **options)
    return out


if _HAS_MUL_RN:
    # Inductor sees the Triton kernel itself and schedules it with the surrounding graph.
    _register = torch.library.triton_op
    _traced_kernel = lambda: torch.library.wrap_triton(_nf4_dequant_kernel)
    _traced_guard = lambda device: contextlib.nullcontext()
else:
    # HIP: an opaque custom op keeps the eager launch (fusion off) and needs its own device guard.
    _register = torch.library.custom_op
    _traced_kernel = lambda: _nf4_dequant_kernel
    _traced_guard = torch.cuda.device


@_register("unsloth::dequantize_nf4", mutates_args = ())
def _dequantize_nf4_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    code: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(shape, dtype = dtype, device = W.device)
    with _traced_guard(W.device):
        return _launch(
            _traced_kernel(),
            W,
            absmax,
            code2,
            absmax2,
            offset,
            code,
            blocksize,
            blocksize2,
            out,
            fp_fusion = _HAS_MUL_RN,
        )


@_register("unsloth::dequantize_nf4_out", mutates_args = ["out"])
def _dequantize_nf4_out_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    code: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    out: torch.Tensor,
) -> None:
    with _traced_guard(W.device):
        _launch(
            _traced_kernel(),
            W,
            absmax,
            code2,
            absmax2,
            offset,
            code,
            blocksize,
            blocksize2,
            out,
            fp_fusion = _HAS_MUL_RN,
        )


if not _HAS_MUL_RN:

    @_dequantize_nf4_op.register_fake
    def _dequantize_nf4_fake(
        W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype
    ):
        return W.new_empty(shape, dtype = dtype)

    @_dequantize_nf4_out_op.register_fake
    def _dequantize_nf4_out_fake(
        W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out
    ):
        return None


tag_compile_cache(__file__)


def dequantize_nf4(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    code: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize a bitsandbytes NF4 weight into ``out`` (allocated when None).

    ``code`` is the quant state's fp32 [16] NF4 codebook. ``code2``, ``absmax2`` and the device
    tensor ``offset`` are None when the absmax is not double quantized (it is then fp32).
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsloth: dequantize_nf4 does not support dtype {dtype}")
    if not _is_pow2(blocksize) or blocksize < 2 or (code2 is not None and not _is_pow2(blocksize2)):
        raise ValueError(f"Unsloth: dequantize_nf4 needs power-of-two blocksizes, got {blocksize}")
    if out is not None and (out.dtype != dtype or out.numel() != math.prod(shape)):
        raise ValueError("Unsloth: dequantize_nf4 out has the wrong dtype or size")
    if torch.compiler.is_compiling():
        if out is None:
            return torch.ops.unsloth.dequantize_nf4(
                W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, list(shape), dtype
            )
        torch.ops.unsloth.dequantize_nf4_out(
            W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out
        )
        return out
    if out is None:
        out = torch.empty(shape, dtype = dtype, device = W.device)
    # Eager launches skip the op dispatcher, which costs about as much as the kernel.
    same_device = W.device.index == torch.cuda.current_device()
    with contextlib.nullcontext() if same_device else torch.cuda.device(W.device):
        return _launch(
            _nf4_dequant_kernel,
            W,
            absmax,
            code2,
            absmax2,
            offset,
            code,
            blocksize,
            blocksize2,
            out,
            fp_fusion = False,
        )
