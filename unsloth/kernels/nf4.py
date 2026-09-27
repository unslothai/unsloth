# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NF4 dequantization in one Triton launch on torch's current stream, bit-exact with bitsandbytes
# kDequantizeBlockwise, which needs three launches for a double quantized weight:
#   absmax = fp32(code2[absmax_u8[k]] * absmax2[k // blocksize2]) + offset   (two roundings)
#   out[j] = T(code[nibble(j)] * absmax[j // blocksize])                     (high nibble first)

import contextlib
import math
from typing import List, Optional

import torch
import triton
import triton.language as tl

__all__ = [
    "dequantize_nf4",
]

# PTX mul.rn.f32 is never contracted into an FMA, so the + offset rounds separately as in
# bitsandbytes. HIP has no PTX: there the launch disables fp fusion instead.
_HAS_MUL_RN = torch.version.hip is None


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
        c2 = tl.load(code2_ptr + a_q, mask = row_mask, other = 0.0)
        s2 = tl.load(absmax2_ptr + (rows >> BLOCKSIZE2_SHIFT), mask = row_mask, other = 0.0)
        if USE_MUL_RN:
            scale = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=r,r,r",
                [c2, s2],
                dtype = tl.float32,
                is_pure = True,
                pack = 1,
            )
        else:
            scale = c2 * s2
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + rows, mask = row_mask, other = 0.0)

    cols = tl.arange(0, HALF)
    byte_offs = rows[:, None] * HALF + cols[None, :]
    byte_mask = byte_offs < n_bytes
    q = tl.load(W_ptr + byte_offs, mask = byte_mask, other = 0)

    scale = scale[:, None]
    # A 64-byte table gather stays in L1; a 16-way select chain made the kernel ALU bound
    # (about 3x slower than bitsandbytes in the rough sweep).
    v_hi = (tl.load(lut_ptr + (q >> 4).to(tl.int32)) * scale).to(out_ptr.dtype.element_ty)
    v_lo = (tl.load(lut_ptr + (q & 15).to(tl.int32)) * scale).to(out_ptr.dtype.element_ty)

    vals = tl.interleave(v_hi, v_lo)  # [ROWS, 2 * HALF], high nibble first
    out_offs = rows[:, None] * (2 * HALF) + tl.arange(0, 2 * HALF)[None, :]
    tl.store(out_ptr + out_offs, vals, mask = out_offs < n_elements)


def _config_for(n_bytes: int, half: int):
    # From a sweep on a B200 over 4096x4096 to 128256x4096; small weights use smaller programs
    # so the grid still fills the GPU.
    if n_bytes <= (1 << 16):
        target, num_warps = 256, 2
    elif n_bytes <= (1 << 20):
        target, num_warps = 1024, 2
    else:
        target, num_warps = 2048, 4
    return max(1, target // half), num_warps


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _launch(kernel, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out, fp_fusion):
    nested = code2 is not None
    n_elements = out.numel()
    n_bytes = (n_elements + 1) // 2
    half = blocksize // 2
    n_blocks = triton.cdiv(n_elements, blocksize)
    rows, num_warps = _config_for(n_bytes, half)
    # The nested-only pointers are never dereferenced for a flat state; absmax fills the slots.
    kernel[(triton.cdiv(n_blocks, rows),)](
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
        HALF = half,
        BLOCKSIZE2_SHIFT = (blocksize2.bit_length() - 1) if nested else 0,
        NESTED = nested,
        USE_MUL_RN = _HAS_MUL_RN,
        ROWS = rows,
        num_warps = num_warps,
        **({} if fp_fusion else {"enable_fp_fusion": False}),
    )
    return out


if _HAS_MUL_RN:
    # Inductor sees the Triton kernel itself and schedules it with the surrounding graph.
    _register = torch.library.triton_op
    _traced_kernel = lambda: torch.library.wrap_triton(_nf4_dequant_kernel)
    _traced_guard = lambda device: contextlib.nullcontext()
else:
    # HIP: Inductor would re-emit a triton_op kernel with fp fusion on. An opaque custom op keeps
    # the eager launch (fusion off), which also needs its own device guard.
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
    def _(W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype):
        return W.new_empty(shape, dtype = dtype)

    @_dequantize_nf4_out_op.register_fake
    def _(W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out):
        return None


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
