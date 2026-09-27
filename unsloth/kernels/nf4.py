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

# Fused NF4 dequantization in one Triton launch: nested absmax decode, + offset, NF4 lookup
# and scale. bitsandbytes does the same work in three launches (cdequantize_blockwise_fp32,
# a torch add, cdequantize_blockwise_*_nf4). Triton launches on torch's current stream, so the
# result is ordered on whatever stream the caller is on, and the op is registered with
# torch.library.triton_op so torch.compile traces it without a graph break.
#
# Bit-exact with bitsandbytes csrc/kernels.cu kDequantizeBlockwise (NF4 branch):
#   absmax = fp32(code2[absmax_u8[k]] * absmax2[k // blocksize2]) + offset   (two roundings)
#   out[j] = T(nf4_lut[nibble(j)] * absmax[j // blocksize])                 (high nibble first)

from typing import List, Optional

import torch
import triton
import triton.language as tl

__all__ = [
    "dequantize_nf4",
    "NF4_LUT",
]

# The table in bitsandbytes csrc/kernels.cu (nf4_dequantization_lut). Written as the same
# decimal literals so every entry rounds to the identical fp32 value.
NF4_LUT = (
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

# On NVIDIA the nested product uses PTX mul.rn.f32, which ptxas never contracts into an FMA,
# so the separate + offset rounds exactly as bitsandbytes does. Inline asm (not a libdevice
# alias) because Inductor re-emits user kernels without their module globals. HIP has no PTX,
# so there the eager launch disables fp fusion instead.
_IS_HIP = torch.version.hip is not None
_HAS_MUL_RN = not _IS_HIP


@triton.jit
def _nf4_dequant_kernel(
    W_ptr,  # uint8, packed: high nibble is the even element
    absmax_ptr,  # uint8 (nested) or fp32, one per absmax block
    code2_ptr,  # fp32 [256], nested only
    absmax2_ptr,  # fp32, nested only
    offset_ptr,  # fp32 [1], nested only
    lut_ptr,  # fp32 [16], NF4_LUT
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


# Rough one-off sweep on a shared B200 (see plans/impl_dequant.md): about 2048 packed bytes per
# program with 4 warps (or 1024 with 2) was fastest from 4096x4096 to 128256x4096, both about
# 2.2-2.5x faster than bitsandbytes' three launches. Small weights use smaller programs so the
# grid still fills the GPU.
def _config_for(n_bytes: int, half: int):
    if n_bytes <= (1 << 16):
        target, num_warps = 256, 2
    elif n_bytes <= (1 << 20):
        target, num_warps = 1024, 2
    else:
        target, num_warps = 2048, 4
    return max(1, target // half), num_warps


# Without mul_rn the add could contract into an FMA, which rounds once where bitsandbytes
# rounds twice.
_NO_FP_FUSION = {"enable_fp_fusion": False}

_LUTS = {}


def _lut_for(device: torch.device) -> torch.Tensor:
    """NF4_LUT on ``device``. Only real tensors are cached: a tensor created while
    torch.compile is tracing is fake, and caching it would leak it into later traces."""
    key = (device.type, device.index)
    lut = _LUTS.get(key)
    if lut is not None:
        return lut
    if torch.compiler.is_compiling():
        return torch.tensor(NF4_LUT, dtype = torch.float32, device = device)
    with torch.inference_mode(False):
        lut = torch.tensor(NF4_LUT, dtype = torch.float32, device = device)
    _LUTS[key] = lut
    return lut


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _check(W, absmax, code2, absmax2, offset, blocksize, blocksize2, dtype):
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"Unsloth: dequantize_nf4 does not support dtype {dtype}")
    if not _is_pow2(blocksize) or blocksize < 2:
        raise ValueError(f"Unsloth: dequantize_nf4 needs a power-of-two blocksize, got {blocksize}")
    nested = code2 is not None
    if nested:
        if absmax2 is None or offset is None:
            raise ValueError("Unsloth: nested NF4 state needs code2, absmax2 and offset")
        if not _is_pow2(blocksize2):
            raise ValueError(
                f"Unsloth: dequantize_nf4 needs a power-of-two blocksize2, got {blocksize2}"
            )
    return nested


def _launch(kernel, W, absmax, code2, absmax2, offset, lut, blocksize, blocksize2, out, nested, eager = False):
    n_elements = out.numel()
    n_bytes = (n_elements + 1) // 2
    half = blocksize // 2
    n_blocks = triton.cdiv(n_elements, blocksize)
    rows, num_warps = _config_for(n_bytes, half)
    grid = (triton.cdiv(n_blocks, rows),)
    if nested:
        args = (W, absmax, code2, absmax2, offset)
    else:
        # The nested-only pointers are never dereferenced; absmax fills the slots.
        args = (W, absmax, absmax, absmax, absmax)
    kernel[grid](
        *args,
        lut,
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
        **(_NO_FP_FUSION if (eager and not _HAS_MUL_RN) else {}),
    )
    return out


if _HAS_MUL_RN:
    # Inductor sees the Triton kernel itself and can schedule it with the surrounding graph.
    _register = torch.library.triton_op
    _traced_kernel = lambda: torch.library.wrap_triton(_nf4_dequant_kernel)
    _traced_eager = False
else:
    # HIP: Inductor re-emits a triton_op kernel with fp fusion on, so the nested product could
    # contract into an FMA. An opaque custom op keeps the eager launch with fusion disabled.
    _register = torch.library.custom_op
    _traced_kernel = lambda: _nf4_dequant_kernel
    _traced_eager = True


@_register("unsloth::dequantize_nf4", mutates_args = ())
def _dequantize_nf4_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    lut: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    nested = code2 is not None
    out = torch.empty(shape, dtype = dtype, device = W.device)
    return _launch(
        _traced_kernel(),
        W,
        absmax,
        code2,
        absmax2,
        offset,
        lut,
        blocksize,
        blocksize2,
        out,
        nested,
        eager = _traced_eager,
    )


@_register("unsloth::dequantize_nf4_out", mutates_args = ["out"])
def _dequantize_nf4_out_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    lut: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    out: torch.Tensor,
) -> None:
    nested = code2 is not None
    _launch(
        _traced_kernel(),
        W,
        absmax,
        code2,
        absmax2,
        offset,
        lut,
        blocksize,
        blocksize2,
        out,
        nested,
        eager = _traced_eager,
    )


if not _HAS_MUL_RN:

    @_dequantize_nf4_op.register_fake
    def _(W, absmax, code2, absmax2, offset, lut, blocksize, blocksize2, shape, dtype):
        return W.new_empty(shape, dtype = dtype)

    @_dequantize_nf4_out_op.register_fake
    def _(W, absmax, code2, absmax2, offset, lut, blocksize, blocksize2, out):
        return None


def _as_offset_tensor(offset, device):
    if offset is None or isinstance(offset, torch.Tensor):
        return offset
    return torch.tensor([float(offset)], dtype = torch.float32, device = device)


def dequantize_nf4(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset,
    blocksize: int,
    blocksize2: int,
    shape,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize a bitsandbytes NF4 weight in one kernel launch.

    ``code2``, ``absmax2`` and ``offset`` are None for a non-nested quant state (absmax is
    then fp32). ``offset`` may be a device tensor (read in the kernel, no host sync) or a float.
    ``code`` is the quant state's NF4 codebook (fp32 [16], bit-identical to NF4_LUT); passing
    it keeps a compiled graph from rebuilding the table on every call.
    Returns ``out`` (allocated when None) with ``shape`` and ``dtype``.
    """
    nested = _check(W, absmax, code2, absmax2, offset, blocksize, blocksize2, dtype)
    offset = _as_offset_tensor(offset, W.device) if nested else None
    if (
        isinstance(code, torch.Tensor)
        and code.dtype == torch.float32
        and code.numel() == 16
        and code.device == W.device
    ):
        lut = code
    else:
        lut = _lut_for(W.device)
    if torch.compiler.is_compiling():
        if out is None:
            return torch.ops.unsloth.dequantize_nf4(
                W, absmax, code2, absmax2, offset, lut, blocksize, blocksize2, list(shape), dtype
            )
        torch.ops.unsloth.dequantize_nf4_out(
            W, absmax, code2, absmax2, offset, lut, blocksize, blocksize2, out
        )
        return out
    if out is None:
        out = torch.empty(shape, dtype = dtype, device = W.device)
    else:
        if out.dtype != dtype or out.numel() != _numel(shape):
            raise ValueError("Unsloth: dequantize_nf4 out has the wrong dtype or size")
    # Eager: launch the kernel directly, skipping the op dispatcher.
    with torch.cuda.device(W.device) if W.device.index != torch.cuda.current_device() else _NULL:
        return _launch(
            _nf4_dequant_kernel,
            W,
            absmax,
            code2,
            absmax2,
            offset,
            lut,
            blocksize,
            blocksize2,
            out,
            nested,
            eager = True,
        )


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


_NULL = _Null()
