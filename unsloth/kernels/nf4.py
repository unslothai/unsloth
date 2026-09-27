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

# STAND-IN for the fused Triton NF4 dequant kernel (to be replaced). Same contract:
#   dequantize_nf4(W_u8, absmax, code2, absmax2, offset, blocksize, blocksize2, shape, dtype, out=None)
# Implemented with the bitsandbytes ctypes kernels on the live stream, and registered as
# unsloth::dequantize_nf4 (functional) / unsloth::dequantize_nf4_out (writes `out`) so
# torch.compile traces through it.

from typing import Optional

import ctypes
import torch
import bitsandbytes.functional as bnb_functional

__all__ = ["dequantize_nf4"]

_lib = bnb_functional.lib
_get_ptr = bnb_functional.get_ptr
_c_int = ctypes.c_int
_DEQUANT_FN = {
    torch.float16: _lib.cdequantize_blockwise_fp16_nf4,
    torch.bfloat16: _lib.cdequantize_blockwise_bf16_nf4,
    torch.float32: _lib.cdequantize_blockwise_fp32_nf4,
}


def _dequantize_nf4_impl(W, absmax, code2, absmax2, offset, blocksize, blocksize2, out):
    device = W.device
    stream = ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(device.index))
    with torch.cuda.device(device):
        if code2 is not None:
            n_absmax = absmax.numel()
            out_absmax = torch.empty(n_absmax, dtype = torch.float32, device = device)
            _lib.cdequantize_blockwise_fp32(
                _get_ptr(code2),
                _get_ptr(absmax),
                _get_ptr(absmax2),
                _get_ptr(out_absmax),
                _c_int(blocksize2),
                _c_int(n_absmax),
                stream,
            )
            out_absmax += offset
        else:
            out_absmax = absmax
        _DEQUANT_FN[out.dtype](
            None,
            _get_ptr(W),
            _get_ptr(out_absmax),
            _get_ptr(out),
            _c_int(blocksize),
            _c_int(out.numel()),
            stream,
        )


@torch.library.custom_op("unsloth::dequantize_nf4", mutates_args = ())
def _dequantize_nf4_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    blocksize: int,
    blocksize2: int,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(shape, dtype = dtype, device = W.device)
    _dequantize_nf4_impl(W, absmax, code2, absmax2, offset, blocksize, blocksize2, out)
    return out


@_dequantize_nf4_op.register_fake
def _(W, absmax, code2, absmax2, offset, blocksize, blocksize2, shape, dtype):
    return W.new_empty(shape, dtype = dtype)


@torch.library.custom_op("unsloth::dequantize_nf4_out", mutates_args = ("out",))
def _dequantize_nf4_out_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    blocksize: int,
    blocksize2: int,
    out: torch.Tensor,
) -> None:
    _dequantize_nf4_impl(W, absmax, code2, absmax2, offset, blocksize, blocksize2, out)


@_dequantize_nf4_out_op.register_fake
def _(W, absmax, code2, absmax2, offset, blocksize, blocksize2, out):
    return None


def dequantize_nf4(
    W, absmax, code2, absmax2, offset, blocksize, blocksize2, shape, dtype, out = None
):
    if torch.compiler.is_compiling():
        if out is None:
            return _dequantize_nf4_op(
                W, absmax, code2, absmax2, offset, blocksize, blocksize2, list(shape), dtype
            )
        _dequantize_nf4_out_op(W, absmax, code2, absmax2, offset, blocksize, blocksize2, out)
        return out
    if out is None:
        out = torch.empty(shape, dtype = dtype, device = W.device)
    _dequantize_nf4_impl(W, absmax, code2, absmax2, offset, blocksize, blocksize2, out)
    return out
