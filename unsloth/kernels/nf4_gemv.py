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

# STAND-IN for the batch-1 NF4 gemv kernel (to be replaced). Same contract:
#   gemv_nf4(X, W_u8, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype, out=None)
# X is (1, 1, in_features); returns (1, 1, out_features). Implemented with the bitsandbytes
# ctypes kernels on the live stream, registered as unsloth::gemv_nf4 / unsloth::gemv_nf4_out.

from typing import Optional

import ctypes
import torch
import bitsandbytes.functional as bnb_functional

__all__ = ["gemv_nf4"]

_lib = bnb_functional.lib
_get_ptr = bnb_functional.get_ptr
_c_int = ctypes.c_int
_c_int32 = ctypes.c_int32
_GEMV_FN = {
    torch.float16: _lib.cgemm_4bit_inference_naive_fp16,
    torch.bfloat16: _lib.cgemm_4bit_inference_naive_bf16,
    torch.float32: _lib.cgemm_4bit_inference_naive_fp32,
}


def _gemv_nf4_impl(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, out):
    device = W.device
    stream = ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(device.index))
    hd = X.shape[-1]
    with torch.cuda.device(device):
        if code2 is not None:
            df = torch.empty(absmax.shape, dtype = torch.float32, device = device)
            _lib.cdequantize_blockwise_fp32(
                _get_ptr(code2),
                _get_ptr(absmax),
                _get_ptr(absmax2),
                _get_ptr(df),
                _c_int(blocksize2),
                _c_int(df.numel()),
                stream,
            )
            df += offset
            absmax = df
        _GEMV_FN[out.dtype](
            _c_int32(shape[0]),
            _c_int32(1),
            _c_int32(shape[1]),
            _get_ptr(X),
            _get_ptr(W),
            _get_ptr(absmax),
            _get_ptr(code),
            _get_ptr(out),
            _c_int32(shape[0]),
            _c_int32((hd + 1) // 2),
            _c_int32(shape[0]),
            _c_int32(blocksize),
            stream,
        )


@torch.library.custom_op("unsloth::gemv_nf4", mutates_args = ())
def _gemv_nf4_op(
    X: torch.Tensor,
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    code: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty((1, 1, shape[0]), dtype = dtype, device = W.device)
    _gemv_nf4_impl(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, out)
    return out


@_gemv_nf4_op.register_fake
def _(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype):
    return W.new_empty((1, 1, shape[0]), dtype = dtype)


@torch.library.custom_op("unsloth::gemv_nf4_out", mutates_args = ("out",))
def _gemv_nf4_out_op(
    X: torch.Tensor,
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    code: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape: list[int],
    out: torch.Tensor,
) -> None:
    _gemv_nf4_impl(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, out)


@_gemv_nf4_out_op.register_fake
def _(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, out):
    return None


def gemv_nf4(
    X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype, out = None
):
    if torch.compiler.is_compiling():
        if out is None:
            return _gemv_nf4_op(
                X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, list(shape), dtype
            )
        _gemv_nf4_out_op(
            X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, list(shape), out
        )
        return out
    if out is None:
        out = torch.empty((1, 1, shape[0]), dtype = dtype, device = W.device)
    _gemv_nf4_impl(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, out)
    return out
