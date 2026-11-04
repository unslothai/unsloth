# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from typing import Optional

import torch

from .nf4 import dequantize_nf4 as _raw


@torch.library.custom_op("unsloth::bnb_nf4_dequantize_out", mutates_args = ("out",))
def _eager(
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
    # The raw eager launcher selects W's device; Fake/Proxy dispatch never launches it.
    _raw(
        W,
        absmax,
        code2,
        absmax2,
        offset,
        code,
        blocksize,
        blocksize2,
        out.shape,
        out.dtype,
        out = out,
    )


@_eager.register_fake
def _eager_fake(W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out):
    return None


def dequantize_nf4(
    W,
    absmax,
    code2,
    absmax2,
    offset,
    code,
    blocksize,
    blocksize2,
    shape,
    dtype,
    out = None,
):
    if torch.compiler.is_compiling():
        return _raw(
            W,
            absmax,
            code2,
            absmax2,
            offset,
            code,
            blocksize,
            blocksize2,
            shape,
            dtype,
            out = out,
        )
    if out is None:
        out = torch.empty(shape, dtype = dtype, device = W.device)
    _eager(W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out)
    return out
