# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Weight-only GGUF layers for text encoders: blocks stay resident, dequantized per call. Imports torch lazily."""

from __future__ import annotations

from typing import Any, Optional


def gguf_tensor(reader_tensor: Any) -> tuple[Any, Optional[int]]:
    """``(tensor, quant_type or None)``: quantized as a raw ``[rows, bytes]`` uint8 copy, floats reshaped ``[rows, cols]``."""
    import gguf
    import numpy as np
    import torch

    qtype = reader_tensor.tensor_type
    data = reader_tensor.data
    shape = tuple(int(x) for x in reversed(list(reader_tensor.shape)))
    if qtype == gguf.GGMLQuantizationType.BF16:
        raw = torch.from_numpy(np.ascontiguousarray(data).view(np.int16).copy()).view(
            torch.bfloat16
        )
        return raw.reshape(shape), None
    if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return torch.from_numpy(np.ascontiguousarray(data).copy()).reshape(shape), None
    return torch.from_numpy(np.ascontiguousarray(data).copy()), int(qtype)


def gguf_weight_only_linear(
    raw: Any, quant_type: int, *, in_features: int, out_features: int, compute_dtype: Any
) -> Any:
    """A ``GGUFLinear`` over ``raw``; raises unless it holds exactly ``out x in`` weights."""
    import gguf
    import torch
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

    block, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(quant_type)]
    if in_features % block or tuple(raw.shape) != (out_features, in_features // block * type_size):
        raise ValueError(
            f"GGUF {gguf.GGMLQuantizationType(quant_type).name} payload {tuple(raw.shape)} does not hold a "
            f"{out_features}x{in_features} weight"
        )
    layer = GGUFLinear(
        in_features, out_features, bias = False, compute_dtype = compute_dtype, device = "meta"
    )
    layer.weight = GGUFParameter(raw, quant_type = gguf.GGMLQuantizationType(quant_type))
    layer.device = None
    layer.requires_grad_(False)
    return layer


def is_dense_linear(module: Any) -> bool:
    import torch
    return (
        isinstance(module, torch.nn.Linear) and getattr(module.weight, "quant_type", None) is None
    )
