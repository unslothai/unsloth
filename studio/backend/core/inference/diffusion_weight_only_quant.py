# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Weight-only GGUF layers for text encoders, any device.

GGUF block formats (Q4_K / Q6_K / ...) load as diffusers' ``GGUFLinear`` over the file's blocks, the layer Studio's
GGUF denoisers already use (its dequant is Studio's compiled / fused one when installed): the blocks stay resident and
each call rebuilds a transient dense weight in the compute dtype for a float GEMM, so any GPU or the CPU runs it. Meant
for modules that run once per request (text encoders), where bytes resident and bytes moved matter and GEMM throughput
does not.

Every helper imports torch lazily: importing this module never imports torch."""

from __future__ import annotations

from typing import Any, Optional


# --- GGUF (llama.cpp block formats) through diffusers' GGUFLinear --------------------------------------------------


def gguf_tensor(reader_tensor: Any) -> tuple[Any, Optional[int]]:
    """``(torch tensor, quant_type or None)`` for one ``gguf.GGUFReader`` tensor. Block-quantized tensors come back as
    their raw ``[rows, bytes]`` uint8 payload (a copy: anonymous memory, the residency the conditioner wants); float
    tensors in their own dtype, shaped ``[rows, cols]`` (GGUF lists dimensions innermost first)."""
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
    """diffusers' ``GGUFLinear`` holding ``raw`` (``[out, bytes]`` uint8) as a ``GGUFParameter``: dequantized per call
    in the compute dtype, so the resident weight stays the file's blocks (Studio's compiled / fused dequant applies when
    installed). Raises when the payload does not hold exactly ``out x in`` weights of that type."""
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
    """A plain float ``nn.Linear`` (diffusers' ``GGUFLinear`` subclasses it but holds blocks, not weights)."""
    import torch
    return (
        isinstance(module, torch.nn.Linear) and getattr(module.weight, "quant_type", None) is None
    )
