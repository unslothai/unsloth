# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 int8 / fp8 denoisers where torchao's runtime does not run (Ampere fp8, ROCm, MPS, stubbed torchao):
the file's codes go to the 1-byte ``NativeWeightOnlyLinear`` twin by the image path's rules instead of widening to
the 40 GB bf16 model. ``UNSLOTH_H3_NATIVE_QUANT=0`` restores the torchao-only refusal."""

from __future__ import annotations

import os
from typing import Any, Optional

H3_NATIVE_QUANT_ENV = "UNSLOTH_H3_NATIVE_QUANT"
H3_FAMILY = "minimax-h3"
# CPU never loads H3 quantized.
_STORAGE_DEVICES = ("cuda", "mps")


def h3_native_quant_enabled() -> bool:
    raw = (os.environ.get(H3_NATIVE_QUANT_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _storage_device(target: Any) -> bool:
    return getattr(target, "device", None) in _STORAGE_DEVICES


def h3_comfy_backends(target: Any, base_repo: Optional[str] = None) -> dict:
    """``load_comfy_quant_transformer`` backends for an H3 file: torchao, else the native twin, else (fp8) storage.
    None / False = no runtime: the load refuses rather than dequantize to 40 GB bf16."""
    from .diffusion_comfy_quant import comfy_fp8_backend, comfy_fp8_storage, comfy_int8_backend

    int8 = comfy_int8_backend(target, H3_FAMILY, base_repo)
    fp8 = comfy_fp8_backend(target, H3_FAMILY, base_repo)
    if not h3_native_quant_enabled():
        return {
            "int8_backend": "torchao" if int8 == "torchao" else None,
            "fp8_backend": "torchao" if fp8 == "torchao" else None,
            "fp8_storage": False,
        }
    return {
        "int8_backend": int8,
        "fp8_backend": fp8,
        "fp8_storage": fp8 is None and _storage_device(target) and comfy_fp8_storage(target),
    }


def h3_fp8_storage(target: Any, base_repo: Optional[str] = None) -> bool:
    """The planner's view of ``h3_comfy_backends``: stored fp8 is priced at 1 byte."""
    try:
        return bool(h3_comfy_backends(target, base_repo)["fp8_storage"])
    except Exception:  # noqa: BLE001 -- unanswerable: priced widened, as before
        return False


def h3_hosted_fp8_native(target: Any, base_repo: Optional[str] = None) -> bool:
    """Hosted fp8 checkpoint as stored fp8: a CUDA card where torchao's fp8 runtime does not run (below sm_89)."""
    if not h3_native_quant_enabled() or getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch

        from .diffusion_comfy_quant import comfy_fp8_storage
        from .diffusion_transformer_quant import TQ_FP8, select_transformer_quant_scheme

        if getattr(target, "dtype", None) is not torch.bfloat16 or not comfy_fp8_storage(target):
            return False
        return (
            select_transformer_quant_scheme(target, TQ_FP8, family = H3_FAMILY, base_repo = base_repo)
            != TQ_FP8
        )
    except Exception:  # noqa: BLE001 -- unanswerable: the torchao rule decides, as before
        return False


def _fp8_parts(weight: Any) -> Optional[tuple]:
    """``(codes, per-row float32 scale)`` of a per-row / per-tensor ``Float8Tensor``, else None."""
    import torch

    if type(weight).__name__ != "Float8Tensor":
        return None
    codes = getattr(weight, "qdata", None)
    scale = getattr(weight, "scale", None)
    if codes is None or scale is None or codes.dim() != 2:
        return None
    if codes.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    rows = int(codes.shape[0])
    if scale.numel() == 1:
        scale = scale.reshape(1).expand(rows)
    elif scale.numel() == rows:
        scale = scale.reshape(rows)
    else:
        return None  # block-wise scales
    return codes, scale.to(torch.float32).contiguous()


def h3_hosted_fp8_to_native(model: Any, logger: Any = None) -> int:
    """Rebuild every ``Float8Tensor`` Linear as a stored-fp8 twin (same codes and per-row scales). Raises
    ``ValueError`` on a layout the twin cannot hold: it could not run here."""
    import torch

    from .diffusion_comfy_quant import FP8_STORED_ATTR
    from .diffusion_native_quant import NATIVE_FP8, native_linear_class

    cls = native_linear_class()
    built = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        if type(weight).__name__ != "Float8Tensor":
            continue
        parts = _fp8_parts(weight)
        if parts is None:
            raise ValueError(f"{name}: a Float8Tensor layout the stored-fp8 runtime cannot hold")
        codes, scale = parts
        dtype = getattr(weight, "dtype", None) or torch.bfloat16
        stand_in = torch.nn.Linear(
            int(module.in_features),
            int(module.out_features),
            bias = False,
            device = "meta",
            dtype = dtype,
        )
        stand_in.bias = module.bias
        layer = cls(stand_in, NATIVE_FP8, codes = codes, scale = scale)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, layer)
        built += 1
    left = [
        n
        for n, m in model.named_modules()
        if type(getattr(m, "weight", None)).__name__ == "Float8Tensor"
    ]
    if left:
        raise ValueError(f"{left[0]}: Float8Tensor weights the stored-fp8 runtime did not take")
    if built:
        try:
            setattr(model, FP8_STORED_ATTR, built)
            model._unsloth_runtime_quant_info = {"scheme": "fp8", "native": True, "storage": True}
        except Exception:  # noqa: BLE001 -- markers are best-effort
            pass
        if logger is not None:
            logger.info(
                "video.h3_native_quant: %d fp8 layers of the hosted checkpoint stored as fp8 and widened per "
                "layer (this card has no fp8 GEMM)",
                built,
            )
    return built
