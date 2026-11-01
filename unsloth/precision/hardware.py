# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GPU class of a device and the refusal check for a recipe on it. Everything here runs before the model is
mutated: a refused request leaves the model as it was loaded."""

from __future__ import annotations

from typing import Optional, Tuple

from .recipes import Gate, Recipe, get_recipe

__all__ = [
    "PrecisionRefused",
    "gpu_class",
    "current_gpu_class",
    "dtype_tag",
    "check_recipe",
    "grad_scaler_kind",
]


class PrecisionRefused(ValueError):
    """A precision request that cannot run here; raised before any layer is swapped."""


def gpu_class(cc: Optional[Tuple[int, int]], hip: bool = False) -> Optional[str]:
    """Compute capability -> the gate table's GPU class (None when unknown / not CUDA)."""
    if hip:
        return "rocm"
    if cc is None:
        return None
    major, minor = int(cc[0]), int(cc[1])
    if major >= 12:
        return "sm120"
    if major == 11 or major == 10:
        return "sm100"
    if major == 9:
        return "sm90"
    if (major, minor) >= (8, 9):
        return "sm89"
    if major == 8:
        return "sm80"
    return "sm75"


def current_gpu_class(device = None) -> Optional[str]:
    import torch
    try:
        if not torch.cuda.is_available():
            return None
        if torch.version.hip is not None:
            return "rocm"
        return gpu_class(torch.cuda.get_device_capability(device))
    except Exception:
        return None


def dtype_tag(dtype) -> Optional[str]:
    """torch.dtype / string -> 'bf16' / 'fp16' / 'fp32' (None when unknown or auto)."""
    if dtype is None:
        return None
    s = str(dtype).replace("torch.", "").lower()
    return {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "half": "fp16", "fp16": "fp16",
            "float32": "fp32", "float": "fp32", "fp32": "fp32"}.get(s)


_DTYPE_WHY = {
    "fp16": "float16 activations are rejected by its quantizer (torchao to_mx / NVFP4 training accept "
            "bfloat16 / float32 only); load with dtype = torch.bfloat16",
    "fp32": "float32 activations are not supported by this recipe; load with dtype = torch.bfloat16",
    "bf16": "bfloat16 activations are not supported by this recipe",
}


def check_recipe(
    recipe,
    *,
    gpu: Optional[str] = None,
    input_dtype = None,
    mode: Optional[str] = None,
    allow_planned: bool = False,
) -> Gate:
    """Refuse `recipe` (a Recipe or a registry name) for this GPU class / activation dtype / training mode.

    Order: a format with no training path at all, then the dtype, then the mode, then the GPU gate, then
    "not built yet". Unknown GPU or dtype (None) skips that check. Returns the gate that applies."""
    r: Recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
    if r.status == "refused":
        raise PrecisionRefused(f"Unsloth: {r.name} training is refused: {r.refusal}")
    tag = dtype_tag(input_dtype)
    if tag is not None and tag not in r.input_dtypes:
        raise PrecisionRefused(f"Unsloth: {r.name} cannot train with {tag} activations: {_DTYPE_WHY[tag]}.")
    if mode is not None and mode not in r.modes:
        raise PrecisionRefused(
            f"Unsloth: {r.name} supports training modes {r.modes}, not {mode!r}."
        )
    gate = r.gate(gpu) if gpu is not None else Gate("unprobed")
    if gate.status == "refused":
        raise PrecisionRefused(
            f"Unsloth: {r.name} is refused on {gpu}: {gate.reason}."
            + (f" (evidence: {gate.evidence})" if gate.evidence else "")
        )
    if r.status == "planned" and not allow_planned:
        where = ", ".join(f"{g}: {v.status}" for g, v in sorted(r.gates.items()))
        raise PrecisionRefused(
            f"Unsloth: {r.name} training is not available in Unsloth yet (planned; per GPU: {where}). "
            "Train in bfloat16, or with float8_training = True on a supported GPU."
        )
    return gate


def grad_scaler_kind(recipe, input_dtype, fsdp2: bool = False) -> str:
    """'none', 'GradScaler' or 'ShardedGradScaler': the loss scaler this recipe needs with these activations."""
    r: Recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
    tag = dtype_tag(input_dtype)
    needs = r.grad_scaler == "dynamic" or (r.grad_scaler == "per_dtype" and tag == "fp16")
    if not needs:
        return "none"
    return "ShardedGradScaler" if fsdp2 else "GradScaler"
