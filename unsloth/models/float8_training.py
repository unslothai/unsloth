# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in torchao float8 training for full finetuning: eligible nn.Linear layers become torchao Float8Linear (fp8 matmuls in forward and backward, high precision master weights)."""

from __future__ import annotations

import torch

__all__ = [
    "FLOAT8_TRAINING_RECIPES",
    "resolve_float8_training",
    "float8_training_filter",
    "convert_model_to_float8_training",
]

FLOAT8_TRAINING_RECIPES = ("rowwise", "tensorwise")

# Never fp8: output heads (Unsloth's fused CE reads lm_head.weight directly), MoE routers (tiny, precision sensitive).
_SKIP_NAME_PARTS = ("lm_head", "embed_out", "output_projection", "router")
_SKIP_LEAF_NAMES = ("gate", "router", "score", "classifier")


def resolve_float8_training(float8_training):
    """`float8_training` kwarg -> recipe name or a torchao Float8LinearConfig; None when off."""
    if float8_training is None or float8_training is False:
        return None
    if float8_training is True:
        return "rowwise"
    if isinstance(float8_training, str):
        recipe = float8_training.strip().lower()
        if recipe not in FLOAT8_TRAINING_RECIPES:
            raise ValueError(
                f"Unsloth: float8_training = '{float8_training}' is not supported. "
                f"Use True or one of {FLOAT8_TRAINING_RECIPES}."
            )
        return recipe
    if type(float8_training).__name__ == "Float8LinearConfig":
        return float8_training
    raise TypeError(
        "Unsloth: float8_training must be True / False, a recipe name "
        f"{FLOAT8_TRAINING_RECIPES} or a torchao Float8LinearConfig, got {type(float8_training)}."
    )


def _size_ok(recipe, out_features, in_features):
    # torchao's H100 thresholds (torchao/float8 README); below them the casts cost more than the fp8 GEMM saves.
    N, K = out_features, in_features
    if recipe == "tensorwise":
        return not (K <= 4096 and N <= 1024)
    if N <= 2048 or K <= 1024:
        return False
    return not (N <= 4096 and K <= 2048)


def float8_training_filter(model, recipe = "rowwise", size_filter = True):
    """Filter for convert_to_float8_training: plain nn.Linear, dims divisible by 16, trainable floating weight, not an output head or router."""
    out_heads = set()
    try:
        out = model.get_output_embeddings()
        if out is not None:
            out_heads.add(id(out))
    except Exception:
        pass
    size_recipe = recipe if isinstance(recipe, str) else "rowwise"

    def module_filter_fn(module, fqn):
        # Exact type: bitsandbytes / QAT / fp8 / PEFT linears subclass nn.Linear and must stay as they are.
        if type(module) is not torch.nn.Linear:
            return False
        if id(module) in out_heads or any(p in fqn for p in _SKIP_NAME_PARTS):
            return False
        if fqn.rsplit(".", 1)[-1] in _SKIP_LEAF_NAMES:
            return False
        weight = module.weight
        if not weight.is_floating_point() or weight.element_size() < 2 or not weight.requires_grad:
            return False
        if weight.dim() != 2 or weight.is_meta:
            return False
        out_features, in_features = weight.shape
        if out_features % 16 != 0 or in_features % 16 != 0:
            return False
        return not size_filter or _size_ok(size_recipe, out_features, in_features)

    return module_filter_fn


def _probe_float8_gemm(device, recipe):
    """One tiny torch._scaled_mm of the recipe's scaling, so an unsupported GPU / torch refuses here, not mid-training."""
    a = torch.ones(16, 32, device = device).to(torch.float8_e4m3fn)
    b = torch.ones(16, 32, device = device).to(torch.float8_e4m3fn).t()
    if recipe == "tensorwise":
        sa = torch.ones((), device = device)
        sb = torch.ones((), device = device)
    else:
        sa = torch.ones(16, 1, device = device)
        sb = torch.ones(1, 16, device = device)
    torch._scaled_mm(a, b, scale_a = sa, scale_b = sb, out_dtype = torch.bfloat16)


def _model_device(model):
    for p in model.parameters():
        return p.device
    return None


def _check_hardware(device, recipe):
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Unsloth: float8_training needs the model on an NVIDIA GPU with compute capability 8.9+ "
            f"(L4, Ada, H100, B200, RTX 40 / 50 series); the model is on {device}."
        )
    major, minor = torch.cuda.get_device_capability(device)
    if torch.version.hip is None and (major, minor) < (8, 9):
        raise RuntimeError(
            "Unsloth: float8_training needs compute capability 8.9+ (L4, Ada, H100, B200, RTX 40 / 50 "
            f"series); {torch.cuda.get_device_name(device)} is {major}.{minor}. Train in bfloat16 / float16 instead."
        )
    if recipe == "tensorwise" and torch.version.hip is None and (major, minor) < (9, 0):
        # Measured on an L4: every gradient is NaN from the first step once gradient checkpointing runs.
        raise RuntimeError(
            "Unsloth: float8_training = 'tensorwise' gives NaN gradients on compute capability 8.9 "
            f"({torch.cuda.get_device_name(device)}); use float8_training = 'rowwise' there."
        )
    probe_recipe = recipe if isinstance(recipe, str) else "rowwise"
    try:
        _probe_float8_gemm(device, probe_recipe)
    except Exception as e:
        hint = " Try float8_training = 'tensorwise'." if probe_recipe == "rowwise" else ""
        raise RuntimeError(
            f"Unsloth: float8_training = '{probe_recipe}' is not supported by torch {torch.__version__} on "
            f"{torch.cuda.get_device_name(device)} ({type(e).__name__}: {str(e)[:200]}).{hint}"
        ) from e


def convert_model_to_float8_training(
    model,
    float8_training = True,
    *,
    size_filter = True,
    module_filter_fn = None,
    check_hardware = True,
):
    """Swap eligible nn.Linear layers of a full finetuning model for torchao Float8Linear in place. Returns the converted module names (also on `model._unsloth_float8_linears`)."""
    recipe = resolve_float8_training(float8_training)
    if recipe is None:
        return []
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        from torchao.float8.float8_linear import Float8Linear
    except Exception as e:
        raise ImportError(
            "Unsloth: float8_training needs torchao with torchao.float8: `pip install torchao`. "
            f"({type(e).__name__}: {e})"
        ) from e
    config = Float8LinearConfig.from_recipe_name(recipe) if isinstance(recipe, str) else recipe
    if check_hardware and not getattr(config, "emulate", False):
        _check_hardware(_model_device(model), recipe)
    if module_filter_fn is None:
        module_filter_fn = float8_training_filter(model, recipe, size_filter = size_filter)
    convert_to_float8_training(model, module_filter_fn = module_filter_fn, config = config)
    converted = [name for name, m in model.named_modules() if isinstance(m, Float8Linear)]
    if not converted:
        import warnings

        warnings.warn(
            "Unsloth: float8_training found no eligible Linear layer (dims divisible by 16 and large enough "
            "for fp8 to pay off); the model trains in its original precision.",
            stacklevel = 2,
        )
    try:
        model._unsloth_float8_linears = converted
    except Exception:
        pass
    return converted
