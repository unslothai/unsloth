# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in torchao float8 training for full finetuning: eligible nn.Linear layers become torchao Float8Linear (fp8 matmuls in forward and backward, high precision master weights).

Usage: `FastModel.from_pretrained(..., full_finetuning = True, float8_training = True)`. True means the
tensorwise recipe. By name it runs only on compute capability 12.x (RTX PRO 6000 / RTX 50 class), where it
was measured faster than bfloat16 at equal loss. Other fp8 GPUs (H100, B200, ROCm) need an explicit
`torchao.float8.Float8LinearConfig`. Works with native FSDP2 (UNSLOTH_FSDP2=1): convert, then shard."""

from __future__ import annotations

import torch

__all__ = [
    "FLOAT8_TRAINING_RECIPES",
    "resolve_float8_training",
    "float8_training_filter",
    "convert_model_to_float8_training",
]

FLOAT8_TRAINING_RECIPES = ("tensorwise",)

# Named recipes run only where they were timed faster than bfloat16 at equal loss (Qwen3-1.7B / 4B full
# finetuning). tensorwise: RTX PRO 6000 (12.0) 15% faster; L4 (8.9) NaN gradients with gradient
# checkpointing. rowwise: 3% slower on the RTX PRO 6000 and 2% slower on the L4, so it is not offered by
# name. Any other GPU needs an explicit torchao Float8LinearConfig.
_MEASURED_FASTER = {"tensorwise": ((12, 0), (12, 1))}

# Never fp8: output heads (Unsloth's fused CE reads lm_head.weight directly), MoE routers (tiny, precision sensitive).
_SKIP_NAME_PARTS = ("lm_head", "embed_out", "output_projection", "router")
_SKIP_LEAF_NAMES = ("gate", "router", "score", "classifier")


def resolve_float8_training(float8_training):
    """`float8_training` kwarg -> recipe name or a torchao Float8LinearConfig; None when off."""
    if float8_training is None or float8_training is False:
        return None
    if float8_training is True:
        return "tensorwise"
    if isinstance(float8_training, str):
        recipe = float8_training.strip().lower()
        if recipe == "rowwise":
            raise ValueError(
                "Unsloth: float8_training = 'rowwise' trained slower than bfloat16 on every GPU it was "
                "timed on (L4, RTX PRO 6000). Use float8_training = True (tensorwise), or pass "
                "torchao.float8.Float8LinearConfig.from_recipe_name('rowwise') to force it."
            )
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


def float8_training_filter(
    model,
    recipe = "tensorwise",
    size_filter = True,
):
    """Filter for convert_to_float8_training: plain nn.Linear, dims divisible by 16, trainable floating weight, not an output head or router."""
    out_heads = set()
    try:
        out = model.get_output_embeddings()
        if out is not None:
            out_heads.add(id(out))
    except Exception:
        pass
    size_recipe = _recipe_scaling(recipe)

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


def _probe_gemms(recipe):
    """The fp8 GEMMs torchao runs for this recipe / config, as ((mat1 dtype, rowwise?), (mat2 dtype, rowwise?)):
    forward input x weight, grad_input grad_output x weight_for_grad_input, grad_weight
    grad_output_for_grad_weight x input_for_grad_weight. Each operand keeps its own cast config's dtype
    (torchao picks e4m3fnuz / e5m2fnuz on MI300) and scaling granularity; a GEMM with a disabled cast runs
    in high precision and is not probed."""
    config = recipe
    if isinstance(recipe, str):
        from torchao.float8 import Float8LinearConfig
        config = Float8LinearConfig.from_recipe_name(recipe)

    def operand(name, fallback, default_dtype):
        cast = getattr(config, name, None)
        if cast is None:
            cast = getattr(config, fallback, None) if fallback else None
        if cast is not None and "DISABLED" in str(getattr(cast, "scaling_type", "")).upper():
            return None
        dtype = getattr(cast, "target_dtype", None)
        dtype = dtype if isinstance(dtype, torch.dtype) else default_dtype
        rowwise = "AXIS" in str(getattr(cast, "scaling_granularity", "")).upper()
        return dtype, rowwise

    e4m3, e5m2 = torch.float8_e4m3fn, torch.float8_e5m2
    x = operand("cast_config_input", None, e4m3)
    w = operand("cast_config_weight", None, e4m3)
    g = operand("cast_config_grad_output", None, e5m2)
    w_gi = operand("cast_config_weight_for_grad_input", "cast_config_weight", e4m3)
    g_gw = operand("cast_config_grad_output_for_grad_weight", "cast_config_grad_output", e5m2)
    x_gw = operand("cast_config_input_for_grad_weight", "cast_config_input", e4m3)
    gemms = []
    for gemm in ((x, w), (g, w_gi), (g_gw, x_gw)):
        if None not in gemm and gemm not in gemms:
            gemms.append(gemm)
    return gemms


def _probe_dtype_pairs(recipe):
    """(mat1, mat2) fp8 dtypes of the GEMMs `_probe_gemms` lists."""
    pairs = []
    for (dtype_a, _), (dtype_b, _) in _probe_gemms(recipe):
        if (dtype_a, dtype_b) not in pairs:
            pairs.append((dtype_a, dtype_b))
    return pairs


def _probe_float8_gemm(device, recipe):
    """Tiny torch._scaled_mm calls with each GEMM's own fp8 dtypes and scaling, so an unsupported GPU / torch refuses here, not mid-training."""
    M, K, N = 16, 32, 16
    for (dtype_a, rowwise_a), (dtype_b, rowwise_b) in _probe_gemms(recipe):
        a = torch.ones(M, K, device = device).to(dtype_a)
        b = torch.ones(N, K, device = device).to(dtype_b).t()
        # torch._scaled_mm takes one granularity for both operands; torchao (float8_ops.preprocess_addmm)
        # repeats a tensorwise scale along the axis when the other operand is rowwise, so the probe does too.
        rowwise = rowwise_a or rowwise_b
        sa = torch.ones(M, 1, device = device) if rowwise else torch.ones((), device = device)
        sb = torch.ones(1, N, device = device) if rowwise else torch.ones((), device = device)
        torch._scaled_mm(a, b, scale_a = sa, scale_b = sb, out_dtype = torch.bfloat16)


def _recipe_scaling(recipe):
    """'tensorwise' or 'rowwise' for a recipe name or a Float8LinearConfig (by its input cast granularity)."""
    if isinstance(recipe, str):
        return recipe
    try:
        granularity = recipe.cast_config_input.scaling_granularity
        return (
            "rowwise"
            if "AXIS" in str(getattr(granularity, "name", granularity)).upper()
            else "tensorwise"
        )
    except Exception:
        return "rowwise"


def _selected_devices(model, module_filter_fn):
    """Devices of the Linear weights that will be converted (an offloaded CPU embedding is not one of them)."""
    devices = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module_filter_fn(module, name):
            if module.weight.device not in devices:
                devices.append(module.weight.device)
    return devices


def _check_hardware(device, recipe):
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Unsloth: float8_training needs the model on an NVIDIA GPU with compute capability 8.9+; "
            f"the model is on {device}."
        )
    major, minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if torch.version.hip is None and (major, minor) < (8, 9):
        raise RuntimeError(
            "Unsloth: float8_training needs compute capability 8.9+ (fp8 tensor cores); "
            f"{name} is {major}.{minor}. Train in bfloat16 / float16 instead."
        )
    if (
        torch.version.hip is None
        and _recipe_scaling(recipe) == "tensorwise"
        and (major, minor) < (9, 0)
    ):
        # Measured on an L4: every gradient is NaN from the first step once gradient checkpointing runs.
        # The scaling is what fails, so explicit Float8LinearConfigs are refused too.
        raise RuntimeError(
            "Unsloth: float8_training with tensorwise scaling gives NaN gradients on compute capability "
            f"8.9 ({name}). Train in bfloat16 instead."
        )
    if isinstance(recipe, str):
        if torch.version.hip is not None or (major, minor) not in _MEASURED_FASTER.get(recipe, ()):
            where = (
                "ROCm" if torch.version.hip is not None else f"compute capability {major}.{minor}"
            )
            raise RuntimeError(
                f"Unsloth: float8_training = {recipe!r} has not been measured faster than bfloat16 on "
                f"{name} ({where}); it is enabled by name only on RTX PRO 6000 / RTX 50 class GPUs "
                "(compute capability 12.x). To try it here, pass "
                f"torchao.float8.Float8LinearConfig.from_recipe_name({recipe!r}) as float8_training."
            )
    probe_recipe = _recipe_scaling(recipe)
    try:
        _probe_float8_gemm(device, recipe)
    except Exception as e:
        raise RuntimeError(
            f"Unsloth: float8_training ({probe_recipe} scaling) is not supported by torch {torch.__version__} on "
            f"{torch.cuda.get_device_name(device)} ({type(e).__name__}: {str(e)[:200]})."
        ) from e


def _reattach_dispatch_hooks(model, names, dispatch_hooks):
    from accelerate.hooks import add_hook_to_module
    modules = dict(model.named_modules())
    for name in names:
        hook = dispatch_hooks.get(name)
        module = modules.get(name)
        if hook is None or module is None or getattr(module, "_hf_hook", None) is not None:
            continue
        # Skip init_hook: re-placing the reused parameters would re-create them and untie shared ones.
        hook.init_hook = lambda module: module
        try:
            add_hook_to_module(module, hook)
        finally:
            del hook.init_hook


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
    from dataclasses import replace

    config = Float8LinearConfig.from_recipe_name(recipe) if isinstance(recipe, str) else recipe
    if getattr(config, "enable_fsdp_float8_all_gather", False):
        # Its weight subclass gives a shared (tied) weight one parameter per user, so AdamW updates it twice,
        # and safetensors cannot save it. Refused before any layer is swapped.
        raise ValueError(
            "Unsloth: float8_training does not support enable_fsdp_float8_all_gather = True (shared weights "
            "would be updated twice and checkpoints cannot be saved). Use a config with it set to False."
        )
    # Flattened token counts of variable-length batches are rarely multiples of 16, which the fp8 GEMM of the
    # weight gradient needs; torchao pads the inner dim when asked. The caller's config is left untouched.
    if not getattr(config, "pad_inner_dim", True):
        config = replace(config, pad_inner_dim = True)
    if module_filter_fn is None:
        module_filter_fn = float8_training_filter(model, recipe, size_filter = size_filter)
    if check_hardware and not getattr(config, "emulate", False):
        devices = _selected_devices(model, module_filter_fn)
        for device in devices or [next((p.device for p in model.parameters()), None)]:
            _check_hardware(device, recipe)
    # torchao's new Float8Linear drops Accelerate dispatch hooks (device_map / offload); re-attach them.
    dispatch_hooks = {
        name: m._hf_hook
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and getattr(m, "_hf_hook", None) is not None
    }
    convert_to_float8_training(model, module_filter_fn = module_filter_fn, config = config)
    converted = [name for name, m in model.named_modules() if isinstance(m, Float8Linear)]
    if dispatch_hooks:
        _reattach_dispatch_hooks(model, converted, dispatch_hooks)
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
