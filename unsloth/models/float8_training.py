# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in torchao float8 training; True = tensorwise, auto only on CC 12.x (else explicit
torchao.float8.Float8LinearConfig); with native FSDP2 convert before sharding."""

from __future__ import annotations

import torch


def _load_precision():
    """unsloth.precision; also when this file is loaded on its own (gloo test workers import it by path
    without the unsloth package), from the sibling directory."""
    try:
        from .. import precision
        return precision
    except ImportError:
        import importlib.util
        import os
        import sys

        name = "_unsloth_precision_standalone"
        if name in sys.modules:
            return sys.modules[name]
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision"
        )
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(root, "__init__.py"), submodule_search_locations = [root]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


_precision = _load_precision()
_PRECISION_NAMES = _precision.FLOAT8_TRAINING_NAMES
_check_precision_recipe = _precision.check_recipe
_find_routers = _precision.find_routers
_precision_measured_faster = _precision.measured_faster
_precision_named_recipes = _precision.named_recipes
_record_engagement = _precision.record
_require_engagement = _precision.require

__all__ = [
    "FLOAT8_TRAINING_RECIPES",
    "resolve_float8_training",
    "float8_training_filter",
    "convert_model_to_float8_training",
    "resolve_fp8_lora",
    "mark_fp8_lora",
    "enable_fp8_lora",
]

# Names and gates come from the precision registry (unsloth/precision/recipes.py).
FLOAT8_TRAINING_RECIPES = _precision_named_recipes("float8_training")

# Named recipes run only where they were timed faster than bfloat16 at equal loss (Qwen3-1.7B / 4B full
# finetuning). tensorwise: RTX PRO 6000 (12.0) 15% faster; L4 (8.9) NaN gradients with gradient
# checkpointing. rowwise: 3% slower on the RTX PRO 6000 and 2% slower on the L4, so it is not offered by
# name. Any other GPU needs an explicit torchao Float8LinearConfig.
_MEASURED_FASTER = _precision_measured_faster("float8_training")

# Never fp8: output heads (Unsloth's fused CE reads lm_head.weight directly), MoE routers (tiny, precision sensitive).
_SKIP_NAME_PARTS = ("lm_head", "embed_out", "output_projection", "router")
_SKIP_LEAF_NAMES = ("gate", "router", "score", "classifier")


def resolve_float8_training(float8_training, dtype = None):
    """`float8_training` kwarg -> recipe name or a torchao Float8LinearConfig; None when off. A format the
    precision registry knows but Unsloth cannot train (MXFP6, or one not built yet) raises with its reason,
    before the model is loaded; `dtype` (the load dtype, None = auto) refuses float16 for formats whose
    quantizer rejects it."""
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
        if recipe not in FLOAT8_TRAINING_RECIPES and recipe in _PRECISION_NAMES:
            # Raises PrecisionRefused (a ValueError) with the format's own reason.
            _check_precision_recipe(
                _PRECISION_NAMES[recipe], gpu = _precision.current_gpu_class(), input_dtype = dtype
            )
            raise ValueError(
                f"Unsloth: float8_training = '{float8_training}' has not been measured faster than bfloat16 "
                "on any GPU, so it is not offered by name. Pass "
                f"torchao.float8.Float8LinearConfig.from_recipe_name({recipe!r}) to force it."
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
    # Routers found by shape beside an experts container (a router named `wg` or in remote code), on top of the
    # name rules below.
    try:
        routers = frozenset(_find_routers(model))
    except Exception:
        routers = frozenset()

    def module_filter_fn(module, fqn):
        # Exact type: bitsandbytes / QAT / fp8 / PEFT linears subclass nn.Linear and must stay as they are.
        if type(module) is not torch.nn.Linear:
            return False
        if id(module) in out_heads or any(p in fqn for p in _SKIP_NAME_PARTS):
            return False
        # A router found structurally can be a gating wrapper: its inner Linear is the router too.
        if fqn.rsplit(".", 1)[-1] in _SKIP_LEAF_NAMES or any(
            fqn == r or fqn.startswith(r + ".") for r in routers
        ):
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


def _probe_float8_gemm(
    device,
    recipe,
    dtype = torch.bfloat16,
):
    """Tiny torch._scaled_mm calls with each GEMM's own fp8 dtypes and scaling, so an unsupported GPU / torch refuses here, not mid-training.
    Rowwise always writes bfloat16 (torchao computes it in bfloat16 and casts)."""
    M, K, N = 16, 32, 16
    for (dtype_a, rowwise_a), (dtype_b, rowwise_b) in _probe_gemms(recipe):
        a = torch.ones(M, K, device = device).to(dtype_a)
        b = torch.ones(N, K, device = device).to(dtype_b).t()
        # torch._scaled_mm takes one granularity for both operands; torchao (float8_ops.preprocess_addmm)
        # repeats a tensorwise scale along the axis when the other operand is rowwise, so the probe does too.
        rowwise = rowwise_a or rowwise_b
        sa = torch.ones(M, 1, device = device) if rowwise else torch.ones((), device = device)
        sb = torch.ones(1, N, device = device) if rowwise else torch.ones((), device = device)
        out_dtype = (
            torch.bfloat16 if rowwise or dtype not in (torch.float16, torch.float32) else dtype
        )
        torch._scaled_mm(a, b, scale_a = sa, scale_b = sb, out_dtype = out_dtype)


def _float8_compute_dtype(model, module_filter_fn):
    """float16 when the converted layers will run in float16 (a float16 load, or one Unsloth upcast to float32
    masters for float16 mixed precision), else bfloat16."""
    if getattr(model, "_unsloth_original_dtype", None) == torch.float16:
        return torch.float16
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module_filter_fn(module, name):
            if module.weight.dtype == torch.float16:
                return torch.float16
    return torch.bfloat16


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


def _check_hardware(
    device,
    recipe,
    dtype = torch.bfloat16,
):
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
    probe_recipe = f"{_recipe_scaling(recipe)} scaling"
    try:
        if dtype == torch.float16:
            # The float16 output GEMM is a different cuBLAS / CUTLASS path from the bfloat16 one.
            probe_recipe += ", float16 output"
            _probe_float8_gemm(device, recipe, torch.float16)
        else:
            _probe_float8_gemm(device, recipe)
    except Exception as e:
        raise RuntimeError(
            f"Unsloth: float8_training ({probe_recipe}) is not supported by torch {torch.__version__} on "
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
        dtype = _float8_compute_dtype(model, module_filter_fn)
        for device in devices or [next((p.device for p in model.parameters()), None)]:
            _check_hardware(device, recipe, dtype)
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
    name = (
        recipe if isinstance(recipe, str) else "Float8LinearConfig(" + _recipe_scaling(recipe) + ")"
    )
    _record_engagement(model, "float8_training:" + name, len(converted))
    # UNSLOTH_STRICT_ENGAGE=1 raises here instead of training in the original precision.
    _require_engagement(
        "float8_training",
        len(converted),
        "no eligible Linear layer: dims divisible by 16 and large enough",
    )
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


# ---- FP8 LoRA: fp8 forward + input gradient on the frozen 16-bit base (kernels/fp8_lora.py) ------------------
FP8_LORA_RECIPE = "fp8_lora_tensorwise"
# float8_training values that mean the frozen-base recipe when the model trains LoRA. True / "tensorwise" run
# only where the recipe is named (measured faster at equal loss); "fp8_lora" runs wherever the GPU can.
_FP8_LORA_BY_NAME = ("tensorwise",)
_FP8_LORA_EXPLICIT = ("fp8_lora", FP8_LORA_RECIPE)


def resolve_fp8_lora(
    float8_training,
    *,
    dtype = None,
    load_in_4bit = False,
    load_in_8bit = False,
    load_in_fp8 = False,
    qat_scheme = None,
    gpu = "current",
):
    """`float8_training` with LoRA / QLoRA -> FP8_LORA_RECIPE or None; refuses (PrecisionRefused, a ValueError)
    before anything loads. `gpu` is a registry GPU class ('sm89', ...); 'current' reads the CUDA device."""
    if float8_training is None or float8_training is False:
        return None
    refused = _precision.PrecisionRefused
    if float8_training is True:
        explicit = False
    elif isinstance(float8_training, str) and float8_training.strip().lower() in (
        _FP8_LORA_BY_NAME + _FP8_LORA_EXPLICIT
    ):
        explicit = float8_training.strip().lower() in _FP8_LORA_EXPLICIT
    else:
        raise ValueError(
            f"Unsloth: float8_training = {float8_training!r} is a full finetuning recipe "
            "(`full_finetuning = True`). With LoRA, float8_training = 'fp8_lora' (or True where it is "
            "measured faster) runs the frozen base's forward and input gradient in fp8."
        )
    if load_in_4bit or load_in_8bit or (load_in_fp8 not in (False, None)):
        raise refused(
            "Unsloth: float8_training with LoRA quantizes a 16-bit base to fp8 itself; pass "
            "load_in_4bit = False (and no load_in_8bit / load_in_fp8). QLoRA bases are already 4 / 8 bit."
        )
    if qat_scheme is not None:
        raise refused(
            "Unsloth: float8_training with LoRA and qat_scheme cannot be combined: QAT fake-quantizes the "
            "base weight that fp8 LoRA would quantize."
        )
    if gpu == "current":
        gpu = _precision.current_gpu_class()
    if gpu is None:
        raise refused(
            "Unsloth: float8_training with LoRA needs an NVIDIA GPU with compute capability 8.9+ "
            "(L4, RTX 40 / 50, RTX PRO 6000, H100, B200); no CUDA device is visible."
        )
    recipe = fp8_lora_recipe_for(gpu)
    gate = _check_precision_recipe(recipe, gpu = gpu, input_dtype = dtype, mode = "lora")
    if not explicit and gate.status != "named":
        raise refused(
            f"Unsloth: float8_training = {float8_training!r} with LoRA has not been measured faster than "
            f"bfloat16 LoRA on {gpu} ({gate.status}), so it is not enabled by name there. Pass "
            "float8_training = 'fp8_lora' to run the fp8 frozen base anyway."
        )
    if gate.status == "unprobed":
        print(f"Unsloth: fp8 LoRA has not been tested on {gpu}; running it as requested.")
    return recipe


def fp8_lora_recipe_for(gpu):
    """The frozen-base recipe the kernels run on this GPU class: per-token scales on sm120, tensorwise
    elsewhere (UNSLOTH_FP8_LORA_SCALING overrides; kernels/fp8_lora.scaling_for_class is the one rule)."""
    scaling = _load_fp8_lora_kernels().scaling_for_class(gpu)
    return "fp8_lora_rowwise" if scaling == "row" else FP8_LORA_RECIPE


def mark_fp8_lora(model, recipe):
    """Remember the request on the loaded model; get_peft_model converts the LoRA bases. A model that is
    already a PeftModel (an adapter repo was loaded, so get_peft_model is skipped) is converted now."""
    if recipe is not None and model is not None:
        try:
            model._unsloth_fp8_lora_recipe = recipe
        except Exception:
            pass
        if getattr(model, "peft_config", None):
            enable_fp8_lora(model, recipe)
    return model


def _fp8_lora_request(model):
    # PeftModel / LoraModel forward attribute reads to the wrapped model; walk explicitly as well.
    stack, seen = [model], set()
    while stack and len(seen) < 8:
        m = stack.pop()
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        recipe = getattr(m, "_unsloth_fp8_lora_recipe", None)
        if isinstance(recipe, str):
            return recipe
        stack.extend(getattr(m, attr, None) for attr in ("base_model", "model"))
    return None


def _load_fp8_lora_kernels():
    try:
        from ..kernels import fp8_lora
        return fp8_lora
    except ImportError:
        import importlib.util
        import os
        import sys

        name = "_unsloth_fp8_lora_standalone"
        if name in sys.modules:
            return sys.modules[name]
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "fp8_lora.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


def _merge_with_master(
    self,
    *args,
    _unsloth_method = "merge",
    **kwargs,
):
    # Resolved from self (not captured), so a deepcopy of the model merges its own layers.
    base = getattr(self, "base_layer", None)
    if base is not None:
        _load_fp8_lora_kernels().restore_fp8_lora_master(base)
    out = getattr(type(self), _unsloth_method)(self, *args, **kwargs)
    state = base.__dict__.get("_unsloth_fp8_lora") if base is not None else None
    if state is not None:
        # PEFT adds the delta into weight.data in place (same storage and version): requantize now, so the
        # next forward (compiled ones too) uses the merged weight.
        state._key = None
        state.refresh(base.weight)
    return out


def _has_variant(lora_module, adapter_name = None) -> bool:
    """A PEFT LoRA variant (DoRA, BD-LoRA, VeLoRA, ...): DoRA reads the base weight in its forward, and variant
    init (after the adapter is placed) allocates its own tensors on base_layer.weight's device."""
    dora = getattr(lora_module, "use_dora", {}) or {}
    variants = getattr(lora_module, "lora_variant", {}) or {}
    if adapter_name is not None:
        return bool(dora.get(adapter_name, False)) or adapter_name in variants
    return any(dora.values()) or bool(variants)


def _move_adapter_with_master(self, adapter_name, *args, **kwargs):
    """PEFT puts a new adapter on base_layer.weight's device, which is the host for an offloaded master: place it
    on the fp8 compute device. A variant adapter (DoRA, BD-LoRA, VeLoRA) brings the master back to that device
    first, so PEFT places it and whatever its variant init builds there."""
    base = getattr(self, "base_layer", None)
    state = base.__dict__.get("_unsloth_fp8_lora") if base is not None else None
    if state is not None and state.offloaded and _has_variant(self, adapter_name):
        _load_fp8_lora_kernels().restore_fp8_lora_master(base)
    out = type(self)._move_adapter_to_device_of_base_layer(self, adapter_name, *args, **kwargs)
    if state is None:
        return out
    # Adapter init can rewrite the base weight (PiSSA / OLoRA keep the residual): requantize now, not at the next
    # eager forward, so compiled calls see it.
    state._key = None
    state.refresh(base.weight)
    if not state.offloaded:
        return out
    for name in tuple(getattr(self, "adapter_layer_names", ())) + tuple(
        getattr(self, "other_param_names", ())
    ):
        layer = getattr(self, name, None)
        if (
            not isinstance(layer, (torch.nn.ModuleDict, torch.nn.ParameterDict))
            or adapter_name not in layer
        ):
            continue
        entry = layer[adapter_name]
        tensors = list(entry.parameters()) if isinstance(entry, torch.nn.Module) else [entry]
        if any(t.device.type == "meta" for t in tensors):
            # low_cpu_mem_usage: PEFT leaves meta parameters for the checkpoint loader to materialize.
            continue
        if isinstance(entry, torch.nn.Module):
            layer[adapter_name] = entry.to(state.device)
        else:
            layer[adapter_name] = torch.nn.Parameter(
                entry.data.to(state.device), requires_grad = entry.requires_grad
            )
    return out


def _restore_master_before_merge(lora_module):
    """PEFT merge / unmerge add to base.weight.data on the LoRA device: bring a host master back first."""
    import functools
    import types

    for method in ("merge", "unmerge"):
        if method in lora_module.__dict__ or not callable(getattr(type(lora_module), method, None)):
            continue
        fn = functools.partial(_merge_with_master, _unsloth_method = method)
        object.__setattr__(lora_module, method, types.MethodType(fn, lora_module))
    move = "_move_adapter_to_device_of_base_layer"
    if move not in lora_module.__dict__ and callable(getattr(type(lora_module), move, None)):
        object.__setattr__(
            lora_module, move, types.MethodType(_move_adapter_with_master, lora_module)
        )


_RAW_READS = {}


def _reads_raw_weight(parent, leaf) -> bool:
    """The parent's class source reads `<leaf>.weight` directly (fused Mamba / Falcon-H1 kernels, WavLM's
    F.multi_head_attention_forward) instead of calling the child: that tensor must stay on the compute device."""
    import inspect
    import re

    key = (type(parent), leaf)
    if key not in _RAW_READS:
        try:
            source = inspect.getsource(type(parent))
        except (OSError, TypeError):
            source = ""
        _RAW_READS[key] = re.search(rf"\b{re.escape(leaf)}\.weight\b", source) is not None
    return _RAW_READS[key]


def _mamba_raw_weight_leaf(model, name) -> bool:
    parent_name, _, leaf = name.rpartition(".")
    try:
        parent = model.get_submodule(parent_name) if parent_name else model
    except AttributeError:
        return False
    if _reads_raw_weight(parent, leaf):
        return True
    try:
        from ._mamba_lora import _RAW_WEIGHT_LEAVES, _is_mamba_mixer
    except Exception:
        return False
    # Unsloth's fused Mamba LoRA path builds W + delta from these leaves of any Mamba-style mixer.
    return leaf in _RAW_WEIGHT_LEAVES and _is_mamba_mixer(parent)


def _block_swapped_modules(model) -> set:
    """ids of every module inside a decoder layer managed by block swap (install_block_swap)."""
    out = set()
    for module in model.modules():
        swapper = module.__dict__.get("_unsloth_block_swap")
        if swapper is None or not isinstance(module, torch.nn.ModuleList):
            continue
        for index in getattr(swapper, "indices", ()) or ():
            if 0 <= index < len(module):
                out.update(id(m) for m in module[index].modules())
    return out


def enable_fp8_lora(
    model,
    recipe = None,
    allow_cpu = False,
):
    """Attach the fp8 frozen-base state to every LoRA base nn.Linear (frozen, 16-bit, dims % 16, not a
    high-precision layer). Returns the converted names. A no-op unless the model was loaded with it."""
    recipe = recipe or _fp8_lora_request(model)
    if recipe is None:
        return []
    kernels = _load_fp8_lora_kernels()
    policy = _precision.get_recipe(recipe).hp_layers
    try:
        is_hp = _precision.hp_filter(model, policy)
    except Exception:
        is_hp = lambda module, fqn: False
    converted, skipped = [], {}
    # How many modules hold each Parameter (tied weights).
    uses = {}
    for _, p in model.named_parameters(remove_duplicate = False):
        uses[id(p)] = uses.get(id(p), 0) + 1
    swapped = _block_swapped_modules(model)
    offload = kernels.offload_supported()
    offloaded = 0
    for name, module in model.named_modules():
        base = getattr(module, "base_layer", None)
        if base is None or not hasattr(module, "lora_A") or not isinstance(base, torch.nn.Module):
            continue
        if is_hp(base, name + ".base_layer") or is_hp(module, name):
            skipped[name] = "high precision layer"
            continue
        why = kernels.eligible_base(base)
        if why is None and not allow_cpu:
            device = base.weight.device
            if device.type != "cuda" or torch.version.hip is not None:
                why = f"weight on {device}"
            elif torch.cuda.get_device_capability(device) < (8, 9):
                why = "compute capability below 8.9"
        if why is None and id(base) in swapped:
            # Block swap (offload_layers) streams these weights in and out and leaves empty placeholders: no fp8
            # copy of a weight that is not there.
            why = "block-swapped layer"
        if why is None and uses.get(id(base.weight), 0) > 1:
            # A Parameter shared by two modules (tied weights): a merge through one reader would leave the other's
            # fp8 copy stale (same storage and version), so it stays a 16-bit base.
            why = "weight shared with another module"
        if why is not None:
            skipped[name] = why
            continue
        # LoRA variants (DoRA's column norm, BD-LoRA / VeLoRA tensors built by base weight device) and parents
        # that read a child's weight directly (fused Mamba / Falcon-H1 kernels, WavLM attention) keep the master
        # on the device.
        reads_raw = _has_variant(module) or _mamba_raw_weight_leaf(model, name)
        state = kernels.attach_fp8_lora_base(base, offload = offload and not reads_raw)
        offloaded += int(state.offloaded)
        _restore_master_before_merge(module)
        converted.append(name)
    if offloaded and torch.cuda.is_available():
        # Return the freed 16-bit blocks to the driver (they are 2 bytes per converted weight).
        torch.cuda.empty_cache()
    detail = {"skipped": len(skipped)} if skipped else {}
    if offloaded:
        detail["master_offloaded"] = offloaded
    _record_engagement(model, recipe, len(converted), detail = detail or None)
    if not _require_engagement(recipe, len(converted), "no frozen 16-bit LoRA base Linear layers"):
        print(
            "Unsloth: float8_training with LoRA found no eligible base layers"
            + (f" ({next(iter(skipped.values()))})" if skipped else "")
            + "; training stays in 16 bit."
        )
    else:
        print(f"Unsloth: fp8 LoRA base on {len(converted)} Linear layers.")
    return converted
