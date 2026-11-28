# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""On-the-fly low-bit loads from a 16-bit checkpoint, selected by a string flag:

- `load_in_4bit = "nvfp4"` / `"mxfp4"`: W4A16. Every eligible `nn.Linear` is quantized as its weight arrives
  from the checkpoint (one layer of 16-bit resident at a time) and becomes `_UnslothNVFP4Linear`, the frozen
  packed layer pre-quantized NVFP4 checkpoints already use; LoRA runs on top through the W4A16 kernels in
  `unsloth.kernels.nvfp4`. 16-bit routed-expert stacks (transformers 5 MoE) are quantized the same way by
  unsloth_zoo's expert quantizer. What NF4 keeps in 16 bit stays 16 bit here (SKIP_QUANTIZATION_MODULES,
  output heads, embeddings, routers).
- `load_in_4bit = "nf4"`: the same as `load_in_4bit = True` (bitsandbytes NF4).
- `load_in_8bit = "fp8"` / `"fp8_block"`: aliases of `load_in_fp8 = True` / `"block"`.

`load_in_4bit = True` and `load_in_8bit = True` are untouched.
"""

import contextlib
import contextvars
import functools
import inspect
import os
import types

import torch

__all__ = [
    "FP4_ONTHEFLY_FORMATS",
    "normalize_lowbit_flags",
    "onthefly_lowbit_load",
    "onthefly_fp4_loading",
    "onthefly_fp4_stats",
]

FP4_ONTHEFLY_FORMATS = ("nvfp4", "mxfp4")
_NF4_NAMES = ("nf4", "bnb", "bnb_nf4", "bitsandbytes")
_FP8_NAMES = {
    "fp8": True,
    "fp8_row": True,
    "fp8_rowwise": True,
    "fp8_block": "block",
    "fp8_blockwise": "block",
}
_GROUP = {"nvfp4": 16, "mxfp4": 32}
# Kill switch: 0 loads 16 bit and keeps it (no FP4 layers), to bisect a model against the 16-bit run.
ONTHEFLY_ENV = "UNSLOTH_ONTHEFLY_FP4"
EXPERTS_ENV = "UNSLOTH_MOE_EXPERTS_QUANTIZE"
_ASYNC_ENV = "HF_DEACTIVATE_ASYNC_LOAD"  # transformers 5: load parameters on the main thread

# Devices whose weights are quantized (the W4A16 kernels run there); others stay 16 bit.
_QUANT_DEVICES = ("cuda", "xpu")

_ACTIVE = contextvars.ContextVar("unsloth_onthefly_fp4", default = None)
# Set while transformers 5 finalizes a load: it materializes missing / mismatched weights through setattr as
# uninitialized storage and initializes them afterwards, so those stay dense until _finalize quantizes them.
_DEFER = contextvars.ContextVar("unsloth_onthefly_fp4_defer", default = False)
_STATS = {"linear": 0, "experts": 0, "skipped": 0, "late": 0}


def onthefly_fp4_stats(reset = False):
    """Counts of the last loads: FP4 linears, FP4 expert stacks, eligible-looking linears kept 16 bit, and
    linears quantized after the load (weights not assigned through the loader)."""
    out = dict(_STATS)
    if reset:
        for k in _STATS:
            _STATS[k] = 0
    return out


def _norm(value):
    return str(value).strip().lower().replace("-", "_")


def normalize_lowbit_flags(
    load_in_4bit,
    load_in_8bit,
    load_in_fp8 = False,
    full_finetuning = False,
):
    """(load_in_4bit, load_in_8bit, load_in_fp8, fp4_format) with every string spelling resolved.
    Booleans pass through unchanged; an unknown string raises instead of turning on bitsandbytes."""
    fmt = None
    if isinstance(load_in_4bit, str):
        name = _norm(load_in_4bit)
        if name in FP4_ONTHEFLY_FORMATS:
            fmt = name
            load_in_4bit = False
        elif name in _NF4_NAMES:
            load_in_4bit = True
        else:
            raise ValueError(
                f"Unsloth: load_in_4bit = {load_in_4bit!r} is not supported. Use True (bitsandbytes NF4), "
                "'nvfp4' or 'mxfp4' (W4A16 quantized on the fly from a 16-bit checkpoint)."
            )
        if fmt is not None and full_finetuning:
            raise ValueError(
                f"Unsloth: load_in_4bit = '{fmt}' is a frozen W4A16 base for LoRA; it cannot be full finetuned. "
                "Use full_finetuning = False, or load_in_16bit = True to full finetune."
            )
    if isinstance(load_in_8bit, str):
        name = _norm(load_in_8bit)
        if name not in _FP8_NAMES:
            raise ValueError(
                f"Unsloth: load_in_8bit = {load_in_8bit!r} is not supported. Use True (bitsandbytes 8-bit), "
                "'fp8' (= load_in_fp8 = True) or 'fp8_block' (= load_in_fp8 = 'block')."
            )
        mode = _FP8_NAMES[name]
        if load_in_fp8 not in (False, None) and load_in_fp8 != mode:
            raise ValueError(
                f"Unsloth: load_in_8bit = {load_in_8bit!r} conflicts with load_in_fp8 = {load_in_fp8!r}."
            )
        load_in_8bit = False
        load_in_fp8 = mode
    return load_in_4bit, load_in_8bit, load_in_fp8, fmt


def onthefly_lowbit_load(fn):
    """Decorator for the from_pretrained entry points: resolve string flags, then run the load under
    `onthefly_fp4_loading` when an FP4 format was asked for. Boolean flags call `fn` unchanged."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn
    var_kw = next(
        (p.name for p in signature.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD),
        None,
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)
        arguments = bound.arguments
        l4 = arguments.get("load_in_4bit", None)
        l8 = arguments.get("load_in_8bit", None)
        if not isinstance(l4, str) and not isinstance(l8, str):
            return fn(*args, **kwargs)
        explicit_4bit = "load_in_4bit" in arguments
        l4, l8, fp8, fmt = normalize_lowbit_flags(
            True if l4 is None else l4,
            False if l8 is None else l8,
            arguments.get("load_in_fp8", False),
            bool(arguments.get("full_finetuning", False)),
        )
        if fmt is not None and arguments.get("fast_inference", False):
            raise NotImplementedError(
                f"Unsloth: load_in_4bit = '{fmt}' does not support fast_inference = True (vLLM) yet."
            )
        if fmt is not None:
            # load_in_4bit is False from here on, so the loader's own conflict checks no longer see the FP4 request.
            extra = arguments.get(var_kw) or {} if var_kw else {}

            def given(name, default):
                return arguments[name] if name in arguments else extra.get(name, default)

            others = [
                name
                for name, on in (
                    ("load_in_8bit", l8),
                    ("load_in_16bit", given("load_in_16bit", False)),
                    ("load_in_fp8", fp8 not in (False, None)),
                    ("float8_training", given("float8_training", None) not in (None, False)),
                )
                if on
            ]
            if others:
                raise ValueError(
                    f"Unsloth: load_in_4bit = '{fmt}' cannot be combined with {', '.join(others)}: pick one "
                    "precision for the base weights."
                )
        arguments["load_in_4bit"] = l4
        arguments["load_in_8bit"] = l8
        if fp8 is not False:
            arguments["load_in_fp8"] = fp8
            if not explicit_4bit:
                # load_in_4bit defaults to True: fp8 asked through load_in_8bit replaces it.
                arguments["load_in_4bit"] = False
        if fmt is None:
            return fn(*bound.args, **bound.kwargs)
        with onthefly_fp4_loading(fmt):
            return fn(*bound.args, **bound.kwargs)

    return wrapper


# ---- the load hook ------------------------------------------------------------------------------------------


def _skip_names():
    try:
        from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
        names = list(SKIP_QUANTIZATION_MODULES)
    except Exception:
        names = ["lm_head", "multi_modal_projector", "router", "mlp.gate", "vision_tower", "visual"]
    return names


def _model_types(config):
    seen, out, stack = set(), [], [config]
    while stack:
        c = stack.pop()
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        mt = getattr(c, "model_type", None)
        if isinstance(mt, str):
            out.append(mt)
        for key in (
            "text_config",
            "vision_config",
            "audio_config",
            "language_config",
            "llm_config",
        ):
            stack.append(getattr(c, key, None))
    return out


def _architecture_skips(config):
    # The per-architecture extras NF4 keeps 16 bit (layers whose kernels read `.weight` as a dense matrix).
    model_types = _model_types(config)
    skip = ["out_proj"] if "falcon_h1" in model_types else []
    try:
        from unsloth.models.vision import _architecture_skip_modules
        skip += list(_architecture_skip_modules(model_types))
    except Exception:
        pass
    return skip


def _name_skipped(name, skip):
    # A skip key matches whole dotted components ("mlp.gate" skips `layers.0.mlp.gate`, not `mlp.gate_proj`).
    dotted = "." + name + "."
    return any(("." + key + ".") in dotted for key in skip)


def _structural_skips(model):
    found = set()
    try:
        from unsloth.precision import selectors
    except Exception:
        return found
    for finder in (selectors.find_output_heads, selectors.find_routers, selectors.find_embeddings):
        try:
            found.update(finder(model))
        except Exception:
            pass
    return found


def _experts_format(fmt):
    raw = os.environ.get(EXPERTS_ENV, None)
    if raw is None or raw.strip() == "":
        return fmt
    value = raw.strip().lower()
    if value in ("0", "off", "false", "none", "no"):
        return None
    return value


class _PendingFP4Linear(torch.nn.Linear):
    """Linear whose weight is quantized the moment the loader assigns it (setattr / load_state_dict(assign))."""

    def __setattr__(self, name, value):
        if (
            name == "weight"
            and isinstance(value, torch.Tensor)
            and value.device.type in _QUANT_DEVICES
            and value.is_floating_point()
            and tuple(value.shape) == (self.out_features, self.in_features)
            and "weight_packed" not in self._parameters
            and not _DEFER.get()
        ):
            _quantize_linear_(self, value)
            return
        super().__setattr__(name, value)


def _quantize_linear_(module, weight):
    from unsloth.kernels.fp4_quantize import fp4_quantize_weight

    fmt = module.__dict__["_unsloth_onthefly_fmt"]
    dtype = weight.dtype if weight.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    with torch.no_grad():
        packed, scale, global_scale = fp4_quantize_weight(weight.detach(), fmt)
    params = module._parameters
    for key, value in (
        ("weight_packed", packed),
        ("weight_scale", scale),
        ("weight_global_scale", global_scale),
    ):
        p = torch.nn.Parameter(value, requires_grad = False)
        p._is_hf_initialized = True
        params[key] = p
    # The loader still reads `weight` (shape checks, init flags) until the load ends: a 0-element stand-in.
    stand_in = torch.nn.Parameter(
        torch.empty(0, dtype = dtype, device = weight.device), requires_grad = False
    )
    stand_in._is_hf_initialized = True
    params["weight"] = stand_in
    module.__dict__["_unsloth_nvfp4_dtype"] = dtype
    del weight


def _experts_pending_class(cls):
    cached = _EXPERT_CLASSES.get(cls)
    if cached is not None:
        return cached

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_unsloth_fp4_base_setattr")(name, value)
        if name in ("gate_up_proj", "down_proj") and not _DEFER.get():
            _maybe_quantize_experts(self)

    pending = type(
        cls.__name__,
        (cls,),
        {
            "__setattr__": __setattr__,
            "__module__": cls.__module__,
            "__qualname__": getattr(cls, "__qualname__", cls.__name__),
            "_unsloth_fp4_pending_experts": True,
        },
    )
    _EXPERT_CLASSES[cls] = pending
    return pending


_EXPERT_CLASSES = {}


def _maybe_quantize_experts(experts):
    params = experts._parameters
    stacks = [params.get("gate_up_proj"), params.get("down_proj")]
    if not all(
        type(p) is torch.nn.Parameter and p.dim() == 3 and p.device.type in _QUANT_DEVICES
        for p in stacks
    ):
        return
    original = experts.__dict__.pop("_unsloth_fp4_original_class", None)
    if original is None:
        return
    object.__setattr__(experts, "__class__", original)
    experts.__dict__.pop("_unsloth_fp4_base_setattr", None)
    fmt = experts.__dict__.pop("_unsloth_onthefly_experts_fmt", None)
    if fmt is None:
        return
    _STATS["experts"] += _quantize_expert_stacks(experts, fmt)


def _quantize_expert_stacks(experts, fmt):
    try:
        from unsloth_zoo import packed_experts as pe
        count = pe.quantize_moe_expert_stacks(experts, fmt)
    except Exception as exc:
        print(f"Unsloth: could not quantize an expert stack to {fmt} ({exc}); it stays 16 bit.")
        return 0
    # Loaded values: transformers' init of missing keys must not overwrite the packed stacks.
    for name in ("gate_up_proj", "down_proj"):
        p = experts._parameters.get(name)
        if p is not None:
            p._is_hf_initialized = True
    if count and not experts.__dict__.get("_unsloth_onthefly_fp4_experts"):
        experts.__dict__["_unsloth_onthefly_fp4_experts"] = True
        experts._register_state_dict_hook(_dense_state_dict_hook)
    return count


def _prepare(model, fmt):
    """Before the weights load (model on meta): mark the Linears and expert stacks to quantize."""
    skip = _skip_names() + _architecture_skips(getattr(model, "config", None))
    structural = _structural_skips(model)
    group = _GROUP[fmt]
    experts_fmt = _experts_format(fmt)
    try:
        from unsloth_zoo import packed_experts as pe
        has_expert_quant = hasattr(pe, "quantize_moe_expert_stacks")
    except Exception:
        has_expert_quant = False
    marked = []
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear:
            if (
                name in structural
                or _name_skipped(name, skip)
                or module.in_features % group
                or module.in_features < group
            ):
                continue
            module.__class__ = _PendingFP4Linear
            module.__dict__["_unsloth_onthefly_fmt"] = fmt
            marked.append(module)
            continue
        params = module._parameters
        gu, down = params.get("gate_up_proj"), params.get("down_proj")
        if (
            experts_fmt is not None
            and has_expert_quant
            and isinstance(gu, torch.nn.Parameter)
            and isinstance(down, torch.nn.Parameter)
            and gu.dim() == 3
            and down.dim() == 3
            and not _name_skipped(name, skip)
        ):
            cls = type(module)
            module.__dict__["_unsloth_fp4_original_class"] = cls
            module.__dict__["_unsloth_fp4_base_setattr"] = functools.partial(
                cls.__setattr__, module
            )
            module.__dict__["_unsloth_onthefly_experts_fmt"] = experts_fmt
            module.__class__ = _experts_pending_class(cls)
            marked.append(module)
    return marked


def _routed_linear_parts():
    """The packed W4A16 layer of pre-quantized NVFP4 checkpoints, its forward, and the PEFT merge / DoRA patch."""
    from unsloth.models.loader_utils import (
        _UnslothNVFP4Linear,
        _patch_peft_for_routed_compressed_tensors,
        _unsloth_compressed_tensors_nvfp4_forward,
    )
    return (
        _UnslothNVFP4Linear,
        _patch_peft_for_routed_compressed_tensors,
        _unsloth_compressed_tensors_nvfp4_forward,
    )


def _finalize(model, fmt):
    """After the load: every marked Linear becomes `_UnslothNVFP4Linear`; ones the loader filled in place
    (not through setattr) are quantized now, layer by layer; still-meta ones go back to nn.Linear."""
    _UnslothNVFP4Linear, _patch_peft, _forward = _routed_linear_parts()
    count = 0
    for module in model.modules():
        if getattr(module, "_unsloth_fp4_pending_experts", False):
            _maybe_quantize_experts(module)
            original = module.__dict__.pop("_unsloth_fp4_original_class", None)
            if original is not None:
                object.__setattr__(module, "__class__", original)
                module.__dict__.pop("_unsloth_fp4_base_setattr", None)
                experts_fmt = module.__dict__.pop("_unsloth_onthefly_experts_fmt", None)
                gu = module._parameters.get("gate_up_proj")
                if experts_fmt is not None and gu is not None and gu.device.type in _QUANT_DEVICES:
                    _STATS["experts"] += _quantize_expert_stacks(module, experts_fmt)
            continue
        if type(module) is not _PendingFP4Linear:
            continue
        params = module._parameters
        if "weight_packed" not in params:
            weight = params.get("weight")
            if (
                isinstance(weight, torch.Tensor)
                and weight.device.type in _QUANT_DEVICES
                and weight.is_floating_point()
                and weight.numel() == module.out_features * module.in_features
            ):
                params.pop("weight")
                _quantize_linear_(module, weight.data)
                del weight
                _STATS["late"] += 1
            else:
                module.__class__ = torch.nn.Linear
                module.__dict__.pop("_unsloth_onthefly_fmt", None)
                _STATS["skipped"] += 1
                continue
        params.pop("weight", None)
        module.__dict__.pop("_unsloth_onthefly_fmt", None)
        module.__class__ = _UnslothNVFP4Linear
        module.forward = _forward.__get__(module)
        # No compressed-tensors scheme here: PEFT merges / SVD inits decode with Unsloth's own kernels.
        module.dequantize_ = _dequantize_onthefly_.__get__(module)
        module._unsloth_onthefly_fp4 = True
        module._register_state_dict_hook(_dense_state_dict_hook)
        count += 1
    if count:
        _patch_peft()
    _STATS["linear"] += count
    experts = sum(1 for m in model.modules() if m.__dict__.get("_unsloth_onthefly_fp4_experts"))
    # Linears + expert modules quantized here: save_pretrained keys on it.
    model._unsloth_onthefly_fp4 = getattr(model, "_unsloth_onthefly_fp4", 0) + count + experts
    try:
        from unsloth.precision import engage
        engage.record(model, f"onthefly_{fmt}_w4a16", count, detail = {"experts": _STATS["experts"]})
    except Exception:
        pass
    return count


# ---- saving: a full-model save_pretrained writes the dequantized 16-bit weights ------------------------------

_SAVING = contextvars.ContextVar("unsloth_onthefly_fp4_saving", default = False)
_EXPERT_STACKS = (("gate_up_proj", "gate_up"), ("down_proj", "down"))
_PACKED_KEYS = ("weight_packed", "weight_scale", "weight_global_scale")


def _is_packed_stack(p):
    return (
        isinstance(p, torch.Tensor)
        and type(p) is not torch.nn.Parameter
        and hasattr(p, "dequantize")
    )


def _dense_weight(module):
    from unsloth.kernels.nvfp4 import nvfp4_dequantize
    return nvfp4_dequantize(
        module.weight_packed,
        module.weight_scale,
        module.weight_global_scale,
        getattr(module, "_unsloth_nvfp4_dtype", torch.bfloat16),
    )


def _dequantize_onthefly_(module):
    """Turn an on-the-fly FP4 layer into a plain nn.Linear holding the weight it decodes to (PEFT merge_and_unload,
    PiSSA / OLoRA / LoftQ inits). compressed-tensors' decompressor skips modules without a quantization scheme."""
    with torch.inference_mode(False), torch.no_grad():
        W = _dense_weight(module)
    for name in ("forward", "dequantize_"):
        module.__dict__.pop(name, None)
    for key in _PACKED_KEYS + ("input_global_scale",):
        module._parameters.pop(key, None)
    module.__class__ = torch.nn.Linear
    module._unsloth_onthefly_fp4 = False
    module.weight = torch.nn.Parameter(W, requires_grad = False)
    return module


def _dense_state_dict_hook(module, state_dict, prefix, local_metadata):
    """Inside save_pretrained only: the packed tensors become the dense `weight` (or expert stack) they decode
    to, moved to the CPU one layer at a time, so the checkpoint reloads as a plain 16-bit model (the config has
    no quantization) holding the FP4-rounded values the model trained with."""
    if not _SAVING.get():
        return state_dict
    with torch.no_grad():
        if "weight_packed" in module._parameters:
            W = _dense_weight(module)
            for key in _PACKED_KEYS:
                state_dict.pop(prefix + key, None)
            state_dict[prefix + "weight"] = W.to("cpu")
            del W
            return state_dict
        for name, _ in _EXPERT_STACKS:
            p = module._parameters.get(name)
            if _is_packed_stack(p) and prefix + name in state_dict:
                state_dict[prefix + name] = p.dequantize().to("cpu")
    return state_dict


def _dense_state_dict(model, state_dict):
    """A copy of `state_dict` whose packed on-the-fly entries are replaced by the dense tensors they decode to."""
    out = dict(state_dict)
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        if module.__dict__.get("_unsloth_onthefly_fp4") and "weight_packed" in module._parameters:
            if prefix + "weight_packed" not in out:
                continue
            # Decode the caller's tensors, not the live module's: the dict may be a snapshot or a gathered copy.
            given = types.SimpleNamespace(
                _unsloth_nvfp4_dtype = getattr(module, "_unsloth_nvfp4_dtype", torch.bfloat16)
            )
            for key in _PACKED_KEYS:
                value = out.pop(prefix + key, None)
                setattr(given, key, getattr(module, key) if value is None else value)
            with torch.no_grad():
                out[prefix + "weight"] = _dense_weight(given).to("cpu")
        elif module.__dict__.get("_unsloth_onthefly_fp4_experts"):
            for stack, _ in _EXPERT_STACKS:
                value = out.get(prefix + stack)
                if _is_packed_stack(value):
                    with torch.no_grad():
                        out[prefix + stack] = value.dequantize().to("cpu")
    return out


def _install_save_hook(PreTrainedModel):
    original = PreTrainedModel.save_pretrained
    if getattr(original, "_unsloth_onthefly_fp4_save", False):
        return

    @functools.wraps(original)
    def save_pretrained(self, *args, **kwargs):
        if not getattr(self, "_unsloth_onthefly_fp4", 0) or _SAVING.get():
            return original(self, *args, **kwargs)
        print(
            "Unsloth: this base was quantized to FP4 at load, so save_pretrained writes its dequantized 16-bit "
            "weights. Save LoRA adapters with the PEFT model, or merge into the original 16-bit weights with "
            "save_pretrained_merged."
        )
        if kwargs.get("state_dict") is not None:
            # A caller-built state dict never runs the module hooks: convert it here.
            kwargs["state_dict"] = _dense_state_dict(self, kwargs["state_dict"])
        token = _SAVING.set(True)
        try:
            return original(self, *args, **kwargs)
        finally:
            _SAVING.reset(token)

    save_pretrained._unsloth_onthefly_fp4_save = True
    PreTrainedModel.save_pretrained = save_pretrained


_HOOKS_INSTALLED = [False]


def _install_hooks():
    if _HOOKS_INSTALLED[0]:
        return True
    try:
        from transformers import modeling_utils as mu
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return False
    original = PreTrainedModel.__dict__.get("_load_pretrained_model")
    if original is None:
        return False
    # transformers 5: staticmethod (model, state_dict, checkpoint_files, load_config); 4.5x: classmethod.
    is_classmethod = isinstance(original, classmethod)
    original_fn = (
        original.__func__ if isinstance(original, (staticmethod, classmethod)) else original
    )
    finalize_raw = PreTrainedModel.__dict__.get("_finalize_model_loading")

    def _around(model, call):
        state = _ACTIVE.get()
        if state is None or getattr(model, "config", None) is None:
            return call()
        qc = getattr(model.config, "quantization_config", None)
        if qc is not None or getattr(model, "hf_quantizer", None) is not None:
            if not state.get("warned"):
                print(
                    f"Unsloth: load_in_4bit = '{state['fmt']}' quantizes 16-bit checkpoints; this one is already "
                    "quantized, so it loads as stored."
                )
                state["warned"] = True
            return call()
        _prepare(model, state["fmt"])
        state["prepared"] = state.get("prepared", 0) + 1
        warmup = mu.__dict__.get("caching_allocator_warmup")
        # The warmup reserves the whole 16-bit model on the GPU up front, the copy this load avoids.
        if callable(warmup):
            mu.caching_allocator_warmup = lambda *args, **kwargs: None
        # transformers 5 loads with worker threads that materialize 16-bit tensors on the GPU faster than the
        # main thread consumes them (it turns them off itself for its own on-the-fly quantizers).
        async_env = os.environ.get(_ASYNC_ENV)
        os.environ[_ASYNC_ENV] = "1"
        try:
            result = call()
        finally:
            if callable(warmup):
                mu.caching_allocator_warmup = warmup
            if async_env is None:
                os.environ.pop(_ASYNC_ENV, None)
            else:
                os.environ[_ASYNC_ENV] = async_env
        if finalize_raw is None:
            _finalize(model, state["fmt"])
        else:
            model.__dict__["_unsloth_fp4_finalize_pending"] = state["fmt"]
        return result

    if is_classmethod:

        @functools.wraps(original_fn)
        def _load_pretrained_model(cls, model, *args, **kwargs):
            return _around(model, lambda: original_fn(cls, model, *args, **kwargs))

    else:

        @functools.wraps(original_fn)
        def _load_pretrained_model(model, *args, **kwargs):
            return _around(model, lambda: original_fn(model, *args, **kwargs))

    _load_pretrained_model.__wrapped__ = original_fn
    PreTrainedModel._load_pretrained_model = (
        classmethod(_load_pretrained_model)
        if is_classmethod
        else staticmethod(_load_pretrained_model)
    )

    if finalize_raw is not None:
        f_is_classmethod = isinstance(finalize_raw, classmethod)
        finalize_fn = (
            finalize_raw.__func__
            if isinstance(finalize_raw, (staticmethod, classmethod))
            else finalize_raw
        )

        def _after(model, call):
            if "_unsloth_fp4_finalize_pending" not in model.__dict__:
                return call()
            token = _DEFER.set(True)
            try:
                result = call()
            finally:
                _DEFER.reset(token)
            fmt = model.__dict__.pop("_unsloth_fp4_finalize_pending", None)
            if fmt is not None:
                _finalize(model, fmt)
            return result

        if f_is_classmethod:

            @functools.wraps(finalize_fn)
            def _finalize_model_loading(cls, model, *args, **kwargs):
                return _after(model, lambda: finalize_fn(cls, model, *args, **kwargs))

        else:

            @functools.wraps(finalize_fn)
            def _finalize_model_loading(model, *args, **kwargs):
                return _after(model, lambda: finalize_fn(model, *args, **kwargs))

        _finalize_model_loading.__wrapped__ = finalize_fn
        PreTrainedModel._finalize_model_loading = (
            classmethod(_finalize_model_loading)
            if f_is_classmethod
            else staticmethod(_finalize_model_loading)
        )
    _install_save_hook(PreTrainedModel)
    _HOOKS_INSTALLED[0] = True
    return True


@contextlib.contextmanager
def onthefly_fp4_loading(fmt):
    """Quantize every model loaded by transformers inside this block to `fmt` (W4A16)."""
    fmt = _norm(fmt)
    if fmt not in FP4_ONTHEFLY_FORMATS:
        raise ValueError(f"Unsloth: unknown on-the-fly FP4 format {fmt!r}")
    if os.environ.get(ONTHEFLY_ENV, "1").strip().lower() in ("0", "off", "false", "no"):
        print(f"Unsloth: {ONTHEFLY_ENV}=0, so load_in_4bit = '{fmt}' loads and keeps 16 bit.")
        yield None
        return
    if not _install_hooks():
        raise RuntimeError(
            f"Unsloth: load_in_4bit = '{fmt}' needs transformers' PreTrainedModel._load_pretrained_model."
        )
    state = {"fmt": fmt}
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)
    if not state.get("prepared") and not state.get("warned"):
        print(
            f"Unsloth: load_in_4bit = '{fmt}' found no transformers checkpoint load to quantize "
            "(this load path bypasses it), so the model stays 16 bit."
        )
