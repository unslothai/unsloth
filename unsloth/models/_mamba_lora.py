# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Keep LoRA adapters on Mamba projections that the fused training kernels read as raw weights.

In training (no cache) transformers' Mamba mixers hand ``self.x_proj.weight``, ``self.dt_proj.weight``
and ``self.out_proj.weight`` (Mamba2-style mixers: ``self.out_proj.weight``) straight to
``mamba_inner_fn`` / ``mamba_split_conv1d_scan_combined``, so a PEFT adapter wrapped around those
modules never runs and gets no gradient.

When every such adapter is plain LoRA (one active adapter, no dropout, no DoRA / variant, no LoRA
bias, dense ``nn.Linear`` base) the fused kernel is kept and receives ``W + delta`` instead of ``W``,
with ``delta`` from PEFT's own ``get_delta_weight`` built inside autograd, so lora_A / lora_B get
exact gradients at fused-kernel memory and speed. Any other adapter routes that call to the mixer's
own non-fused path, which calls the modules. Mixers without raw-weight adapters are unchanged.
"""

import functools
import inspect
import sys
import warnings

import torch

__all__ = ["patch_mamba_fused_lora"]

# Mixer leaves a fused kernel reads as `.weight` instead of calling the module.
_RAW_WEIGHT_LEAVES = ("x_proj", "dt_proj", "out_proj")
# Fused entry points in the modeling modules; in transformers 5 each mixer falls back to its torch
# path when one returns None.
_FUSED_FNS = (
    "mamba_inner_fn",
    "mamba_split_conv1d_scan_combined",
    "mamba2_split_conv1d_scan_combined",
)
_MARK = "_unsloth_mamba_fused_lora"
_force_slow = [0]
# Stack of {id(base weight): (base weight, merged weight)} for the mixer forward in flight.
_merged_stack = []
_warned_dt_proj = [False]


def _has_active_adapter(module):
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return False
    if not isinstance(module, BaseTunerLayer):
        return False
    # Merged adapters already live in base_layer.weight, which is what the kernel reads.
    if module.disable_adapters or module.merged:
        return False
    for layer_name in module.adapter_layer_names:
        layer = getattr(module, layer_name, None)
        if layer is not None and any(name in layer for name in module.active_adapters):
            return True
    return False


def mamba_raw_weight_adapters(mixer):
    """Names of the mixer's raw-weight leaves that carry an active, unmerged adapter."""
    return [name for name in _RAW_WEIGHT_LEAVES if _has_active_adapter(mixer._modules.get(name))]


def _merged_weight(module):
    """``W + delta`` when that is exactly what the LoRA module computes, else None."""
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:
        return None
    if not isinstance(module, LoraLinear):
        return None
    active = [name for name in module.active_adapters if name in module.lora_A]
    if len(active) != 1:
        return None
    # Other adapter layers (embedding LoRA, a mixed tuner) for the active adapter are not covered.
    for layer_name in module.adapter_layer_names:
        if layer_name not in ("lora_A", "lora_B"):
            layer = getattr(module, layer_name, None)
            if layer is not None and active[0] in layer:
                return None
    name = active[0]
    if name in getattr(module, "lora_variant", {}) or getattr(module, "use_dora", {}).get(name):
        return None
    if getattr(module, "fan_in_fan_out", False):
        return None
    if getattr(module.lora_B[name], "bias", None) is not None:
        return None
    dropout = module.lora_dropout[name]
    if not isinstance(dropout, (torch.nn.Identity, torch.nn.Dropout)):
        return None
    if isinstance(dropout, torch.nn.Dropout) and dropout.training and dropout.p > 0:
        return None
    base = module.get_base_layer()
    # Exact nn.Linear only: bitsandbytes / other quantized subclasses do not hold a dense weight.
    if type(base) is not torch.nn.Linear:
        return None
    weight = base.weight
    if not weight.is_floating_point() or weight.is_meta:
        return None
    return weight + module.get_delta_weight(name).to(weight.dtype)


def _merged_weights(mixer, leaves):
    merged = {}
    for leaf in leaves:
        module = mixer._modules[leaf]
        weight = _merged_weight(module)
        if weight is None:
            return None
        base = module.get_base_layer().weight
        merged[id(base)] = (base, weight)
    return merged


def _swap(value, merged):
    if isinstance(value, torch.Tensor):
        entry = merged.get(id(value))
        if entry is not None and entry[0] is value:
            return entry[1]
    return value


def _gate(fn):
    if getattr(fn, _MARK, False):
        return fn

    @functools.wraps(fn)
    def gated(*args, **kwargs):
        if _force_slow[0]:
            return None
        if _merged_stack:
            merged = _merged_stack[-1]
            args = tuple(_swap(value, merged) for value in args)
            kwargs = {key: _swap(value, merged) for key, value in kwargs.items()}
        return fn(*args, **kwargs)

    setattr(gated, _MARK, True)
    return gated


def _is_mamba_mixer(module):
    name = type(module).__name__
    if not (name.endswith("Mixer") or hasattr(module, "ssm_state_size")):
        return False
    return any(module._modules.get(leaf) is not None for leaf in _RAW_WEIGHT_LEAVES)


def _calls_fused(fn):
    code = getattr(inspect.unwrap(fn), "__code__", None)
    return code is not None and any(name in code.co_names for name in _FUSED_FNS)


def _patch_class(cls):
    forward = cls.forward
    if getattr(forward, _MARK, False):
        return True
    params = list(inspect.signature(forward).parameters)
    if "cache_params" not in params or len(params) < 2:
        return False
    # Position of cache_params within *args, after (self, hidden_states).
    cache_index = params.index("cache_params") - 2

    # transformers 4.x: forward picks cuda_kernels_forward or a same-signature torch path.
    slow = getattr(cls, "slow_forward", None) or getattr(cls, "torch_forward", None)
    cuda_forward = getattr(cls, "cuda_kernels_forward", None)
    if callable(cuda_forward) and callable(slow):
        if not _calls_fused(cuda_forward):
            return False
        if list(inspect.signature(slow).parameters)[2:] != params[2:]:
            return False
        legacy = True
    else:
        # transformers 5: one forward whose fused branch short-circuits unless the kernel returns None.
        if not _calls_fused(forward):
            return False
        legacy = False
    # Both lines look the fused function up in the modeling module's globals at call time.
    module_globals = getattr(sys.modules.get(cls.__module__), "__dict__", {})
    names = [name for name in _FUSED_FNS if callable(module_globals.get(name))]
    if not names and not legacy:
        return False
    for name in names:
        module_globals[name] = _gate(module_globals[name])

    def non_fused(self, hidden_states, *args, **kwargs):
        if legacy:
            return slow(self, hidden_states, *args, **kwargs)
        if "dt_proj" in mamba_raw_weight_adapters(self) and not _warned_dt_proj[0]:
            # transformers 5's torch path reads dt_proj.weight raw as well.
            _warned_dt_proj[0] = True
            warnings.warn(
                "Unsloth: a dt_proj LoRA adapter that uses dropout, DoRA, several active adapters "
                "or a quantized base is not applied by transformers' Mamba torch path; use "
                "lora_dropout = 0 or leave dt_proj out of target_modules.",
                stacklevel = 3,
            )
        _force_slow[0] += 1
        try:
            return forward(self, hidden_states, *args, **kwargs)
        finally:
            _force_slow[0] -= 1

    @functools.wraps(forward)
    def forward_keep_adapters(self, hidden_states, *args, **kwargs):
        if self.training:
            if "cache_params" in kwargs:
                cache_params = kwargs["cache_params"]
            else:
                cache_params = args[cache_index] if 0 <= cache_index < len(args) else None
            leaves = mamba_raw_weight_adapters(self) if cache_params is None else None
            if leaves:
                merged = _merged_weights(self, leaves) if names else None
                if merged is None:
                    return non_fused(self, hidden_states, *args, **kwargs)
                # Fused kernel kept; it receives W + delta in place of each adapted W.
                _merged_stack.append(merged)
                try:
                    return forward(self, hidden_states, *args, **kwargs)
                finally:
                    _merged_stack.pop()
        return forward(self, hidden_states, *args, **kwargs)

    setattr(forward_keep_adapters, _MARK, True)
    cls.forward = forward_keep_adapters
    return True


def patch_mamba_fused_lora(model):
    """Keep LoRA on Mamba mixers whose fused kernels read x_proj / dt_proj / out_proj raw."""
    if not isinstance(model, torch.nn.Module):
        return model
    seen = set()
    for module in model.modules():
        cls = type(module)
        if cls in seen or not _is_mamba_mixer(module):
            continue
        seen.add(cls)
        try:
            _patch_class(cls)
        except Exception:
            pass
    return model
