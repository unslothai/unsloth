# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Keep LoRA adapters on Mamba projections that the fused training kernels read as raw weights.

In training (no cache) transformers' Mamba mixers hand ``self.x_proj.weight``, ``self.dt_proj.weight``
and ``self.out_proj.weight`` (Mamba2-style mixers: ``self.out_proj.weight``) straight to
``mamba_inner_fn`` / ``mamba_split_conv1d_scan_combined``, so a PEFT adapter wrapped around those
modules never runs and gets no gradient. When one of them carries an active adapter we route that
call to the mixer's own non-fused path, which calls the modules; otherwise the fused path is unchanged.
"""

import functools
import inspect
import sys

import torch

__all__ = ["patch_mamba_fused_lora"]

# Mixer leaves a fused kernel reads as `.weight` instead of calling the module.
_RAW_WEIGHT_LEAVES = ("x_proj", "dt_proj", "out_proj")
# transformers 5 fused entry points; each mixer falls back to its torch path when one returns None.
_FUSED_FNS = (
    "mamba_inner_fn",
    "mamba_split_conv1d_scan_combined",
    "mamba2_split_conv1d_scan_combined",
)
_MARK = "_unsloth_mamba_fused_lora"
_force_slow = [0]


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


def _gate(fn):
    if getattr(fn, _MARK, False):
        return fn

    @functools.wraps(fn)
    def gated(*args, **kwargs):
        if _force_slow[0]:
            return None
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
        module_globals = getattr(sys.modules.get(cls.__module__), "__dict__", {})
        names = [name for name in _FUSED_FNS if callable(module_globals.get(name))]
        if not names or not _calls_fused(forward):
            return False
        for name in names:
            module_globals[name] = _gate(module_globals[name])
        legacy = False

    @functools.wraps(forward)
    def forward_keep_adapters(self, hidden_states, *args, **kwargs):
        if self.training:
            if "cache_params" in kwargs:
                cache_params = kwargs["cache_params"]
            else:
                cache_params = args[cache_index] if 0 <= cache_index < len(args) else None
            if cache_params is None and mamba_raw_weight_adapters(self):
                if legacy:
                    return slow(self, hidden_states, *args, **kwargs)
                _force_slow[0] += 1
                try:
                    return forward(self, hidden_states, *args, **kwargs)
                finally:
                    _force_slow[0] -= 1
        return forward(self, hidden_states, *args, **kwargs)

    setattr(forward_keep_adapters, _MARK, True)
    cls.forward = forward_keep_adapters
    return True


def patch_mamba_fused_lora(model):
    """Route Mamba mixers whose raw-weight projections carry adapters off the fused kernel."""
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
