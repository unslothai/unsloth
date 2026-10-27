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
exact gradients at fused-kernel memory and speed. Otherwise the call routes to the mixer's own
non-fused path when that path calls more of the adapted modules (transformers 5's Mamba torch path
still reads dt_proj.weight raw, so it does not help a dt_proj adapter). An adapter neither path
applies warns once. Mixers without raw-weight adapters are unchanged.
"""

import functools
import inspect
import sys
import threading
import warnings

import torch

__all__ = ["patch_mamba_fused_lora"]

# Mixer leaves a fused kernel reads as `.weight` instead of calling the module.
_RAW_WEIGHT_LEAVES = ("x_proj", "dt_proj", "out_proj")
# Fused entry points; transformers 5 falls back to its torch path when one returns None.
_FUSED_FNS = (
    "mamba_inner_fn",
    "mamba_split_conv1d_scan_combined",
    "mamba2_split_conv1d_scan_combined",
)
_MARK = "_unsloth_mamba_fused_lora"


class _Routing(threading.local):
    # Per thread: nn.DataParallel replicas run their forwards concurrently.
    def __init__(self):
        self.force_slow = 0
        self.stack = []


_routing = _Routing()
_warned = set()


class _Frame:
    __slots__ = ("merged", "applied")

    def __init__(self, merged):
        # {id(base weight): (base weight, W + delta, leaf)} handed to the fused kernel.
        self.merged = merged
        # Leaves whose adapter ran: merged into a kernel call that returned, or called as a module.
        self.applied = set()


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


def _swap(value, merged, used):
    if isinstance(value, torch.Tensor):
        entry = merged.get(id(value))
        if entry is not None and entry[0] is value:
            used.append(entry[2])
            return entry[1]
    return value


def _gate(fn):
    if getattr(fn, _MARK, False):
        return fn

    @functools.wraps(fn)
    def gated(*args, **kwargs):
        if _routing.force_slow:
            return None
        stack = _routing.stack
        if not stack or not stack[-1].merged:
            return fn(*args, **kwargs)
        frame, used = stack[-1], []
        args = tuple(_swap(value, frame.merged, used) for value in args)
        kwargs = {key: _swap(value, frame.merged, used) for key, value in kwargs.items()}
        out = fn(*args, **kwargs)
        if out is not None:
            frame.applied.update(used)
        return out

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


def _leaves_called(fn):
    """Raw-weight leaves the non-fused path calls as modules, so an adapter on them runs there.

    transformers 5's MambaMixer torch path computes ``self.dt_proj.weight @ time_step``, so routing a
    dt_proj adapter there would trade the fused kernel for nothing. Unknown source: assume all.
    """
    try:
        source = inspect.getsource(inspect.unwrap(fn))
    except (OSError, TypeError):
        return frozenset(_RAW_WEIGHT_LEAVES)
    return frozenset(leaf for leaf in _RAW_WEIGHT_LEAVES if f"self.{leaf}(" in source)


def _mark_applied(applied, leaf, module, args):
    applied.add(leaf)


def _warn_not_applied(cls, leaves):
    leaves = [leaf for leaf in leaves if (cls, leaf) not in _warned]
    if not leaves:
        return
    _warned.update((cls, leaf) for leaf in leaves)
    names = ", ".join(f"{cls.__name__}.{leaf}" for leaf in leaves)
    warnings.warn(
        f"Unsloth: the LoRA adapter on {names} is not applied in this Mamba training forward and "
        "will not train. The fused kernel reads that weight raw and only plain LoRA (one active "
        "adapter, lora_dropout = 0, no DoRA, no LoRA bias, dense base) can be merged into it, and "
        "the non-fused path does not call it either. Use lora_dropout = 0 or remove it from "
        "target_modules.",
        stacklevel = 3,
    )


def _named_params(fn):
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _keywords_taken(fn):
    """Keyword names ``fn`` accepts, or None when it takes **kwargs."""
    params = inspect.signature(fn).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params if p.kind is not p.VAR_POSITIONAL}


def _gate_module_fns(cls):
    # Both lines look the fused function up in the modeling module's globals at call time, and
    # transformers 5.5 / 5.10 MambaMixer.__init__ reassigns them: re-gate on every use.
    module_globals = getattr(sys.modules.get(cls.__module__), "__dict__", {})
    names = [name for name in _FUSED_FNS if callable(module_globals.get(name))]
    for name in names:
        module_globals[name] = _gate(module_globals[name])
    return names


def _patch_class(cls):
    forward = cls.forward
    if getattr(forward, _MARK, False):
        _gate_module_fns(cls)
        return True
    params = list(inspect.signature(forward).parameters)
    if "cache_params" not in params or len(params) < 2:
        return False
    # Position of cache_params within *args, after (self, hidden_states).
    cache_index = params.index("cache_params") - 2

    # transformers 4.x: forward picks cuda_kernels_forward or a same-signature torch path.
    slow_takes = None
    slow = getattr(cls, "slow_forward", None) or getattr(cls, "torch_forward", None)
    cuda_forward = getattr(cls, "cuda_kernels_forward", None)
    if callable(cuda_forward) and callable(slow):
        if not _calls_fused(cuda_forward):
            return False
        # transformers 5.5 / 5.10 forward adds **kwargs that it drops before calling slow_forward.
        if _named_params(slow)[2:] != _named_params(forward)[2:]:
            return False
        slow_takes = _keywords_taken(slow)
        legacy = True
    else:
        # transformers 5: one forward whose fused branch short-circuits unless the kernel returns None.
        if not _calls_fused(forward):
            return False
        legacy = False
    if not _gate_module_fns(cls) and not legacy:
        return False
    called = _leaves_called(slow if legacy else forward)

    def non_fused(self, hidden_states, *args, **kwargs):
        if legacy:
            if slow_takes is not None:
                kwargs = {k: v for k, v in kwargs.items() if k in slow_takes}
            return slow(self, hidden_states, *args, **kwargs)
        _routing.force_slow += 1
        try:
            return forward(self, hidden_states, *args, **kwargs)
        finally:
            _routing.force_slow -= 1

    @functools.wraps(forward)
    def forward_keep_adapters(self, hidden_states, *args, **kwargs):
        if self.training:
            if "cache_params" in kwargs:
                cache_params = kwargs["cache_params"]
            else:
                cache_params = args[cache_index] if 0 <= cache_index < len(args) else None
            leaves = mamba_raw_weight_adapters(self) if cache_params is None else None
            if leaves:
                names = _gate_module_fns(cls)
                modules = {leaf: self._modules[leaf] for leaf in leaves}
                weights = {leaf: _merged_weight(modules[leaf]) for leaf in leaves} if names else {}
                fused_cover = [leaf for leaf in leaves if weights.get(leaf) is not None]
                # Leave the fused kernel only when the non-fused path applies more adapters.
                use_slow = sum(leaf in called for leaf in leaves) > len(fused_cover)
                merged = {}
                if not use_slow:
                    for leaf in fused_cover:
                        base = modules[leaf].get_base_layer().weight
                        merged[id(base)] = (base, weights[leaf], leaf)
                frame = _Frame(merged)
                hooks = [
                    module.register_forward_pre_hook(
                        functools.partial(_mark_applied, frame.applied, leaf)
                    )
                    for leaf, module in modules.items()
                ]
                _routing.stack.append(frame)
                try:
                    if use_slow:
                        out = non_fused(self, hidden_states, *args, **kwargs)
                    else:
                        # Fused kernel kept; it receives W + delta in place of each adapted W.
                        out = forward(self, hidden_states, *args, **kwargs)
                finally:
                    _routing.stack.pop()
                    for hook in hooks:
                        hook.remove()
                _warn_not_applied(cls, [leaf for leaf in leaves if leaf not in frame.applied])
                return out
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
