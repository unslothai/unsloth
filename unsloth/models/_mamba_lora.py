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
non-fused path when that path calls more of the adapted modules. An adapter neither path applies
warns once. Mixers without raw-weight adapters are unchanged.

The torch path (transformers 5 always, 4.x ``cuda_kernels_forward`` outside fused training) also
reads ``self.dt_proj.weight @ time_step`` raw, in eval and cached decode too. Adapted leaves therefore
get a ``weight`` property that, while their mixer's forward runs, returns ``W + delta`` (built inside
autograd and outside autocast, rebuilt on every read so no weight write can leave it stale), so
unmerged eval and generate match ``merge_and_unload`` and ``disable_adapter()`` gives the base model.
"""

import copy
import copyreg
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
        # [(adapted leaf, W + delta or None = build on read, "Mixer.leaf", batch size or None)]
        self.raw = []


_routing = _Routing()
_warned = set()
_TUNER_LAYER = None
_RAW_READ_CLASSES = {}


class _Frame:
    __slots__ = ("merged", "applied")

    def __init__(self, merged):
        # [(weight the forward passes, W + delta handed to the kernel instead, leaf)]; matched by
        # identity, not id(), which torch.compile cannot trace for a tensor built in the forward.
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


def _tuner_layer_class():
    global _TUNER_LAYER
    if _TUNER_LAYER is None:
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
        except Exception:
            return ()
        _TUNER_LAYER = BaseTunerLayer
    return _TUNER_LAYER


def mamba_raw_weight_adapters(mixer):
    """Names of the mixer's raw-weight leaves that carry an active, unmerged adapter."""
    modules, tuner = mixer._modules, _TUNER_LAYER or _tuner_layer_class()
    leaves = [name for name in _RAW_WEIGHT_LEAVES if isinstance(modules.get(name), tuner)]
    return [name for name in leaves if _has_active_adapter(modules[name])] if leaves else leaves


def _merged_weight(
    module,
    allow_multiple = False,
    names = None,
):
    """``W + delta`` when that is exactly what the LoRA module computes, else None.

    ``names`` replaces the active adapters (one row of a PEFT mixed-adapter batch).
    """
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:
        return None
    if not isinstance(module, LoraLinear):
        return None
    if names is None:
        names = module.active_adapters
    active = [name for name in names if name in module.lora_A]
    if not active or (len(active) != 1 and not allow_multiple):
        return None
    for name in active:
        # Other adapter layers (embedding LoRA, a mixed tuner) for the active adapter are not covered.
        for layer_name in module.adapter_layer_names:
            if layer_name not in ("lora_A", "lora_B"):
                layer = getattr(module, layer_name, None)
                if layer is not None and name in layer:
                    return None
        if name in getattr(module, "lora_variant", {}) or getattr(module, "use_dora", {}).get(name):
            return None
        if getattr(module.lora_B[name], "bias", None) is not None:
            return None
        dropout = module.lora_dropout[name]
        if not isinstance(dropout, (torch.nn.Identity, torch.nn.Dropout)):
            return None
        if isinstance(dropout, torch.nn.Dropout) and dropout.training and dropout.p > 0:
            return None
    if getattr(module, "fan_in_fan_out", False):
        return None
    base = module.get_base_layer()
    # Exact nn.Linear only: bitsandbytes / other quantized subclasses do not hold a dense weight.
    if type(base) is not torch.nn.Linear:
        return None
    weight = base.weight
    if not weight.is_floating_point() or weight.is_meta:
        return None
    # Same order as PEFT's merge (base += delta per adapter), so merged and unmerged match bitwise.
    for name in active:
        weight = weight + module.get_delta_weight(name).to(weight.dtype)
    return weight


def _full_precision_merged(module, names = None):
    # Built outside autocast, as merge_and_unload does, so autocast eval matches the merged model.
    device_type = module.get_base_layer().weight.device.type
    try:
        # Autocast off already: same tensor without the context's cost, which shows on decode reads.
        context = None
        if torch.compiler.is_compiling() or torch.is_autocast_enabled(device_type):
            context = torch.autocast(device_type, enabled = False)
    except Exception:
        context = None
    if context is None:
        return _merged_weight(module, allow_multiple = True, names = names)
    with context:
        return _merged_weight(module, allow_multiple = True, names = names)


def _mixed_batch_adapter_names(module):
    """Per-row adapter names of a PEFT mixed-adapter batch (``adapter_names=``), else None.

    PEFT delivers them through a forward pre-hook on each LoRA layer, which a raw ``.weight``
    read never runs.
    """
    for hook in module._forward_pre_hooks.values():
        if (
            isinstance(hook, functools.partial)
            and getattr(hook.func, "__name__", "") == "_adapter_names_pre_forward_hook"
        ):
            names = hook.keywords.get("adapter_names")
            if names is not None:
                return list(names)
    return None


def _per_row_weight(
    module,
    names,
    label,
    batch = None,
):
    """[batch, out, in] effective weights, one per row of a mixed-adapter batch.

    ``W @ x`` with a [batch, in, seq] input broadcasts per row, so each row sees its own adapter and
    ``"__base__"`` / unknown names see the base weight, as PEFT's mixed-batch forward gives them.

    PEFT repeats the names only for ``num_beams`` (peft/tuners/lora/model.py
    ``_enable_peft_forward_hooks``); ``generate`` also expands each prompt ``num_return_sequences``
    times with ``repeat_interleave``, so a running batch that is a multiple of the names gets them
    repeated in that same order. Any other size cannot be routed and raises.
    """
    if batch is not None and batch != len(names):
        if not names or batch % len(names):
            raise ValueError(
                f"Unsloth: {label} got {len(names)} adapter_names for a batch of {batch} rows; "
                "pass one adapter name per input row."
            )
        repeat = batch // len(names)
        names = [name for name in names for _ in range(repeat)]
    base = module.get_base_layer().weight
    weights = {}
    for name in dict.fromkeys(names):
        if name == "__base__" or name not in module.lora_A:
            weights[name] = base
            continue
        weight = _full_precision_merged(module, names = [name])
        if weight is None:
            if not torch.compiler.is_compiling():
                _warn_raw_read(label)
            weight = base
        weights[name] = weight
    return torch.stack([weights[name] for name in names])


def _warn_raw_read(label):
    if ("raw_read", label) in _warned:
        return
    _warned.add(("raw_read", label))
    warnings.warn(
        f"Unsloth: the adapter on {label} is not applied here: this Mamba forward reads that "
        "weight raw and only plain LoRA (no DoRA, no LoRA bias, no active dropout, dense base) can "
        "be folded into it. Merge the adapter (merge_and_unload) for inference.",
        stacklevel = 3,
    )


def _raw_read_class(cls):
    """Subclass of an adapter layer whose ``weight`` is ``W + delta`` while its mixer forward runs."""
    sub = _RAW_READ_CLASSES.get(cls)
    if sub is not None:
        return sub

    def weight(self):
        # Per thread: nn.DataParallel replicas share their leaves' __dict__ contents.
        for module, effective, label, batch in reversed(_routing.raw):
            if module is not self:
                continue
            if effective is None:
                names = _mixed_batch_adapter_names(self)
                if names is not None:
                    return _per_row_weight(self, names, label, batch)
                # Never cached: hotswap_adapter / FlashAdamW `.data` writes skip version counters.
                effective = _full_precision_merged(self)
            if effective is not None:
                return effective
            if not torch.compiler.is_compiling():
                _warn_raw_read(label)
            break
        return super(sub, self).weight

    def __reduce_ex__(self, protocol):
        # Pickle as the plain adapter layer so it loads without Unsloth.
        return copyreg._reconstructor, (cls, object, None), self.__dict__.copy()

    # copy / deepcopy keep this class: a trace cannot install it and would read the base weight.
    def __copy__(self):
        new = object.__new__(type(self))
        new.__setstate__(self.__dict__.copy())
        return new

    def __deepcopy__(self, memo):
        new = object.__new__(type(self))
        memo[id(self)] = new
        new.__setstate__(copy.deepcopy(self.__dict__, memo))
        return new

    qualname = "_RawRead_" + "_".join(f"{cls.__module__}.{cls.__qualname__}".split("."))
    sub = type(
        cls.__name__,
        (cls,),
        {
            "weight": property(weight),
            "__reduce_ex__": __reduce_ex__,
            "__copy__": __copy__,
            "__deepcopy__": __deepcopy__,
            "__module__": __name__,
            "__qualname__": qualname,
            _MARK: True,
        },
    )
    globals()[qualname] = sub  # resolvable by qualname (torch.compile, introspection)
    _RAW_READ_CLASSES[cls] = sub
    return sub


def _install_raw_read(mixer, leaf):
    """Give the adapted leaf the effective-weight ``weight``; False when it cannot take one."""
    module = mixer._modules.get(leaf)
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return False
    if not isinstance(module, BaseTunerLayer):
        return False
    if getattr(type(module), _MARK, False):
        return True
    if torch.compiler.is_compiling():
        # Dynamo cannot trace the class swap; skipping it would silently drop the adapter, so fail loudly.
        raise RuntimeError(
            f"Unsloth: {type(mixer).__name__}.{leaf} has an active LoRA adapter that was attached "
            "after Unsloth prepared this Mamba model, so a compiled forward would read its base "
            "weight and drop the adapter. Run one eager forward, or call "
            "unsloth.models._mamba_lora.patch_mamba_fused_lora(model), before torch.compile."
        )
    if not isinstance(inspect.getattr_static(type(module), "weight", None), property):
        return False
    module.__class__ = _raw_read_class(type(module))
    return True


def _prepare_leaves(mixer):
    """Install the effective-weight class on every adapter leaf of a mixer, active or not."""
    tuner = _TUNER_LAYER or _tuner_layer_class()
    modules = mixer.__dict__.get("_modules") or {}
    for leaf in _RAW_WEIGHT_LEAVES:
        if tuner and isinstance(modules.get(leaf), tuner):
            _install_raw_read(mixer, leaf)


def _swap(value, merged, used):
    if isinstance(value, torch.Tensor):
        for source, weight, leaf in merged:
            if source is value:
                used.append(leaf)
                return weight
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
        leaves = mamba_raw_weight_adapters(self)
        if not leaves:
            return forward(self, hidden_states, *args, **kwargs)
        if "cache_params" in kwargs:
            cache_params = kwargs["cache_params"]
        else:
            cache_params = args[cache_index] if 0 <= cache_index < len(args) else None
        compiling = torch.compiler.is_compiling()
        readable = [leaf for leaf in leaves if _install_raw_read(self, leaf)]
        raw, depth = _routing.raw, len(_routing.raw)
        if not (self.training and cache_params is None):
            # Eval / cached decode: the torch path reads dt_proj.weight raw; serve W + delta lazily.
            batch = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) else None
            for leaf in readable:
                raw.append((self._modules[leaf], None, f"{cls.__name__}.{leaf}", batch))
            try:
                return forward(self, hidden_states, *args, **kwargs)
            finally:
                while len(raw) > depth:
                    raw.pop()
        names = _gate_module_fns(cls)
        modules = {leaf: self._modules[leaf] for leaf in leaves}
        weights = {leaf: _merged_weight(modules[leaf]) for leaf in leaves} if names else {}
        fused_cover = [leaf for leaf in leaves if weights.get(leaf) is not None]
        # Leave the fused kernel only when the non-fused path applies more adapters.
        use_slow = sum(leaf in called for leaf in leaves) > len(fused_cover)
        merged = []
        frame = _Frame(merged)
        for leaf in fused_cover:
            module = modules[leaf]
            if leaf in readable:
                # `.weight` now returns this tensor, so the torch path's raw read applies it too.
                raw.append((module, weights[leaf], f"{cls.__name__}.{leaf}", None))
                frame.applied.add(leaf)
            if not use_slow:
                base = module.get_base_layer().weight
                merged.append((base, weights[leaf], leaf))
                merged.append((weights[leaf], weights[leaf], leaf))
        hooks = []
        if not compiling:
            hooks = [
                module.register_forward_pre_hook(
                    functools.partial(_mark_applied, frame.applied, leaf)
                )
                for leaf, module in modules.items()
                if leaf not in frame.applied
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
            while len(raw) > depth:
                raw.pop()
        if not compiling:
            _warn_not_applied(cls, [leaf for leaf in leaves if leaf not in frame.applied])
        return out

    setattr(forward_keep_adapters, _MARK, True)
    cls.forward = forward_keep_adapters
    _prepare_on_assign(cls)
    return True


def _prepare_on_assign(cls):
    # PEFT attaches adapters via setattr on the mixer: install the class there, before any compile.
    original = cls.__setattr__
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def __setattr__(self, name, value):
        original(self, name, value)
        if name in _RAW_WEIGHT_LEAVES and not torch.compiler.is_compiling():
            _prepare_leaves(self)

    setattr(__setattr__, _MARK, True)
    cls.__setattr__ = __setattr__


def patch_mamba_fused_lora(model):
    """Keep LoRA on Mamba mixers whose fused kernels read x_proj / dt_proj / out_proj raw."""
    if not isinstance(model, torch.nn.Module):
        return model
    patched = {}
    for module in model.modules():
        cls = type(module)
        if not _is_mamba_mixer(module):
            continue
        if cls not in patched:
            try:
                patched[cls] = _patch_class(cls)
            except Exception:
                patched[cls] = False
        if patched[cls]:
            _prepare_leaves(module)
    return model
