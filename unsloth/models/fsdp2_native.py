# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 for every Unsloth model, built directly on `torch.distributed.fsdp.fully_shard`.

    for unit in find_shard_units(model): fully_shard(unit, mp_policy = mp)
    fully_shard(model, mp_policy = mp)

with `MixedPrecisionPolicy(param_dtype = compute dtype, reduce_dtype = float32)`. Neither accelerate's
nor transformers' FSDP code wraps the model: Unsloth shards it before the optimizer exists and the
Trainer / accelerate only see an already sharded `FSDPModule` (no second wrap, no DDP).

Opt in with `UNSLOTH_FSDP2=1` (or `native`) under torchrun (WORLD_SIZE > 1).
`UNSLOTH_FSDP2_BACKEND=accelerate` (or an `accelerate launch` FSDP config) keeps the accelerate
path in `unsloth/models/fsdp2.py`. Nothing changes without the flag.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import os
import sys
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "FSDP2Config",
    "STORAGE_DTYPE",
    "find_shard_units",
    "shard_model",
    "native_fsdp2_requested",
    "is_native_fsdp2",
    "install_native_fsdp2_hooks",
    "native_fsdp2_trainer_setup",
    "native_clip_grad_norm_",
    "native_full_state_dict",
    "native_fsdp2_unsharded",
    "register_shard_context",
]

_PATCHED = "_unsloth_fsdp2_native_patched"
_NATIVE_FLAG = "_unsloth_fsdp2_native"
# Leaf module types that are never a shard unit on their own, even when repeated in a ModuleList.
_LEAF_TYPES = (
    nn.Linear,
    nn.Embedding,
    nn.LayerNorm,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Dropout,
    nn.Identity,
)


@dataclasses.dataclass
class FSDP2Config:
    param_dtype: Optional[torch.dtype] = None  # None = model compute dtype (bf16; fp16 on T4)
    reduce_dtype: torch.dtype = torch.float32
    reshard_after_forward: Union[bool, int] = True  # last block always False
    cpu_offload: bool = False  # CPUOffloadPolicy
    mesh: Optional[object] = None  # DeviceMesh; None = 1-D over WORLD_SIZE
    # Knobs for perf tuning (defaults = the plan's contract behaviour).
    # The last unit's reshard_after_forward (backward re-gathers it first).
    reshard_last_block: bool = False
    # unit_reshard_after_forward(index, unit) -> bool | int | None (None = the default), per unit.
    unit_reshard_after_forward: Optional[Callable] = None
    # False: frozen weights (LoRA / QLoRA base) stay replicated and are never gathered.
    shard_frozen: bool = True
    # Extra parameters to leave replicated (trainable ones get their gradients averaged).
    ignored_params: Optional[set] = None
    # Explicit shard units instead of find_shard_units(model).
    units: Optional[list] = None
    # All-gather the next N units during a unit's forward (0 = FSDP2's implicit prefetch).
    forward_prefetch: int = 0
    # Prefetch the previous N units in backward (0 = FSDP2 default of 1).
    backward_prefetch: int = 0
    # Passed to fully_shard (per-parameter Shard(dim) placement).
    shard_placement_fn: Optional[Callable] = None
    # True: a small minority-dtype trainable group (Unsloth full FT's fp32 norms next to bf16 linears) stays
    # replicated in its own dtype with DDP-style gradient averaging (matches 1 GPU). False: every parameter
    # becomes a DTensor and mixed units get float32 masters (measured 0.25 loss drift vs 1 GPU by step 10).
    replicate_minority_dtype: bool = True

    @classmethod
    def from_env(cls):
        """Defaults, overridden by UNSLOTH_FSDP2_PARAM_DTYPE (bfloat16 / float16 / float32 / storage =
        gather each weight in its own dtype), UNSLOTH_FSDP2_REDUCE_DTYPE, UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD
        (0 / 1) and UNSLOTH_FSDP2_CPU_OFFLOAD (0 / 1)."""
        config = cls()
        dtypes = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "storage": STORAGE_DTYPE,
            "none": STORAGE_DTYPE,
        }
        value = os.environ.get("UNSLOTH_FSDP2_PARAM_DTYPE", "").strip().lower()
        if value:
            if value not in dtypes:
                raise ValueError(
                    f"Unsloth: UNSLOTH_FSDP2_PARAM_DTYPE={value} is not one of {sorted(dtypes)}."
                )
            config.param_dtype = dtypes[value]
        value = os.environ.get("UNSLOTH_FSDP2_REDUCE_DTYPE", "").strip().lower()
        if value:
            if value not in dtypes or dtypes[value] is STORAGE_DTYPE:
                raise ValueError(f"Unsloth: UNSLOTH_FSDP2_REDUCE_DTYPE={value} is not a dtype.")
            config.reduce_dtype = dtypes[value]
        value = os.environ.get("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", "").strip().lower()
        if value:
            config.reshard_after_forward = value not in ("0", "false", "no", "off")
        value = os.environ.get("UNSLOTH_FSDP2_CPU_OFFLOAD", "").strip().lower()
        if value:
            config.cpu_offload = value in ("1", "true", "yes", "on")
        value = os.environ.get("UNSLOTH_FSDP2_SHARD_FROZEN", "").strip().lower()
        if value:
            config.shard_frozen = value not in ("0", "false", "no", "off")
        for name in ("forward_prefetch", "backward_prefetch"):
            value = os.environ.get(f"UNSLOTH_FSDP2_{name.upper()}", "").strip()
            if value:
                setattr(config, name, int(value))
        return config


# `FSDP2Config(param_dtype = STORAGE_DTYPE)`: no gather cast, each weight computes in its own dtype.
STORAGE_DTYPE = "storage"


def native_fsdp2_requested() -> bool:
    """UNSLOTH_FSDP2=1 (or "native") under WORLD_SIZE > 1. UNSLOTH_FSDP2_BACKEND=accelerate, or an
    `accelerate launch` FSDP config (ACCELERATE_USE_FSDP) with no explicit backend, keeps the
    accelerate path. Read from the launcher env only, so every rank agrees."""
    from ..fsdp2_launch import unsloth_fsdp2_native_requested
    return unsloth_fsdp2_native_requested()


def _fsdp_api():
    try:
        from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import (  # torch 2.4 / 2.5
            FSDPModule,
            MixedPrecisionPolicy,
            fully_shard,
        )
    return FSDPModule, MixedPrecisionPolicy, fully_shard


def _fsdp_module_class():
    try:
        return _fsdp_api()[0]
    except ImportError:
        return None


def is_native_fsdp2(model) -> bool:
    if model is None:
        return False
    if getattr(model, _NATIVE_FLAG, False):
        return True
    for attr in ("module", "_orig_mod"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module) and getattr(inner, _NATIVE_FLAG, False):
            return True
    return False


def _rank0():
    try:
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def _print_rank0(message):
    if _rank0():
        print(message, flush = True)


# --------------------------------------------------------------------------------------------------
# Unit discovery
# --------------------------------------------------------------------------------------------------


def _no_split_names(model):
    names = set()
    for module in model.modules():
        for name in getattr(module, "_no_split_modules", None) or ():
            if isinstance(name, str):
                names.add(name)
    return names


def _mro_names(module):
    return {c.__name__ for c in type(module).__mro__}


def _has_params(module):
    return next(module.parameters(), None) is not None


def _is_block_list(container, no_split):
    children = list(container.children())
    if not children:
        return False
    for child in children:
        if isinstance(child, _LEAF_TYPES) or not any(True for _ in child.children()):
            return False
        if not _has_params(child):
            return False
    if len({type(c) for c in children}) == 1:
        return True
    # Hybrid stacks (Jamba: attention + mamba layers) mix classes; accept when the model lists them.
    return all(_mro_names(c) & no_split for c in children)


def find_shard_units(model) -> list:
    """Repeated decoder / vision / audio blocks, outermost first in module order: every
    nn.ModuleList of identical-class blocks (model.layers, language_model.layers,
    vision_tower.*.layers, ...) not inside another unit, then any `_no_split_modules` instance
    outside them as a fallback. MoE experts stay inside their decoder block's unit."""
    no_split = _no_split_names(model)
    units = []
    unit_ids = set()
    inside = set()  # ids of modules inside a chosen unit

    def _take(module):
        units.append(module)
        unit_ids.add(id(module))
        for sub in module.modules():
            inside.add(id(sub))

    for _, module in model.named_modules():
        if id(module) in inside:
            continue
        if isinstance(module, nn.ModuleList) and _is_block_list(module, no_split):
            for child in module.children():
                if id(child) not in inside:
                    _take(child)
    if no_split:
        for _, module in model.named_modules():
            if id(module) in inside or module is model:
                continue
            if _mro_names(module) & no_split and _has_params(module):
                _take(module)
    # Module (pre-)order = forward order for the usual layouts; the fallback units are re-sorted in.
    order = {id(m): i for i, m in enumerate(model.modules())}
    units.sort(key = lambda m: order.get(id(m), 0))
    return units


# --------------------------------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------------------------------


def _device_of(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    for b in model.buffers():
        if b.device.type != "meta":
            return b.device
    return torch.device("cpu")


def _ensure_process_group(device):
    import torch.distributed as dist

    if dist.is_initialized():
        return dist
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend = backend)
    return dist


def _default_mesh(device, dist):
    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh(device.type, (dist.get_world_size(),))


def _floating(p):
    return p.is_floating_point() and not _is_params4bit(p)


def _is_params4bit(p):
    return any(c.__name__ == "Params4bit" for c in type(p).__mro__)


def _compute_dtype(model):
    """bfloat16 / float16 the model computes in (its config / most common 16-bit weight)."""
    for attr in ("config",):
        cfg = getattr(model, attr, None)
        for key in ("torch_dtype", "dtype"):
            value = getattr(cfg, key, None) if cfg is not None else None
            if isinstance(value, str):
                value = getattr(torch, value, None)
            if value in (torch.bfloat16, torch.float16):
                return value
    counts = {}
    for p in model.parameters():
        if p.dtype in (torch.bfloat16, torch.float16) and not _is_params4bit(p):
            counts[p.dtype] = counts.get(p.dtype, 0) + p.numel()
    for p in model.parameters():
        quant_state = getattr(p, "quant_state", None)
        dtype = getattr(quant_state, "dtype", None)
        if dtype in (torch.bfloat16, torch.float16):
            counts[dtype] = counts.get(dtype, 0) + p.numel()
    if counts:
        return max(counts, key = counts.get)
    return None


def _resolve_param_dtype(model, config):
    if config.param_dtype == STORAGE_DTYPE:
        return None
    if config.param_dtype is not None:
        return config.param_dtype
    compute = _compute_dtype(model)
    if compute == torch.float16 and any(
        p.requires_grad and p.dtype == torch.float32 for p in model.parameters()
    ):
        # float16 gathers of float32 masters move the embedding / norm / lm_head backward into
        # float16, where scaled gradients overflow and the GradScaler skips steps one GPU takes;
        # gather the masters as they are and let autocast cast like one GPU does.
        return None
    return compute


def _fsdp2_checks_frozen_dtypes():
    """torch < 2.8 asserts one original dtype over every parameter of a unit, frozen ones included."""
    try:
        import inspect

        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
        return "requires_grad" not in inspect.getsource(FSDPParamGroup._init_mp_dtypes)
    except Exception:
        pass
    try:
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        return (major, minor) < (2, 8)
    except Exception:
        return False


def _supports_ignored_params(fully_shard):
    import inspect
    try:
        return "ignored_params" in inspect.signature(fully_shard).parameters
    except (TypeError, ValueError):
        return False


# A minority dtype group up to this fraction of a unit's trainable elements stays replicated.
_REPLICATE_MINORITY_FRACTION = 0.10


def _unify_dtypes(
    model,
    units,
    param_dtype,
    can_ignore = True,
):
    """FSDP2 wants one original dtype over a unit's trainable parameters (all of them on torch < 2.8).
    Unsloth's full finetuning keeps 16-bit linears next to float32 norms; one GPU updates each in its own
    dtype. To keep that, a small minority dtype group (norms) is left out of the unit: replicated, its
    gradient averaged over ranks like DDP. A large one, or a torch without `ignored_params`, gets float32
    masters instead (gathered in `param_dtype`). Returns (ignored parameter set, number upcast)."""
    frozen_too = _fsdp2_checks_frozen_dtypes()
    groups = [list(u.parameters()) for u in units]
    inside = set()
    for u in units:
        for p in u.parameters():
            inside.add(id(p))
    groups.append([p for p in model.parameters() if id(p) not in inside])
    ignored = set()
    frozen_kept = []
    n = 0
    seen = set()
    for params in groups:
        params = [p for p in params if id(p) not in seen and not seen.add(id(p))]
        trainable = [p for p in params if _floating(p) and p.requires_grad]
        by_dtype = {}
        for p in trainable:
            by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel()
        if len(by_dtype) > 1:
            majority = max(by_dtype, key = by_dtype.get)
            total = sum(by_dtype.values())
            minority = total - by_dtype[majority]
            if can_ignore and minority <= _REPLICATE_MINORITY_FRACTION * total:
                ignored.update(p for p in trainable if p.dtype != majority)
            else:
                for p in trainable:
                    if p.dtype != torch.float32 and type(p) is nn.Parameter:
                        p.data = p.data.to(torch.float32)
                        n += 1
        if can_ignore and param_dtype in (torch.bfloat16, torch.float16):
            # Frozen float32 weights next to 16-bit ones (GDN A_log / dt_bias, fp32 norms, routers): the
            # gather would cast them to 16-bit, which one GPU never does. Keep small ones replicated.
            floats = [p for p in params if _floating(p)]
            frozen32 = [
                p
                for p in floats
                if not p.requires_grad and p.dtype == torch.float32 and p not in ignored
            ]
            total = sum(p.numel() for p in floats)
            if frozen32 and len(frozen32) < len(floats):
                if sum(p.numel() for p in frozen32) <= _REPLICATE_MINORITY_FRACTION * total:
                    ignored.update(frozen32)
                    frozen_kept.extend(frozen32)
        if frozen_too:
            checked = [p for p in params if _floating(p) and p not in ignored]
            if len({p.dtype for p in checked}) > 1:
                for p in checked:
                    if p.dtype != torch.float32 and type(p) is nn.Parameter:
                        p.data = p.data.to(torch.float32)
                        n += 1
    if n:
        _print_rank0(
            f"Unsloth: native FSDP2 needs one dtype per unit, so {n} 16-bit parameters are kept in "
            f"float32 and gathered in {str(param_dtype).replace('torch.', '')} for compute."
        )
    if frozen_kept:
        _print_rank0(
            f"Unsloth: native FSDP2 keeps {len(frozen_kept)} frozen float32 parameters "
            f"({sum(p.numel() for p in frozen_kept)} elements) replicated, so they are not cast to "
            f"{str(param_dtype).replace('torch.', '')} as one GPU never does."
        )
    trainable_kept = [p for p in ignored if p.requires_grad]
    if trainable_kept:
        ignored_trainable = trainable_kept
        _print_rank0(
            f"Unsloth: native FSDP2 keeps {len(ignored_trainable)} trainable "
            f"{'/'.join(sorted({str(p.dtype).replace('torch.', '') for p in ignored_trainable}))} parameters "
            f"({sum(p.numel() for p in ignored_trainable)} elements, e.g. norms) replicated in their own dtype, as one "
            "GPU trains them; their gradients are averaged over ranks."
        )
    return ignored, n


# Context-manager factories entered around the fully_shard calls, in order: factory(model) -> context
# manager yielding an info dict or None. Workstream B (quantized params: Params4bit FSDPParam patches) registers
# here; the default is main's bitsandbytes QLoRA wrap.
_SHARD_CONTEXTS = {}


def register_shard_context(name, factory):
    """Register (or replace, same `name`) a context entered around every native fully_shard pass."""
    _SHARD_CONTEXTS[name] = factory
    return factory


def _qlora_context(model):
    try:
        from .fsdp2_qlora import fsdp2_qlora_wrap
    except Exception:
        return contextlib.nullcontext(None)
    return fsdp2_qlora_wrap(model, None)


register_shard_context("bnb_qlora", _qlora_context)


def _fully_shard(fully_shard, module, **kwargs):
    """fully_shard, retrying with an ABC-compatible FSDPModule for ABC-derived models on torch builds where the
    dynamic `FSDP<cls>` subclass hits a metaclass conflict (verl `maybe_patch_fsdp_module`)."""
    try:
        return fully_shard(module, **kwargs)
    except TypeError as exc:
        if "metaclass" not in str(exc):
            raise
    import abc

    from torch.distributed.fsdp._fully_shard import _fully_shard as fully_shard_module

    original = fully_shard_module.FSDPModule

    class FSDPModuleABC(abc.ABC, original):
        pass

    fully_shard_module.FSDPModule = FSDPModuleABC
    try:
        return fully_shard(module, **kwargs)
    finally:
        fully_shard_module.FSDPModule = original


def _post_shard(model, units, device):
    """Unsloth-side bookkeeping shared with the accelerate path: fused LoRA gate, direct-read units,
    gradient averaging for anything FSDP2 left unmanaged."""
    FSDPModule = _fsdp_module_class()
    try:
        from . import fsdp2 as _acc

        parents = _acc._parents(model)
        shim = type("_Shim", (), {"device": device})()
        # Fused LoRA kernels exist only once llama.py patched the model; never import it here.
        if __package__ and f"{__package__}.llama" in sys.modules:
            _acc._gate_fused_lora(shim, model, parents, FSDPModule, units)
        model._unsloth_fsdp2_direct_read_units = _acc._gather_units_read_directly(
            model, units, parents, FSDPModule
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        _print_rank0(
            f"Unsloth: native FSDP2 post-shard step skipped ({type(exc).__name__}: {exc})."
        )
    try:
        from .fsdp2_save import reduce_unsharded_trainable_grads
        replicated = reduce_unsharded_trainable_grads(model)
    except Exception:
        replicated = []
    if replicated:
        _print_rank0(
            f"Unsloth: native FSDP2 leaves {len(replicated)} trainable parameters unsharded (e.g. "
            f"{replicated[0]}); their gradients are averaged over ranks like DDP."
        )


def _patch_accumulated_grad_guard():
    """torch's FSDPParam.to_accumulated_grad_if_needed reads `_unsharded_param`, which a unit never
    unsharded in this step lacks (a tower skipped on a text batch, a block an MoE route skipped), so float32
    reduce_dtype + accumulation without sync raises AttributeError. Same guard as upstream torch nightly,
    miles and NeMo Automodel; a wrapper, so torch's own logic runs otherwise. Native FSDP2 only."""
    try:
        import inspect

        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return False
    original = getattr(FSDPParam, "to_accumulated_grad_if_needed", None)
    if original is None or getattr(original, _PATCHED, False):
        return original is not None
    try:
        if "getattr" in inspect.getsource(original):
            return True
    except (OSError, TypeError):
        pass

    @functools.wraps(original)
    def to_accumulated_grad_if_needed(self):
        if getattr(self, "_unsharded_param", None) is None:
            return None
        return original(self)

    setattr(to_accumulated_grad_if_needed, _PATCHED, True)
    FSDPParam.to_accumulated_grad_if_needed = to_accumulated_grad_if_needed
    return True


def _set_prefetch(units, config):
    """Explicit prefetch (FSDP2 `set_modules_to_{forward,backward}_prefetch`, torch >= 2.5)."""
    for i, unit in enumerate(units):
        if config.forward_prefetch > 0 and hasattr(unit, "set_modules_to_forward_prefetch"):
            nxt = units[i + 1 : i + 1 + config.forward_prefetch]
            if nxt:
                unit.set_modules_to_forward_prefetch(nxt)
        if config.backward_prefetch > 0 and hasattr(unit, "set_modules_to_backward_prefetch"):
            prev = units[max(0, i - config.backward_prefetch) : i][::-1]
            if prev:
                unit.set_modules_to_backward_prefetch(prev)


def shard_model(model, config: Optional[FSDP2Config] = None):
    """fully_shard(unit) for each unit of `find_shard_units`, then fully_shard(model): the root holds
    embeddings, final norm, lm_head and any leftover parameters, so Unsloth's direct lm_head.weight
    read in the fused CE runs while the root is unsharded. Pass the outermost module the training
    loop calls (the PeftModel for LoRA): peft calls the inner model's `.forward` directly, which
    would skip an inner root's hooks. Idempotent."""
    if getattr(model, _NATIVE_FLAG, False):
        return model
    FSDPModule, MixedPrecisionPolicy, fully_shard = _fsdp_api()
    if isinstance(model, FSDPModule):
        raise RuntimeError(
            "Unsloth: this model is already wrapped by FSDP2 outside Unsloth; native FSDP2 will not "
            "wrap it again."
        )
    config = config or FSDP2Config.from_env()
    device = _device_of(model)
    dist = _ensure_process_group(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mesh = config.mesh if config.mesh is not None else _default_mesh(device, dist)
    param_dtype = _resolve_param_dtype(model, config)
    mp_policy = MixedPrecisionPolicy(param_dtype = param_dtype, reduce_dtype = config.reduce_dtype)
    kwargs = {"mesh": mesh, "mp_policy": mp_policy}
    if config.cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory = device.type == "cuda")
    if config.shard_placement_fn is not None:
        kwargs["shard_placement_fn"] = config.shard_placement_fn
    units = list(config.units) if config.units is not None else find_shard_units(model)
    if not units:
        _print_rank0(
            "Unsloth: native FSDP2 found no repeated blocks in this model; only the root is sharded "
            "(every forward gathers all weights)."
        )
    with contextlib.ExitStack() as stack:
        infos = {
            name: stack.enter_context(factory(model)) for name, factory in _SHARD_CONTEXTS.items()
        }
        qlora = infos.get("bnb_qlora")
        ignored, _ = _unify_dtypes(
            model,
            units,
            param_dtype,
            can_ignore = config.replicate_minority_dtype and _supports_ignored_params(fully_shard),
        )
        ignored = set(ignored)
        if config.ignored_params:
            ignored.update(config.ignored_params)
        if not config.shard_frozen:
            ignored.update(p for p in model.parameters() if not p.requires_grad)
        if ignored:
            if not _supports_ignored_params(fully_shard):
                raise RuntimeError(
                    f"Unsloth: native FSDP2 needs torch with fully_shard(ignored_params) (2.7+) to keep "
                    f"parameters replicated; this is torch {torch.__version__}."
                )
            kwargs["ignored_params"] = ignored
        last = len(units) - 1
        for i, unit in enumerate(units):
            # Backward starts with the last block, so resharding it after forward only re-gathers it.
            reshard = config.reshard_last_block if i == last else config.reshard_after_forward
            if config.unit_reshard_after_forward is not None:
                chosen = config.unit_reshard_after_forward(i, unit)
                reshard = reshard if chosen is None else chosen
            _fully_shard(fully_shard, unit, reshard_after_forward = reshard, **kwargs)
            unit._unsloth_reshard_after_forward = reshard
        # The root stays gathered from forward to backward (FSDP2 does this for a root anyway; explicit).
        _fully_shard(fully_shard, model, reshard_after_forward = False, **kwargs)
        model._unsloth_reshard_after_forward = False
        _set_prefetch(units, config)
    setattr(model, _NATIVE_FLAG, True)
    model._unsloth_fsdp2_blocks = len(units)
    model._unsloth_fsdp2_units = units
    model._unsloth_fsdp2_mp_policy = mp_policy
    model._unsloth_fsdp2_replicated = len(kwargs.get("ignored_params", ()))
    _post_shard(model, units, device)
    _patch_accumulated_grad_guard()
    install_native_fsdp2_hooks()
    classes = sorted({type(u).__name__.removeprefix("FSDP") for u in units})
    note = ""
    if qlora is not None:
        note = f", {qlora.get('params4bit', 0)} 4-bit weights packed as {'/'.join(qlora.get('storage', []))}"
    _print_rank0(
        f"Unsloth: native FSDP2 sharded {len(units)} blocks ({', '.join(classes) or 'none'}) + root over "
        f"{dist.get_world_size()} ranks; gather dtype {str(param_dtype).replace('torch.', '')}, reduce "
        f"dtype {str(config.reduce_dtype).replace('torch.', '')}{note}."
    )
    return model


# --------------------------------------------------------------------------------------------------
# Training-loop pieces: grad clipping, grad accumulation, full state dict
# --------------------------------------------------------------------------------------------------


def native_clip_grad_norm_(
    parameters,
    max_norm,
    norm_type = 2.0,
):
    """Global gradient norm over DTensor (sharded) and plain gradients; clips in place and returns a
    plain float32 tensor (the same value on every rank)."""
    import torch.distributed as dist

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p is not None and p.grad is not None]
    norm_type = float(norm_type)
    device = None
    sharded, replicated = [], []
    for p in params:
        g = p.grad
        if isinstance(g, DTensor):
            sharded.append(g.to_local())
            device = device or g.to_local().device
        else:
            replicated.append(g)
            device = device or g.device
    if device is None:
        return torch.tensor(0.0)
    if device.type == "cpu" and torch.cuda.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
    initialized = dist.is_available() and dist.is_initialized()

    def _local(grads, op):
        if not grads:
            return torch.zeros((), device = device, dtype = torch.float64)
        if norm_type == math.inf:
            return torch.stack([g.detach().abs().max().double().to(device) for g in grads]).max()
        # Per-tensor norms in the gradient dtype, accumulated in float64 (NeMo Automodel does the same).
        norms = torch._foreach_norm([g.detach() for g in grads], norm_type)
        return torch.stack([n.double().to(device) for n in norms]).pow(norm_type).sum()

    if norm_type == math.inf:
        total = torch.maximum(_local(sharded, "max"), _local(replicated, "max"))
        if initialized and sharded:
            dist.all_reduce(total, op = dist.ReduceOp.MAX)
    else:
        total = _local(sharded, "sum")
        if initialized and sharded:
            dist.all_reduce(total, op = dist.ReduceOp.SUM)
        total = (total + _local(replicated, "sum")).pow(1.0 / norm_type)
    clip_coef = float(max_norm) / (total + 1e-6)
    clip_coef = torch.clamp(clip_coef, max = 1.0).float()
    total = total.float()
    grads = sharded + replicated
    if grads:
        by_device = {}
        for g in grads:
            by_device.setdefault((g.device, g.dtype), []).append(g)
        for (dev, _), group in by_device.items():
            torch._foreach_mul_(group, clip_coef.to(dev))
    return total


def native_full_state_dict(
    model,
    trainable_only = False,
    cpu_offload = True,
):
    """Collective: every rank must call. Full (unsharded) state dict on rank 0, {} on the others."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(
        full_state_dict = True,
        cpu_offload = cpu_offload,
        ignore_frozen_params = bool(trainable_only),
    )
    return get_model_state_dict(model, options = options)


@contextlib.contextmanager
def native_fsdp2_unsharded(model):
    """Collective (every rank must enter): gather every unit's whole weights and keep them gathered,
    for code that reads weights without calling the blocks (Unsloth's decode loop, `PeftModel.generate`,
    which skips the root's hooks, merges). Resharded on exit. A no-op for a model not sharded natively."""
    FSDPModule = _fsdp_module_class()
    if FSDPModule is None or not is_native_fsdp2(model):
        yield model
        return
    units = [m for m in model.modules() if isinstance(m, FSDPModule)]
    for unit in units:
        unit.set_reshard_after_forward(False, recurse = False)
    try:
        for unit in units:
            unit.unshard()
        yield model
    finally:
        for unit in units:
            unit.reshard()
            unit.set_reshard_after_forward(
                getattr(unit, "_unsloth_reshard_after_forward", True), recurse = False
            )


@contextlib.contextmanager
def _no_grad_sync(model):
    model.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        model.set_requires_gradient_sync(True)


def _native_models(accelerator):
    return [m for m in getattr(accelerator, "_models", []) if is_native_fsdp2(m)]


def _split_mixed_param_groups(optimizer):
    """foreach / fused optimizer kernels cannot take DTensor (sharded) and plain (replicated) tensors in
    one list: split such groups in two with the same hyperparameters. Before the first step only."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    groups = getattr(optimizer, "param_groups", None)
    if not groups or getattr(optimizer, "state", None):
        return 0
    new_groups = []
    splits = 0
    for group in groups:
        sharded = [p for p in group["params"] if isinstance(p, DTensor)]
        plain = [p for p in group["params"] if not isinstance(p, DTensor)]
        if sharded and plain:
            rest = {k: v for k, v in group.items() if k != "params"}
            new_groups.append({**rest, "params": sharded})
            new_groups.append({**rest, "params": plain})
            splits += 1
        else:
            new_groups.append(group)
    if splits:
        optimizer.param_groups[:] = new_groups
    return splits


def _install_accelerate_hooks():
    try:
        from accelerate import Accelerator
    except Exception:
        return False
    if getattr(Accelerator, _PATCHED, False):
        return True

    original_prepare_model = Accelerator.prepare_model

    @functools.wraps(original_prepare_model)
    def prepare_model(
        self,
        model,
        device_placement = None,
        evaluation_mode = False,
    ):
        if not is_native_fsdp2(model):
            return original_prepare_model(
                self, model, device_placement = device_placement, evaluation_mode = evaluation_mode
            )
        # Already sharded by Unsloth: keep accelerate's autocast wrap of forward (as one GPU gets),
        # skip device placement and the DDP / FSDP wrap (evaluation_mode skips exactly those).
        if getattr(model, "_is_accelerate_prepared", False):
            if model not in self._models:
                self._models.append(model)
            return model
        return original_prepare_model(self, model, device_placement = False, evaluation_mode = True)

    original_no_sync = Accelerator.no_sync

    @contextlib.contextmanager
    def no_sync(self, model):
        target = model
        if not is_native_fsdp2(target):
            with original_no_sync(self, model):
                yield
            return
        root = (
            getattr(target, "module", None) if not getattr(target, _NATIVE_FLAG, False) else target
        )
        root = root if root is not None else target
        with _no_grad_sync(root):
            yield

    no_sync.__doc__ = original_no_sync.__doc__

    original_clip = Accelerator.clip_grad_norm_

    @functools.wraps(original_clip)
    def clip_grad_norm_(
        self,
        parameters,
        max_norm,
        norm_type = 2,
    ):
        if not _native_models(self):
            return original_clip(self, parameters, max_norm, norm_type = norm_type)
        self.unscale_gradients()
        return native_clip_grad_norm_(list(parameters), max_norm, norm_type)

    original_prepare_optimizer = Accelerator.prepare_optimizer

    @functools.wraps(original_prepare_optimizer)
    def prepare_optimizer(self, optimizer, *args, **kwargs):
        if _native_models(self) and _split_mixed_param_groups(
            getattr(optimizer, "optimizer", optimizer)
        ):
            _print_rank0(
                "Unsloth: native FSDP2 split optimizer groups mixing sharded and replicated parameters "
                "after the optimizer was built; an LR scheduler built before this keeps the old groups."
            )
        return original_prepare_optimizer(self, optimizer, *args, **kwargs)

    Accelerator.prepare_model = prepare_model
    Accelerator.prepare_optimizer = prepare_optimizer
    Accelerator.no_sync = no_sync
    Accelerator.clip_grad_norm_ = clip_grad_norm_
    setattr(Accelerator, _PATCHED, True)
    return True


def _install_trainer_hooks():
    try:
        from transformers import Trainer
    except Exception:
        return False
    if getattr(Trainer, _PATCHED, False):
        return True
    original_save_model = Trainer.save_model

    @functools.wraps(original_save_model)
    def save_model(
        self,
        output_dir = None,
        _internal_call = False,
    ):
        model = getattr(self, "model", None)
        if not is_native_fsdp2(model) or getattr(self, "is_fsdp_enabled", False):
            return original_save_model(self, output_dir, _internal_call = _internal_call)
        if output_dir is None:
            output_dir = self.args.output_dir
        trainable_only = _is_peft(model)
        # Collective gather on every rank (the Trainer would only call _save on rank 0).
        state_dict = native_full_state_dict(model, trainable_only = trainable_only)
        if self.args.should_save:
            self._save(output_dir, state_dict = state_dict)
        if getattr(self.args, "push_to_hub", False) and not _internal_call:
            self.push_to_hub(commit_message = "Model save", revision = self.args.hub_revision)

    original_create_optimizer = Trainer.create_optimizer

    @functools.wraps(original_create_optimizer)
    def create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        if is_native_fsdp2(getattr(self, "model", None)):
            # Before the LR scheduler records one base LR per group.
            _split_mixed_param_groups(getattr(optimizer, "optimizer", optimizer))
        return optimizer

    Trainer.save_model = save_model
    Trainer.create_optimizer = create_optimizer
    setattr(Trainer, _PATCHED, True)
    return True


def _is_peft(model):
    try:
        from peft import PeftModel
        return isinstance(model, PeftModel)
    except Exception:
        return False


def install_native_fsdp2_hooks():
    """Idempotent: accelerate treats a native FSDP2 model as prepared (no DDP / FSDP wrap), no_sync
    turns gradient reduction off for accumulation, clip_grad_norm_ handles DTensor gradients, and
    Trainer.save_model gathers collectively."""
    a = _install_accelerate_hooks()
    t = _install_trainer_hooks()
    return a and t


def _scaler_for_native(trainer):
    """float16: the GradScaler's inf check must agree over ranks, or one rank skips a step the
    others take. ShardedGradScaler all-reduces it."""
    accelerator = getattr(trainer, "accelerator", None)
    scaler = getattr(accelerator, "scaler", None)
    if scaler is None:
        return False
    try:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    except ImportError:
        return False
    if isinstance(scaler, ShardedGradScaler):
        return False
    kwargs = {}
    handler = getattr(accelerator, "scaler_handler", None)
    if handler is not None and hasattr(handler, "to_kwargs"):
        kwargs = handler.to_kwargs()
    accelerator.scaler = ShardedGradScaler(**kwargs)
    return True


_BNB_OPTIMS = {
    "adamw_8bit": "adamw_torch_fused",
    "adamw_bnb_8bit": "adamw_torch_fused",
    "paged_adamw_8bit": "adamw_torch_fused",
    "paged_adamw_32bit": "adamw_torch_fused",
    "lion_8bit": "lion_32bit",
    "paged_lion_8bit": "lion_32bit",
}


def _swap_bnb_optimizer(training_args):
    """bitsandbytes' optimizers step plain CUDA tensors; native FSDP2 parameters are DTensors."""
    optim = getattr(training_args, "optim", None)
    if optim is None:
        return False
    name = str(getattr(optim, "value", optim)).strip().lower()
    target = _BNB_OPTIMS.get(name)
    if target is None:
        return False
    if target == "adamw_torch_fused" and not torch.cuda.is_available():
        target = "adamw_torch"
    _print_rank0(
        f"Unsloth: native FSDP2 shards parameters as DTensors, which bitsandbytes' optimizers cannot "
        f"step. Switching optim = '{name}' to '{target}'."
    )
    try:
        from transformers.training_args import OptimizerNames
        training_args.optim = OptimizerNames(target)
    except Exception:
        training_args.optim = target
    return True


def native_fsdp2_trainer_setup(
    trainer = None,
    model = None,
    phase = "pre",
    optimizers = None,
    training_args = None,
):
    """Called around Trainer.__init__. pre: shard `model` (LoRA or full finetuning) before the
    optimizer and accelerate exist. post: float16 scaler. Returns the (sharded) model."""
    if not native_fsdp2_requested():
        return model
    install_native_fsdp2_hooks()
    if phase == "pre":
        if model is None or not isinstance(model, nn.Module) or is_native_fsdp2(model):
            return model
        optimizer = optimizers[0] if isinstance(optimizers, (tuple, list)) and optimizers else None
        if optimizer is not None:
            raise RuntimeError(
                "Unsloth: native FSDP2 shards the model inside Trainer.__init__, which replaces its "
                "parameters, so an optimizer built before the Trainer would update stale tensors. Call "
                "`unsloth.models.fsdp2_native.shard_model(model)` before building the optimizer, or "
                "let the Trainer build it (optimizers = (None, None))."
            )
        _swap_bnb_optimizer(training_args)
        return shard_model(model)
    if trainer is not None and is_native_fsdp2(getattr(trainer, "model", None)):
        _scaler_for_native(trainer)
    return model
