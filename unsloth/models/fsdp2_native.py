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
    "check_shard_units",
    "native_save_optimizer",
    "native_load_optimizer",
    "native_load_model_checkpoint",
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
    # None = auto: when the whole model fits on one GPU (fsdp2_perf.plan_fsdp2_layout), frozen weights stay
    # replicated (never gathered), only trainable ones are sharded, kept gathered from forward to backward
    # (False) with next-block prefetch, and a small trainable set (LoRA) is one root unit; otherwise every
    # block reshards after forward (True). An explicit bool / int is used as is. Last block always False.
    reshard_after_forward: Optional[Union[bool, int]] = None
    cpu_offload: bool = False  # CPUOffloadPolicy
    mesh: Optional[object] = None  # DeviceMesh; None = 1-D over WORLD_SIZE
    # "cpu": shard on a CPU mesh (gloo) even with CUDA present (TrainingArguments(use_cpu = True)).
    device: Optional[Union[str, torch.device]] = None
    # Process group backend / timeout when shard_model has to create the group (Trainer setup passes
    # TrainingArguments.ddp_backend / ddp_timeout). None = NCCL for CUDA (gloo without NCCL), default.
    backend: Optional[str] = None
    timeout: Optional[object] = None  # datetime.timedelta
    # Knobs for perf tuning (defaults = the plan's contract behaviour).
    # The last unit's reshard_after_forward (backward re-gathers it first).
    reshard_last_block: bool = False
    # unit_reshard_after_forward(index, unit) -> bool | int | None (None = the default), per unit.
    unit_reshard_after_forward: Optional[Callable] = None
    # False: frozen weights (LoRA / QLoRA base) stay replicated and are never gathered. True: sharded.
    # None = auto (replicated when the model fits, see reshard_after_forward).
    shard_frozen: Optional[bool] = None
    # None = auto by device memory; True / False forces the fit / sharded layout (env UNSLOTH_FSDP2_LAYOUT).
    fit_layout: Optional[bool] = None
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
    # Root-level embedding tables (not the input / output embeddings) holding at least this fraction of all
    # parameters get their own unit (Gemma4 / Gemma3n `embed_tokens_per_layer`: ~40% of an E-model would sit
    # gathered in the root for the whole step). 0 disables.
    split_large_root_embeddings: float = 0.05
    # Wrap `generate` so it runs inside native_fsdp2_unsharded (Unsloth's decode loops read layer weights
    # without calling the blocks; PeftModel.generate skips the root's hooks). Collective: every rank generates.
    wrap_generate: bool = True

    @classmethod
    def from_env(cls):
        """Defaults, overridden by UNSLOTH_FSDP2_PARAM_DTYPE (bfloat16 / float16 / float32 / storage =
        gather each weight in its own dtype / auto = the default), UNSLOTH_FSDP2_REDUCE_DTYPE, UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD
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
        if value and value != "auto":
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
        if value and value != "auto":
            config.reshard_after_forward = value not in ("0", "false", "no", "off")
        value = os.environ.get("UNSLOTH_FSDP2_CPU_OFFLOAD", "").strip().lower()
        if value:
            config.cpu_offload = value in ("1", "true", "yes", "on")
        value = os.environ.get("UNSLOTH_FSDP2_SHARD_FROZEN", "").strip().lower()
        if value and value != "auto":
            config.shard_frozen = value not in ("0", "false", "no", "off")
        value = os.environ.get("UNSLOTH_FSDP2_LAYOUT", "").strip().lower()
        if value in ("fit", "sharded"):
            config.fit_layout = value == "fit"
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


def _fsdp_attr(name):
    """A public FSDP2 name (fully_shard, CPUOffloadPolicy, OffloadPolicy, MixedPrecisionPolicy,
    FSDPModule, register_fsdp_forward_method, ...): `torch.distributed.fsdp` on torch >= 2.6,
    `torch.distributed._composable.fsdp` on 2.4 / 2.5."""
    import importlib

    try:
        public = importlib.import_module("torch.distributed.fsdp")
        if hasattr(public, name):
            return getattr(public, name)
    except ImportError:
        pass
    composable = importlib.import_module("torch.distributed._composable.fsdp")  # torch 2.4 / 2.5
    return getattr(composable, name)


# FSDP2's private modules, by static import on either layout: torch.distributed.fsdp._fully_shard.<name>
# (torch >= 2.6) or torch.distributed._composable.fsdp.<name> (2.4 / 2.5).


def _load_fsdp_param():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_param as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_param as module
    return module


def _load_fsdp_state():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_state as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_state as module
    return module


def _load_fsdp_common():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_common as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_common as module
    return module


def _load_fsdp_param_group():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_param_group as module
    return module


def _load_fsdp_init():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_init as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_init as module
    return module


def _load_fsdp_collectives():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_collectives as module
    return module


def _load_fully_shard():
    try:
        from torch.distributed.fsdp._fully_shard import _fully_shard as module
    except (
        ImportError
    ):  # torch 2.4 / 2.5: the module is `fully_shard` (the package attribute is the function)
        import torch.distributed._composable.fsdp.fully_shard  # noqa: F401
        module = sys.modules["torch.distributed._composable.fsdp.fully_shard"]
    return module


_FSDP_PRIVATE_LOADERS = {
    "_fsdp_param": _load_fsdp_param,
    "_fsdp_state": _load_fsdp_state,
    "_fsdp_common": _load_fsdp_common,
    "_fsdp_param_group": _load_fsdp_param_group,
    "_fsdp_init": _load_fsdp_init,
    "_fsdp_collectives": _load_fsdp_collectives,
    "_fully_shard": _load_fully_shard,
}


def _fsdp_private(submodule):
    """FSDP2's private module `submodule` (_fsdp_param, _fsdp_state, _fsdp_common, _fsdp_param_group,
    _fsdp_init, _fsdp_collectives, _fully_shard) on either layout (see _FSDP_PRIVATE_LOADERS)."""
    loader = _FSDP_PRIVATE_LOADERS.get(submodule)
    if loader is None:
        raise ImportError(f"no FSDP2 module {submodule}")
    return loader()


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


def _leafish(module):
    """A Linear / norm / Embedding-like module, also when PEFT wrapped it (lora.Linear keeps `base_layer`)."""
    if isinstance(module, _LEAF_TYPES) or not any(True for _ in module.children()):
        return True
    base = getattr(module, "base_layer", None)
    return isinstance(base, nn.Module) and _leafish(base)


def _is_block_list(container, no_split):
    """A ModuleList whose children are all blocks (modules with submodules and parameters). Classes may mix
    (Jamba, Zamba / Zamba2, OlmoHybrid, Mllama text); lists of bare Linear / norm modules (Gemma3n altup
    projections, also after PEFT wraps them) are not blocks."""
    children = list(container.children())
    if not children:
        return False
    for child in children:
        if _leafish(child) or not _has_params(child):
            return False
    return True


def find_shard_units(model, ignored_params = ()) -> list:
    """Repeated decoder / vision / audio blocks in module order: every child of the outermost
    nn.ModuleLists of blocks (model.layers, language_model.layers, vision_tower.*.layers, ...); never a
    unit inside a unit (MoE experts, fused-LoRA Linears stay in their block's unit). Only when no such list
    exists, the existing `_no_split_modules` instances."""
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
    if no_split and not units:
        # Fallback only when no block list exists: the existing `_no_split_modules` classes (transformers 5.x
        # merges children's names into the parent, so absent names are simply never matched).
        for _, module in model.named_modules():
            if id(module) in inside or module is model:
                continue
            if _mro_names(module) & no_split and _has_params(module):
                _take(module)
    # Module (pre-)order = forward order for the usual layouts; the fallback units are re-sorted in.
    order = {id(m): i for i, m in enumerate(model.modules())}
    units.sort(key = lambda m: order.get(id(m), 0))
    units, dropped = _drop_units_sharing_params(model, units, ignored_params)
    if dropped:
        _print_rank0(
            f"Unsloth: native FSDP2 keeps {len(dropped)} blocks ({', '.join(sorted(set(dropped)))}) in the "
            "root unit: they share parameters with other blocks or modules (Zamba2's shared transformer), "
            "which one FSDP2 unit each would silently untie."
        )
    check_shard_units(model, units, ignored_params)
    return units


def _units_sharing_params(
    model,
    units,
    ignored = (),
):
    """Indices of units sharing a parameter with another unit or an outside module: fully_shard(unit)
    would untie it. Counted per unit, so one module instance reused by two units is caught too."""
    skip = {id(p) for p in ignored}
    holders = {}
    for i, unit in enumerate(units):
        for p in unit.parameters():
            holders.setdefault(id(p), set()).add(i)
    unit_ids = {id(u) for u in units}
    outside = set()
    stack = [model]
    while stack:
        module = stack.pop()
        if id(module) in unit_ids:
            continue
        outside.update(id(p) for p in module.parameters(recurse = False))
        stack.extend(module.children())
    bad = []
    for i, unit in enumerate(units):
        for p in unit.parameters():
            if id(p) not in skip and (id(p) in outside or len(holders[id(p)]) > 1):
                bad.append(i)
                break
    return bad


def _drop_units_sharing_params(
    model,
    units,
    ignored_params = (),
):
    """Drop sharing units into the root, repeated: the root may then share with another unit."""
    dropped = []
    while True:
        bad = set(_units_sharing_params(model, units, ignored_params))
        if not bad:
            return units, dropped
        dropped += [type(units[i]).__name__ for i in bad]
        units = [u for i, u in enumerate(units) if i not in bad]


def _block_compiled(module):
    """Unsloth's block compile installs a compiled `forward` on the decoder-layer class (_block_compile)."""
    return bool(getattr(getattr(type(module), "forward", None), "_unsloth_block_compiled", False))


def check_shard_units(
    model,
    units,
    ignored_params = (),
):
    """The one hard torch.compile rule: no FSDP unit inside a (compiled) decoder block or inside another
    unit. Unsloth compiles each decoder block's forward (fullgraph=False); FSDP2's hooks are dynamo
    disabled, so a unit nested inside the block (an expert, a router, a LoRA / norm submodule) is a silent
    graph break per block (an error under fullgraph). Experts, routers and LoRA stay inside their block's
    unit. Raises RuntimeError naming the offending unit."""
    blocks = set()
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and _is_block_list(module, ()):
            blocks.update(id(child) for child in module.children())
        elif _block_compiled(module):
            blocks.add(id(module))
    parents = {}
    names = {}
    for name, module in model.named_modules(remove_duplicate = False):
        names.setdefault(id(module), name)
        for child in module.children():
            parents.setdefault(id(child), module)
    unit_ids = {id(u) for u in units}
    for unit in units:
        if unit is model:
            raise RuntimeError(
                "Unsloth: native FSDP2 shard units must not include the root module."
            )
        ancestor = parents.get(id(unit))
        while ancestor is not None and ancestor is not model:
            if id(ancestor) in unit_ids or id(ancestor) in blocks:
                kind = "shard unit" if id(ancestor) in unit_ids else "decoder block"
                raise RuntimeError(
                    f"Unsloth: native FSDP2 would make `{names.get(id(unit), type(unit).__name__)}` its own "
                    f"unit inside the {kind} `{names.get(id(ancestor), type(ancestor).__name__)}`. Units nested "
                    "in a compiled block break its graph (FSDP2 hooks are dynamo disabled); keep experts, "
                    "routers and LoRA inside their block's unit."
                )
            ancestor = parents.get(id(ancestor))
    if ignored_params is not None:
        _refuse_units_sharing_params(model, units, ignored_params, names)
    return True


def _refuse_units_sharing_params(
    model,
    units,
    ignored_params,
    names = None,
):
    shared = _units_sharing_params(model, units, ignored_params)
    if not shared:
        return
    unit = units[shared[0]]
    if names is None:
        names = {id(m): n for n, m in model.named_modules(remove_duplicate = False)}
    raise RuntimeError(
        f"Unsloth: native FSDP2 shard unit `{names.get(id(unit), type(unit).__name__)}` holds parameters "
        "that another unit or module also holds (tied / shared weights). One FSDP2 unit per holder would "
        "silently untie them: keep every holder of a shared parameter in one unit (or the root)."
    )


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


def _offloaded_device_map(model):
    """cpu / disk entries of an accelerate device map (from_pretrained(device_map = ...)) anywhere
    in the model: the user placed those weights off the GPU on purpose."""
    found = []
    for module in model.modules():
        device_map = module.__dict__.get("hf_device_map", None)
        if isinstance(device_map, dict):
            found.extend(
                f"{k}: {v}" for k, v in device_map.items() if str(v).lower() in ("cpu", "disk")
            )
    return found


def _refuse_gpu_quantized_on_cpu_mesh(model):
    """A CPU mesh (use_cpu) does not replicate frozen weights, so fully_shard would move packed
    bitsandbytes storage off the GPU it was quantized on, away from its quantization state."""
    if any(p.device.type == "cuda" and _is_quantized_param(p) for p in model.parameters()):
        raise RuntimeError(
            "Unsloth: native FSDP2 was asked for a CPU mesh (use_cpu = True), but the model's "
            "bitsandbytes quantized weights are on the GPU; moving them would separate them from "
            "their quantization state. Train on the GPU (use_cpu = False), or load the model "
            "without load_in_4bit / load_in_8bit."
        )


def _shard_device(model, requested = None):
    """The device FSDP2 shards onto: the model's own, except that a model still entirely on the CPU
    (loaded with device_map = None; meta tensors stay meta) goes to this rank's GPU when CUDA is used,
    like the Trainer would have placed it: fully_shard moves each unit's states to the mesh device as
    it shards it, _place_replicated the replicated ones. A deliberate CPU / disk offload (any
    device_map entry on cpu / disk) and CPU 4-bit weights are refused instead. requested = "cpu" (the
    Trainer's use_cpu) keeps the CPU mesh."""
    if requested is not None and torch.device(requested).type == "cpu":
        _refuse_gpu_quantized_on_cpu_mesh(model)
        return torch.device("cpu")
    if requested is not None and torch.device(requested).type != "cuda":
        _refuse_non_cuda(torch.device(requested).type)
    if str(os.environ.get("ACCELERATE_USE_CPU", "")).strip().lower() in ("1", "true", "yes", "on"):
        _refuse_gpu_quantized_on_cpu_mesh(model)
        return torch.device("cpu")
    elsewhere = sorted(
        {
            t.device.type
            for t in list(model.parameters()) + list(model.buffers())
            if t.device.type not in ("cpu", "cuda", "meta")
        }
    )
    if elsewhere:
        _refuse_non_cuda(elsewhere[0])
    offloaded = _offloaded_device_map(model)
    if offloaded:
        raise RuntimeError(
            "Unsloth: native FSDP2 got a model whose device_map offloads weights to the CPU or disk "
            f"(e.g. {offloaded[0]}). FSDP2 shards the whole model over the GPUs; load it without "
            "CPU / disk offload (device_map = None or each rank's GPU)."
        )
    device = _device_of(model)
    if not torch.cuda.is_available():
        accelerator = _non_cuda_accelerator()
        if accelerator is not None:
            # The Trainer / accelerate would run the batches on it while the mesh stays on the CPU.
            _refuse_non_cuda(accelerator)
        return device
    if device.type not in ("cpu", "cuda"):
        return device  # meta
    # Only an explicit CPU request keeps the CPU (above). A gloo process group alone does not: CUDA
    # builds without NCCL (Windows) and _ensure_process_group's fallback run gloo over CUDA tensors.
    tensors = [
        t for t in list(model.parameters()) + list(model.buffers()) if t.device.type != "meta"
    ]
    if any(t.device.type == "cuda" for t in tensors):
        return _placed_cuda_device(model, tensors, device)
    if any(_is_params4bit(p) for p in model.parameters()):
        raise RuntimeError(
            "Unsloth: native FSDP2 got a 4-bit model on the CPU; moving bitsandbytes weights would "
            "re-quantize them. Load it onto each rank's GPU (device_map = {'': local_rank})."
        )
    if any(_is_int8_param(p) for p in model.parameters()):
        raise RuntimeError(
            "Unsloth: native FSDP2 got an 8-bit model on the CPU; moving bitsandbytes Int8Params "
            "would leave their CB / SCB quantization state behind. Load it onto each rank's GPU "
            "(device_map = {'': local_rank})."
        )
    index = _local_cuda_index()
    if index is None:
        index = torch.cuda.current_device()
    _print_rank0(
        "Unsloth: native FSDP2 got a model on the CPU; it is sharded onto each rank's GPU "
        "(cuda:LOCAL_RANK)."
    )
    return torch.device("cuda", index)


def _non_cuda_accelerator():
    """The device type of an available accelerator other than CUDA / ROCm (xpu, mps, npu, hpu, ...),
    else None."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "current_accelerator"):  # torch 2.6+
        try:
            device = accelerator.current_accelerator(check_available = True)
        except Exception:
            device = None
        return None if device is None or device.type == "cuda" else device.type
    for name in ("xpu", "npu", "hpu", "mlu", "musa"):
        try:
            if getattr(torch, name).is_available():
                return name
        except Exception:
            pass
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def _refuse_non_cuda(device_type):
    raise RuntimeError(
        f"Unsloth: native FSDP2 (UNSLOTH_FSDP2=1) supports NVIDIA / AMD GPUs (CUDA / ROCm) and the CPU "
        f"only; this model or host uses {device_type}, which Unsloth's sharding, process group and "
        "memory planning do not handle. Unset UNSLOTH_FSDP2, or train on the CPU with "
        "TrainingArguments(use_cpu = True)."
    )


def _local_cuda_index():
    try:
        index = int(os.environ.get("LOCAL_RANK", ""))
    except ValueError:
        return None
    return index if 0 <= index < torch.cuda.device_count() else None


def _is_quantized_param(t):
    """bitsandbytes 4-bit / 8-bit weights: moving them re-quantizes (or loses their state)."""
    return any(c.__name__ in ("Params4bit", "Int8Params") for c in type(t).__mro__) or (
        getattr(t, "quant_state", None) is not None
    )


def _is_int8_param(t):
    """bitsandbytes LLM.int8() weight (Linear8bitLt): int8 data plus CB / SCB attributes."""
    return any(c.__name__ == "Int8Params" for c in type(t).__mro__)


def _int8_params(model):
    """bitsandbytes 8-bit weights stay replicated, never sharded: FSDP2 turns a parameter into a
    DTensor shard and gathers it back as a plain tensor, which drops the Int8Params CB / SCB state
    Linear8bitLt's forward reads. They are frozen (int8), so no gradient needs reducing."""
    found = {p for p in model.parameters() if _is_int8_param(p)}
    if found:
        _print_rank0(
            f"Unsloth: native FSDP2 keeps {len(found)} bitsandbytes 8-bit weights replicated (FSDP2 "
            "would drop their CB / SCB quantization state)."
        )
    return found


def _placed_cuda_device(model, tensors, device):
    """A model already (partly) on CUDA shards onto this rank's GPU, cuda:LOCAL_RANK, not wherever its
    first weight sits: device_map = {"": 0} on every rank would put every rank's mesh on GPU 0 (NCCL
    duplicate device). Plain tensors elsewhere are moved by fully_shard / _place_replicated as they are
    sharded; quantized weights on another device, or a model split over several GPUs, are refused."""
    index = _local_cuda_index()
    if index is None:
        return device  # no launcher local rank: keep the model's own device
    target = torch.device("cuda", index)
    gpus = set()
    for module in model.modules():
        device_map = module.__dict__.get("hf_device_map", None)
        if isinstance(device_map, dict):
            gpus.update(
                str(torch.device(v) if not isinstance(v, int) else torch.device("cuda", v))
                for v in device_map.values()
            )
    if len(gpus) > 1:
        raise RuntimeError(
            f"Unsloth: native FSDP2 got a model split over several GPUs ({', '.join(sorted(gpus))}). "
            "FSDP2 shards a whole model per rank; load it onto each rank's own GPU "
            "(device_map = {'': local_rank})."
        )
    misplaced = [t for t in tensors if _is_quantized_param(t) and t.device != target]
    if misplaced:
        raise RuntimeError(
            f"Unsloth: native FSDP2 runs this rank on {target}, but the model's quantized weights are on "
            f"{misplaced[0].device}; moving them would re-quantize. Load it with device_map = "
            "{'': local_rank} so each rank's weights start on its own GPU."
        )
    if any(t.device != target for t in tensors):
        _print_rank0(
            "Unsloth: native FSDP2 shards onto each rank's GPU (cuda:LOCAL_RANK); weights loaded on "
            "another device are moved there as they are sharded."
        )
    return target


def _ensure_process_group(
    device,
    backend = None,
    timeout = None,
):
    """The default process group, created here only when nothing created it yet (transformers 4.x sets
    it up lazily; Trainer setup first runs TrainingArguments' own device setup). An explicit `backend`
    (TrainingArguments.ddp_backend) and `timeout` (ddp_timeout) are used as given; only the default
    picks NCCL for CUDA and falls back to gloo when NCCL is missing (Windows CUDA builds)."""
    import torch.distributed as dist

    if dist.is_initialized():
        return dist
    if backend is None:
        backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    kwargs = {"backend": backend}
    if timeout is not None:
        kwargs["timeout"] = timeout
    dist.init_process_group(**kwargs)
    return dist


def _default_mesh(device, dist):
    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh(device.type, (dist.get_world_size(),))


def _floating(p):
    return p.is_floating_point() and p.dtype not in _FLOAT8 and not _is_params4bit(p)


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
        FSDPParamGroup = _fsdp_private("_fsdp_param_group").FSDPParamGroup
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


# Context-manager factories entered around the fully_shard calls, in order:
# factory(model, param_dtype = None) -> context manager yielding an info dict or None. The default is the
# bitsandbytes 4-bit context (fsdp2_native_quant: Params4bit FSDPParam patches, packed storage dtype).
_SHARD_CONTEXTS = {}


def register_shard_context(name, factory):
    """Register (or replace, same `name`) a context entered around every native fully_shard pass."""
    _SHARD_CONTEXTS[name] = factory
    return factory


def _qlora_context(model, param_dtype = None):
    """bitsandbytes 4-bit weights through the fully_shard calls (fsdp2_native_quant): packed bytes
    stored as `param_dtype`, QuantState and Params4bit class kept on the all-gathered weight, no
    accelerate source check (this wrap never upcasts the model)."""
    try:
        from .fsdp2_native_quant import native_quant_context
    except Exception:
        return contextlib.nullcontext(None)
    return native_quant_context(model, dtype = param_dtype)


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

    # The module defining fully_shard (and FSDPModule) on this torch's layout: already imported.
    fully_shard_module = sys.modules.get(getattr(fully_shard, "__module__", ""), None)
    if fully_shard_module is None or not hasattr(fully_shard_module, "FSDPModule"):
        fully_shard_module = _fsdp_private("_fully_shard")

    original = fully_shard_module.FSDPModule

    class FSDPModuleABC(abc.ABC, original):
        pass

    fully_shard_module.FSDPModule = FSDPModuleABC
    try:
        return fully_shard(module, **kwargs)
    finally:
        fully_shard_module.FSDPModule = original


def _post_shard(
    model,
    units,
    device,
    root_only = False,
):
    """Unsloth-side bookkeeping shared with the accelerate path: fused LoRA gate, direct-read units,
    gradient averaging for anything FSDP2 left unmanaged."""
    FSDPModule = _fsdp_module_class()
    try:
        from . import fsdp2 as _acc

        parents = _acc._parents(model)
        shim = type("_Shim", (), {"device": device})()
        # Fused LoRA kernels exist only once llama.py patched the model; never import it here.
        if __package__ and f"{__package__}.llama" in sys.modules:
            if root_only:
                # One root unit = the outermost module the loop calls: its pre-forward gathers every
                # adapter before any block runs and keeps them gathered through backward.
                from .llama import _fused_lora_modules
                model._unsloth_fsdp2_fused_lora = len(_fused_lora_modules(model))
            else:
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
        FSDPParam = _fsdp_private("_fsdp_param").FSDPParam
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


def _recompute_skips_input_cast(original):
    """torch < 2.13: FSDPState._pre_forward returns early for an activation-checkpoint recompute (training
    state PRE_BACKWARD) without the mixed precision cast of the forward inputs."""
    import inspect

    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return False
    branch = source.split("TrainingState.PRE_BACKWARD:", 1)
    if len(branch) < 2:
        return False
    first_return = branch[1].split("return", 1)
    return len(first_return) == 2 and first_return[1].lstrip().startswith("args, kwargs")


def _patch_recompute_input_cast():
    """torch < 2.13 skips FSDP2's forward-input cast when a checkpointed unit is recomputed in backward, so
    the recompute sees the uncast (e.g. float32) inputs while the forward saw param_dtype ones: anything
    cast to the input dtype (LoRA A / B, a dequantized 4-bit weight) is saved in another dtype and
    non-reentrant checkpointing raises CheckpointError (Kaggle T4, torch 2.10, fp16 MoE / VLM 4-bit).
    torch 2.13 casts there (`return self._cast_forward_inputs(args, kwargs)`); do the same on older
    builds. Must run before fully_shard registers the hook. Native FSDP2 only;
    UNSLOTH_FSDP2_RECOMPUTE_CAST=0 turns it off (diagnostics)."""
    if os.environ.get("UNSLOTH_FSDP2_RECOMPUTE_CAST", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    try:
        _common = _fsdp_private("_fsdp_common")
        TrainingState, _cast_fp_tensor = _common.TrainingState, _common._cast_fp_tensor
        FSDPState = _fsdp_private("_fsdp_state").FSDPState
        from torch.distributed.utils import _apply_to_tensors
    except Exception:
        return False
    original = getattr(FSDPState, "_pre_forward", None)
    if original is None or getattr(original, _PATCHED, False):
        return original is not None
    if not _recompute_skips_input_cast(original):
        return False

    @functools.wraps(original)
    def _pre_forward(self, module, args, kwargs):
        recompute = self._training_state == TrainingState.PRE_BACKWARD
        args, kwargs = original(self, module, args, kwargs)
        mp = getattr(self, "_mp_policy", None)
        if recompute and mp is not None and mp.cast_forward_inputs and mp.param_dtype:
            cast = functools.partial(_cast_fp_tensor, mp.param_dtype)
            args, kwargs = _apply_to_tensors(cast, args), _apply_to_tensors(cast, kwargs)
        return args, kwargs

    disable = getattr(torch, "_disable_dynamo", None)
    if callable(disable):
        _pre_forward = functools.wraps(original)(disable(_pre_forward))
    setattr(_pre_forward, _PATCHED, True)
    FSDPState._pre_forward = _pre_forward
    return True


def _large_root_embeddings(model, units, fraction):
    """nn.Embedding tables outside every unit, other than the input / output embeddings (the fused CE reads
    lm_head.weight directly; tied tables must share one unit), holding >= `fraction` of all parameters. They
    are called as modules, so their own unit's hooks gather them."""
    if not fraction or fraction <= 0:
        return []
    total = sum(p.numel() for p in model.parameters())
    if total == 0:
        return []
    inside = set()
    for u in units:
        for p in u.parameters():
            inside.add(id(p))
    skip = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        try:
            module = getattr(model, getter)()
        except Exception:
            module = None
        if isinstance(module, nn.Module):
            skip.update(id(p) for p in module.parameters())
    owners = {}
    for module in model.modules():
        for p in module.parameters(recurse = False):
            owners.setdefault(id(p), set()).add(id(module))
    # Tables inside a decoder block (one kept in the root for sharing weights) would nest in it.
    in_block = set()
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and _is_block_list(module, ()):
            for child in module.children():
                in_block.update(id(m) for m in child.modules())
    extra = []
    for _, module in model.named_modules():
        if not isinstance(module, nn.Embedding) or id(module) in in_block:
            continue
        params = list(module.parameters())
        if not params or any(id(p) in inside or id(p) in skip for p in params):
            continue
        if any(len(owners.get(id(p), ())) > 1 for p in params):  # tied elsewhere
            continue
        if sum(p.numel() for p in params) >= fraction * total:
            extra.append(module)
    return extra


_FLOAT8 = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
)


def _float8_params(model):
    """FP8 weights: FSDP2's mixed precision would cast them to the gather dtype (a floating dtype), dropping
    the scale semantics. Frozen ones stay replicated in float8; trainable ones are refused."""
    found = [(n, p) for n, p in model.named_parameters() if p.dtype in _FLOAT8]
    trainable = [n for n, p in found if p.requires_grad]
    if trainable:
        raise NotImplementedError(
            f"Unsloth: native FSDP2 cannot shard trainable float8 parameters (e.g. {trainable[0]}); FSDP2's "
            "gather would cast them. Train LoRA on a frozen FP8 base, or load in 16-bit."
        )
    if found:
        _print_rank0(
            f"Unsloth: native FSDP2 keeps {len(found)} frozen float8 weights replicated (FSDP2 would cast them "
            "to the gather dtype)."
        )
    return {p for _, p in found}


_GENERATE_DEPTH = [0]


def _wrap_generate(model):
    """Run `generate` (the root's and every inner module's that has one) inside native_fsdp2_unsharded."""
    import weakref

    root = weakref.ref(model)
    for module in model.modules():
        generate = getattr(module, "generate", None)
        if not callable(generate) or getattr(generate, _PATCHED, False):
            continue
        if not hasattr(type(module), "generate"):
            continue

        def wrapped(
            *args,
            _generate = generate,
            **kwargs,
        ):
            target = root()
            if target is None or _GENERATE_DEPTH[0] > 0:
                return _generate(*args, **kwargs)
            _GENERATE_DEPTH[0] += 1
            try:
                with native_fsdp2_unsharded(target), torch.no_grad():
                    return _generate(*args, **kwargs)
            finally:
                _GENERATE_DEPTH[0] -= 1

        functools.update_wrapper(wrapped, generate)
        setattr(wrapped, _PATCHED, True)
        module.generate = wrapped


def _all_trainable_float32(model):
    """Every trainable floating parameter is float32 (LoRA masters), none 16-bit (full FT, embeddings)."""
    trainable = [p for p in model.parameters() if p.requires_grad and _floating(p)]
    return bool(trainable) and all(p.dtype == torch.float32 for p in trainable)


def _agree_over_ranks(
    dist,
    device,
    flags,
    mesh = None,
):
    """Every rank must pick the same layout (a per-rank memory heuristic could differ on mixed GPUs):
    elementwise AND of the boolean flags over the ranks of `mesh` (FSDP2Config.mesh; the world for
    the default mesh), so disjoint sub-meshes decide on their own and a sub-group shards without the
    other ranks calling in. At shard time only."""
    if not flags:
        return flags
    from .fsdp2_save import mesh_all_reduce

    t = torch.tensor([1 if f else 0 for f in flags], dtype = torch.int32, device = device)
    ranks = mesh_all_reduce(t, mesh, dist)
    return [int(v) == ranks for v in t.tolist()]


def _place_replicated(params, device):
    """Parameters left out of FSDP (ignored_params) are never moved by it: they must already sit on this
    rank's device. Moves plain ones that do not; refuses meta (a meta / broadcast load must materialise
    them first) and bitsandbytes 4-bit weights (moving them would re-quantize)."""
    moved = 0
    for p in params:
        if p.device == device:
            continue
        if p.device.type == "meta" or _is_quantized_param(p):
            raise RuntimeError(
                f"Unsloth: native FSDP2 keeps this {tuple(p.shape)} parameter replicated, but it is on "
                f"{p.device}, not {device}. Load the model onto each rank's device first."
            )
        p.data = p.data.to(device)
        moved += 1
    return moved


def _refuse_block_swap(model):
    """Unsloth block swap (offload_layers = N or "auto", `_unsloth_block_swap`) keeps decoder layers in
    host RAM behind 0-element placeholders it rebinds each step; fully_shard would turn those into
    DTensors. Refused before any change. Every "auto" rank refuses, also one that swapped nothing, so
    no rank shards alone (the peers would hang)."""
    swapped = 0
    auto = False
    for module in model.modules():
        d = module.__dict__
        swapper = d.get("_unsloth_block_swap", None)
        if swapper is not None:
            swapped = max(swapped, len(getattr(swapper, "indices", None) or ()))
        auto = auto or bool(d.get("_unsloth_offload_layers_auto_ddp", False))
        auto = auto or d.get("_unsloth_offload_layers_auto", None) is not None
    if auto or swapped:
        what = "'auto'" if auto else str(swapped)
        raise ValueError(
            f"Unsloth: offload_layers = {what} (this rank swapped {swapped} decoder layers to host "
            "RAM) cannot be combined with native FSDP2 (UNSLOTH_FSDP2=1), which shards the weights "
            "itself and cannot wrap swapped layers. Load with offload_layers = 0 (and "
            "get_peft_model(offload_layers = 0)), or launch without UNSLOTH_FSDP2."
        )


def shard_model(model, config: Optional[FSDP2Config] = None):
    """fully_shard(unit) for each unit of `find_shard_units`, then fully_shard(model): the root holds
    embeddings, final norm, lm_head and any leftover parameters, so Unsloth's direct lm_head.weight
    read in the fused CE runs while the root is unsharded. Pass the outermost module the training
    loop calls (the PeftModel for LoRA): peft calls the inner model's `.forward` directly, which
    would skip an inner root's hooks. Idempotent.

    Layout (fsdp2_perf.plan_fsdp2_layout): when the model fits on one GPU and reshard_after_forward is
    auto, frozen weights stay replicated (never gathered, like DDP), only trainable ones are sharded and
    kept gathered from forward to backward with next-block prefetch, and a small trainable set (LoRA) is
    one root unit. Otherwise each block reshards after forward."""
    if getattr(model, _NATIVE_FLAG, False):
        return model
    _refuse_block_swap(model)
    FSDPModule, MixedPrecisionPolicy, fully_shard = _fsdp_api()
    if isinstance(model, FSDPModule):
        raise RuntimeError(
            "Unsloth: this model is already wrapped by FSDP2 outside Unsloth; native FSDP2 will not "
            "wrap it again."
        )
    config = config or FSDP2Config.from_env()
    if config.units is not None:
        # Sharing is checked once the replicated set is known (ignored and, if replicated, frozen params).
        check_shard_units(model, list(config.units), ignored_params = None)
    _refuse_tensor_parallel_model(model)
    device = _shard_device(model, config.device)
    dist = _ensure_process_group(device, config.backend, config.timeout)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mesh = config.mesh if config.mesh is not None else _default_mesh(device, dist)
    param_dtype = _resolve_param_dtype(model, config)
    mp_policy = MixedPrecisionPolicy(param_dtype = param_dtype, reduce_dtype = config.reduce_dtype)
    _patch_recompute_input_cast()  # before any fully_shard registers FSDPState._pre_forward
    offload_policy = None
    if config.cpu_offload:
        CPUOffloadPolicy = _fsdp_attr("CPUOffloadPolicy")
        offload_policy = CPUOffloadPolicy(pin_memory = device.type == "cuda")
    from .fsdp2_perf import (
        _env_tristate,
        plan_fsdp2_layout,
        resolve_reshard_after_forward,
        shard_model_layout,
    )

    layout = plan_fsdp2_layout(
        model,
        param_dtype = param_dtype,
        reshard_after_forward = config.reshard_after_forward,
        device = device,  # a CPU mesh has no GPU capacity: the per-block sharded layout
        cpu_offload = config.cpu_offload,
        fits = config.fit_layout,
        replicate_frozen = None if config.shard_frozen is None else not config.shard_frozen,
        allow_root_only = config.units is None,
    )
    if (
        layout.replicate_frozen
        and config.shard_frozen is None
        and _env_tristate("UNSLOTH_FSDP2_REPLICATE_FROZEN") is not True
        and not _supports_ignored_params(fully_shard)
    ):
        # Automatic fit layout on a torch whose fully_shard has no ignored_params (2.6): shard the
        # frozen weights too. An explicit replication request still fails below. Before the agreement,
        # which every rank then reaches with the same answer.
        layout.replicate_frozen = False
        layout.root_only = False
        layout.reasons.append("replicate_frozen=False (fully_shard has no ignored_params)")
    layout.fits, layout.replicate_frozen, layout.root_only, layout.prefetch = _agree_over_ranks(
        dist,
        device,
        [layout.fits, layout.replicate_frozen, layout.root_only, layout.prefetch],
        config.mesh,
    )
    # reshard_after_forward is derived from `fits` (when not set by config / env): derive it again from
    # the agreed value, or a rank that fit alone keeps its blocks gathered while the others reshard, and
    # their backward collectives no longer match.
    resolve_reshard_after_forward(layout, config.reshard_after_forward, note = False)
    # Units after the layout: frozen weights it replicates may be shared by blocks that keep their own unit.
    ignored_params = set(config.ignored_params or ())
    if layout.replicate_frozen and _supports_ignored_params(fully_shard):
        ignored_params.update(p for p in model.parameters() if not p.requires_grad)
    units = (
        list(config.units) if config.units is not None else find_shard_units(model, ignored_params)
    )
    extra_units = (
        _large_root_embeddings(model, units, config.split_large_root_embeddings)
        if config.units is None
        else []
    )
    if extra_units:
        check_shard_units(model, units + extra_units, ignored_params = None)
    gather_dtype = param_dtype
    if (
        config.param_dtype is None
        and layout.replicate_frozen
        and param_dtype in (torch.bfloat16, torch.float16)
        and _all_trainable_float32(model)
    ):
        # Fit layout: FSDP2 manages only trainable parameters. One GPU keeps float32 masters (LoRA) and
        # lets autocast cast them at each op; a 16-bit gather casts them first, which moved Gemma4's step-0
        # grad norm 0.2% off 1 GPU (A/A 0.01%). Gather them as they are; the 4-bit storage stays 16-bit.
        gather_dtype = None
        mp_policy = MixedPrecisionPolicy(param_dtype = None, reduce_dtype = config.reduce_dtype)
    if not units and not layout.root_only:
        _print_rank0(
            "Unsloth: native FSDP2 found no repeated blocks in this model; only the root is sharded "
            "(every forward gathers all weights)."
        )
    with contextlib.ExitStack() as stack:
        infos = {
            name: stack.enter_context(factory(model, param_dtype = param_dtype))
            for name, factory in _SHARD_CONTEXTS.items()
        }
        qlora = infos.get("bnb_qlora")
        ignored, _ = _unify_dtypes(
            model,
            # Root only: every trainable param is in the root group. Otherwise the split root
            # embedding tables are units of their own, not part of the root's dtype group.
            [] if layout.root_only else units + list(extra_units),
            param_dtype,
            can_ignore = config.replicate_minority_dtype and _supports_ignored_params(fully_shard),
        )
        ignored = set(ignored)
        ignored.update(_float8_params(model))
        int8 = _int8_params(model)
        if int8 and not _supports_ignored_params(fully_shard):
            raise RuntimeError(
                "Unsloth: native FSDP2 keeps bitsandbytes 8-bit (load_in_8bit) weights replicated, which "
                f"needs torch with fully_shard(ignored_params) (2.7+); this is torch {torch.__version__}. "
                "Upgrade torch, or load the model in 4-bit or 16-bit."
            )
        ignored.update(int8)
        if config.ignored_params:
            ignored.update(config.ignored_params)
        replicated = set(ignored)
        if layout.replicate_frozen:
            replicated.update(p for p in model.parameters() if not p.requires_grad)
        _refuse_units_sharing_params(model, units + list(extra_units), replicated)
        if replicated:
            if not _supports_ignored_params(fully_shard):
                raise RuntimeError(
                    f"Unsloth: native FSDP2 needs torch with fully_shard(ignored_params) (2.7+) to keep "
                    f"parameters replicated; this is torch {torch.__version__}."
                )
            _place_replicated(replicated, device)
        wrapped = shard_model_layout(
            model,
            units,
            mp_policy,
            layout,
            mesh = mesh,
            offload_policy = offload_policy,
            ignored_params = ignored,
            extra_units = extra_units,
            fully_shard_fn = functools.partial(_fully_shard, fully_shard),
            reshard_last_block = config.reshard_last_block,
            unit_reshard_after_forward = config.unit_reshard_after_forward,
            forward_prefetch = config.forward_prefetch,
            backward_prefetch = config.backward_prefetch,
            shard_kwargs = (
                {"shard_placement_fn": config.shard_placement_fn}
                if config.shard_placement_fn is not None
                else None
            ),
        )
        model._unsloth_reshard_after_forward = False
    setattr(model, _NATIVE_FLAG, True)
    for param in model.parameters():
        # An optimizer over these is a native one even before (or without) its model is prepared.
        try:
            param._unsloth_fsdp2_native_param = True
        except Exception:
            pass
    model._unsloth_fsdp2_blocks = len(wrapped)
    model._unsloth_fsdp2_units = wrapped
    model._unsloth_fsdp2_extra_units = [u for u in extra_units if isinstance(u, FSDPModule)]
    model._unsloth_fsdp2_root_only = layout.root_only
    if config.wrap_generate:
        _wrap_generate(model)
    model._unsloth_fsdp2_mp_policy = mp_policy
    model._unsloth_fsdp2_replicated = sum(1 for p in model.parameters() if p in replicated)
    # The replicated-gradient average and the media agreement run over this mesh's ranks only.
    model.__dict__["_unsloth_fsdp2_mesh"] = mesh
    _post_shard(model, units, device, root_only = layout.root_only)
    from .fsdp2_vision import install_vision_dummy

    install_vision_dummy(model)  # mixed image / text ranks
    _patch_accumulated_grad_guard()
    if qlora is not None:
        # Trainer / DCP checkpoints of the sharded 4-bit weights (fsdp2_save).
        from .fsdp2_save import install_dcp_bnb_state_dict
        install_dcp_bnb_state_dict()
    install_native_fsdp2_hooks()
    classes = sorted({type(u).__name__.removeprefix("FSDP") for u in wrapped})
    note = ""
    if qlora is not None:
        note = f", {qlora.get('params4bit', 0)} 4-bit weights packed as {'/'.join(qlora.get('storage', []))}"
    if layout.fits:
        shape = (
            "trainable params in one root unit"
            if layout.root_only
            else f"{len(wrapped)} blocks ({', '.join(classes) or 'none'}) + root, kept gathered"
        )
        frozen = "frozen weights replicated (never gathered)" if layout.replicate_frozen else ""
        what = f"(the model fits one GPU) {shape}{', ' + frozen if frozen else ''}"
    else:
        what = f"{len(wrapped)} blocks ({', '.join(classes) or 'none'}) + root"
    _print_rank0(
        f"Unsloth: native FSDP2 sharded {what}, over {dist.get_world_size()} ranks; gather dtype "
        f"{str(gather_dtype).replace('torch.', '') if gather_dtype is not None else 'as stored'}, reduce dtype "
        f"{str(config.reduce_dtype).replace('torch.', '')}{note}."
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
    # Per (mesh, mesh dims the gradient is sharded over): a user mesh may be 2-D (HSDP: replicate x
    # shard) or a submesh, and a local shard repeats along Replicate dims, so its partial norm reduces
    # over the Shard dims' groups only (what DTensor's own norm does, _NormPartial).
    groups = {}
    for p in params:
        g = p.grad
        if isinstance(g, DTensor):
            local = g.to_local()
            dims = tuple(i for i, pl in enumerate(g.placements) if pl.is_shard())
            groups.setdefault((g.device_mesh, dims), []).append(local)
            sharded.append(local)
            device = device or local.device
        else:
            replicated.append(g)
            device = device or g.device
    if device is None:
        return torch.tensor(0.0)
    if device.type == "cpu" and torch.cuda.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
    initialized = dist.is_available() and dist.is_initialized()
    p = norm_type

    def _per_tensor(grads):
        """One float64 norm per tensor (torch.linalg.vector_norm semantics); an empty local shard
        (uneven sharding: dim 0 smaller than the world) is the identity of the combine. 0 for p >= 0;
        +inf for every negative p (its p-th power is 0, and -inf takes the min), where vector_norm
        itself refuses an empty tensor."""
        if not grads:
            return torch.zeros(0, device = device, dtype = torch.float64)
        empty = math.inf if p < 0 else 0.0
        live = [g.detach() for g in grads if g.numel()]
        try:
            norms = list(torch._foreach_norm(live, p)) if live else []
        except Exception:
            norms = [torch.linalg.vector_norm(g, p) for g in live]
        it = iter(norms)
        return torch.stack(
            [
                next(it).double().to(device)
                if g.numel()
                else torch.tensor(empty, device = device, dtype = torch.float64)
                for g in grads
            ]
        )

    # torch.nn.utils.clip_grad_norm_ is vector_norm(stack(per-tensor norms), p): for p = 0 the number
    # of tensors with a nonzero element, for +-inf the max / min, else the p-th root of the sum of
    # p-th powers. A sharded tensor's norm combines over ranks first (counts and powers add up).
    replicated_norms = _per_tensor(replicated)

    def _over_shards(value, op, mesh, dims):
        if initialized:
            for dim in dims:
                dist.all_reduce(value, op = op, group = mesh.get_group(dim))
        return value

    if p == 0:
        total = (replicated_norms > 0).sum().double()
        for (mesh, dims), grads in groups.items():
            counts = _over_shards(_per_tensor(grads), dist.ReduceOp.SUM, mesh, dims)
            total = total + (counts > 0).sum().double()
    elif p in (math.inf, -math.inf):
        pick, op = (torch.max, dist.ReduceOp.MAX) if p > 0 else (torch.min, dist.ReduceOp.MIN)
        total = torch.tensor(0.0 if p > 0 else math.inf, device = device, dtype = torch.float64)
        parts = [pick(replicated_norms)] if replicated_norms.numel() else []
        for (mesh, dims), grads in groups.items():
            parts.append(_over_shards(pick(_per_tensor(grads)), op, mesh, dims))
        if parts:
            total = pick(torch.stack(parts))
    else:
        total = replicated_norms.pow(p).sum()
        for (mesh, dims), grads in groups.items():
            powers = _per_tensor(grads).pow(p).sum()
            total = total + _over_shards(powers, dist.ReduceOp.SUM, mesh, dims)
        total = total.pow(1.0 / p)
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
    # Re-entrant: the wrapped generate() enters again inside an outer block. Only the outermost
    # enter gathers and only the outermost exit reshards (per sharded root).
    root = _native_root(model)
    depth = root.__dict__.get("_unsloth_unsharded_depth", 0)
    root.__dict__["_unsloth_unsharded_depth"] = depth + 1
    if depth > 0:
        try:
            yield model
        finally:
            root.__dict__["_unsloth_unsharded_depth"] -= 1
        return
    units = [m for m in model.modules() if isinstance(m, FSDPModule)]
    try:
        for unit in units:
            unit.set_reshard_after_forward(False, recurse = False)
        for unit in units:
            unit.unshard()
        yield model
    finally:
        root.__dict__["_unsloth_unsharded_depth"] = 0
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


# Set by Trainer.create_accelerator_and_postprocess on the accelerator a Trainer owns.
_TRAINER_ACCELERATOR = "_unsloth_fsdp2_trainer_accelerator"

_CUSTOM_LOOP_REFUSAL = (
    "Unsloth: native FSDP2 (UNSLOTH_FSDP2=1 / shard_model) supports the Hugging Face / TRL Trainer, "
    "and plain PyTorch loops over shard_model(model) with the native helpers "
    "(unsloth.models.fsdp2_native: native_clip_grad_norm_, native_full_state_dict, "
    "native_save_optimizer, native_load_model_checkpoint, native_load_optimizer). "
    "accelerator.prepare() of a natively sharded model or its optimizer outside a Trainer is not "
    "supported: accelerate would treat the sharded model as unsharded (FP8 conversion, save_state / "
    "load_state, device placement). Drop accelerator.prepare for the model and optimizer, or use "
    "UNSLOTH_FSDP2_BACKEND=accelerate for an accelerate-managed FSDP2 loop."
)


def _native_object(obj):
    """A natively sharded model (or a wrapper of one), an optimizer over its parameters, or an LR
    scheduler over such an optimizer."""
    if isinstance(obj, nn.Module):
        return is_native_fsdp2(obj)
    if isinstance(obj, torch.optim.Optimizer) or hasattr(obj, "param_groups"):
        return _native_optimizer(obj)
    inner = getattr(obj, "optimizer", None)
    return inner is not None and hasattr(inner, "param_groups") and _native_optimizer(inner)


def _accelerator_fp8(accelerator):
    """accelerate converts Linear layers to FP8 (Transformer Engine / torchao / MS-AMP) inside
    Accelerator.prepare, before prepare_model."""
    if str(getattr(accelerator, "mixed_precision", "no") or "no").lower() == "fp8":
        return True
    try:
        backend = accelerator.fp8_backend
    except Exception:
        return False
    backend = getattr(backend, "value", backend)
    return backend is not None and str(backend).strip().upper() not in ("", "NO", "NONE")


def _refuse_untrusted_accelerator(accelerator, objects):
    """Native objects are prepared only by a Trainer's accelerator, never with FP8 conversion."""
    if not any(_native_object(obj) for obj in objects):
        return
    if not getattr(accelerator, _TRAINER_ACCELERATOR, False):
        raise RuntimeError(_CUSTOM_LOOP_REFUSAL)
    if _accelerator_fp8(accelerator):
        raise RuntimeError(
            "Unsloth: native FSDP2 cannot train with accelerate's FP8 (mixed_precision = 'fp8' or "
            "ACCELERATE_FP8_BACKEND): accelerate converts the layers after Unsloth sharded them. Use "
            "bf16 / fp16, or UNSLOTH_FSDP2_BACKEND=accelerate."
        )


def split_param_groups(params_or_groups):
    """Optimizer parameter groups for a natively sharded model: foreach / fused optimizer kernels cannot
    take DTensor (sharded) and plain (replicated) tensors in one list, so each group comes back as one
    group per kind with the same hyperparameters. Takes `model.parameters()` (or any iterable of
    tensors) or a list of group dicts. Build the optimizer from this before an LR scheduler:
    `torch.optim.AdamW(split_param_groups(p for p in model.parameters() if p.requires_grad), lr = ...)`."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    items = list(params_or_groups)
    if items and not isinstance(items[0], dict):
        items = [{"params": items}]
    out = []
    for group in items:
        params = list(group["params"])
        sharded = [p for p in params if isinstance(p, DTensor)]
        plain = [p for p in params if not isinstance(p, DTensor)]
        rest = {k: v for k, v in group.items() if k != "params"}
        out.extend({**rest, "params": kind} for kind in (sharded, plain) if kind)
        if not params:
            out.append({**rest, "params": []})
    return out


def _mixed_param_groups(optimizer):
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    groups = getattr(_inner_optimizer(optimizer), "param_groups", None) or []
    return sum(
        1
        for g in groups
        if any(isinstance(p, DTensor) for p in g["params"])
        and any(not isinstance(p, DTensor) for p in g["params"])
    )


def _split_mixed_param_groups(optimizer):
    """Split mixed DTensor / plain groups in place (split_param_groups). Before the first step only,
    and only where no LR scheduler was built yet (the Trainer refuses a prebuilt pair)."""
    groups = getattr(optimizer, "param_groups", None)
    if not groups or getattr(optimizer, "state", None):
        return 0
    splits = _mixed_param_groups(optimizer)
    if splits:
        optimizer.param_groups[:] = split_param_groups(groups)
    return splits


def _install_accelerate_hooks():
    try:
        from accelerate import Accelerator
    except Exception:
        return False
    if getattr(Accelerator, _PATCHED, False):
        return True

    original_prepare = Accelerator.prepare

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        # Before accelerate's FP8 conversion, which runs here ahead of prepare_model.
        _refuse_untrusted_accelerator(self, args)
        return original_prepare(self, *args, **kwargs)

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
        _refuse_untrusted_accelerator(self, [model])
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
        # From the optimizer's own parameters too: prepare(optimizer, model) prepares in the order
        # given, so the native model may not be registered in `_models` yet.
        native = bool(_native_models(self)) or _native_optimizer(optimizer)
        if native:
            _refuse_untrusted_accelerator(self, [optimizer] if _native_optimizer(optimizer) else [])
            _refuse_bnb_optimizer_instance(optimizer)
        if native and _mixed_param_groups(optimizer):
            if getattr(_inner_optimizer(optimizer), "state", None):
                # The split carries no per-parameter state over; a mixed group stepping a foreach /
                # fused kernel over DTensor and plain tensors fails.
                raise RuntimeError(_MIXED_GROUPS_REFUSAL)
        if native and _split_mixed_param_groups(getattr(optimizer, "optimizer", optimizer)):
            _print_rank0(
                "Unsloth: native FSDP2 split optimizer groups mixing sharded and replicated parameters "
                "after the optimizer was built; an LR scheduler built before this keeps the old groups."
            )
        return original_prepare_optimizer(self, optimizer, *args, **kwargs)

    original_get_state_dict = Accelerator.get_state_dict

    @functools.wraps(original_get_state_dict)
    def get_state_dict(
        self,
        model,
        unwrap = True,
    ):
        if not is_native_fsdp2(model):
            return original_get_state_dict(self, model, unwrap = unwrap)
        # accelerate's non-FSDP branch would return this rank's DTensor shards. Collective: the full
        # state dict on rank 0, {} elsewhere (save_model writes from the main process only).
        return native_full_state_dict(_native_root(model))

    def _refuse_state(name, original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if _native_models(self) or any(
                _native_optimizer(o) for o in getattr(self, "_optimizers", [])
            ):
                raise RuntimeError(
                    f"Unsloth: accelerator.{name}() would save / load each natively sharded model and "
                    "optimizer as this rank's DTensor shards (accelerate treats the model as unsharded). "
                    "Use unsloth.models.fsdp2_native.native_full_state_dict / native_save_optimizer and "
                    "native_load_model_checkpoint / native_load_optimizer (collective), "
                    "accelerator.save_model / get_state_dict (full weights), or Trainer checkpoints."
                )
            return original(self, *args, **kwargs)

        return wrapped

    Accelerator.get_state_dict = get_state_dict
    Accelerator.save_state = _refuse_state("save_state", Accelerator.save_state)
    Accelerator.load_state = _refuse_state("load_state", Accelerator.load_state)
    Accelerator.prepare = prepare
    Accelerator.prepare_model = prepare_model
    Accelerator.prepare_optimizer = prepare_optimizer
    Accelerator.no_sync = no_sync
    Accelerator.clip_grad_norm_ = clip_grad_norm_
    setattr(Accelerator, _PATCHED, True)
    return True


# --------------------------------------------------------------------------------------------------
# Checkpoint resume: optimizer state and model weights of a sharded model
# --------------------------------------------------------------------------------------------------

_OPTIMIZER_NAME = "optimizer.pt"
_SCHEDULER_NAME = "scheduler.pt"


def _native_root(model):
    """The module shard_model wrapped (the Trainer may hold it directly, under `.module`, or under
    torch.compile's `_orig_mod`). Wrappers are peeled by their registered child, not by attribute
    lookup: OptimizedModule (and PeftModel) forward unknown attributes, so the wrapper itself would
    answer the native flag."""
    current = model
    for _ in range(8):
        children = getattr(current, "_modules", None) or {}
        inner = next(
            (
                children[attr]
                for attr in ("_orig_mod", "module")
                if isinstance(children.get(attr), nn.Module)
                and getattr(children[attr], _NATIVE_FLAG, False)
            ),
            None,
        )
        if inner is None:
            break
        current = inner
    return current


def _inner_optimizer(optimizer):
    """accelerate's AcceleratedOptimizer wraps the torch optimizer as `.optimizer`."""
    while isinstance(getattr(optimizer, "optimizer", None), torch.optim.Optimizer):
        optimizer = optimizer.optimizer
    return optimizer


def native_save_optimizer(
    model,
    optimizer,
    path = None,
    should_save = None,
):
    """Collective (every rank must call): the full, unsharded optimizer state keyed by parameter FQN
    (`get_optimizer_state_dict(full_state_dict, cpu_offload)`), written to `path` on rank 0. A plain
    `optimizer.state_dict()` on rank 0 holds only rank 0's DTensor shards, so a resume from it is wrong."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    model = _native_root(model)
    optimizer = _inner_optimizer(optimizer)
    state = get_optimizer_state_dict(
        model,
        optimizer,
        options = StateDictOptions(full_state_dict = True, cpu_offload = True),
    )
    if should_save is None:
        should_save = _rank0()
    if path is None:
        return state
    from .fsdp2_save import _dist, _wait_for_writer

    dist = _dist()
    writer = bool(should_save)
    try:
        if writer:
            torch.save(state, path)
    finally:
        if dist is not None:
            # The other ranks wait for the write, so no later collective overtakes it. A store key,
            # not a barrier: a large full-finetuning state can take longer to write than the
            # collective timeout. Every rank passes the same path (the Trainer does).
            _wait_for_writer(dist, writer)
    return state


def native_load_optimizer(model, optimizer, path_or_state):
    """Collective: every rank reads the full optimizer state (CPU) and keeps only its own shards."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    state = path_or_state
    if not isinstance(state, dict):
        state = torch.load(state, map_location = "cpu", weights_only = True)
    set_optimizer_state_dict(
        _native_root(model),
        _inner_optimizer(optimizer),
        state,
        options = StateDictOptions(full_state_dict = True),
    )
    return state


_WEIGHT_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
_WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _read_checkpoint_weights(checkpoint):
    import json

    def _read(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path, device = "cpu")
        return torch.load(path, map_location = "cpu", weights_only = True)

    for name in _WEIGHT_FILES:
        path = os.path.join(checkpoint, name)
        if os.path.isfile(path):
            return _read(path), name.startswith("adapter_")
    for name in _WEIGHT_INDEXES:
        path = os.path.join(checkpoint, name)
        if os.path.isfile(path):
            with open(path, encoding = "utf-8") as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
            state = {}
            for shard in files:
                state.update(_read(os.path.join(checkpoint, shard)))
            return state, False
    raise ValueError(f"Unsloth: native FSDP2 found no model or adapter weights in {checkpoint}.")


def _peft_saved_key(key, adapter):
    """PEFT's own model-key -> checkpoint-key rule (_remove_adapter_name_from_state_dict_key): a
    ParameterDict entry (lora_embedding_A.<adapter>) drops the terminal `.<adapter>`, a module-backed
    one (lora_A.<adapter>.weight) the `.<adapter>` before the last segment, never one mid-path."""
    if "." not in key:
        return key
    if key.endswith(f".{adapter}"):
        return key[: -len(adapter) - 1]
    head, _, suffix = key.rpartition(".")
    if head.endswith(f".{adapter}"):
        head = head[: -len(adapter) - 1]
    key = f"{head}.{suffix}"
    if key.endswith(".lora_magnitude_vector.weight"):
        # LoraModel._get_adapter_state_dict: DoRA's magnitude (a DoraLinearLayer's weight since the
        # ParameterDict -> ModuleDict refactor) is still saved without ".weight".
        key = key[: -len(".weight")]
    return key


def _peft_save_keys(root, adapter):
    """model key -> checkpoint key for `adapter`, from PEFT's own save path
    (peft.utils.get_peft_model_state_dict, peft/utils/save_and_load.py): every key rule the installed
    PEFT applies on save (adapter name removal, DoRA's lora_magnitude_vector, modules_to_save, later
    ones) is inverted here. PEFT is handed empty placeholder tensors (no collective, no copy) and the
    returned keys are matched back by tensor identity; {} when PEFT cannot be asked, and the rule above
    stands in."""
    try:
        from peft.utils import get_peft_model_state_dict
    except Exception:
        return {}
    import warnings

    names = [n for n, _ in root.named_parameters()] + [n for n, _ in _persistent_buffers(root)]
    placeholders = {n: torch.empty(0) for n in names}
    by_id = {id(t): n for n, t in placeholders.items()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_peft_model_state_dict(
                root,
                state_dict = dict(placeholders),
                adapter_name = adapter,
                save_embedding_layers = False,
            )
    except Exception:
        return {}
    return {by_id[id(v)]: k for k, v in out.items() if id(v) in by_id}


def _has_checkpoint_weights(directory):
    return any(
        os.path.isfile(os.path.join(directory, name)) for name in _WEIGHT_FILES + _WEIGHT_INDEXES
    )


def _adapter_owner(root, adapters):
    """name -> the adapter a parameter belongs to, or None: PEFT keys adapter weights by name under
    a tuner layer's `adapter_layer_names` (lora_A.<adapter>.weight, lora_embedding_A.<adapter>) and
    under modules_to_save.<adapter>. Matching only those segments keeps a module path that merely
    equals an adapter name (layers.0, mlp) from counting."""
    containers = {"modules_to_save"}
    for module in root.modules():
        names = getattr(module, "adapter_layer_names", None)
        if isinstance(names, (list, tuple)):
            containers.update(n for n in names if isinstance(n, str))
    adapters = set(adapters)

    def owner(name):
        parts = name.split(".")
        for i in range(len(parts) - 1):
            if parts[i] in containers and parts[i + 1] in adapters:
                return parts[i + 1]
        return None

    return owner


def _persistent_buffers(root):
    out = []
    for module_name, module in root.named_modules():
        skip = getattr(module, "_non_persistent_buffers_set", ())
        for name, buf in module.named_buffers(recurse = False):
            if buf is not None and name not in skip:
                out.append((f"{module_name}.{name}" if module_name else name, buf))
    return out


def native_load_model_checkpoint(model, checkpoint):
    """Collective: load a Trainer checkpoint's weights (PEFT adapter or full model, as save_model wrote
    them) into a natively sharded model with `set_model_state_dict(full_state_dict)`. PEFT's saved keys
    drop the adapter name; it is put back for each trainable parameter. Raises if a trainable parameter
    has no saved tensor (the resume would silently keep the fresh value)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    root = _native_root(model)
    adapters = []
    active = getattr(root, "active_adapters", None)
    if isinstance(active, (list, tuple)):
        adapters = [a for a in active if isinstance(a, str)]
    elif isinstance(getattr(root, "active_adapter", None), str):
        adapters = [root.active_adapter]
    peft_config = getattr(root, "peft_config", None) if _is_peft(root) else None
    absent = []
    if isinstance(peft_config, dict) and peft_config:
        # PeftModel.save_pretrained writes every adapter in peft_config: "default" at the checkpoint
        # root, every other one under <checkpoint>/<adapter>/; each file drops its adapter's name from
        # the keys. Active adapters first, so shared tensors (resized embeddings) come from them.
        active_names = [a for a in adapters if a in peft_config]
        adapters = active_names + [a for a in peft_config if a not in active_names]
        sources = []
        for adapter in adapters:
            directory = checkpoint if adapter == "default" else os.path.join(checkpoint, adapter)
            if adapter not in active_names and not _has_checkpoint_weights(directory):
                absent.append(adapter)  # an inactive adapter this checkpoint did not save
                continue
            sources.append((adapter, _read_checkpoint_weights(directory)[0]))
        is_adapter = True
    else:
        saved, is_adapter = _read_checkpoint_weights(checkpoint)
        sources = [(adapter, saved) for adapter in adapters] or [(None, saved)]
    owner_of = _adapter_owner(root, adapters)
    loaded = {adapter for adapter, _ in sources}
    save_keys = (
        {adapter: _peft_save_keys(root, adapter) for adapter, _ in sources if adapter is not None}
        if _is_peft(root)
        else {}
    )

    mapped = {}
    missing = []
    # Persistent buffers too (e.g. DeepSeek's routing correction bias): the state dict holds them.
    tensors = list(root.named_parameters()) + _persistent_buffers(root)
    for name, param in tensors:
        owner = owner_of(name)
        if owner in absent:
            continue
        key = None
        # An adapter's own tensors come only from that adapter's file; shared ones (resized
        # embeddings, buffers) from the first file holding them.
        for adapter, saved in sources:
            if owner is not None and adapter != owner:
                continue
            candidates = [name]
            if save_keys.get(adapter, {}).get(name) is not None:
                candidates.insert(0, save_keys[adapter][name])
            if adapter is not None:
                candidates.append(name.replace(f".modules_to_save.{adapter}.", "."))
                candidates.append(_peft_saved_key(name, adapter))
            key = next((c for c in candidates if c in saved), None)
            if key is not None:
                mapped[name] = saved[key]
                break
        # A saved adapter must restore all its tensors, frozen (inactive) ones included.
        if key is None and (param.requires_grad or (owner is not None and owner in loaded)):
            missing.append(name)
    if absent:
        _print_rank0(
            f"Unsloth: native FSDP2 resume: {checkpoint} holds no weights for the inactive adapters "
            f"{absent}; they keep their current values."
        )
    if missing:
        raise RuntimeError(
            f"Unsloth: native FSDP2 resume found no saved tensor for {len(missing)} parameters "
            f"(e.g. {missing[0]}) in {checkpoint}."
        )
    n_loaded = len(mapped)  # set_model_state_dict may add entries to the dict it is given
    set_model_state_dict(
        root,
        mapped,
        options = StateDictOptions(full_state_dict = True, strict = False),
    )
    _print_rank0(
        f"Unsloth: native FSDP2 resumed {n_loaded} {'adapter ' if is_adapter else ''}tensors from "
        f"{checkpoint}."
    )
    return n_loaded


def _save_unwrapped(
    trainer,
    root,
    output_dir,
    state_dict = None,
):
    """Trainer._save with `trainer.model` set to the sharded root for the call: on a wrapper (a
    torch.compile OptimizedModule) _save would take `self.model.state_dict()`, whose `_orig_mod.`
    keys PEFT's adapter selection does not match, and write an empty adapter."""
    held = trainer.model
    trainer.model = root
    try:
        if state_dict is None:
            trainer._save(output_dir)
        else:
            trainer._save(output_dir, state_dict = state_dict)
    finally:
        trainer.model = held


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
        # Collective gather on every rank (the Trainer would only call _save on rank 0). The sharded
        # root, not a torch.compile (`_orig_mod`) / DDP-style (`.module`) wrapper around it: a
        # compiled PeftModel is still an adapter checkpoint, and its state dict keys carry no prefix.
        root = _native_root(model)
        if _is_peft(root):
            # The adapter save also writes frozen tensors (resized embeddings, inactive adapters,
            # modules_to_save): gather those with the trainable ones, PEFT picks what it writes.
            from .fsdp2_save import fsdp2_gathered
            from ..save import _fsdp2_peft_saved_frozen

            include = _fsdp2_peft_saved_frozen(root, {})
            with fsdp2_gathered(root, trainable_only = True, include = include) as writer:
                if writer is not False and self.args.should_save:
                    _save_unwrapped(self, root, output_dir)
        else:
            from .fsdp2_save import _dist, _wait_for_writer

            state_dict = native_full_state_dict(root)
            writer = bool(self.args.should_save)
            try:
                if writer:
                    _save_unwrapped(self, root, output_dir, state_dict = state_dict)
            finally:
                # As in fsdp2_gathered: the other ranks wait for the (possibly multi-GB) write
                # through the store, so they cannot run ahead into the optimizer save's or the next
                # step's collectives and hit their timeout while rank 0 is still writing.
                dist = _dist()
                if dist is not None:
                    _wait_for_writer(dist, writer)
        if getattr(self.args, "push_to_hub", False) and not _internal_call:
            self.push_to_hub(commit_message = "Model save", revision = self.args.hub_revision)

    original_move = getattr(Trainer, "_move_model_to_device", None)

    if original_move is not None:

        @functools.wraps(original_move)
        def _move_model_to_device(self, model, device):
            # shard_model already put every shard and replicated tensor on the mesh device (or keeps
            # the shards on the CPU under CPUOffloadPolicy); Trainer.__init__ only skips this placement
            # for is_fsdp_enabled, which native FSDP2 leaves False. `.to(device)` would move offloaded
            # shards to the GPU.
            if is_native_fsdp2(model):
                return None
            return original_move(self, model, device)

        Trainer._move_model_to_device = _move_model_to_device

    original_create_optimizer = Trainer.create_optimizer

    @functools.wraps(original_create_optimizer)
    def create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        if is_native_fsdp2(getattr(self, "model", None)):
            # Before the LR scheduler records one base LR per group.
            _split_mixed_param_groups(getattr(optimizer, "optimizer", optimizer))
        return optimizer

    original_create_scheduler = Trainer.create_scheduler

    @functools.wraps(original_create_scheduler)
    def create_scheduler(
        self,
        num_training_steps,
        optimizer = None,
        *args,
        **kwargs,
    ):
        # A subclass create_optimizer that builds its own optimizer (UnslothTrainer with
        # embedding_learning_rate / loraplus_lr_ratio, DecisionTrainer) never reaches the base hook
        # above, and transformers 4.x builds the scheduler before Accelerator.prepare splits the
        # groups. Split here, before the scheduler records one base LR (and lambda) per group.
        target = optimizer if optimizer is not None else getattr(self, "optimizer", None)
        if (
            target is not None
            and getattr(self, "lr_scheduler", None) is None
            and is_native_fsdp2(getattr(self, "model", None))
            and _split_mixed_param_groups(_inner_optimizer(target))
        ):
            _print_rank0(
                "Unsloth: native FSDP2 split optimizer groups mixing sharded and replicated "
                "parameters (each kind needs its own foreach / fused group) before the LR scheduler."
            )
        return original_create_scheduler(self, num_training_steps, optimizer, *args, **kwargs)

    original_save_optimizer = Trainer._save_optimizer_and_scheduler

    @functools.wraps(original_save_optimizer)
    def _save_optimizer_and_scheduler(self, output_dir):
        model = getattr(self, "model", None)
        if not is_native_fsdp2(model) or getattr(self, "is_fsdp_enabled", False):
            return original_save_optimizer(self, output_dir)
        # Collective full optimizer state (the original saves rank 0's shards only), then the scheduler.
        native_save_optimizer(
            model,
            self.optimizer,
            os.path.join(output_dir, _OPTIMIZER_NAME),
            should_save = self.args.should_save,
        )
        scheduler = getattr(self, "lr_scheduler", None)
        if self.args.should_save and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(output_dir, _SCHEDULER_NAME))

    original_load_optimizer = Trainer._load_optimizer_and_scheduler

    @functools.wraps(original_load_optimizer)
    def _load_optimizer_and_scheduler(self, checkpoint, *args, **kwargs):
        model = getattr(self, "model", None)
        if (
            checkpoint is None
            or not is_native_fsdp2(model)
            or getattr(self, "is_fsdp_enabled", False)
            or not os.path.isfile(os.path.join(checkpoint, _OPTIMIZER_NAME))
            or not os.path.isfile(os.path.join(checkpoint, _SCHEDULER_NAME))
        ):
            return original_load_optimizer(self, checkpoint, *args, **kwargs)
        native_load_optimizer(model, self.optimizer, os.path.join(checkpoint, _OPTIMIZER_NAME))
        self.lr_scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint, _SCHEDULER_NAME), weights_only = True)
        )

    original_load_from_checkpoint = Trainer._load_from_checkpoint

    @functools.wraps(original_load_from_checkpoint)
    def _load_from_checkpoint(
        self,
        resume_from_checkpoint,
        model = None,
    ):
        target = model if model is not None else getattr(self, "model", None)
        if not is_native_fsdp2(target) or getattr(self, "is_fsdp_enabled", False):
            return original_load_from_checkpoint(self, resume_from_checkpoint, model)
        # Copying full tensors into DTensor parameters (load_state_dict / PEFT load_adapter) cannot work.
        native_load_model_checkpoint(target, resume_from_checkpoint)

    original_load_best_model = getattr(Trainer, "_load_best_model", None)

    if original_load_best_model is not None:

        @functools.wraps(original_load_best_model)
        def _load_best_model(self, *args, **kwargs):
            model = getattr(self, "model", None)
            if not is_native_fsdp2(model) or getattr(self, "is_fsdp_enabled", False):
                return original_load_best_model(self, *args, **kwargs)
            # load_best_model_at_end: the same native load as a resume (collective, every rank).
            native_load_model_checkpoint(model, self.state.best_model_checkpoint)

        Trainer._load_best_model = _load_best_model

    original_create_accelerator = getattr(Trainer, "create_accelerator_and_postprocess", None)
    if original_create_accelerator is not None:

        @functools.wraps(original_create_accelerator)
        def create_accelerator_and_postprocess(self, *args, **kwargs):
            out = original_create_accelerator(self, *args, **kwargs)
            accelerator = getattr(self, "accelerator", None)
            if accelerator is not None:
                # The Trainer owns this accelerator: its prepare / save paths are the supported ones.
                setattr(accelerator, _TRAINER_ACCELERATOR, True)
            return out

        Trainer.create_accelerator_and_postprocess = create_accelerator_and_postprocess
    Trainer.save_model = save_model
    Trainer.create_optimizer = create_optimizer
    Trainer.create_scheduler = create_scheduler
    Trainer._save_optimizer_and_scheduler = _save_optimizer_and_scheduler
    Trainer._load_optimizer_and_scheduler = _load_optimizer_and_scheduler
    Trainer._load_from_checkpoint = _load_from_checkpoint
    setattr(Trainer, _PATCHED, True)
    return True


def _is_peft(model):
    model = _native_root(model)
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

    class _ShardedGradScaler(ShardedGradScaler):
        # ShardedGradScaler.unscale_ makes found_inf shape [1]; fused optimizers subtract it in place
        # from their 0-dim step counts, which fails. Keep it 0-dim as GradScaler does.
        def unscale_(self, optimizer):
            super().unscale_(optimizer)
            if not self.is_enabled():
                return
            state = self._per_optimizer_states[id(optimizer)]
            state["found_inf_per_device"] = {
                d: t.reshape(()) for d, t in state["found_inf_per_device"].items()
            }

    kwargs = {}
    handler = getattr(accelerator, "scaler_handler", None)
    if handler is not None and hasattr(handler, "to_kwargs"):
        kwargs = handler.to_kwargs()
    kwargs.setdefault("device", getattr(scaler, "_device", "cuda"))
    accelerator.scaler = _ShardedGradScaler(**kwargs)
    return True


def _native_optimizer(optimizer):
    """Whether an optimizer steps parameters shard_model produced (marked on each of them)."""
    groups = getattr(_inner_optimizer(optimizer), "param_groups", None) or []
    return any(
        getattr(p, "_unsloth_fsdp2_native_param", False) is True
        for g in groups
        for p in g["params"]
    )


# bitsandbytes optimizers (transformers' OptimizerNames) step plain tensors through custom ops with no
# DTensor sharding strategy. Those with a torch equivalent are switched to it; the others are refused.
_BNB_OPTIMS = {
    "adamw_8bit": "adamw_torch_fused",
    "adamw_bnb_8bit": "adamw_torch_fused",
    "paged_adamw_8bit": "adamw_torch_fused",
    "paged_adamw_32bit": "adamw_torch_fused",
    "rmsprop_bnb": "rmsprop",
    "rmsprop_bnb_8bit": "rmsprop",
    "rmsprop_bnb_32bit": "rmsprop",
}
_BNB_UNSUPPORTED = (
    "lion_8bit",
    "lion_32bit",
    "paged_lion_8bit",
    "paged_lion_32bit",
    "ademamix",
    "ademamix_8bit",
    "paged_ademamix_32bit",
    "paged_ademamix_8bit",
)


def _swap_bnb_optimizer(training_args):
    """bitsandbytes' optimizers step plain CUDA tensors; native FSDP2 parameters are DTensors."""
    optim = getattr(training_args, "optim", None)
    if optim is None:
        return False
    name = str(getattr(optim, "value", optim)).strip().lower()
    if name in _BNB_UNSUPPORTED:
        raise ValueError(
            f"Unsloth: native FSDP2 shards parameters as DTensors, which bitsandbytes' optimizers cannot "
            f"step (optim = '{name}'), and torch has no equivalent to switch to. Use optim = "
            f"'adamw_torch' (or 'adamw_torch_fused'), or build a DTensor-compatible optimizer after "
            f"`unsloth.models.fsdp2_native.shard_model(model)` and pass it to the Trainer."
        )
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


# Trainer.__init__'s positional order (the same in transformers 4.57 and 5.x), for an __init__ whose
# signature cannot be read (a wrapper without functools.wraps).
_TRAINER_INIT_ORDER = (
    "model",
    "args",
    "data_collator",
    "train_dataset",
    "eval_dataset",
    "processing_class",
    "model_init",
    "compute_loss_func",
    "compute_metrics",
    "callbacks",
    "optimizers",
)


def native_fsdp2_trainer_init_arguments(init, args, kwargs):
    """`model`, `args`, `model_init` and `optimizers` of a Trainer.__init__(self, *args, **kwargs)
    call, passed by keyword or by position (read from the signature of `init`)."""
    import inspect

    names = ()
    try:
        params = list(inspect.signature(init).parameters.values())
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        names = tuple(p.name for p in params if p.kind in kinds)
        if names and names[0] == "self":
            names = names[1:]
    except (TypeError, ValueError):
        pass
    if "model" not in names:
        names = _TRAINER_INIT_ORDER
    wanted = ("model", "args", "model_init", "optimizers", "optimizer_cls_and_kwargs")
    out = {name: kwargs.get(name) for name in wanted}
    for name, value in zip(names, args):
        if name in out and name not in kwargs:
            out[name] = value
    return out


def _trainer_fsdp_requested(training_args):
    """TrainingArguments(fsdp = ...): transformers 5 keeps it on the arguments (`fsdp`,
    `fsdp_plugin_args`) and the Trainer builds accelerate's FSDP plugin from them; 4.x also exports
    ACCELERATE_USE_FSDP from __post_init__."""
    if training_args is None:
        return False
    return bool(getattr(training_args, "fsdp", None)) or (
        getattr(training_args, "fsdp_plugin_args", None) is not None
    )


def native_fsdp2_defers_to_trainer_fsdp(training_args):
    """TrainingArguments(fsdp = ...) asks for accelerate's FSDP. With no UNSLOTH_FSDP2_BACKEND that
    request wins, as an `accelerate launch` FSDP config does (fsdp2_launch): True, nothing is sharded
    natively. With UNSLOTH_FSDP2_BACKEND=native the two requests conflict: refused before any model
    change (accelerate would wrap the natively sharded model again)."""
    if not _trainer_fsdp_requested(training_args):
        return False
    backend = os.environ.get("UNSLOTH_FSDP2_BACKEND", "").strip().lower()
    if backend in ("native", "unsloth", "torch"):
        raise RuntimeError(
            "Unsloth: UNSLOTH_FSDP2_BACKEND=native shards the model with Unsloth's own FSDP2, but "
            "TrainingArguments(fsdp = ...) also asks accelerate to wrap it. Pick one: drop `fsdp` "
            "(and `fsdp_config`) from the TrainingArguments, or set UNSLOTH_FSDP2_BACKEND=accelerate."
        )
    _print_rank0(
        "Unsloth: TrainingArguments(fsdp = ...) selects accelerate's FSDP, so Unsloth's native FSDP2 "
        "sharding is skipped (set UNSLOTH_FSDP2_BACKEND=native and drop `fsdp` to use it)."
    )
    return True


def _trainer_uses_cpu(training_args):
    """TrainingArguments(use_cpu = True) (4.x `no_cuda`, copied to use_cpu in __post_init__) or
    ACCELERATE_USE_CPU: the Trainer's accelerator runs on the CPU. Read without `args.device`, whose
    first access sets up the distributed state."""
    if getattr(training_args, "use_cpu", False) or getattr(training_args, "no_cuda", False):
        return True
    return str(os.environ.get("ACCELERATE_USE_CPU", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_MIXED_GROUPS_REFUSAL = (
    "Unsloth: native FSDP2 got an optimizer whose parameter groups mix sharded (DTensor) and "
    "replicated parameters, together with an LR scheduler built on those groups or with optimizer state "
    "already loaded, so the groups cannot be split. Build the optimizer from "
    "`unsloth.models.fsdp2_native.split_param_groups(...)` (before the scheduler and before loading "
    "state), or let the Trainer build both (optimizers = (None, None))."
)


def _is_bnb_optimizer_class(cls):
    """A bitsandbytes optimizer class, or a subclass of one (Optimizer8bit / Optimizer2State /
    Optimizer1State), read from the MRO's modules so bitsandbytes is not imported."""
    for klass in getattr(cls, "__mro__", None) or (cls,):
        module = str(getattr(klass, "__module__", "") or "")
        if module == "bitsandbytes" or module.startswith("bitsandbytes."):
            return True
    return False


def _refuse_bnb_optimizer_instance(optimizer):
    """A prebuilt optimizer passed as Trainer(optimizers = (opt, ...)) or to accelerator.prepare:
    _swap_bnb_optimizer only edits args.optim, so a bitsandbytes instance would be kept and step
    DTensor shards it cannot handle. Torch optimizers pass."""
    if optimizer is None:
        return
    inner = _inner_optimizer(optimizer)
    if _is_bnb_optimizer_class(type(inner)):
        raise ValueError(
            f"Unsloth: native FSDP2 shards parameters as DTensors, which bitsandbytes' optimizers cannot "
            f"step (got a {type(inner).__name__}). Build a torch optimizer (torch.optim.AdamW over "
            "`unsloth.models.fsdp2_native.split_param_groups(...)`), or let the Trainer build it with "
            "optim = 'adamw_8bit', which native FSDP2 switches to torch's AdamW."
        )


def _refuse_bnb_optimizer_cls(optimizer_cls_and_kwargs):
    """Trainer(optimizer_cls_and_kwargs = (cls, kwargs)) overrides args.optim, so the bitsandbytes
    switch in _swap_bnb_optimizer never sees it; bitsandbytes' optimizers step plain CUDA tensors, not
    DTensors. Torch optimizer classes pass. Refused rather than switched: the kwargs are the class's."""
    if not isinstance(optimizer_cls_and_kwargs, (tuple, list)) or not optimizer_cls_and_kwargs:
        return
    cls = optimizer_cls_and_kwargs[0]
    if _is_bnb_optimizer_class(cls):
        raise ValueError(
            f"Unsloth: native FSDP2 shards parameters as DTensors, which bitsandbytes' optimizers cannot "
            f"step (optimizer_cls_and_kwargs = ({getattr(cls, '__name__', cls)}, ...)). Pass a torch "
            "optimizer class (torch.optim.AdamW), or drop optimizer_cls_and_kwargs and set optim = "
            "'adamw_8bit' / 'paged_adamw_8bit', which native FSDP2 switches to torch's AdamW."
        )


def _trainer_process_group(training_args, config):
    """Create the process group the way the Trainer would, before sharding needs it: TrainingArguments'
    own device setup (PartialState with ddp_backend, ddp_timeout and use_cpu), which the Trainer runs
    first thing in __init__ anyway. Without it (a namespace), pass ddp_backend / ddp_timeout on."""
    import datetime

    import torch.distributed as dist

    if training_args is None or (dist.is_available() and dist.is_initialized()):
        return
    if isinstance(getattr(type(training_args), "_setup_devices", None), functools.cached_property):
        training_args._setup_devices
        return
    backend = getattr(training_args, "ddp_backend", None)
    if backend:
        config.backend = backend
    timeout = getattr(training_args, "ddp_timeout", None)
    if timeout:
        config.timeout = datetime.timedelta(seconds = int(timeout))


def _refuse_parallelism_config(training_args):
    """transformers 5 TrainingArguments(parallelism_config = accelerate.ParallelismConfig(...)) is
    separate from `fsdp`: accelerate would prepare tensor / context / sequence parallelism or a
    replicate x shard mesh over the natively sharded model. Only a plain shard over every rank (what
    native FSDP2 does) passes. Absent on 4.x."""
    pc = getattr(training_args, "parallelism_config", None)
    if pc is None:
        return
    sizes = {
        name: int(getattr(pc, name, None) or 1)
        for name in ("tp_size", "cp_size", "sp_size", "dp_replicate_size", "dp_shard_size")
    }
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    extra = {k: v for k, v in sizes.items() if k != "dp_shard_size" and v > 1}
    if extra or sizes["dp_shard_size"] not in (1, world):
        raise RuntimeError(
            f"Unsloth: native FSDP2 shards over all {world} ranks itself, but TrainingArguments("
            f"parallelism_config = ...) asks accelerate for {', '.join(f'{k}={v}' for k, v in sizes.items())}"
            ". Drop parallelism_config, or set UNSLOTH_FSDP2_BACKEND=accelerate to let accelerate "
            "build that layout."
        )


def _refuse_tensor_parallel_model(model):
    """A model loaded with a tensor-parallel plan (from_pretrained(tp_plan = ...)) already holds
    DTensor parameters on its own mesh; the Trainer would also build a ParallelismConfig from its
    tp_size. Native FSDP2 cannot shard it again."""
    if not isinstance(model, nn.Module) or getattr(model, _NATIVE_FLAG, False):
        return
    tp_size = getattr(model, "tp_size", None)
    try:
        tp_size = int(tp_size) if tp_size is not None else None
    except (TypeError, ValueError):
        tp_size = None
    DTensor = None
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        pass
    dtensor = DTensor is not None and any(isinstance(p, DTensor) for p in model.parameters())
    if (tp_size or 1) > 1 or dtensor:
        raise RuntimeError(
            "Unsloth: native FSDP2 got a model that is already distributed (tensor-parallel tp_plan, "
            f"tp_size = {tp_size}, or DTensor parameters). Load it without tp_plan for native FSDP2, or "
            "use UNSLOTH_FSDP2_BACKEND=accelerate with a ParallelismConfig."
        )


def _refuse_clef(model):
    """DecisionTrainer's Clef models save (adapters + head through staging folders, written by
    `_save` on the rank that calls it) and resume (`_load_clef_checkpoint`, plain tensor copies) outside
    the native collective hooks: rank 0's `_save` would wait in the encoder's gathered save_pretrained
    the other ranks never enter."""
    if getattr(model, "is_clef", False) is True:
        raise RuntimeError(
            "Unsloth: native FSDP2 (UNSLOTH_FSDP2=1) does not support Clef / DecisionTrainer models: "
            "their checkpoint save and resume are not collective, so a multi-rank save would hang. "
            "Train them without UNSLOTH_FSDP2 (DDP or a single GPU)."
        )


def _refuse_q_galore(training_args):
    """UnslothTrainer builds QGaLoreAdamW8bit in its own create_optimizer when q_galore_config is set,
    bypassing args.optim (so _swap_bnb_optimizer never sees it). Its 8-bit states and 2-D projections
    expect plain tensors, not DTensor shards."""
    if getattr(training_args, "q_galore_config", None) is not None:
        raise ValueError(
            "Unsloth: native FSDP2 (UNSLOTH_FSDP2=1) cannot train with q_galore_config: Q-GaLore's "
            "8-bit optimizer projects whole 2-D weights, and native FSDP2 hands it DTensor shards. Set "
            "q_galore_config = None, or launch without UNSLOTH_FSDP2."
        )


def _refuse_projection_optim(training_args):
    """GaLore and APOLLO (optim = 'galore_*' / 'apollo_*', layerwise or not) project whole 2-D weights,
    like Q-GaLore, and the layerwise variants register per-parameter post-gradient hooks that
    transformers documents as single-GPU only. Native FSDP2 hands them DTensor shards."""
    optim = getattr(training_args, "optim", None) if training_args is not None else None
    if optim is None:
        return
    name = str(getattr(optim, "value", optim)).strip().lower()
    if name.startswith(("galore_", "apollo_")):
        raise ValueError(
            f"Unsloth: native FSDP2 (UNSLOTH_FSDP2=1) cannot train with optim = '{name}': GaLore and "
            "APOLLO project whole 2-D weights (the layerwise variants are single-GPU only), and native "
            "FSDP2 hands the optimizer DTensor shards. Use optim = 'adamw_torch', or launch without "
            "UNSLOTH_FSDP2."
        )


def _refuse_accelerate_fp8(training_args):
    """accelerate's FP8 training (mixed_precision = "fp8": Transformer Engine, torchao or MS-AMP)
    converts the model's Linear layers in Accelerator.prepare, after native sharding captured the
    DTensors, so the optimizer would target replacement parameters FSDP2 does not manage. The Trainer
    gets it from ACCELERATE_MIXED_PRECISION (accelerate launch / config) unless fp16 / bf16 is set:
    transformers 5 stores the result as args.mixed_precision, 4.x leaves it in the env. Unsloth's own
    float8 weights are not affected (kept replicated, never converted by accelerate)."""
    mode = getattr(training_args, "mixed_precision", None) if training_args is not None else None
    if mode is None:
        mode = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if training_args is not None and (
            getattr(training_args, "fp16", False) or getattr(training_args, "bf16", False)
        ):
            mode = "fp16"
    if str(mode).strip().lower() == "fp8":
        raise RuntimeError(
            "Unsloth: native FSDP2 cannot train with accelerate's FP8 mixed precision (mixed_precision = "
            "'fp8', from ACCELERATE_MIXED_PRECISION or the accelerate config): accelerate converts the "
            "layers after Unsloth sharded them. Use bf16 / fp16 (TrainingArguments(bf16 = True)), or "
            "UNSLOTH_FSDP2_BACKEND=accelerate."
        )


def _refuse_accelerate_backends(training_args):
    """Every accelerate backend native FSDP2 cannot run under (fsdp2_launch.NATIVE_BACKEND_TABLE:
    DeepSpeed, Megatron-LM, XLA, NPU / MLU / SDAA / MUSA / HPU / XPU / Neuron devices, SageMaker DP, a
    ParallelismConfig env), from the env and the TrainingArguments (`deepspeed`, `deepspeed_plugin`,
    `hf_deepspeed_config`, ...): the import-time check only sees the launch env, and accelerate would
    otherwise build that backend over the DTensor model. Before any model change."""
    from ..fsdp2_launch import refuse_native_backend_conflicts
    refuse_native_backend_conflicts(training_args, _trainer_uses_cpu(training_args))


def _refuse_save_on_each_node(training_args):
    """save_on_each_node makes every node's local rank 0 a writer, but the collective gathers
    (native_full_state_dict, native_save_optimizer, fsdp2_gathered) leave the full state on global
    rank 0 only: the other nodes would write empty model and optimizer files. One node is fine."""
    if not getattr(training_args, "save_on_each_node", False):
        return
    try:
        multi_node = int(os.environ.get("WORLD_SIZE", "1")) > int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError):
        multi_node = True  # cannot tell: assume the launch spans nodes
    if multi_node:
        raise RuntimeError(
            "Unsloth: native FSDP2 gathers the full model and optimizer state on global rank 0 only, "
            "so `save_on_each_node = True` would make the other nodes write empty checkpoints. Set "
            "`save_on_each_node = False` and save to storage every node can read."
        )


def native_fsdp2_trainer_setup(
    trainer = None,
    model = None,
    phase = "pre",
    optimizers = None,
    training_args = None,
    model_init = None,
    optimizer_cls_and_kwargs = None,
):
    """Called around Trainer.__init__. pre: shard `model` (LoRA or full finetuning) before the
    optimizer and accelerate exist. post: float16 scaler. Returns the (sharded) model."""
    if not native_fsdp2_requested():
        return model
    install_native_fsdp2_hooks()
    if phase == "pre":
        if native_fsdp2_defers_to_trainer_fsdp(training_args):
            return model
        _refuse_accelerate_backends(training_args)
        _refuse_accelerate_fp8(training_args)
        _refuse_save_on_each_node(training_args)
        _refuse_bnb_optimizer_cls(optimizer_cls_and_kwargs)
        _refuse_q_galore(training_args)
        _refuse_projection_optim(training_args)
        _refuse_clef(model)
        _refuse_parallelism_config(training_args)
        _refuse_tensor_parallel_model(model)
        if model_init is not None:
            # Trainer.train / hyperparameter_search call model_init again for a fresh model (and
            # rebuild the accelerator per trial), which would train unsharded.
            raise RuntimeError(
                "Unsloth: native FSDP2 does not support `Trainer(model_init = ...)` or "
                "`hyperparameter_search`: the model model_init returns would not be sharded. Build "
                "the model first and pass it as `Trainer(model = model)`."
            )
        optimizer = optimizers[0] if isinstance(optimizers, (tuple, list)) and optimizers else None
        scheduler = (
            optimizers[1] if isinstance(optimizers, (tuple, list)) and len(optimizers) > 1 else None
        )
        _refuse_bnb_optimizer_instance(optimizer)
        if is_native_fsdp2(model) and optimizer is not None and _mixed_param_groups(optimizer):
            # The Trainer would split the groups after a scheduler recorded one base LR (and lambda)
            # per group (the new groups keep their initial LR; a resumed scheduler state no longer
            # matches), and a split cannot carry loaded per-parameter state, so it is skipped and the
            # mixed group reaches a foreach / fused kernel. Only a fresh optimizer alone is split.
            if scheduler is not None or getattr(_inner_optimizer(optimizer), "state", None):
                raise RuntimeError(_MIXED_GROUPS_REFUSAL)
        if is_native_fsdp2(model):
            _swap_bnb_optimizer(
                training_args
            )  # the Trainer still builds the optimizer from args.optim
        if model is None or not isinstance(model, nn.Module) or is_native_fsdp2(model):
            return model
        if optimizer is not None:
            raise RuntimeError(
                "Unsloth: native FSDP2 shards the model inside Trainer.__init__, which replaces its "
                "parameters, so an optimizer built before the Trainer would update stale tensors. Call "
                "`unsloth.models.fsdp2_native.shard_model(model)` before building the optimizer, or "
                "let the Trainer build it (optimizers = (None, None))."
            )
        _swap_bnb_optimizer(training_args)
        config = FSDP2Config.from_env()
        if _trainer_uses_cpu(training_args):
            config.device = "cpu"  # the Trainer will run on the CPU: no GPU mesh
        _trainer_process_group(training_args, config)
        return shard_model(model, config)
    if trainer is not None and is_native_fsdp2(getattr(trainer, "model", None)):
        _scaler_for_native(trainer)
    return model
