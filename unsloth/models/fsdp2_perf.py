# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Throughput layout for native FSDP2 when the model fits on one GPU.

Plain FSDP2 is slower than DDP for a model that fits on each GPU because it moves weights DDP never
moves: every block's frozen LoRA base weights are all-gathered in forward and again for the backward
(or gradient checkpointing) recompute, and full fine-tuning all-gathers each block twice per step.
When the model fits, the layout here keeps:

  * frozen parameters replicated (passed as ``ignored_params``): never gathered, never reduced, exactly
    like DDP, and bitsandbytes 4-bit base weights stay plain ``Params4bit``;
  * trainable parameters, their gradients and optimizer state sharded as usual;
  * the gathered trainable weights resident from forward to backward (``reshard_after_forward=False``),
    one all-gather per step instead of two;
  * explicit next-block prefetch in forward and backward;
  * when only small trainable params are left (LoRA adapters), a single root unit: one all-gather and
    one reduce-scatter per step, no per-block hooks.

Units holding no trainable parameter are not wrapped (no hooks, no empty collectives). Peak memory stays
at or below DDP: frozen weights are what DDP holds, trainable weights are held once gathered plus one
shard, and gradients plus optimizer state are sharded.

Env overrides (read only when native FSDP2 is used): ``UNSLOTH_FSDP2_REPLICATE_FROZEN`` = auto | 1 | 0,
``UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD`` = auto | 1 | 0, ``UNSLOTH_FSDP2_PREFETCH`` = 1 | 0,
``UNSLOTH_FSDP2_FIT_FRACTION`` (default 0.4 of one device's memory), ``UNSLOTH_FSDP2_FREE_FIT_FRACTION``
(default max(0.5, fit fraction / 0.8) of free + cached + resident model memory, read after process group init).
"""

from dataclasses import dataclass, field
from typing import Optional
import os

import torch

__all__ = [
    "FSDP2Layout",
    "plan_fsdp2_layout",
    "unit_ignored_params",
    "resolve_reshard_after_forward",
    "apply_fsdp2_prefetch",
    "shard_model_layout",
]

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _env_tristate(name):
    value = os.environ.get(name, "auto").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _param_bytes(params, dtype = None):
    total = 0
    seen = set()
    for p in params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        # bitsandbytes Params4bit keeps its packed storage: count what is really resident.
        size = dtype.itemsize if dtype is not None and p.is_floating_point() else p.element_size()
        total += p.numel() * size
    return total


def _device_bytes(device = None):
    """Total memory of the GPU the model trains on; 0 for a CPU (or other non-CUDA) mesh, which plans
    the per-block sharded layout as a CPU-only host does."""
    if device is not None and torch.device(device).type != "cuda":
        return 0
    try:
        if device is None:
            device = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return 0


def _device_free_bytes(device = None):
    """``mem_get_info`` free plus torch's unallocated cache; None off CUDA or if unreadable."""
    if device is not None and torch.device(device).type != "cuda":
        return None
    try:
        if device is None:
            device = torch.cuda.current_device()
        free, _ = torch.cuda.mem_get_info(device)
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return int(free) + max(int(cached), 0)
    except Exception:
        return None


def _resident_bytes(params, device):
    """Bytes of `params` already stored on `device`."""
    if device is None:
        return 0
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        try:
            target = torch.device("cuda", torch.cuda.current_device())
        except Exception:
            return 0
    return _param_bytes(p for p in params if p.device == target)


@dataclass
class FSDP2Layout:
    replicate_frozen: bool = False
    reshard_after_forward: bool = True
    prefetch: bool = False
    root_only: bool = False
    fits: bool = False
    frozen_bytes: int = 0
    trainable_bytes: int = 0
    device_bytes: int = 0
    available_bytes: Optional[int] = None
    reasons: list = field(default_factory = list)

    def as_dict(self):
        return {
            "replicate_frozen": self.replicate_frozen,
            "reshard_after_forward": self.reshard_after_forward,
            "prefetch": self.prefetch,
            "root_only": self.root_only,
            "fits": self.fits,
            "frozen_gib": round(self.frozen_bytes / 2**30, 3),
            "trainable_gib": round(self.trainable_bytes / 2**30, 3),
            "device_gib": round(self.device_bytes / 2**30, 1),
            "available_gib": None
            if self.available_bytes is None
            else round(self.available_bytes / 2**30, 2),
            "reasons": list(self.reasons),
        }


def plan_fsdp2_layout(
    model,
    param_dtype = None,
    reshard_after_forward = None,
    device = None,
    cpu_offload = False,
    fits = None,
    replicate_frozen = None,
    allow_root_only = True,
):
    """Decide the layout. ``reshard_after_forward`` None = auto; an explicit bool / int is kept.
    ``fits`` None = by device memory (``UNSLOTH_FSDP2_LAYOUT`` = fit / sharded forces it), True / False
    forces it. ``replicate_frozen`` None = auto (``UNSLOTH_FSDP2_REPLICATE_FROZEN``), else forced.
    ``allow_root_only`` False keeps per-unit wrapping (explicit units)."""
    params = list(model.parameters())
    frozen = [p for p in params if not p.requires_grad]
    trainable = [p for p in params if p.requires_grad]
    layout = FSDP2Layout(
        frozen_bytes = _param_bytes(frozen),
        trainable_bytes = _param_bytes(trainable, param_dtype),
        device_bytes = _device_bytes(device),
    )
    try:
        fraction = float(os.environ.get("UNSLOTH_FSDP2_FIT_FRACTION", "0.4"))
    except ValueError:
        fraction = 0.4
    # Scales with `fraction`, so the free-memory rule binds only once > 20% of the card is in use.
    default_free_fraction = max(0.5, fraction / 0.8)
    try:
        free_fraction = float(
            os.environ.get("UNSLOTH_FSDP2_FREE_FIT_FRACTION", str(default_free_fraction))
        )
    except ValueError:
        free_fraction = default_free_fraction
    model_bytes = layout.frozen_bytes + layout.trainable_bytes
    # "Fits" = the whole model (what DDP keeps on every GPU) uses at most `fraction` of one device,
    # leaving the rest for activations, gradients and optimizer state.
    # CPU offload asks for memory, not speed: keep the plain sharded layout.
    forced = os.environ.get("UNSLOTH_FSDP2_LAYOUT", "auto").strip().lower()
    if fits is None and forced in ("fit", "sharded"):
        fits = forced == "fit"
    if fits is not None:
        layout.fits = bool(fits) and not cpu_offload
    else:
        layout.fits = (
            not cpu_offload
            and layout.device_bytes > 0
            and model_bytes <= fraction * layout.device_bytes
        )
        if layout.fits:
            # Crowded GPUs shard; ranks AND their decision, so one crowded rank shards everyone.
            free = _device_free_bytes(device)
            if free is not None:
                layout.available_bytes = free + _resident_bytes(params, device)
                if model_bytes > free_fraction * layout.available_bytes:
                    layout.fits = False
                    layout.reasons.append(
                        f"fits=False: only {layout.available_bytes / 2**30:.2f} GiB usable on this GPU "
                        f"(free + cached + resident model), model needs <= {free_fraction} of it"
                    )
    layout.reasons.append(
        f"fits={layout.fits} (model {model_bytes / 2**30:.2f} GiB vs {fraction} x {layout.device_bytes / 2**30:.1f} GiB)"
    )

    replicate = replicate_frozen
    if replicate is None:
        replicate = _env_tristate("UNSLOTH_FSDP2_REPLICATE_FROZEN")
    if replicate is None:
        replicate = layout.fits and len(frozen) > 0 and len(trainable) > 0
    layout.replicate_frozen = bool(replicate and frozen and trainable)

    resolve_reshard_after_forward(layout, reshard_after_forward)
    # Explicit prefetch is part of the fit layout; the sharded layout keeps the caller's prefetch knobs.
    layout.prefetch = (
        layout.fits and os.environ.get("UNSLOTH_FSDP2_PREFETCH", "1").strip().lower() not in _FALSE
    )
    # With the frozen weights replicated, what is left to shard (LoRA adapters) is small: one root
    # unit = one all-gather + one reduce-scatter per step instead of a hook pair and two collectives
    # per block, and the per-block saving it gives up is a few MB.
    units = os.environ.get("UNSLOTH_FSDP2_TRAINABLE_UNITS", "auto").strip().lower()
    small = layout.trainable_bytes <= min(1 << 30, 0.01 * layout.device_bytes)
    layout.root_only = (
        allow_root_only
        and layout.replicate_frozen
        and (units == "root" or (units == "auto" and small))
    )
    layout.reasons.append(
        f"root_only={layout.root_only} (trainable {layout.trainable_bytes / 2**20:.1f} MiB)"
    )
    return layout


def resolve_reshard_after_forward(
    layout,
    reshard_after_forward = None,
    note = True,
):
    """Set ``layout.reshard_after_forward``: the explicit config value, else the env override, else
    derived from ``layout.fits`` (kept gathered when it fits). Call again after ``fits`` changes (the
    ranks agreeing on one layout) so every rank derives the same value."""
    env_reshard = _env_tristate("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD")
    if reshard_after_forward is not None:
        layout.reshard_after_forward = reshard_after_forward
        source = "config"
    elif env_reshard is not None:
        layout.reshard_after_forward = env_reshard
        source = "env"
    else:
        layout.reshard_after_forward = not layout.fits
        source = None
    if note and source is not None:
        layout.reasons.append(f"reshard_after_forward from {source}")
    return layout.reshard_after_forward


def unit_ignored_params(module, layout):
    """Frozen params of `module` to pass as fully_shard(ignored_params = ...), or None."""
    if not layout.replicate_frozen:
        return None
    ignored = {p for p in module.parameters() if not p.requires_grad}
    return ignored or None


def _has_managed_params(module, ignored):
    return any(ignored is None or p not in ignored for p in module.parameters())


def apply_fsdp2_prefetch(
    units,
    forward_depth = 1,
    backward_depth = 1,
):
    """Explicit prefetch: forward gathers blocks i+1..i+forward_depth while i runs, backward gathers
    i-1..i-backward_depth (NeMo Automodel defaults to forward 1 / backward 2)."""
    units = [u for u in units if hasattr(u, "set_modules_to_forward_prefetch")]
    for i, unit in enumerate(units):
        ahead = units[i + 1 : i + 1 + forward_depth]
        behind = units[max(0, i - backward_depth) : i][::-1]
        if ahead:
            unit.set_modules_to_forward_prefetch(ahead)
            _skip_forward_prefetch_in_backward(unit, ahead)
        if behind:
            unit.set_modules_to_backward_prefetch(behind)
    return len(units)


def _in_backward():
    task = getattr(torch._C, "_current_graph_task_id", None)
    return task is not None and task() != -1


def _skip_forward_prefetch_in_backward(unit, ahead):
    """Gradient checkpointing re-runs a block's forward inside backward, which fires its pre-forward
    hook and with it the explicit forward prefetch of block i+1: block i+1 has already run its backward
    and been resharded, so it is gathered again for nothing (one wasted all-gather per block per
    micro-batch). Clear the forward prefetch while autograd is running a backward and restore it for
    the next real forward. Public FSDPModule API only; `UNSLOTH_FSDP2_RECOMPUTE_PREFETCH=1` keeps torch's
    behaviour."""
    if os.environ.get("UNSLOTH_FSDP2_RECOMPUTE_PREFETCH", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    if getattr(unit, "_unsloth_recompute_prefetch_hooked", False):
        unit._unsloth_forward_prefetch = list(ahead)
        return
    unit._unsloth_forward_prefetch = list(ahead)
    state = {"cleared": False}

    def _pre_forward(module, args):
        if _in_backward():
            if not state["cleared"]:
                module.set_modules_to_forward_prefetch([])
                state["cleared"] = True
        elif state["cleared"]:
            module.set_modules_to_forward_prefetch(module._unsloth_forward_prefetch)
            state["cleared"] = False

    # prepend: runs before FSDP's own pre-forward hook, which issues the prefetch.
    unit.register_forward_pre_hook(_pre_forward, prepend = True)
    unit._unsloth_recompute_prefetch_hooked = True


def _env_int(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def shard_model_layout(
    model,
    units,
    mp_policy,
    layout,
    mesh = None,
    offload_policy = None,
    ignored_params = None,
    extra_units = (),
    fully_shard_fn = None,
    reshard_last_block = False,
    unit_reshard_after_forward = None,
    forward_prefetch = 0,
    backward_prefetch = 0,
    shard_kwargs = None,
):
    """fully_shard each unit then the root following `layout`. Returns the units that were wrapped.

    The last unit never reshards after forward (backward starts there) unless `reshard_last_block`.
    Units with no parameter left to manage (all frozen and replicated) are not wrapped.
    `ignored_params`: extra parameters every fully_shard call leaves replicated (minority dtype norms,
    float8 weights, ...). `extra_units` (large root embedding tables) are wrapped after the blocks with
    the layout's reshard_after_forward. `unit_reshard_after_forward(i, unit)` -> bool | int | None
    overrides one block. `forward_prefetch` / `backward_prefetch` > 0 set explicit prefetch depths;
    0 / 0 uses the layout's (one ahead, env UNSLOTH_FSDP2_PREFETCH_FWD / _BWD) when it prefetches.
    """
    if fully_shard_fn is None:
        try:
            from torch.distributed.fsdp import fully_shard as fully_shard_fn
        except ImportError:  # torch 2.4 / 2.5
            from torch.distributed._composable.fsdp import fully_shard as fully_shard_fn

    kwargs = {"mp_policy": mp_policy}
    if mesh is not None:
        kwargs["mesh"] = mesh
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy
    if shard_kwargs:
        kwargs.update(shard_kwargs)
    extra_ignored = set(ignored_params or ())

    def _ignored(module):
        own = unit_ignored_params(module, layout) or set()
        if extra_ignored:
            own = own | {p for p in module.parameters() if p in extra_ignored}
        return own or None

    def _wrap(module, reshard, ignored):
        kw = dict(kwargs)
        if ignored:
            kw["ignored_params"] = ignored
        fully_shard_fn(module, reshard_after_forward = reshard, **kw)
        module._unsloth_reshard_after_forward = reshard

    targets = []
    for unit in [] if layout.root_only else units:
        ignored = _ignored(unit)
        if _has_managed_params(unit, ignored):
            targets.append((unit, ignored))
    wrapped = []
    last = len(targets) - 1
    for i, (unit, ignored) in enumerate(targets):
        reshard = reshard_last_block if i == last else layout.reshard_after_forward
        if unit_reshard_after_forward is not None:
            chosen = unit_reshard_after_forward(i, unit)
            reshard = reshard if chosen is None else chosen
        _wrap(unit, reshard, ignored)
        wrapped.append(unit)
    for unit in [] if layout.root_only else extra_units:
        ignored = _ignored(unit)
        if _has_managed_params(unit, ignored):
            _wrap(unit, layout.reshard_after_forward, ignored)
    # The root holds embeddings / final norm / lm_head; it is used again first in backward.
    _wrap(model, False, _ignored(model))
    if len(wrapped) > 1:
        if forward_prefetch > 0 or backward_prefetch > 0:
            apply_fsdp2_prefetch(wrapped, forward_prefetch, backward_prefetch)
        elif layout.prefetch:
            apply_fsdp2_prefetch(
                wrapped,
                forward_depth = _env_int("UNSLOTH_FSDP2_PREFETCH_FWD", 1),
                backward_depth = _env_int("UNSLOTH_FSDP2_PREFETCH_BWD", 1),
            )
    model._unsloth_fsdp2_layout = layout.as_dict()
    return wrapped
