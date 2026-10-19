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
``UNSLOTH_FSDP2_FIT_FRACTION`` (default 0.4 of one device's memory).
"""

from dataclasses import dataclass, field
import os

import torch

__all__ = [
    "FSDP2Layout",
    "plan_fsdp2_layout",
    "unit_ignored_params",
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
    try:
        if device is None:
            device = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return 0


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
            "reasons": list(self.reasons),
        }


def plan_fsdp2_layout(
    model,
    param_dtype = None,
    reshard_after_forward = None,
    device = None,
):
    """Decide the layout. ``reshard_after_forward`` None = auto; an explicit bool / int is kept."""
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
    # "Fits" = the whole model (what DDP keeps on every GPU) uses at most `fraction` of one device,
    # leaving the rest for activations, gradients and optimizer state.
    layout.fits = (
        layout.device_bytes > 0
        and (layout.frozen_bytes + layout.trainable_bytes) <= fraction * layout.device_bytes
    )
    layout.reasons.append(
        f"fits={layout.fits} (model {(layout.frozen_bytes + layout.trainable_bytes) / 2**30:.2f} GiB vs {fraction} x {layout.device_bytes / 2**30:.1f} GiB)"
    )

    replicate = _env_tristate("UNSLOTH_FSDP2_REPLICATE_FROZEN")
    if replicate is None:
        replicate = layout.fits and len(frozen) > 0 and len(trainable) > 0
    layout.replicate_frozen = bool(replicate and frozen and trainable)

    env_reshard = _env_tristate("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD")
    if reshard_after_forward is not None:
        layout.reshard_after_forward = reshard_after_forward
        layout.reasons.append("reshard_after_forward from config")
    elif env_reshard is not None:
        layout.reshard_after_forward = env_reshard
        layout.reasons.append("reshard_after_forward from env")
    else:
        layout.reshard_after_forward = not layout.fits
    layout.prefetch = os.environ.get("UNSLOTH_FSDP2_PREFETCH", "1").strip().lower() not in _FALSE
    # With the frozen weights replicated, what is left to shard (LoRA adapters) is small: one root
    # unit = one all-gather + one reduce-scatter per step instead of a hook pair and two collectives
    # per block, and the per-block saving it gives up is a few MB.
    units = os.environ.get("UNSLOTH_FSDP2_TRAINABLE_UNITS", "auto").strip().lower()
    small = layout.trainable_bytes <= min(1 << 30, 0.01 * layout.device_bytes)
    layout.root_only = layout.replicate_frozen and (units == "root" or (units == "auto" and small))
    layout.reasons.append(
        f"root_only={layout.root_only} (trainable {layout.trainable_bytes / 2**20:.1f} MiB)"
    )
    return layout


def unit_ignored_params(module, layout):
    """Frozen params of `module` to pass as fully_shard(ignored_params = ...), or None."""
    if not layout.replicate_frozen:
        return None
    ignored = {p for p in module.parameters() if not p.requires_grad}
    return ignored or None


def _has_managed_params(module, ignored):
    return any(ignored is None or p not in ignored for p in module.parameters())


def apply_fsdp2_prefetch(units):
    """Explicit one-ahead prefetch: forward gathers block i+1 while i runs, backward gathers i-1."""
    units = [u for u in units if hasattr(u, "set_modules_to_forward_prefetch")]
    for i, unit in enumerate(units):
        if i + 1 < len(units):
            unit.set_modules_to_forward_prefetch([units[i + 1]])
        if i > 0:
            unit.set_modules_to_backward_prefetch([units[i - 1]])
    return len(units)


def shard_model_layout(
    model,
    units,
    mp_policy,
    layout,
    mesh = None,
    offload_policy = None,
):
    """fully_shard each unit then the root following `layout`. Returns the units that were wrapped.

    The last unit never reshards after forward (backward starts there). Units with no parameter left
    to manage (all frozen and replicated) are not wrapped.
    """
    from torch.distributed.fsdp import fully_shard

    kwargs = {"mp_policy": mp_policy}
    if mesh is not None:
        kwargs["mesh"] = mesh
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy
    wrapped = []
    for unit in [] if layout.root_only else units:
        ignored = unit_ignored_params(unit, layout)
        if not _has_managed_params(unit, ignored):
            continue
        fully_shard(
            unit,
            reshard_after_forward = layout.reshard_after_forward,
            ignored_params = ignored,
            **kwargs,
        )
        wrapped.append(unit)
    if (
        wrapped
        and layout.reshard_after_forward is not False
        and hasattr(wrapped[-1], "set_reshard_after_forward")
    ):
        wrapped[-1].set_reshard_after_forward(False, recurse = False)
    # The root holds embeddings / final norm / lm_head; it is used again first in backward.
    fully_shard(
        model,
        reshard_after_forward = False,
        ignored_params = unit_ignored_params(model, layout),
        **kwargs,
    )
    if layout.prefetch and len(wrapped) > 1:
        apply_fsdp2_prefetch(wrapped)
    model._unsloth_fsdp2_layout = layout.as_dict()
    return wrapped
