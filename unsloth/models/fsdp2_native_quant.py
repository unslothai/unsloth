# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bnb 4-bit shardable under native FSDP2; after_native_shard checks no trainable param skips FSDP2."""

__all__ = [
    "native_quant_context",
    "after_native_shard",
    "sharding_report",
]

import contextlib

import torch

from .fsdp2_qlora import fsdp2_qlora_wrap, is_params4bit


def _dtensor_class():
    try:
        from torch.distributed.tensor import DTensor
        return DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
            return DTensor
        except ImportError:
            return None


@contextlib.contextmanager
def native_quant_context(model, dtype = None):
    """Enter around the fully_shard calls. Yields None for a model without bitsandbytes 4-bit weights
    (nothing is touched), else the `prepare_fsdp2_qlora` info (`params4bit`, `converted`, `patched`,
    `storage`, `reverted`). `dtype`: the float storage of the packed bytes, the mixed precision
    param_dtype (default: the weights' compute dtype). load_in_4bit = False models are left exactly
    as they are."""
    with fsdp2_qlora_wrap(model, native = True, dtype = dtype) as info:
        yield info


def sharding_report(model):
    """What FSDP2 manages after the wrap: counts of DTensor parameters, bitsandbytes 4-bit weights
    sharded, and the names of parameters FSDP2 does not manage (trainable ones get no reduction)."""
    DTensor = _dtensor_class()
    report = {
        "dtensor": 0,
        "params4bit_sharded": 0,
        "unsharded": [],
        "unsharded_trainable": [],
    }
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if DTensor is not None and isinstance(param, DTensor):
            report["dtensor"] += 1
            if getattr(param, "quant_state", None) is not None:
                report["params4bit_sharded"] += 1
            continue
        report["unsharded"].append(name)
        if param.requires_grad:
            report["unsharded_trainable"].append(name)
    return report


def after_native_shard(model):
    """After `fully_shard(model)`: install the DCP state dict handling of bitsandbytes 4-bit weights,
    and average the gradient of any trainable parameter FSDP2 does not manage over ranks (like DDP),
    so no replica drifts. Returns `sharding_report(model)` plus `reduced` (those parameter names)."""
    from .fsdp2_save import install_dcp_bnb_state_dict, reduce_unsharded_trainable_grads

    if any(
        is_params4bit(p) or getattr(p, "quant_state", None) is not None for p in model.parameters()
    ):
        install_dcp_bnb_state_dict()
    report = sharding_report(model)
    report["reduced"] = (
        reduce_unsharded_trainable_grads(model) if report["unsharded_trainable"] else []
    )
    return report
