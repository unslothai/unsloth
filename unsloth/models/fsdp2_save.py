# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Saving an FSDP2-sharded model. Every rank must call the save: the shards are all-gathered one
parameter at a time, rank 0 swaps the full tensors (on CPU) into the modules, runs the ordinary
single-GPU save code, and puts the sharded parameters back. bitsandbytes 4-bit weights are gathered
as raw bytes and rebuilt as Params4bit with their QuantState, so a merge can still dequantize them.
"""

__all__ = [
    "fsdp2_units",
    "fsdp2_gathered",
    "fsdp2_gather_active",
    "install_dcp_bnb_state_dict",
    "reduce_unsharded_trainable_grads",
]

import contextlib
import datetime
import functools
import os

import torch

# Writer flag of the gather in progress: a nested save (save_pretrained_merged -> save_pretrained,
# or on a submodule) must neither gather again nor reshard what rank 0 swapped in.
_ACTIVE = []


def _fsdp_module_class():
    try:
        from torch.distributed.fsdp import FSDPModule
        return FSDPModule
    except ImportError:
        pass
    try:
        from torch.distributed._composable.fsdp import FSDPModule
        return FSDPModule
    except ImportError:
        return None


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


def fsdp2_units(model):
    FSDPModule = _fsdp_module_class()
    if FSDPModule is None or not isinstance(model, torch.nn.Module):
        return []
    return [m for m in model.modules() if isinstance(m, FSDPModule)]


def _dist():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist
    return None


_SAVE_COUNTER = [0]


def _save_timeout():
    try:
        return datetime.timedelta(seconds = float(os.environ.get("UNSLOTH_FSDP2_SAVE_TIMEOUT", 7 * 86400)))
    except ValueError:
        return datetime.timedelta(days = 7)


def _wait_for_writer(dist, writer):
    """The other ranks wait for rank 0's write, so no later collective overtakes it. A store key, not
    a barrier: a merged or quantized export (and a Hub upload) can outlast the process group's
    collective timeout, and NCCL ranks blocked in a barrier would abort before rank 0 arrives."""
    _SAVE_COUNTER[0] += 1
    key = f"unsloth_fsdp2_save_{_SAVE_COUNTER[0]}"
    try:
        store = dist.distributed_c10d._get_default_store()
    except Exception:
        store = None
    if store is None:
        dist.barrier()
        return
    if writer:
        store.set(key, "done")
    else:
        store.wait([key], _save_timeout())


def _gather_full(param, DTensor):
    """All ranks: the full tensor of a sharded parameter. Packed 4-bit bytes travel as uint8, so no
    float copy kernel can rewrite NaN bit patterns inside them."""
    if getattr(param, "quant_state", None) is None or param.dtype == torch.uint8:
        return param.full_tensor()
    local = param.to_local()
    as_bytes = local.view(torch.uint8)
    shape = list(param.shape)
    shape[-1] = shape[-1] * param.element_size()
    stride = []
    acc = 1
    for size in reversed(shape):
        stride.insert(0, acc)
        acc *= size
    gathered = DTensor.from_local(
        as_bytes, param.device_mesh, param.placements,
        shape = torch.Size(shape), stride = tuple(stride),
    ).full_tensor()
    return gathered.view(param.dtype)


def _rebuild(param, full, module):
    """A full parameter that looks like the one FSDP2 sharded (Params4bit keeps its QuantState)."""
    quant_state = getattr(param, "quant_state", None)
    if quant_state is None:
        return torch.nn.Parameter(full, requires_grad = param.requires_grad)
    from bitsandbytes.nn import Params4bit
    new = torch.Tensor._make_subclass(Params4bit, full, False)
    for key in ("quant_state", "blocksize", "compress_statistics", "quant_type", "quant_storage", "bnb_quantized"):
        if hasattr(param, key):
            setattr(new, key, getattr(param, key))
    new.module = module
    return new


@contextlib.contextmanager
def fsdp2_gathered(model, trainable_only = False, offload_to_cpu = True, include = None):
    """Yields None for a model FSDP2 does not shard (nothing happens). Otherwise every rank must
    enter; rank 0 sees whole parameters (and yields True), the other ranks yield False and should
    not write. `trainable_only` gathers just the trainable parameters (enough for an adapter save),
    plus those `include(name, param)` selects (frozen ones the save still writes).
    Packed 4-bit weights stay on their device: the merge dequantizes them there."""
    if _ACTIVE:
        yield _ACTIVE[-1]
        return
    units = fsdp2_units(model)
    DTensor = _dtensor_class()
    if not units or DTensor is None:
        yield None
        return
    dist = _dist()
    rank = dist.get_rank() if dist is not None else 0
    writer = rank == 0
    for unit in units:
        # An eval forward leaves the last block and the lm_head unit gathered; make every
        # parameter a DTensor again so all ranks gather the same list.
        if hasattr(unit, "reshard"):
            unit.reshard()
    owners = {}
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is not None:
                owners.setdefault(id(param), []).append((module, name))
    replaced = {}
    seen = set()
    pushed = False
    try:
        for name, param in model.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if not isinstance(param, DTensor):
                continue
            if trainable_only and not param.requires_grad and not (include and include(name, param)):
                continue
            full = _gather_full(param, DTensor)
            if not writer:
                del full
                continue
            module = owners[id(param)][0][0]
            if offload_to_cpu and getattr(param, "quant_state", None) is None:
                full = full.cpu()
            replaced[id(param)] = (param, _rebuild(param, full.detach(), module))
        if writer:
            for key, (old, new) in replaced.items():
                for module, name in owners[key]:
                    module._parameters[name] = new
        _ACTIVE.append(writer)
        pushed = True
        yield writer
    finally:
        if pushed:
            _ACTIVE.pop()
        for key, (old, new) in replaced.items():
            for module, name in owners[key]:
                module._parameters[name] = old
        replaced.clear()
        if dist is not None:
            _wait_for_writer(dist, writer)


def fsdp2_gather_active():
    return bool(_ACTIVE)


_DCP_DEPTH = [0]
_ADAPTER_ONLY = [0]
_DCP_FUNCTIONS = (
    "get_model_state_dict",
    "set_model_state_dict",
    "get_state_dict",
    "set_state_dict",
    "get_optimizer_state_dict",
    "set_optimizer_state_dict",
)


def _drops_frozen_weights(kwargs):
    """Only adapter-only state dicts (accelerate's `adapter_only` path, or an explicit
    `ignore_frozen_params`) may skip the QuantState: they never keep the frozen 4-bit weights. A full
    DCP checkpoint still needs it to dequantize them."""
    if _ADAPTER_ONLY[0]:
        return True
    return bool(getattr(kwargs.get("options", None), "ignore_frozen_params", False))


def _wrap_accelerate_adapter_only():
    try:
        import accelerate.utils.fsdp_utils as fsdp_utils
    except Exception:
        return
    for name in ("_get_model_state_dict", "_set_model_state_dict"):
        fn = getattr(fsdp_utils, name, None)
        if fn is None or getattr(fn, "_unsloth_adapter_only_flag", False):
            continue

        @functools.wraps(fn)
        def wrapper(*args, _fn = fn, **kwargs):
            if not kwargs.get("adapter_only", False):
                return _fn(*args, **kwargs)
            _ADAPTER_ONLY[0] += 1
            try:
                return _fn(*args, **kwargs)
            finally:
                _ADAPTER_ONLY[0] -= 1

        wrapper._unsloth_adapter_only_flag = True
        setattr(fsdp_utils, name, wrapper)


def install_dcp_bnb_state_dict():
    """torch.distributed.checkpoint resolves every state dict key as an attribute path, and
    bitsandbytes' `weight.absmax` / `weight.quant_state.bitsandbytes__nf4` entries are not one
    (AttributeError in `_get_fqns`). That breaks every FSDP2 Trainer adapter checkpoint of a 4-bit
    model. Inside adapter-only DCP calls (`_drops_frozen_weights`) Linear4bit reports just its weight
    and bias; the QuantState is unchanged in memory. Other DCP calls keep bitsandbytes' entries."""
    try:
        import torch.distributed.checkpoint.state_dict as dcp_state_dict
        from bitsandbytes.nn import Linear4bit
    except Exception:
        return False
    _wrap_accelerate_adapter_only()
    original_save = Linear4bit._save_to_state_dict
    if getattr(original_save, "_unsloth_fsdp2_patched", False):
        return True

    @functools.wraps(original_save)
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if _DCP_DEPTH[0]:
            return torch.nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
        return original_save(self, destination, prefix, keep_vars)

    _save_to_state_dict._unsloth_fsdp2_patched = True
    Linear4bit._save_to_state_dict = _save_to_state_dict

    def _wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _drops_frozen_weights(kwargs):
                return fn(*args, **kwargs)
            _DCP_DEPTH[0] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                _DCP_DEPTH[0] -= 1

        wrapper._unsloth_fsdp2_patched = True
        return wrapper

    for name in _DCP_FUNCTIONS:
        fn = getattr(dcp_state_dict, name, None)
        if fn is not None and not getattr(fn, "_unsloth_fsdp2_patched", False):
            setattr(dcp_state_dict, name, _wrap(fn))
    return True


def reduce_unsharded_trainable_grads(model):
    """Trainable parameters FSDP2 does not manage (e.g. in `ignored_modules`) get no gradient
    reduction, so every rank would step on its own gradient and the replicas drift apart. Average
    their gradient over ranks as it is produced, like DDP. Returns the parameter names."""
    dist = _dist()
    DTensor = _dtensor_class()
    if dist is None or DTensor is None or dist.get_world_size() == 1:
        return []
    names = []
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen or not param.requires_grad or isinstance(param, DTensor):
            continue
        seen.add(id(param))
        if getattr(param, "_unsloth_grad_all_reduce", False):
            continue
        world = dist.get_world_size()

        def _average(grad, world = world):
            grad = grad.clone()
            dist.all_reduce(grad)
            return grad.div_(world)

        param.register_hook(_average)
        param._unsloth_grad_all_reduce = True
        names.append(name)
    return names
