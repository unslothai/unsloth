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
    "mesh_writer_rank",
    "writes_off_rank0",
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


_SAVE_COUNTER = {}


def _save_timeout():
    try:
        return datetime.timedelta(
            seconds = float(os.environ.get("UNSLOTH_FSDP2_SAVE_TIMEOUT", 7 * 86400))
        )
    except ValueError:
        return datetime.timedelta(days = 7)


def _unit_mesh(unit):
    """The mesh an FSDP2 unit shards its parameters over, whatever state the parameters are in."""
    try:
        group = unit._get_fsdp_state()._fsdp_param_group
        return group.mesh_info.mesh if group is not None else None
    except Exception:
        return None


def mesh_writer_rank(model):
    """Global rank that writes a collective save of `model`: the lowest rank of the mesh its FSDP2
    parameters live on. Rank 0 for the default mesh over every rank (and without a process group); a
    sub-mesh that leaves out rank 0 (several models on disjoint ranks, FSDP2Config.mesh) writes from
    its own first rank, since rank 0 never sees that model's gathered weights."""
    dist = _dist()
    if dist is None:
        return 0
    DTensor = _dtensor_class()
    mesh = None
    if isinstance(model, torch.nn.Module):
        if DTensor is not None:
            mesh = next(
                (p.device_mesh for p in model.parameters() if isinstance(p, DTensor)), None
            )
        if mesh is None:
            # An eval forward or set_reshard_after_forward(False) can leave every parameter
            # unsharded (plain tensors): the FSDP units still hold their mesh.
            mesh = next(filter(None, map(_unit_mesh, fsdp2_units(model))), None)
        if mesh is None:
            mesh = model.__dict__.get("_unsloth_fsdp2_mesh", None)
    try:
        return int(mesh.mesh.min()) if mesh is not None else 0
    except Exception:
        return 0


@contextlib.contextmanager
def writes_off_rank0(model, writer_rank):
    """transformers 5 `PreTrainedModel.save_pretrained` writes only on global rank 0 once a process
    group exists (`should_save_on_this_rank`, transformers/distributed/mixin.py), whatever
    `is_main_process` says: a sub-mesh writer (`writer_rank` != 0) would silently write nothing.
    Lets every PreTrainedModel inside `model` write on this rank for the block; a no-op on rank 0 and
    on transformers without that check."""
    patched = []
    if writer_rank != 0 and isinstance(model, torch.nn.Module):
        for module in model.modules():
            if callable(getattr(module, "should_save_on_this_rank", None)) and (
                "should_save_on_this_rank" not in module.__dict__
            ):
                module.should_save_on_this_rank = lambda is_main_process: bool(is_main_process)
                patched.append(module)
    try:
        yield
    finally:
        for module in patched:
            module.__dict__.pop("should_save_on_this_rank", None)


def _wait_for_writer(dist, writer, writer_rank = 0):
    """The other ranks wait for the writer's write, so no later collective overtakes it. A store key,
    not a barrier: a merged or quantized export (and a Hub upload) can outlast the process group's
    collective timeout, and NCCL ranks blocked in a barrier would abort before the writer arrives.
    Keys are per writer rank (`mesh_writer_rank`), so disjoint sub-meshes saving at the same time do
    not release each other."""
    _SAVE_COUNTER[writer_rank] = _SAVE_COUNTER.get(writer_rank, 0) + 1
    count = _SAVE_COUNTER[writer_rank]
    key = (
        f"unsloth_fsdp2_save_{count}"
        if writer_rank == 0
        else f"unsloth_fsdp2_save_r{writer_rank}_{count}"
    )
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
    local = param.to_local()
    mesh_type = param.device_mesh.device_type
    if local.device.type == "cpu" and mesh_type != "cpu":
        # CPUOffloadPolicy keeps shards on the host; NCCL can only gather device tensors.
        device = torch.device(mesh_type, getattr(torch, mesh_type).current_device())
        local = local.to(device)
        if getattr(param, "quant_state", None) is None or param.dtype == torch.uint8:
            return DTensor.from_local(
                local, param.device_mesh, param.placements, shape = param.shape, stride = param.stride()
            ).full_tensor()
    elif getattr(param, "quant_state", None) is None or param.dtype == torch.uint8:
        return param.full_tensor()
    as_bytes = local.view(torch.uint8)
    shape = list(param.shape)
    shape[-1] = shape[-1] * param.element_size()
    stride = []
    acc = 1
    for size in reversed(shape):
        stride.insert(0, acc)
        acc *= size
    gathered = DTensor.from_local(
        as_bytes,
        param.device_mesh,
        param.placements,
        shape = torch.Size(shape),
        stride = tuple(stride),
    ).full_tensor()
    return gathered.view(param.dtype)


def _rebuild(param, full, module):
    """A full parameter that looks like the one FSDP2 sharded (Params4bit keeps its QuantState)."""
    quant_state = getattr(param, "quant_state", None)
    if quant_state is None:
        return torch.nn.Parameter(full, requires_grad = param.requires_grad)
    from bitsandbytes.nn import Params4bit

    as_uint8 = (
        full.is_floating_point()
        and full.dim() == 2
        and full.shape[-1] == 1
        and full.is_contiguous()
    )
    if as_uint8:
        # FSDP2 packs the 4-bit bytes in the compute dtype only to shard them. Saved (merged_4bit
        # requantizes with the weight's own quant_storage) they are the uint8 bitsandbytes writes
        # without FSDP2, so the checkpoint matches a single-GPU one and its quantization_config.
        full = full.reshape(-1).view(torch.uint8).reshape(-1, 1)
    new = torch.Tensor._make_subclass(Params4bit, full, False)
    for key in (
        "quant_state",
        "blocksize",
        "compress_statistics",
        "quant_type",
        "quant_storage",
        "bnb_quantized",
        "_original_shape",
    ):
        if hasattr(param, key):
            setattr(new, key, getattr(param, key))
    if as_uint8:
        new.quant_storage = torch.uint8
    new.module = module
    return new


@contextlib.contextmanager
def fsdp2_gathered(
    model,
    trainable_only = False,
    offload_to_cpu = True,
    include = None,
):
    """Yields None for a model FSDP2 does not shard (nothing happens). Otherwise every rank must
    enter; rank 0 sees whole parameters (and yields True), the other ranks yield False and should
    not write. `trainable_only` gathers just the trainable parameters (enough for an adapter save),
    plus those `include(name, param)` selects (frozen ones the save still writes).
    `offload_to_cpu` may be a predicate `(name, param) -> bool`.
    Packed 4-bit weights stay on their device: the merge dequantizes them there."""
    if _ACTIVE:
        yield _ACTIVE[-1]
        return
    units = fsdp2_units(model)
    DTensor = _dtensor_class()
    if not units or DTensor is None:
        yield None
        return
    for unit in units:
        # An eval forward leaves the last block and the lm_head unit gathered; make every
        # parameter a DTensor again so all ranks gather the same list.
        if hasattr(unit, "reshard"):
            unit.reshard()
    # After the reshard: an unsharded parameter is a plain tensor and carries no mesh.
    dist = _dist()
    rank = dist.get_rank() if dist is not None else 0
    writer_rank = mesh_writer_rank(model)
    writer = rank == writer_rank
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
            if (
                trainable_only
                and not param.requires_grad
                and not (include and include(name, param))
            ):
                continue
            full = _gather_full(param, DTensor)
            if not writer:
                del full
                continue
            module = owners[id(param)][0][0]
            offload = offload_to_cpu(name, param) if callable(offload_to_cpu) else offload_to_cpu
            if offload and getattr(param, "quant_state", None) is None:
                full = full.cpu()
            replaced[id(param)] = (param, _rebuild(param, full.detach(), module))
        if writer:
            for key, (old, new) in replaced.items():
                for module, name in owners[key]:
                    module._parameters[name] = new
        _ACTIVE.append(writer)
        pushed = True
        with writes_off_rank0(model, writer_rank if writer else 0):
            yield writer
    finally:
        if pushed:
            _ACTIVE.pop()
        for key, (old, new) in replaced.items():
            for module, name in owners[key]:
                module._parameters[name] = old
        replaced.clear()
        if dist is not None:
            _wait_for_writer(dist, writer, writer_rank)


def _is_4bit_base(base_layer):
    weight = getattr(base_layer, "weight", None)
    if weight is not None and getattr(weight, "quant_state", None) is not None:
        return True
    # The weight object an eval forward left unsharded need not carry the QuantState attribute.
    return any(c.__name__ == "Linear4bit" for c in type(base_layer).__mro__)


def offload_all_but_4bit_merge_operands(model):
    """`offload_to_cpu` for a merged 4-bit save: PEFT merges a LoRA layer where its packed 4-bit base
    weight stays (its GPU), so only the adapter weights of those layers stay there too. Everything
    else (embeddings, unquantized towers, LoRA on unquantized layers with their base) goes to the CPU.
    Selected by parameter name, not identity: `fsdp2_gathered` reshards first, and a parameter an eval
    forward left unsharded is a different object from the DTensor the gather then sees."""
    prefixes = []
    modules = model.named_modules() if isinstance(model, torch.nn.Module) else ()
    for module_name, module in modules:
        base_layer = getattr(module, "base_layer", None)
        if not isinstance(base_layer, torch.nn.Module) or not _is_4bit_base(base_layer):
            continue
        prefix = f"{module_name}." if module_name else ""
        for attr in (
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
            "lora_magnitude_vector",
        ):
            if isinstance(getattr(module, attr, None), torch.nn.Module):
                prefixes.append(f"{prefix}{attr}.")
    prefixes = tuple(prefixes)
    return lambda name, param: not name.startswith(prefixes)


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


def _has_sharded_4bit(args, kwargs):
    """A Linear4bit whose weight FSDP2 made a DTensor: DCP cannot resolve `weight.absmax` on it at all,
    so keeping bitsandbytes' entries only raises (Trainer.save_model -> accelerate get_state_dict)."""
    model = args[0] if args else kwargs.get("model", None)
    DTensor = _dtensor_class()
    if DTensor is None or not isinstance(model, torch.nn.Module):
        return False
    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:
        return False
    return any(
        isinstance(m, Linear4bit) and isinstance(getattr(m, "weight", None), DTensor)
        for m in model.modules()
    )


def _has_native_replicated_4bit(args, kwargs):
    """A natively sharded model (fsdp2_native, fit layout) whose Linear4bit weights FSDP2 left replicated:
    they keep bitsandbytes' own `weight.absmax` / `quant_state` entries, which a strict DCP load rejects
    as unexpected keys although the in-memory QuantState already equals them."""
    model = args[0] if args else kwargs.get("model", None)
    if not isinstance(model, torch.nn.Module) or not getattr(model, "_unsloth_fsdp2_native", False):
        return False
    DTensor = _dtensor_class()
    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:
        return False
    return any(
        isinstance(m, Linear4bit)
        and not (DTensor is not None and isinstance(getattr(m, "weight", None), DTensor))
        for m in model.modules()
    )


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
        def wrapper(
            *args,
            _fn = fn,
            **kwargs,
        ):
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
    and bias; the QuantState is unchanged in memory. A full DCP state dict of a sharded 4-bit model is
    built the same way and then gets bitsandbytes' entries back under their usual keys
    (`_add_quant_state_entries`), which loading drops again. Other DCP calls are untouched."""
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

    def _wrap(fn, name):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            adapter_only = _drops_frozen_weights(kwargs)
            full_4bit = not adapter_only and _has_sharded_4bit(args, kwargs)
            if not (adapter_only or full_4bit):
                if name.startswith("set_") and _has_native_replicated_4bit(args, kwargs):
                    args, kwargs = _without_quant_state_entries(name, args, kwargs)
                return fn(*args, **kwargs)
            if full_4bit and name.startswith("set_"):
                args, kwargs = _without_quant_state_entries(name, args, kwargs)
            _DCP_DEPTH[0] += 1
            try:
                out = fn(*args, **kwargs)
            finally:
                _DCP_DEPTH[0] -= 1
            if full_4bit and name in ("get_model_state_dict", "get_state_dict"):
                model = args[0] if args else kwargs.get("model")
                state_dict = out[0] if name == "get_state_dict" else out
                _add_quant_state_entries(model, state_dict, kwargs.get("options", None))
            return out

        wrapper._unsloth_fsdp2_patched = True
        return wrapper

    for name in _DCP_FUNCTIONS:
        fn = getattr(dcp_state_dict, name, None)
        if fn is not None and not getattr(fn, "_unsloth_fsdp2_patched", False):
            setattr(dcp_state_dict, name, _wrap(fn, name))
    return True


def _linear4bit_entries(model):
    """(state dict key prefix, module) of every bitsandbytes Linear4bit with a QuantState, keyed the way
    torch.distributed.checkpoint names them (no torch.compile `_orig_mod.` segment)."""
    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:
        return []
    entries = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        if getattr(getattr(module, "weight", None), "quant_state", None) is None:
            continue
        prefix = name.replace("_orig_mod.", "")
        entries.append((prefix + "." if prefix else "", module))
    return entries


def _add_quant_state_entries(model, state_dict, options):
    """A full (not adapter-only) DCP state dict of a sharded 4-bit model: DCP cannot name bitsandbytes'
    `weight.absmax` / `weight.quant_state.*` entries (they are not attribute paths of a DTensor), so
    they are built without them and added here, next to each packed weight the dict holds. Without them
    the packed bytes cannot be dequantized after a reload. The QuantState is replicated on every rank."""
    if not isinstance(state_dict, dict) or not isinstance(model, torch.nn.Module):
        return
    cpu = bool(getattr(options, "cpu_offload", False))
    for prefix, module in _linear4bit_entries(model):
        if prefix + "weight" not in state_dict:
            continue
        for key, value in module.weight.quant_state.as_dict(packed = True).items():
            value = value.detach()
            state_dict[prefix + "weight." + key] = value.cpu() if cpu else value


def _without_quant_state_entries(name, args, kwargs):
    """Loading: drop the entries `_add_quant_state_entries` wrote (DCP would reject them as unexpected
    keys); the QuantState is rebuilt identically when the 4-bit model is loaded again."""
    model = args[0] if args else kwargs.get("model")
    if not isinstance(model, torch.nn.Module):
        return args, kwargs
    drop = set()
    for prefix, module in _linear4bit_entries(model):
        drop.update(
            prefix + "weight." + key for key in module.weight.quant_state.as_dict(packed = True)
        )
    if not drop:
        return args, kwargs

    def strip(state_dict):
        if not isinstance(state_dict, dict) or not drop.intersection(state_dict):
            return state_dict
        return {k: v for k, v in state_dict.items() if k not in drop}

    if name == "set_model_state_dict":
        if len(args) > 1:
            args = (args[0], strip(args[1])) + tuple(args[2:])
        elif "model_state_dict" in kwargs:
            kwargs = dict(kwargs, model_state_dict = strip(kwargs["model_state_dict"]))
    elif "model_state_dict" in kwargs:
        kwargs = dict(kwargs, model_state_dict = strip(kwargs["model_state_dict"]))
    return args, kwargs


def _in_backward():
    try:
        return torch._C._current_graph_task_id() != -1
    except Exception:
        return False


def model_mesh(model):
    """The device mesh `model` is sharded over: the one shard_model recorded, else its first DTensor
    parameter's; None when neither exists (the default world group applies)."""
    mesh = model.__dict__.get("_unsloth_fsdp2_mesh", None)
    if mesh is not None:
        return mesh
    DTensor = _dtensor_class()
    if DTensor is None:
        return None
    for p in model.parameters():
        if isinstance(p, DTensor):
            return p.device_mesh
    return None


def mesh_all_reduce(tensor, mesh, dist):
    """Sum `tensor` in place over every rank of `mesh` (the world when None), one all_reduce per mesh
    dimension (the groups of each dimension partition the mesh, so an HSDP mesh is summed whole).
    Returns the number of ranks summed. A sub-mesh (several models or replicas on disjoint ranks)
    reduces only over its own ranks."""
    if mesh is None:
        dist.all_reduce(tensor)
        return dist.get_world_size()
    for dim in range(mesh.ndim):
        dist.all_reduce(tensor, group = mesh.get_group(dim))
    return mesh.size()


def _sync_replicated_grads(
    params,
    dist,
    mesh = None,
):
    """One all_reduce per (device, reduce dtype) over the fixed parameter list, in its order, issued
    by every rank: a gradient this rank did not produce counts as zeros, and a per-parameter
    "has a gradient" flag rides along so a parameter no rank used keeps grad None (as with DDP).
    Sums in float32 (float64 stays float64), as FSDP2's reduce_dtype does: two fp16 gradients of
    40000 overflow an fp16 sum although their mean is finite. Linear, so averaging the running
    accumulation after each micro-step equals averaging the accumulated sum once. Over the ranks of
    `mesh`, the model's own (mesh_all_reduce), not the whole world."""
    groups = {}
    for p in params:
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        groups.setdefault((str(p.device), dtype), []).append(p)
    for (device, dtype), group in groups.items():
        sizes = [p.numel() for p in group]
        total = sum(sizes)
        flat = torch.zeros(total + len(group), dtype = dtype, device = group[0].device)
        offset = 0
        for i, (p, n) in enumerate(zip(group, sizes)):
            if p.grad is not None:
                flat[offset : offset + n].copy_(p.grad.detach().reshape(-1))
                flat[total + i] = 1
            offset += n
        world = mesh_all_reduce(flat, mesh, dist)
        flat[:total].div_(world)
        used = flat[total:].tolist()  # one device sync, not one per parameter
        offset = 0
        for i, (p, n) in enumerate(zip(group, sizes)):
            if used[i] > 0:
                value = flat[offset : offset + n].view(p.shape)
                if p.grad is None:
                    p.grad = value.to(p.dtype).clone()
                else:
                    p.grad.copy_(value)
            offset += n


def reduce_unsharded_trainable_grads(model):
    """Trainable parameters FSDP2 does not manage (e.g. in `ignored_modules`) get no gradient
    reduction, so every rank would step on its own gradient and the replicas drift apart. Average
    their gradients over the model's mesh ranks like DDP, once per backward, at its end, on every rank: a per-parameter
    hook would make a rank that did not use the parameter (a branch, a routed MoE expert) skip the
    collective the others enter, and hang. The sync is queued as an autograd engine callback from a
    hook on `model`'s forward outputs (and, as a fallback, on each of these parameters), so it runs
    after all gradients are accumulated and before the optimizer step / clip. Returns the names."""
    dist = _dist()
    DTensor = _dtensor_class()
    if dist is None or DTensor is None:
        return []
    mesh = model_mesh(model)
    if (mesh.size() if mesh is not None else dist.get_world_size()) == 1:
        return []
    names = []
    seen = set()
    params = model.__dict__.setdefault("_unsloth_replicated_grad_params", [])
    state = model.__dict__.setdefault("_unsloth_replicated_grad_state", {"queued": False})

    def _sync():
        state["queued"] = False
        _sync_replicated_grads(params, dist, mesh)

    def _queue(*_):
        if not state["queued"]:
            state["queued"] = True
            torch.autograd.Variable._execution_engine.queue_callback(_sync)

    for name, param in model.named_parameters():
        if id(param) in seen or not param.requires_grad or isinstance(param, DTensor):
            continue
        seen.add(id(param))
        if getattr(param, "_unsloth_grad_all_reduce", False):
            continue
        param.register_hook(_queue)
        param._unsloth_grad_all_reduce = True
        params.append(param)
        names.append(name)
    if params and not getattr(model, "_unsloth_replicated_grad_hooked", False):

        def _pre_forward(module, args):
            if not _in_backward():
                state["queued"] = False  # a backward that raised never ran its callback

        def _post_forward(module, args, output):
            if not torch.is_grad_enabled():
                return
            from torch.utils._pytree import tree_flatten

            leaves = tree_flatten(output)[0]
            if not any(isinstance(t, torch.Tensor) for t in leaves) and isinstance(output, dict):
                leaves = list(output.values())
            for t in leaves:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    t.register_hook(_queue)

        model.register_forward_pre_hook(_pre_forward)
        model.register_forward_hook(_post_forward)
        model.__dict__["_unsloth_replicated_grad_hooked"] = True
    return names
