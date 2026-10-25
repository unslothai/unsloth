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
import copy
import datetime
import functools
import logging
import os

import torch

# Writer flag of the gather in progress: a nested save (save_pretrained_merged -> save_pretrained,
# or on a submodule) must neither gather again nor reshard what rank 0 swapped in.
_ACTIVE = []
# Per gather in progress: id of each packed 4-bit weight staged on the CPU -> the device it was
# gathered on, where `merge_4bit_lora_layer_by_layer` merges it.
_STAGED_4BIT = []


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
        return datetime.timedelta(
            seconds = float(os.environ.get("UNSLOTH_FSDP2_SAVE_TIMEOUT", 7 * 86400))
        )
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


def _quant_state_on_cpu(quant_state, device = "cpu"):
    """A copy of a bitsandbytes QuantState on `device` (the CPU by default). `QuantState.to` moves in
    place (its nested state too), and the live one must stay where the sharded weight computes with it."""
    if not hasattr(quant_state, "to"):
        return quant_state
    state = copy.copy(quant_state)
    if getattr(state, "state2", None) is not None:
        state.state2 = copy.copy(state.state2)
    state.to(device)
    return state


def _restage_4bit(weight, device):
    """The packed Params4bit `weight` on `device`, with its own copy of the QuantState there. Every
    other attribute is carried over as is, which is what PEFT's 4-bit merge rebuilds the weight from."""
    from bitsandbytes.nn import Params4bit

    moved = torch.Tensor._make_subclass(Params4bit, weight.data.to(device), False)
    moved.__dict__.update(weight.__dict__)
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        moved.quant_state = _quant_state_on_cpu(quant_state, device)
    return moved


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
    offload_4bit = False,
):
    """Yields None for a model FSDP2 does not shard (nothing happens). Otherwise every rank must
    enter; rank 0 sees whole parameters (and yields True), the other ranks yield False and should
    not write. `trainable_only` gathers just the trainable parameters (enough for an adapter save),
    plus those `include(name, param)` selects (frozen ones the save still writes).
    `offload_to_cpu` may be a predicate `(name, param) -> bool`.
    Packed 4-bit weights stay on their device (a merge dequantizes them there) unless `offload_4bit`:
    they then go, with a CPU copy of their QuantState, to the CPU right after each one is gathered, so
    rank 0 never holds the whole quantized base on its accelerator. A merge brings them back one layer
    at a time (`merge_4bit_lora_layer_by_layer`)."""
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
    staged = {}
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
            quant_state = getattr(param, "quant_state", None)
            home = None
            if offload and quant_state is None:
                full = full.cpu()
            elif offload and offload_4bit:
                home = full.device
                full = full.cpu()
                quant_state = _quant_state_on_cpu(quant_state)
            rebuilt = _rebuild(param, full.detach(), module)
            if quant_state is not None and rebuilt.quant_state is not quant_state:
                rebuilt.quant_state = quant_state
            if home is not None:
                staged[id(rebuilt)] = home
            replaced[id(param)] = (param, rebuilt)
        if writer:
            for key, (old, new) in replaced.items():
                for module, name in owners[key]:
                    module._parameters[name] = new
        _ACTIVE.append(writer)
        _STAGED_4BIT.append(staged)
        pushed = True
        yield writer
    finally:
        if pushed:
            _ACTIVE.pop()
            _STAGED_4BIT.pop()
        staged.clear()
        for key, (old, new) in replaced.items():
            for module, name in owners[key]:
                module._parameters[name] = old
        replaced.clear()
        if dist is not None:
            _wait_for_writer(dist, writer)


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


_MERGED_4BIT_METHODS = ("merged_4bit", "merged_4bit_forced", "forced_merged_4bit")


def _merges_onto_dequantized_4bit(model):
    """unsloth_zoo folds a merged_16bit LoRA onto dequant(W4) for a few architectures (falcon_h1),
    dequantizing the live packed weight where it sits."""
    try:
        from unsloth_zoo.saving_utils import (
            _model_type_needs_dequant_merge_base,
            find_lora_base_model,
        )
    except Exception:
        config = getattr(model, "config", None)
        return (getattr(config, "model_type", "") or "").lower() == "falcon_h1"
    try:
        model = find_lora_base_model(model)
    except Exception:
        pass
    return bool(_model_type_needs_dequant_merge_base(model))


def merged_save_gather_options(model, save_method):
    """`fsdp2_gathered` options for save_pretrained_merged / push_to_hub_merged.

    merged_16bit reads its base weights from the 16-bit checkpoint and only the LoRA weights from the
    model, so the packed 4-bit base goes to the CPU like everything else.
    merged_4bit stages the packed base on the CPU too, keeps the adapters of 4-bit layers on their
    device, and `merge_4bit_lora_layer_by_layer` brings one packed layer at a time back there to merge.
    Either way rank 0's accelerator holds one gathered layer above its shards, not the whole base.
    A merged_16bit that dequantizes the live 4-bit base (`_merges_onto_dequantized_4bit`) and other
    export methods keep the previous placement."""
    if save_method in _MERGED_4BIT_METHODS:
        return dict(offload_to_cpu = offload_all_but_4bit_merge_operands(model), offload_4bit = True)
    if save_method == "merged_16bit" and not _merges_onto_dequantized_4bit(model):
        return dict(offload_to_cpu = True, offload_4bit = True)
    return dict(offload_to_cpu = True)


def merge_4bit_lora_layer_by_layer(model):
    """Writer, inside `fsdp2_gathered` with packed 4-bit weights staged on the CPU: merges each LoRA
    layer over such a base with PEFT's own `merge`, one layer at a time, on the device that weight
    was gathered on (where the merge ran before, so the requantized bytes are the same), then stages
    the merged weight back on the CPU. PEFT's `merge_and_unload` afterwards finds those layers merged
    and only unloads them. Returns the merged layers (see `unmerge_flags`); [] outside such a gather."""
    staged = _STAGED_4BIT[-1] if _STAGED_4BIT else None
    if not staged or not isinstance(model, torch.nn.Module):
        return []
    merged = []
    for module in model.modules():
        base_layer = getattr(module, "base_layer", None)
        if not isinstance(base_layer, torch.nn.Module) or not callable(
            getattr(module, "merge", None)
        ):
            continue
        weight = base_layer._parameters.get("weight", None)
        home = staged.get(id(weight)) if weight is not None else None
        if home is None:
            continue
        before = list(getattr(module, "merged_adapters", None) or ())
        try:
            base_layer._parameters["weight"] = _restage_4bit(weight, home)
            module.merge()
            result = base_layer._parameters.get("weight", None)
            if result is not None and getattr(result, "quant_state", None) is not None:
                result = _restage_4bit(result, "cpu")
                base_layer._parameters["weight"] = result
        except BaseException:
            base_layer._parameters["weight"] = weight
            # Layers merged earlier in this loop get their unmerged shards back from the gather too.
            if list(getattr(module, "merged_adapters", None) or ()) != before:
                merged.append((module, before))
            unmerge_flags(merged)
            raise
        if list(getattr(module, "merged_adapters", None) or ()) != before:
            merged.append((module, before))
    return merged


def unmerge_flags(merged):
    """Undo `merge_4bit_lora_layer_by_layer`'s bookkeeping when the export fails before it unloads:
    the gather puts the sharded (unmerged) base weights back, so the layers must not claim a merge."""
    for module, before in merged:
        module.merged_adapters[:] = before


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


_LEGACY_4BIT_NOTICE = [False]
logger = logging.getLogger(__name__)


def _as_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _same_packed_bytes(live, loaded, broadcast_dist):
    """Whether the packed 4-bit bytes being loaded equal the live (possibly sharded) weight's. Every
    rank gathers the live weight (a collective); with `broadcast_dist` only rank 0 holds `loaded`,
    decides, and sends the verdict, so all ranks agree."""
    DTensor = _dtensor_class()
    if DTensor is not None and isinstance(live, DTensor):
        live = _gather_full(live, DTensor)
    if DTensor is not None and isinstance(loaded, DTensor):
        loaded = loaded.full_tensor()
    same = False
    if isinstance(loaded, torch.Tensor):
        a, b = _as_bytes(live), _as_bytes(loaded)
        same = a.numel() == b.numel() and torch.equal(a.cpu(), b.cpu())
    if broadcast_dist is not None:
        verdict = [same]
        broadcast_dist.broadcast_object_list(verdict, src = 0)
        same = bool(verdict[0])
    return same


def _restore_quant_state(model, state_dict, options):
    """Loading a full DCP state dict: the saved QuantState entries (`_add_quant_state_entries`) are
    copied into each live QuantState in place, so the packed bytes DCP loads are dequantized with the
    scales they were saved with. A saved QuantState that does not fit the live one (other shape,
    blocksize, quant type or nesting) raises instead of pairing the bytes with the wrong scales, as
    does one without its packed weight unless strict=False (DCP keeps the live bytes then). With
    `broadcast_from_rank0` only rank 0 holds the entries and sends them to the others."""
    entries = _linear4bit_entries(model)
    if not entries:
        return
    dist = _dist()
    broadcast = dist is not None and bool(getattr(options, "broadcast_from_rank0", False))
    if not isinstance(state_dict, dict):
        state_dict = {}
    if broadcast:
        # An object broadcast, not a tensor one: a CPU tensor cannot go through an NCCL group, and
        # broadcast_object_list stages on whatever device the group's backend needs.
        has = [
            any(
                k == prefix + "weight" or k.startswith(prefix + "weight.")
                for prefix, _ in entries
                for k in state_dict
            )
        ]
        dist.broadcast_object_list(has, src = 0)
        if not has[0]:
            return
    from bitsandbytes.functional import QuantState

    strict = bool(getattr(options, "strict", True)) if options is not None else True
    orphans = []
    for prefix, module in entries:
        live = module.weight.quant_state
        head = prefix + "weight."
        saved = {
            k[len(head) :]: v
            for k, v in state_dict.items()
            if k.startswith(head) and isinstance(v, torch.Tensor)
        }
        loads_weight = head[:-1] in state_dict
        if broadcast:
            payload = [(loads_weight, {k: v.detach().cpu() for k, v in saved.items()})]
            dist.broadcast_object_list(payload, src = 0)
            loads_weight, saved = payload[0]
        if not saved:
            if loads_weight:
                # A checkpoint written before QuantState was saved (a frozen QLoRA base): its packed
                # bytes are the live ones, so the live scales are theirs. Other bytes would be
                # dequantized with the wrong scales.
                same = _same_packed_bytes(
                    module.weight, state_dict.get(head[:-1]), dist if broadcast else None
                )
                if not same:
                    raise ValueError(
                        f"Unsloth: the checkpoint holds the 4-bit weight `{head[:-1]}` but none of "
                        "its quantization state (`weight.absmax`, `weight.quant_state.*`), and its "
                        "packed bytes differ from the model's, so they cannot be dequantized "
                        "correctly. Save the full state dict again with Unsloth, or load it without "
                        "the frozen 4-bit base weights."
                    )
                if not _LEGACY_4BIT_NOTICE[0]:
                    _LEGACY_4BIT_NOTICE[0] = True
                    logger.info(
                        "Unsloth: the checkpoint has 4-bit weights without their quantization state "
                        "(saved by an older Unsloth); their packed bytes match the model's, so its "
                        "quantization state is kept."
                    )
            continue
        if not loads_weight:
            if strict:
                raise ValueError(
                    f"Unsloth: the state dict has the 4-bit quantization state of `{head[:-1]}` but "
                    "not its packed weight. Load the weight with it, or pass strict=False to keep "
                    "the model's own weight and quantization state."
                )
            orphans.append(head[:-1])
            continue
        expected = set(live.as_dict(packed = True))
        saved_nested = any(k.startswith("nested_") for k in saved)
        if saved_nested != bool(live.nested):
            raise ValueError(
                f"Unsloth: the checkpoint's 4-bit quantization state for `{head[:-1]}` (nested "
                f"{saved_nested}) does not match the model's (nested {bool(live.nested)}). Load it "
                "into a model quantized with the same bitsandbytes settings "
                "(bnb_4bit_use_double_quant)."
            )
        if set(saved) != expected:
            raise ValueError(
                f"Unsloth: the checkpoint's 4-bit quantization state for `{head[:-1]}` has entries "
                f"{sorted(saved)}, the model's has {sorted(expected)}."
            )
        loaded = QuantState.from_dict(saved, device = live.absmax.device)
        pairs = [("absmax", loaded.absmax, live.absmax), ("code", loaded.code, live.code)]
        if live.nested:
            pairs += [
                ("offset", loaded.offset, live.offset),
                ("state2.absmax", loaded.state2.absmax, live.state2.absmax),
                ("state2.code", loaded.state2.code, live.state2.code),
            ]
        same = (
            loaded.quant_type == live.quant_type
            and loaded.blocksize == live.blocksize
            and loaded.nested == live.nested
            and tuple(loaded.shape) == tuple(live.shape)
            and loaded.dtype == live.dtype
            and (not live.nested or loaded.state2.blocksize == live.state2.blocksize)
            and all(
                isinstance(a, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                for _, a, b in pairs
            )
        )
        if not same:
            raise ValueError(
                f"Unsloth: the checkpoint's 4-bit quantization state for `{head[:-1]}` does not "
                "match the model's (shape, blocksize, quant type or nesting differ)."
            )
        with torch.no_grad():
            for _, a, b in pairs:
                b.copy_(a.to(b.device))
    if orphans and (not broadcast or dist.get_rank() == 0):
        import warnings
        warnings.warn(
            f"Unsloth: skipped the 4-bit quantization state of {', '.join(orphans)}: the state dict "
            "does not have the packed weight, so the model keeps its own weight and quantization "
            "state.",
            stacklevel = 3,
        )


# Every key `QuantState.as_dict(packed = True)` can write, so skipped metadata never reaches DCP.
_QUANT_STATE_KEYS = frozenset(
    (
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "quant_state.bitsandbytes__nf4",
        "quant_state.bitsandbytes__fp4",
    )
)


def _without_quant_state_entries(name, args, kwargs):
    """Loading: the entries `_add_quant_state_entries` wrote are restored into the live QuantState
    (`_restore_quant_state`) and then dropped, as DCP would reject them as unexpected keys."""
    model = args[0] if args else kwargs.get("model")
    if not isinstance(model, torch.nn.Module):
        return args, kwargs
    drop = set()
    for prefix, module in _linear4bit_entries(model):
        drop.update(
            prefix + "weight." + key
            for key in _QUANT_STATE_KEYS.union(module.weight.quant_state.as_dict(packed = True))
        )
    if not drop:
        return args, kwargs
    if name == "set_model_state_dict" and len(args) > 1:
        loading = args[1]
    else:
        loading = kwargs.get("model_state_dict")
    if name in ("set_model_state_dict", "set_state_dict"):
        _restore_quant_state(model, loading, kwargs.get("options", None))

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
    buffers = None,
):
    """Average gradients over `mesh` (DDP-style), one all_reduce per (device, reduce dtype), entered
    by every rank: a missing local gradient counts as zeros, and a "has a gradient" flag keeps grad
    None for a parameter no rank used. Sums in float32 (fp16 sums overflow). The summed flags are
    read (a host sync) only when some local gradient is None. `buffers` caches one flat buffer per
    (device, dtype); gradients never alias it."""
    groups = {}
    for p in params:
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        groups.setdefault((str(p.device), dtype), []).append(p)
    for (device, dtype), group in groups.items():
        sizes = [p.numel() for p in group]
        total = sum(sizes)
        key = tuple((id(p), n) for p, n in zip(group, sizes))
        cached = buffers.get((device, dtype)) if buffers is not None else None
        if cached is not None and cached[0] == key:
            flat = cached[1]
        else:
            flat = torch.empty(total + len(group), dtype = dtype, device = group[0].device)
            if buffers is not None:
                buffers[(device, dtype)] = (key, flat)
        grads = [p.grad for p in group]
        local_all = all(g is not None for g in grads)
        flags = flat[total:]
        if local_all:
            flags.fill_(1)
        else:
            flags.zero_()
        offset = 0
        for i, (g, n) in enumerate(zip(grads, sizes)):
            if g is not None:
                flat[offset : offset + n].copy_(g.detach().reshape(-1))
                if not local_all:
                    # Not flags[i] = 1: that is a host sync on CUDA.
                    flags[i].fill_(1)
            else:
                flat[offset : offset + n].zero_()
            offset += n
        world = mesh_all_reduce(flat, mesh, dist)
        flat[:total].div_(world)
        used = None if local_all else flags.tolist()
        offset = 0
        for i, (p, g, n) in enumerate(zip(group, grads, sizes)):
            if g is not None or used[i] > 0:
                value = flat[offset : offset + n].view(p.shape)
                if p.grad is None:
                    p.grad = value.to(p.dtype, copy = True)
                else:
                    p.grad.copy_(value)
            offset += n


def _fsdp_grad_sync_deferred(model, cache):
    """True when every FSDP2 param group has set_requires_gradient_sync(False), i.e. a non-final
    micro-batch. False when no group is found (sync every backward)."""
    states = cache.get("fsdp_states")
    if states is None:
        states = []
        try:
            from torch.distributed.fsdp import FSDPModule
        except Exception:
            try:
                from torch.distributed._composable.fsdp import FSDPModule
            except Exception:
                FSDPModule = None
        if FSDPModule is not None:
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    try:
                        states.append(module._get_fsdp_state())
                    except Exception:
                        pass
        cache["fsdp_states"] = states
    found = False
    for st in states:
        groups = getattr(st, "_fsdp_param_groups", None)  # a list on newer torch (singular on 2.10)
        if groups is None:
            group = getattr(st, "_fsdp_param_group", None)
            groups = [group] if group is not None else []
        for group in groups:
            reduce = getattr(group, "reduce_grads", None)
            if reduce is None:
                return False
            if reduce:
                return False
            found = True
    return found


def reduce_unsharded_trainable_grads(model):
    """Average gradients of trainable params FSDP2 does not manage (e.g. `ignored_modules`) over the
    mesh, else replicas drift. Runs once per backward as an autograd engine callback on every rank;
    a per-parameter hook would hang when a rank skips the parameter (branch, routed MoE expert).
    Skipped while FSDP2 grad sync is off (GA); the first synced backward averages the accumulated
    sum. Returns the names."""
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
    buffers = state.setdefault("buffers", {})

    def _sync():
        state["queued"] = False
        # Every rank enters the same branch: the GA flag is set alike on all ranks.
        if _fsdp_grad_sync_deferred(model, state):
            return
        _sync_replicated_grads(params, dist, mesh, buffers)

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
