# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inside fsdp2_sharded_loading() or UNSLOTH_FSDP2_SHARDED_LOAD=1, each rank reads only its dim-0
chunk of nn.Linear / nn.Embedding weights (output head included); others whole. 16-bit and pre-quantized bitsandbytes 4-bit
checkpoints: a 4-bit weight's chunk is its byte range of the packed storage (the QuantState stays whole),
which FSDP2 shards the same way whatever float dtype later holds the packed bytes."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import os

import torch
import torch.nn as nn

__all__ = [
    "fsdp2_sharded_loading",
    "load_fsdp2_sharded",
    "fsdp2_sharded_load_requested",
]

_PRELOAD = "_unsloth_fsdp2_preload"
_GUARD = "_unsloth_fsdp2_load_guard"
_STATS = "_unsloth_fsdp2_load_stats"
_ACTIVE = [0]

_QUANTIZED_HINT = (
    "QLoRA quantizing on load (load_in_4bit / 8-bit of a 16-bit checkpoint) under native FSDP2 loads the full "
    "model on every rank: drop fsdp2_sharded_loading / UNSLOTH_FSDP2_SHARDED_LOAD for it, or load a "
    "pre-quantized bnb-4bit checkpoint."
)
# A 4-bit chunk is a byte range: equal for every float storage FSDP2 may view the packed bytes in (up to
# 4-byte float32) when the packed size divides by this many bytes per rank.
_PACKED_UNIT = 4


def _chunk_rows(rows, world, rank):
    """Rows [start, stop) of dim-0 chunk `rank` as fully_shard cuts it (torch.chunk: ceil(rows / world) per
    chunk, so the last ranks may hold fewer rows or none)."""
    size = math.ceil(rows / world) if rows else 0
    start = min(rank * size, rows)
    return start, min(start + size, rows)


_DORA_REFUSAL = (
    "Unsloth: LoraConfig(use_dora = True) initializes DoRA's magnitude from the whole base weight, which "
    "native FSDP2 sharded loading holds only as this rank's rows; load the full model (drop "
    "fsdp2_sharded_loading / UNSLOTH_FSDP2_SHARDED_LOAD) to train DoRA under native FSDP2."
)


def _sharded_chunk(weight):
    info = _preload(weight)
    return info is not None and info["world"] > 1


def _guard_dora_init():
    """DoRA's magnitude init (peft DoraLinearLayer.update_layer, every PEFT version with DoRA) adds a full
    lora_B @ lora_A to the base weight: refuse it on a sharded-loaded chunk with a clear message instead of
    PEFT's shape mismatch. Installed once, inert for any other base layer."""
    try:
        from peft.tuners.lora import dora
    except Exception:
        return
    cls = getattr(dora, "DoraLinearLayer", None)
    update = getattr(cls, "update_layer", None)
    if update is None or getattr(update, "_unsloth_fsdp2_load_guard", False):
        return

    @functools.wraps(update)
    def update_layer(self, *args, **kwargs):
        base = kwargs.get("base_layer", args[0] if args else None)
        if _sharded_chunk(getattr(base, "weight", None)):
            raise ValueError(_DORA_REFUSAL)
        return update(self, *args, **kwargs)

    update_layer._unsloth_fsdp2_load_guard = True
    cls.update_layer = update_layer


def _refuse_unsharded_forward(
    module,
    args,
    kwargs = None,
):
    raise RuntimeError(
        "Unsloth: this model was loaded with native FSDP2 sharded loading (each rank holds only its shard "
        "of the linear weights); shard it first (unsloth.models.fsdp2_native.shard_model(model), or let the "
        "Trainer do it under UNSLOTH_FSDP2=1) before running it."
    )


def _materialize(model, name, kwargs, mesh, plan, device):
    """Fill a meta model: nn.Linear / nn.Embedding weights as this rank's dim-0 chunk, the rest whole."""
    from .tp_native import _fill_from_checkpoint

    world, rank = mesh.size(), mesh.get_local_rank()
    ranks = mesh.mesh.tolist()
    linear = {
        id(m.weight)
        for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Embedding))
        and isinstance(getattr(m, "weight", None), torch.Tensor)
    }
    owners = {}

    def read_local(old, reader, key):
        if id(old) not in linear or old.ndim < 2:
            return None
        module = reader.quant.module(old) if reader.quant is not None else None
        if module is not None:
            rows = reader.shape(key)[0]
            itemsize = reader.quant.packed_dtype(key).itemsize
            if (rows * itemsize) % (_PACKED_UNIT * world):
                return None  # read whole: fully_shard cuts it as from a full load
            start, stop = rank * rows // world, (rank + 1) * rows // world
            new = reader.quant.rows(key, start, stop, module, device)
            # Packed metadata counts bytes: a float quant storage holds itemsize bytes per stored row.
            shape, packed = (rows * itemsize, 1), True
            start, stop = start * itemsize, stop * itemsize
        else:
            rows = old.shape[0]
            start, stop = _chunk_rows(rows, world, rank)
            local = reader.rows_to(key, start, stop, old.dtype, device)
            new = nn.Parameter(local.contiguous(), requires_grad = old.requires_grad)
            shape, packed = tuple(old.shape), False
        setattr(
            new,
            _PRELOAD,
            {
                "shape": shape,
                "ranks": ranks,
                "world": world,
                "rank": rank,
                "rows": (start, stop),
                "packed": packed,
            },
        )
        owners[id(new)] = new
        return new

    reader, sharded_numel, full_numel = _fill_from_checkpoint(
        model,
        name,
        kwargs,
        device,
        read_local,
        label = "native FSDP2",
        fallback = "a full load, then shard_model",
    )
    guards = [model.register_forward_pre_hook(_refuse_unsharded_forward)]
    for module in model.modules():
        if isinstance(module, nn.Linear) and id(module._parameters.get("weight")) in owners:
            guards.append(module.register_forward_pre_hook(_refuse_unsharded_forward))
    model.__dict__[_GUARD] = guards
    stats = {
        "world": world,
        "rank": rank,
        "sharded_tensors": len(owners),
        "sharded_numel": sharded_numel,
        "replicated_numel": full_numel,
        "read_bytes_full": reader.full_bytes,
        "read_bytes_sliced": reader.slice_bytes,
    }
    model.__dict__[_STATS] = stats
    if owners and world > 1:
        _guard_dora_init()
    return stats


@contextlib.contextmanager
def fsdp2_sharded_loading(mesh = None):
    """Every transformers `from_pretrained` inside reads only this rank's native FSDP2 shard of the linear
    weights (see the module docstring). `mesh`: the 1-D DeviceMesh shard_model will use (default: the
    world, which is shard_model's default too)."""
    with _fsdp2_loading(mesh) as live:
        yield live


@contextlib.contextmanager
def _fsdp2_loading(mesh, auto = False):
    """fsdp2_sharded_loading; `auto`: only checkpoints larger than the GPU (native FSDP2 without the flag)."""
    from .tp_native import _sharded_from_pretrained

    if mesh is None:
        mesh = _default_load_mesh()
    state = {
        "mesh": mesh,
        "plan": None,
        "materialize": _materialize,
        "label": "native FSDP2",
        "quantized_hint": _QUANTIZED_HINT,
        "prequantized_hint": _QUANTIZED_HINT,
        "mesh_hint": "Unsloth: native FSDP2 sharded loading takes a 1-D mesh (HSDP / 2-D meshes load in full).",
        "auto": auto,
    }
    _ACTIVE[0] += 1
    try:
        with _sharded_from_pretrained(state) as live:
            yield live
    finally:
        _ACTIVE[0] -= 1


def _default_load_mesh():
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        from .fsdp2_native import _ensure_process_group, _local_cuda_index

        index = _local_cuda_index() if torch.cuda.is_available() else None
        if torch.cuda.is_available() and index is None:
            index = torch.cuda.current_device()
        device = torch.device("cuda", index) if index is not None else torch.device("cpu")
        _ensure_process_group(device, None, None)  # sets this rank's CUDA device first
    from torch.distributed.device_mesh import init_device_mesh

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(device_type, (dist.get_world_size(),))


def load_fsdp2_sharded(
    model_name,
    mesh = None,
    loader = None,
    **kwargs,
):
    """`loader(model_name, **kwargs)` under fsdp2_sharded_loading (default FastLanguageModel.from_pretrained,
    16-bit). Raises if no model went through the sharded path."""
    if loader is None:
        from unsloth import FastLanguageModel
        loader = FastLanguageModel.from_pretrained
        kwargs.setdefault("load_in_4bit", False)
    with fsdp2_sharded_loading(mesh) as state:
        out = loader(model_name, **kwargs)
    if not state["models"]:
        raise RuntimeError(
            "Unsloth: load_fsdp2_sharded's loader did not call transformers from_pretrained; nothing was "
            "loaded sharded."
        )
    return out


def fsdp2_sharded_load_requested():
    """UNSLOTH_FSDP2_SHARDED_LOAD=1 with native FSDP2 requested (UNSLOTH_FSDP2=1 under WORLD_SIZE > 1)."""
    if os.environ.get("UNSLOTH_FSDP2_SHARDED_LOAD", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    try:
        from .fsdp2_native import native_fsdp2_requested
        return bool(native_fsdp2_requested())
    except Exception:
        return False


def _auto_sharded_load():
    """Native FSDP2 outside the tensor / pipeline parallel environment (which loads itself), with
    UNSLOTH_FSDP2_SHARDED_LOAD unset: sharded loading for a checkpoint larger than the GPU only."""
    if os.environ.get("UNSLOTH_FSDP2_SHARDED_LOAD", "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    try:
        from .fsdp2_native import native_fsdp2_requested
        from .tp_env import env_layout
        return bool(native_fsdp2_requested()) and env_layout() is None
    except Exception:
        return False


def sharded_load_env(loader):
    """Decorator for Unsloth's top-level from_pretrained: under UNSLOTH_FSDP2_SHARDED_LOAD=1 (+ native FSDP2)
    the outermost call runs inside fsdp2_sharded_loading; under native FSDP2 alone, a checkpoint larger than
    the GPU does. Quantizing on load (load_in_4bit of a 16-bit checkpoint, 8-bit) is refused there."""

    @functools.wraps(loader)
    def wrapper(*args, **kwargs):
        if _ACTIVE[0]:
            return loader(*args, **kwargs)
        if os.environ.get("UNSLOTH_PARALLEL", "").strip().lower() == "auto":
            # Before anything below (or inside the load) reads UNSLOTH_FSDP2: DDP or FSDP2 by size.
            from ..parallel_auto import resolve_for_load
            resolve_for_load(loader, args, kwargs)
        if fsdp2_sharded_load_requested():
            with fsdp2_sharded_loading():
                return loader(*args, **kwargs)
        if not _auto_sharded_load():
            return loader(*args, **kwargs)
        with _fsdp2_loading(None, auto = True):
            return loader(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------------------------
# shard_model side
# --------------------------------------------------------------------------------------------------


def _preload(p):
    return getattr(p, _PRELOAD, None) if isinstance(p, torch.Tensor) else None


def has_preloaded_shards(model):
    return any(_preload(p) is not None for p in model.parameters())


def _refuse_weight_dependent_lora_init(model):
    configs = {}
    m, seen = model, 0
    while isinstance(m, nn.Module) and seen < 8:
        configs.update(getattr(m, "peft_config", None) or {})
        modules = m.__dict__.get("_modules") or {}
        m = modules.get("base_model") or modules.get("model")
        seen += 1
    # Random inits (True / "gaussian" / "orthogonal") read only the module's full in / out features and the
    # weight dtype, so a rank's rows give the one-process factors; the others read the weight itself.
    bad = {
        name: cfg.init_lora_weights
        for name, cfg in configs.items()
        if getattr(cfg, "init_lora_weights", True) not in (True, False, "gaussian", "orthogonal")
    }
    if any(getattr(cfg, "use_dora", False) for cfg in configs.values()) and any(
        _sharded_chunk(p) for p in model.parameters()
    ):
        raise ValueError(_DORA_REFUSAL)
    tokens = [
        name for name, cfg in configs.items() if getattr(cfg, "trainable_token_indices", None)
    ]
    if tokens and any(_sharded_chunk(p) for p in model.parameters()):
        raise ValueError(
            f"Unsloth: trainable_token_indices ({tokens}) copies rows of the embedding, which native FSDP2 sharded "
            "loading holds only as this rank's rows; load the full model (no sharded loading) to train them."
        )
    if bad:
        raise ValueError(
            f"Unsloth: init_lora_weights {bad} reads the base weight, which native FSDP2 sharded loading "
            "holds only as this rank's rows; use init_lora_weights = True / 'gaussian' / 'orthogonal', or load "
            "the full model (no sharded loading)."
        )


def _tag_modules_to_save_copies(model):
    """PEFT modules_to_save deep-copies a sharded-loaded linear, but Parameter.__deepcopy__ drops the chunk's
    tag: give each still-identical copy its original's, so it is sharded like the full load's copy."""
    for module in model.modules():
        original = getattr(module, "original_module", None)
        copies = getattr(module, "modules_to_save", None)
        if not isinstance(original, nn.Module) or not isinstance(copies, nn.ModuleDict):
            continue
        params = dict(original.named_parameters())
        for copy in copies.values():
            for name, q in copy.named_parameters():
                p = params.get(name)
                info = _preload(p)
                if info is not None and _preload(q) is None and q.shape == p.shape:
                    q.__dict__[_PRELOAD] = info


def begin_shard(model, mesh, config):
    """Before shard_model's layout: check the preloaded chunks match `mesh`, drop the forward guards and
    replace every chunk by a meta Parameter of the full shape. Returns ({(id(module), name): (local, info)}
    (empty without preloaded chunks), config): FSDP2Config.ignored_params entries naming a chunk now name its
    placeholder (the caller's set is updated in place, a non-set copied), so gather_replicated keeps them whole."""
    _tag_modules_to_save_copies(model)
    slots = {}
    for module in model.modules():
        for name, p in module._parameters.items():
            info = _preload(p)
            if info is not None:
                slots.setdefault(id(p), (p, info, []))[2].append((module, name))
    if not slots:
        return {}, config
    if mesh.ndim != 1:
        raise RuntimeError(
            "Unsloth: native FSDP2 sharded loading needs a 1-D shard mesh (the one it loaded over); this "
            f"FSDP2Config.mesh has {mesh.ndim} dims. Load in full for HSDP."
        )
    ranks = mesh.mesh.tolist()
    if config.shard_placement_fn is not None:
        raise RuntimeError(
            "Unsloth: native FSDP2 sharded loading holds dim-0 chunks; FSDP2Config.shard_placement_fn is "
            "not supported with it."
        )
    if config.cpu_offload:
        raise RuntimeError(
            "Unsloth: native FSDP2 sharded loading does not support cpu_offload yet; load in full for it."
        )
    for p, info, _ in slots.values():
        if info["ranks"] != ranks:
            raise RuntimeError(
                f"Unsloth: this model was sharded-loaded over ranks {info['ranks']}, but shard_model shards "
                f"over {ranks}; pass the same mesh to fsdp2_sharded_loading and FSDP2Config."
            )
    _refuse_weight_dependent_lora_init(model)
    for level in list(model.modules()):
        for handle in level.__dict__.pop(_GUARD, None) or ():
            handle.remove()
        # PEFT modules_to_save deep-copies a guarded linear, hook included, with no handle on the root.
        hooks = level.__dict__.get("_forward_pre_hooks") or {}
        for key in [k for k, fn in hooks.items() if fn is _refuse_unsharded_forward]:
            del hooks[key]
    pending, swap = {}, {}
    for p, info, where in slots.values():
        shape = info["shape"]
        if info.get("packed"):
            # Bytes in the checkpoint; the load may already hold them as float storage (prepare_fsdp2_qlora).
            shape = (shape[0] // p.element_size(), 1)
        empty = torch.empty(shape, dtype = p.dtype, device = "meta")
        if info.get("packed"):
            # The packed 4-bit weight's full shape, still a Params4bit: the QLoRA wrap repacks it as float storage.
            meta = torch.Tensor._make_subclass(type(p), empty, p.requires_grad)
        else:
            meta = nn.Parameter(empty, requires_grad = p.requires_grad)
        for attr, value in p.__dict__.items():
            if attr != _PRELOAD:
                meta.__dict__[attr] = value
        meta.__dict__["_unsloth_fsdp2_pending"] = True
        local = p.detach()
        swap[id(p)] = meta
        for module, name in where:
            module._parameters[name] = meta
            pending[(id(module), name)] = (local, info)
    ignored = config.ignored_params
    if ignored and any(id(q) in swap for q in ignored):
        if isinstance(ignored, set):
            for q in [q for q in ignored if id(q) in swap]:
                ignored.discard(q)
                ignored.add(swap[id(q)])
        else:
            config = dataclasses.replace(
                config, ignored_params = {swap.get(id(q), q) for q in ignored}
            )
    return pending, config


def _as_storage(local, info, dtype):
    """A chunk in the dtype of its placeholder: packed 4-bit bytes are reinterpreted, never converted."""
    if not info.get("packed") or local.dtype == dtype:
        return local.to(dtype)
    return local.contiguous().reshape(-1).view(torch.uint8).view(dtype).reshape(-1, 1)


# Rows per all-gather slice of a replicated chunk: about this many bytes per rank.
_GATHER_SLICE_BYTES = 64 << 20


def _full_from_chunks(
    local,
    info,
    mesh,
    dtype,
    device = None,
):
    """All-gather dim-0 chunks slice by slice; whole-chunk gather (5x the chunk) OOMed a T4."""
    import torch.distributed as dist

    if info.get("packed"):
        raw = local.contiguous().reshape(-1).view(torch.uint8).reshape(-1, 1)
        full = _full_from_chunks(raw, dict(info, packed = False), mesh, torch.uint8, device)
        return _as_storage(full, info, dtype)
    rows, world = info["shape"][0], info["world"]
    size = math.ceil(rows / world)
    tail = tuple(info["shape"][1:])
    row_bytes = max(1, math.prod(tail) * local.element_size())
    step = max(1, min(size, _GATHER_SLICE_BYTES // row_bytes)) if size else 1
    full = torch.empty(
        (rows,) + tail, dtype = local.dtype, device = local.device if device is None else device
    )
    group = mesh.get_group()
    for start in range(0, size, step):
        n = min(step, size - start)
        piece = local.new_zeros((n,) + tail)
        have = max(0, min(n, local.shape[0] - start))
        if have:
            piece[:have].copy_(local[start : start + have])
        pieces = [torch.empty_like(piece) for _ in range(world)]
        dist.all_gather(pieces, piece, group = group)
        for r, got in enumerate(pieces):
            first = r * size + start
            keep = min(n, rows - first)
            if keep > 0:
                full[first : first + keep].copy_(got[:keep])
    return full if full.dtype == dtype else full.to(dtype)


def gather_replicated(
    model,
    pending,
    replicated,
    mesh,
    also = (),
    device = None,
):
    """Gather replicated (non-FSDP2) chunks whole onto `device`. Collective: every rank passes the same set."""
    if not pending:
        return 0
    keep = {id(p) for p in replicated}
    done = {}
    for module in model.modules():
        for name, p in list(module._parameters.items()):
            key = (id(module), name)
            if key not in pending or id(p) not in keep:
                continue
            local, info = pending.pop(key)
            if id(p) not in done:
                with torch.no_grad():
                    full = _full_from_chunks(local, info, mesh, p.dtype, device)
                if info.get("packed"):
                    new = torch.Tensor._make_subclass(type(p), full, p.requires_grad)
                else:
                    new = nn.Parameter(full, requires_grad = p.requires_grad)
                for attr, value in p.__dict__.items():
                    if attr != "_unsloth_fsdp2_pending":
                        new.__dict__[attr] = value
                done[id(p)] = (p, new)
            module._parameters[name] = done[id(p)][1]
    if done:
        swap = {id(old): new for old, new in done.values()}
        for params in (replicated, *also):
            for p in list(params):
                if id(p) in swap:
                    params.discard(p)
                    params.add(swap[id(p)])
    return len(done)


def finish_shard(model, pending):
    """After fully_shard: each pending chunk becomes its FSDPParam's local shard (padded like fully_shard)."""
    if not pending:
        return 0
    from .fsdp2_native import _fsdp_module_class

    FSDPModule = _fsdp_module_class()
    installed = 0
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        group = getattr(state, "_fsdp_param_group", None)
        for fsdp_param in getattr(group, "fsdp_params", None) or ():
            info = fsdp_param._module_info
            entry = pending.pop((id(info.module), info.param_name), None)
            if entry is None:
                continue
            for shared, shared_name in zip(info.shared_modules, info.shared_param_names):
                pending.pop((id(shared), shared_name), None)
            local, chunk = entry
            old = fsdp_param.sharded_param
            if chunk.get("packed"):
                with torch.no_grad():
                    local = _as_storage(local, chunk, old.dtype)
            if tuple(local.shape) != tuple(fsdp_param.sharded_size):
                raise RuntimeError(
                    f"Unsloth: native FSDP2 sharded loading read {tuple(local.shape)} rows for "
                    f"'{info.param_name}', but fully_shard keeps {tuple(fsdp_param.sharded_size)} on this rank."
                )
            with torch.no_grad():
                local = local.to(dtype = old.dtype)
            new = nn.Parameter(
                fsdp_param.to_sharded_dtensor(local), requires_grad = old.requires_grad
            )
            for attr, value in old.__dict__.items():
                if attr != "_unsloth_fsdp2_pending":
                    new.__dict__[attr] = value
            fsdp_param._setattr_on_modules(new)
            fsdp_param.reset_sharded_param()
            installed += 1
    if pending:
        names = sorted({name for (_, name) in pending})
        raise RuntimeError(
            f"Unsloth: native FSDP2 sharded loading left {len(pending)} chunks ({names[:4]}) outside every "
            "FSDP2 unit; load this model in full."
        )
    return installed
