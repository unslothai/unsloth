# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Layout-portable Trainer checkpoints for tensor / pipeline / expert parallel runs.

A tensor, pipeline or expert parallel checkpoint holds each rank's own shards (unsloth_tp/ or unsloth_ep/) and the
1-GPU adapter. Next to them rank 0 also writes optimizer.pt in the 1-GPU form native FSDP2 writes: every state
tensor whole, keyed by parameter name, param_groups listing names in 1-GPU order, Adam moments as float32 exp_avg /
exp_avg_sq (bitsandbytes 8-bit state dequantized). The trainable state is small (LoRA), so this costs one gather.

Resume on the saved layout reads the per-rank files as before (bitwise). Resume on another layout (another world
size, tp / pp / ep size, or one GPU) reads the adapter and optimizer.pt instead and each rank keeps its slice; a
bitsandbytes 8-bit optimizer requantizes the float32 moments into its own blocks (within 8-bit rounding of an
uninterrupted run, not bitwise). One GPU (or DDP) needs no layout code: Trainer.__init__ makes the optimizer load
name-aware (install_portable_resume). A 1-GPU checkpoint (int-keyed optimizer.pt) also resumes under tp / pp / ep."""

from __future__ import annotations

import os

import torch

__all__ = [
    "PORTABLE_OPTIMIZER",
    "portable_optimizer_state",
    "write_portable_optimizer",
    "portable_available",
    "portable_load_weights",
    "portable_load_optimizer",
    "install_portable_resume",
]

PORTABLE_OPTIMIZER = "optimizer.pt"  # transformers' OPTIMIZER_NAME
_TORCH_ADAM_KEYS = {"state1": "exp_avg", "state2": "exp_avg_sq"}
# bitsandbytes per-parameter keys rebuilt by the target optimizer (quantization maps / scales, clipping history).
_BNB_AUX = (
    "qmap1",
    "qmap2",
    "absmax1",
    "absmax2",
    "max1",
    "max2",
    "new_max1",
    "new_max2",
    "gnorm_vec",
    "unorm_vec",
)
_BNB_WRAPPED = "__bnb_optimizer_quant_state__"  # Optimizer8bit._FSDP_WRAPPED_QUANT_STATE_KEY
_ID_BASE = 2**32  # slice maps: global rank * _ID_BASE + local flat index, exact in float64


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def _rank():
    dist = _dist()
    return dist.get_rank() if dist is not None else 0


def _is_dtensor(t):
    from .tp_native import _is_dtensor as is_dtensor
    return is_dtensor(t)


def _inner(optimizer):
    from .fsdp2_native import _inner_optimizer
    return _inner_optimizer(optimizer)


def _local(p):
    return p._local_tensor if _is_dtensor(p) else p


def _ep_layout(p):
    from .ep_native import _EP
    layout = getattr(p, _EP, None)
    # A per-expert module (ep_expert_lists) is whole on the rank holding it, like a pipeline stage's layer.
    return None if layout is None or layout[0] in ("4bit", "list") else layout


def _all_gather_object(obj):
    dist = _dist()
    if dist is None:
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _agree(flag):
    """Collective: True only when `flag` holds on every rank."""
    if _dist() is None:
        return bool(flag)
    from .fsdp2_native import _all_ranks_agree

    return _all_ranks_agree(bool(flag), None)


def _trainables(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _tree_keys(model):
    """{parameter name: sort key} reproducing one GPU's named_parameters order from any rank: the position of every
    module along the path in its parent, then the parameter's own position (a module's parameters before its
    children). A pipeline stage keeps the other stages' layers as stand-ins at the same list positions, so keys of
    names held on different ranks compare as on one GPU."""
    keys = {}

    def walk(module, prefix, path, seen):
        if id(module) in seen:
            return
        seen.add(id(module))
        for j, (name, p) in enumerate(module._parameters.items()):
            if p is not None:
                keys.setdefault(prefix + name, path + ((0, j),))
        # Per-expert containers (ep_expert_lists) hold only this rank's experts: order them by global index.
        experts = getattr(module, "_unsloth_global_expert_keys", False)
        for c, (name, child) in enumerate(module._modules.items()):
            if child is not None:
                walk(child, f"{prefix}{name}.", path + ((1, int(name) if experts else c),), seen)

    walk(model, "", (), set())
    return keys


def _full_value(model, p, value):
    """Collective over the group sharding `p`: the 1-GPU tensor of `value` (`p` itself, a state tensor of `p`, or
    an id map), laid out as `p` is (tensor parallel DTensor, expert parallel shard, or whole)."""
    if _is_dtensor(p):
        from .tp_native import _dtensor_api, _full, _param_layouts
        if not _is_dtensor(value):
            value = _dtensor_api()[0].from_local(
                value.to(_local(p).device),
                p.device_mesh,
                p.placements,
                run_check = False,
                shape = p.shape,
                stride = p.stride(),
            )
        return _full(value, p, _param_layouts(model).get(id(p))).detach()
    if _ep_layout(p) is not None:
        from .ep_native import _gather_expert, _mesh
        return _gather_expert(p, _mesh(model).get_group(), value = value.to(p.device))
    return value.detach()


def _slice_map(model, p):
    """Collective (sharded `p`): (full shape, positions in the flattened full tensor, their positions in this
    rank's flattened local tensor, local shape); None when `p` is whole on this rank."""
    if not _is_dtensor(p) and _ep_layout(p) is None:
        return None
    local = _local(p)
    rank = _rank()
    ids = torch.arange(local.numel(), dtype = torch.float64, device = local.device) + float(
        rank * _ID_BASE
    )
    full = _full_value(model, p, ids.reshape(local.shape)).cpu()
    flat = full.reshape(-1)
    owner = torch.div(flat, _ID_BASE, rounding_mode = "floor")
    mine = owner == rank
    src = mine.nonzero().squeeze(1)
    dst = (flat[mine] - float(rank * _ID_BASE)).long()
    return tuple(full.shape), src, dst, tuple(local.shape)


def _slice_maps(model):
    return {n: _slice_map(model, p) for n, p in _trainables(model)}


def _to_local(name, full, smap, like):
    """This rank's part of the 1-GPU tensor `full` (as dtype / device of `like`, the local parameter)."""
    want = tuple(like.shape) if smap is None else smap[0]
    if tuple(full.shape) != want:
        raise RuntimeError(
            f"Unsloth: checkpoint tensor {name} has shape {tuple(full.shape)}, but this model's 1-GPU parameter "
            f"has {want}: the checkpoint was not written from this model."
        )
    if smap is None:
        return full
    _, src, dst, shape = smap
    out = torch.zeros(int(torch.Size(shape).numel()), dtype = full.dtype)  # padded experts stay 0
    out[dst] = full.reshape(-1)[src]
    return out.reshape(shape)


# --------------------------------------------------------------------------------------------------------------
# Optimizer state: normalised to torch Adam naming, float32 moments
# --------------------------------------------------------------------------------------------------------------


def _is_bnb(optimizer):
    module = type(optimizer).__module__ or ""
    if module.startswith("bitsandbytes"):
        return True
    try:
        from bitsandbytes.optim.optimizer import Optimizer8bit
    except Exception:
        return False
    return isinstance(optimizer, Optimizer8bit)


def _is_bnb_adam(optimizer):
    """bitsandbytes Adam / AdamW (8-bit blockwise or 32-bit, paged or not, Unsloth's fused 8-bit AdamW)."""
    try:
        from bitsandbytes.optim.optimizer import Optimizer2State
    except Exception:
        return False
    if (
        not isinstance(optimizer, Optimizer2State)
        or getattr(optimizer, "optimizer_name", None) != "adam"
    ):
        return False
    if "ademamix" in type(optimizer).__name__.lower():
        return False
    args = getattr(optimizer, "args", None)
    return not (
        args is not None
        and int(getattr(args, "optim_bits", 32)) == 8
        and not getattr(args, "block_wise", True)
    )


def _normalize_entry(entry):
    """A parameter's optimizer state with bitsandbytes Adam keys / 8-bit codes turned into torch Adam's float32
    exp_avg / exp_avg_sq and a float step; None if it holds state this cannot map (a 1-moment bnb state)."""
    from .fsdp2_bnb_optim import _dequantize

    wrapped = entry.get(_BNB_WRAPPED)
    if isinstance(
        wrapped, dict
    ):  # bitsandbytes >= 0.48 nests the quantized state (FSDP compatibility)
        entry = {**{k: v for k, v in entry.items() if k != _BNB_WRAPPED}, **wrapped}
    if ("state1" in entry) != ("state2" in entry):
        return None
    out = {}
    for key, value in entry.items():
        if key in _BNB_AUX:
            continue
        if isinstance(value, dict):
            return None  # a nested state this does not know
        if key in _TORCH_ADAM_KEYS:
            if _is_dtensor(value):
                value = value.float()
            elif torch.is_tensor(value) and value.dtype == torch.uint8:
                if f"qmap{key[-1]}" not in entry or f"absmax{key[-1]}" not in entry:
                    return None
                value = _dequantize(entry, key, value.device)
            else:
                value = value.float()
            key = _TORCH_ADAM_KEYS[key]
        elif key == "step":
            if _is_dtensor(value):
                value = value.to_local()
            value = torch.tensor(
                float(value.item() if torch.is_tensor(value) else value), dtype = torch.float32
            )
        out[key] = value
    return out


def _local_state(optimizer):
    """{parameter: normalised state} of this rank's optimizer, or (None, why)."""
    inner = _inner(optimizer)
    if _is_bnb(inner) and not _is_bnb_adam(inner):
        return None, f"{type(inner).__name__} (only bitsandbytes Adam / AdamW state is converted)"
    raw = inner.state_dict()
    flat = [p for group in inner.param_groups for p in group["params"]]
    out = {}
    for index, entry in raw["state"].items():
        p = flat[index]
        entry = _normalize_entry(entry)
        if entry is None:
            return None, f"{type(inner).__name__} keeps state that is not two Adam moments"
        local = _local(p)
        for key, value in entry.items():
            if torch.is_tensor(value) and value.ndim > 0:
                shape = tuple(_local(value).shape) if _is_dtensor(value) else tuple(value.shape)
                if shape != tuple(local.shape):
                    return (
                        None,
                        f"{type(inner).__name__} state '{key}' is not shaped like its parameter",
                    )
        out[p] = entry
    return out, None


def _group_layout(
    model,
    inner,
    names,
    with_keys = False,
):
    """Collective: [(hyperparameters, [names in 1-GPU order], exact)] per optimizer group, over every rank (a
    pipeline stage holds only its own parameters); None if the ranks' optimizers have different group counts.
    Ranks holding the same names (tensor / expert / data parallel) keep their optimizer's own order; stages are
    merged in one GPU's module order. `exact`: that order is the optimizer's own (every rank holds the same list,
    or every rank's list already follows module order), so a positional 1-GPU state maps onto it."""
    keys = _tree_keys(model)
    mine = []
    for group in inner.param_groups:
        group_names = [names[id(p)] for p in group["params"] if id(p) in names]
        hyper = {k: v for k, v in group.items() if k != "params"}
        mine.append((hyper, group_names, {n: keys.get(n, ((2, 0),)) for n in group_names}))
    every = _all_gather_object(mine)
    if len({len(x) for x in every}) != 1:
        return (None, {}) if with_keys else None
    every_key = {}
    for x in every:
        for group in x:
            for n, key in group[2].items():
                every_key.setdefault(n, key)
    out = []
    for gi in range(len(every[0])):
        hyper = next((x[gi][0] for x in every if x[gi][1]), every[0][gi][0])
        lists = [x[gi][1] for x in every]
        if all(names_ == lists[0] for names_ in lists):
            out.append((hyper, list(lists[0]), True))
            continue
        order = {}
        for x in every:
            for n in x[gi][1]:
                order.setdefault(n, x[gi][2][n])
        exact = all(names_ == sorted(names_, key = order.__getitem__) for names_ in lists)
        out.append((hyper, sorted(order, key = order.__getitem__), exact))
    return (out, every_key) if with_keys else out


_PORTABLE_ENV = "UNSLOTH_PORTABLE_RESUME"  # 1: always write the 1-GPU optimizer.pt, 0: never
_PORTABLE_MAX_GB_ENV = "UNSLOTH_PORTABLE_RESUME_MAX_GB"
_PORTABLE_MAX_FRACTION = (
    0.05  # trainable share of the model below which the 1-GPU form is always written
)
_PORTABLE_MAX_GB = 2.0  # else: written while its float32 Adam moments stay under this


def _logical_numel(p):
    """Elements of this rank's part of `p`; a bitsandbytes 4-bit weight counts its unpacked elements."""
    shape = getattr(getattr(p, "quant_state", None), "shape", None)
    if shape is not None:
        return int(torch.Size(shape).numel())
    return int(_local(p).numel())


def portable_size_decision(model):
    """Collective (one all-reduce, before any gather): (write, why). The 1-GPU optimizer state is gathered onto
    rank 0's host memory, cheap for LoRA but the whole model's float32 Adam moments for full finetuning (64 GiB
    at 8B), so it is written only when the trainable parameters are at most 5% of the model or their moments fit
    UNSLOTH_PORTABLE_RESUME_MAX_GB (2). UNSLOTH_PORTABLE_RESUME=1 / 0 forces it on / off. Sums over ranks count
    replicated parameters once per rank, which only errs towards skipping."""
    value = os.environ.get(_PORTABLE_ENV, "").strip().lower()
    force_on, force_off = value in ("1", "true", "yes", "on"), value in ("0", "false", "no", "off")
    trainable = total = 0
    for p in model.parameters():
        n = _logical_numel(p)
        total += n
        if p.requires_grad:
            trainable += n
    try:
        budget = float(os.environ.get(_PORTABLE_MAX_GB_ENV, _PORTABLE_MAX_GB)) * 2**30
    except ValueError:
        budget = _PORTABLE_MAX_GB * 2**30
    counts = [float(trainable), float(total), float(force_on), float(force_off)]
    dist = _dist()
    if dist is not None:
        from .tp_native import _flag_device

        t = torch.tensor(counts, dtype = torch.float64, device = _flag_device())
        dist.all_reduce(t)
        counts = t.tolist()
    trainable, total, force_on, force_off = counts
    if force_off:
        return False, f"this run ({_PORTABLE_ENV}=0)"
    if force_on:
        return True, None
    moments = 8.0 * trainable  # float32 exp_avg + exp_avg_sq
    if trainable <= _PORTABLE_MAX_FRACTION * total or moments <= budget:
        return True, None
    return False, (
        f"{trainable / 1e6:,.0f}M trainable parameters ({100 * trainable / max(total, 1):.0f}% of the model, about "
        f"{moments / 2**30:,.1f} GiB of float32 Adam moments on rank 0; {_PORTABLE_ENV}=1 writes it anyway)"
    )


def portable_optimizer_state(model, optimizer):
    """Collective (every rank): the 1-GPU optimizer state keyed by parameter name on global rank 0 (None on the
    others), or (None, why) on every rank when it is not written (too large, or not convertible)."""
    write, why = portable_size_decision(model)
    if not write:
        return None, why
    dist = _dist()
    inner = _inner(optimizer)
    names = {id(p): n for n, p in model.named_parameters()}
    state, why = _local_state(optimizer)
    why = (
        None
        if _agree(why is None)
        else (why or "another rank's optimizer state cannot be converted")
    )
    if why is not None:
        return None, why
    groups = _group_layout(model, inner, names)
    if not _agree(groups is not None):
        return None, "the ranks' optimizers have different parameter group counts"
    # Every rank: which keys each name has (tensor-shaped or not), so peers missing a state still enter the gather.
    schema = {}
    for n, p in _trainables(model):
        entry = state.get(p)
        if entry is not None:
            schema[n] = {k: torch.is_tensor(v) and v.ndim > 0 for k, v in entry.items()}
    every = _all_gather_object(schema)
    union, owner = {}, {}
    for rank, part in enumerate(every):
        for n, keys in part.items():
            union.setdefault(n, {}).update(keys)
            owner.setdefault(n, rank)
    rank = _rank()
    # Rank 0 keeps the names it owns (every name under tensor / expert parallel); another owner (a pipeline stage)
    # keeps only its own, then sends them one parameter at a time. Non-owners drop each gathered tensor at once.
    merged, held = {}, {}
    for n, p in _trainables(model):
        keys = union.get(n)
        if keys is None:
            continue
        entry = state.get(p) or {}
        full = {}
        for key in sorted(keys):
            if keys[key]:
                value = entry.get(key)
                if value is None:  # a peer has this state, this rank never stepped its shard
                    value = torch.zeros(
                        _local(p).shape, dtype = torch.float32, device = _local(p).device
                    )
                value = _full_value(model, p, value)
            else:
                value = entry.get(key)
                if value is None:
                    continue
            if owner[n] == rank:
                full[key] = value.cpu() if torch.is_tensor(value) else value
        if owner[n] == rank:
            (merged if rank == 0 else held)[n] = full
    if dist is not None:
        for n in sorted(n for n, r in owner.items() if r != 0):
            if rank == owner[n]:
                dist.send_object_list([held.pop(n)], dst = 0)
            elif rank == 0:
                box = [None]
                dist.recv_object_list(box, src = owner[n])
                merged[n] = box[0]
    if rank != 0:
        return None, None
    param_groups = [{**hyper, "params": group_names} for hyper, group_names, _ in groups]
    return {"state": merged, "param_groups": param_groups}, None


def write_portable_optimizer(
    output_dir,
    portable,
    why = None,
):
    """Global rank 0: write optimizer.pt, or remove a stale one when the state could not be converted."""
    if _rank() != 0:
        return
    from .tp_native import _atomic_torch_save

    path = os.path.join(output_dir, PORTABLE_OPTIMIZER)
    if portable is not None:
        os.makedirs(output_dir, exist_ok = True)
        _atomic_torch_save(portable, path)
        return
    if os.path.lexists(path):
        os.remove(path)
    if why:
        from .tp_native import _print_rank0
        _print_rank0(
            f"Unsloth: {output_dir} resumes only on the layout it was saved on: the optimizer state of {why} "
            "is not written in the 1-GPU form."
        )


def _is_named(state_dict):
    groups = state_dict.get("param_groups") if isinstance(state_dict, dict) else None
    if not isinstance(groups, list) or not groups:
        return False
    ids = [i for g in groups if isinstance(g, dict) for i in (g.get("params") or [])]
    return bool(ids) and all(isinstance(i, str) for i in ids)


def portable_available(checkpoint, model = None):
    """Collective: every rank sees optimizer.pt and saved weights in `checkpoint` (at its root, or under an
    adapter's own directory: PEFT saves an adapter not named "default" in <checkpoint>/<name>)."""
    from .fsdp2_native import _has_checkpoint_weights

    directories = [checkpoint]
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict):
        directories += [os.path.join(checkpoint, name) for name in peft_config if name != "default"]
    ok = os.path.isfile(os.path.join(checkpoint, PORTABLE_OPTIMIZER)) and any(
        _has_checkpoint_weights(d) for d in directories
    )
    return _agree(ok)


# --------------------------------------------------------------------------------------------------------------
# Load on another layout
# --------------------------------------------------------------------------------------------------------------


def _grouped_lora_b(configs):
    """Whether an adapter config records fused expert lora_B saved grouped_by_expert (Zoo repacks those in place
    on the whole tensor after a load, which a rank's slice cannot do)."""
    if not configs:
        return False
    try:
        from unsloth_zoo.temporary_patches.moe_utils import (
            LORA_B_LAYOUT_GROUPED_BY_EXPERT,
            saved_lora_b_layouts,
        )
    except Exception:
        return False
    return any(
        LORA_B_LAYOUT_GROUPED_BY_EXPERT in (saved_lora_b_layouts(config) or {}).values()
        for config in configs.values()
    )


def _restorable(p):
    """A parameter a checkpoint's saved tensor is copied into: trainable, or a frozen float tensor PEFT saved
    (resized embeddings, an inactive adapter); never a quantized base weight."""
    if p.requires_grad:
        return True
    from .ep_native import _EP

    layout = getattr(p, _EP, None)
    if layout is not None and layout[0] == "4bit":
        return False
    return type(p).__name__ != "Params4bit" and _local(p).is_floating_point()


def portable_load_weights(model, checkpoint):
    """Collective: every parameter the checkpoint's 1-GPU adapter holds (the trainable ones, and frozen ones PEFT
    saved such as resized embeddings or an inactive adapter), each rank keeping its slice."""
    from .fsdp2_native import checkpoint_state_by_name

    mapped, legacy, _ = checkpoint_state_by_name(model, checkpoint)
    if not _agree(not _grouped_lora_b(legacy)):
        raise RuntimeError(
            f"Unsloth: {checkpoint} holds fused MoE expert lora_B saved grouped_by_expert (an older layout); "
            "resume it on the layout it was saved on, or set UNSLOTH_MOE_LORA_B_LAYOUT to that layout."
        )
    # The same names in the same order on every rank of a shard group (one file, one model): slice maps
    # are collectives over that group.
    targets = [(n, p) for n, p in model.named_parameters() if n in mapped and _restorable(p)]
    from .fsdp2_native import _persistent_buffers

    buffers = [(n, b) for n, b in _persistent_buffers(model) if n in mapped]
    with torch.no_grad():
        for n, p in targets:
            local = _local(p)
            smap = _slice_map(model, p)
            local.copy_(
                _to_local(n, mapped[n], smap, local).to(device = local.device, dtype = local.dtype)
            )
        # Persistent buffers (a routing correction bias) are whole on every rank; a sharded one is refused.
        for n, b in buffers:
            local = _local(b)
            b_full = mapped[n]
            if tuple(b_full.shape) != tuple(local.shape):
                raise RuntimeError(
                    f"Unsloth: the checkpoint buffer {n} has shape {tuple(b_full.shape)}, this rank holds "
                    f"{tuple(local.shape)} of it; resume this checkpoint on the layout it was saved on."
                )
            local.copy_(b_full.to(device = local.device, dtype = local.dtype))
    return len(targets) + len(buffers)


def _regroup(groups, sizes, keys):
    """The target's groups merged back into the 1-GPU checkpoint's: tensor parallel / native FSDP2 split each
    mixed DTensor / plain group into adjacent groups with the same hyperparameters (split_param_groups), so
    consecutive equal groups are merged until each saved group's size is met, names in module order (one GPU's
    create_optimizer order). None if they cannot be matched."""

    def same(a, b):
        # lr follows the scheduler; param_names (torch >= 2.6 named groups) lists each part's own names.
        skip = ("lr", "initial_lr", "param_names")
        return {k: v for k, v in a.items() if k not in skip} == {
            k: v for k, v in b.items() if k not in skip
        }

    def key(n):
        return keys.get(n, ((2, 0),))

    out, i = [], 0
    for size in sizes:
        start = i
        if size == 0 and (i >= len(groups) or groups[i][1]):
            out.append(({}, [], True))  # an empty saved group this optimizer does not keep
            continue
        if i >= len(groups):
            return None
        hyper, merged = groups[i][0], list(groups[i][1])
        i += 1
        while len(merged) < size and i < len(groups) and same(groups[i][0], hyper):
            merged += groups[i][1]
            i += 1
        if len(merged) != size:
            return None
        parts = groups[start:i]
        if len(parts) == 1:
            out.append(parts[0])
            continue
        # Each part keeps the original group's relative order; merged back in module order, which is exact when
        # every part already follows it.
        exact = all(g[2] and g[1] == sorted(g[1], key = key) for g in parts)
        out.append((hyper, sorted(merged, key = key), exact))
    return out if i == len(groups) else None


def _named_from_positional(model, inner, state_dict):
    """Collective: a 1-GPU (int-keyed) optimizer state keyed by name, through the 1-GPU parameter order."""
    names = {id(p): n for n, p in model.named_parameters()}
    groups, keys = _group_layout(model, inner, names, with_keys = True)
    saved = state_dict.get("param_groups") or []
    sizes = [len(g.get("params") or []) for g in saved]
    if groups is not None and [len(g[1]) for g in groups] != sizes:
        groups = _regroup(groups, sizes, keys)
    if groups is None:
        raise RuntimeError(
            "Unsloth: this checkpoint's optimizer.pt was written by one GPU with parameter groups of sizes "
            f"{sizes}, which this run's optimizer groups {[len(g['params']) for g in inner.param_groups]} do "
            "not match; resume with the same trainable parameters and optimizer settings."
        )
    if not all(exact for _, _, exact in groups):
        raise RuntimeError(
            "Unsloth: this checkpoint's optimizer.pt was written by one GPU (keyed by position), and this run's "
            "optimizer orders its parameters other than module order, so the positions cannot be matched to "
            "parameters; resume it on one GPU, or with an optimizer built over model.named_parameters()."
        )
    name_of = {}
    param_groups = []
    for (_, group_names, _), group in zip(groups, saved):
        name_of.update(zip(group["params"], group_names))
        param_groups.append(
            {**{k: v for k, v in group.items() if k != "params"}, "params": group_names}
        )
    state = {
        name_of[i]: entry for i, entry in (state_dict.get("state") or {}).items() if i in name_of
    }
    return {"state": state, "param_groups": param_groups}


def as_named_state(model, inner, state_dict):
    """Collective: `state_dict` keyed by parameter name. A name-keyed one is returned as is; one GPU's int-keyed
    one is mapped through the 1-GPU parameter order, its entries in torch Adam form (float32 moments)."""
    if _is_named(state_dict) or not (state_dict.get("state") or state_dict.get("param_groups")):
        return state_dict
    named = _named_from_positional(model, inner, state_dict)
    state = {}
    for name, entry in named["state"].items():
        entry = _normalize_entry(entry)
        if entry is None:
            raise RuntimeError(f"Unsloth: the optimizer state of {name} is not two Adam moments.")
        state[name] = entry
    named["state"] = state
    return named


def _bnb_import(inner, state_dict):
    """torch Adam state (float32 moments, whole or this rank's slices) into a bitsandbytes Adam: each moment is
    quantized into the 8-bit blocks (or copied into the 32-bit state) the optimizer allocates itself."""
    from .fsdp2_bnb_optim import _LocalShardMixin, is_local_shard_bnb_optimizer

    if is_local_shard_bnb_optimizer(inner):
        return type(inner).load_state_dict(inner, state_dict)
    vars(inner).pop(
        "_unsloth_flat_plans", None
    )  # Unsloth's fused 8-bit AdamW repacks from the new state
    saved_groups = state_dict["param_groups"]
    torch.optim.Optimizer.load_state_dict(inner, {"state": {}, "param_groups": saved_groups})
    slots = [
        (group, p, gindex, pindex)
        for gindex, group in enumerate(inner.param_groups)
        for pindex, p in enumerate(group["params"])
    ]
    with torch.no_grad():
        for index, saved in state_dict["state"].items():
            _LocalShardMixin._import_state(inner, *slots[index], saved)
    return None


def _load_named(
    model,
    inner,
    named,
    maps = None,
):
    """Load a name-keyed, normalised optimizer state into `inner` (whole parameters, or this rank's slices)."""
    from .tp_native import _dtensor_api

    names = {id(p): n for n, p in model.named_parameters()}
    group_of = {n: g for g in named["param_groups"] for n in g["params"]}
    bnb = _is_bnb(inner)
    if bnb and not _is_bnb_adam(inner):
        raise RuntimeError(
            f"Unsloth: resuming a checkpoint written on another layout needs a torch optimizer or a bitsandbytes "
            f"Adam / AdamW (its state is two Adam moments); this run uses {type(inner).__name__}."
        )
    unknown = [
        names.get(id(p), "?")
        for g in inner.param_groups
        for p in g["params"]
        if p.requires_grad and names.get(id(p)) not in group_of
    ]
    if unknown:
        raise RuntimeError(
            f"Unsloth: this checkpoint's optimizer has no entry for {unknown[0]} ({len(unknown)} parameters), so "
            "it was not saved from this model's optimizer."
        )
    state, groups, index = {}, [], 0
    saved_index = {id(g): i for i, g in enumerate(named["param_groups"])}
    empty = [i for i, g in enumerate(named["param_groups"]) if not g["params"]]
    origin = []  # the saved group each live group takes its hyperparameters (and scheduler entries) from
    for group in inner.param_groups:
        group_names = [names.get(id(p)) for p in group["params"]]
        saved = next((group_of[n] for n in group_names if n in group_of), None)
        if saved is None and not group_names and empty:
            saved = named["param_groups"][empty.pop(0)]
        origin.append(
            saved_index[id(saved)] if saved is not None else (origin[-1] if origin else 0)
        )
        hyper = {
            k: (saved[k] if saved is not None and k in saved else v)
            for k, v in group.items()
            if k != "params"
        }
        ids = []
        for p, n in zip(group["params"], group_names):
            entry = named["state"].get(n)
            if entry is not None:
                entry = _normalize_entry(entry)
                if entry is None:
                    raise RuntimeError(
                        f"Unsloth: the optimizer state of {n} is not two Adam moments."
                    )
                local = _local(p)
                smap = None if maps is None else maps.get(n)
                out = {}
                for key, value in entry.items():
                    if torch.is_tensor(value) and value.ndim > 0:
                        value = _to_local(f"{n} ({key})", value, smap, local).to(local.device)
                        if _is_dtensor(p) and not bnb:
                            value = _dtensor_api()[0].from_local(
                                value,
                                p.device_mesh,
                                p.placements,
                                run_check = False,
                                shape = p.shape,
                                stride = p.stride(),
                            )
                    out[key] = value
                state[index] = out
            ids.append(index)
            index += 1
        groups.append({**hyper, "params": ids})
    loaded = {"state": state, "param_groups": groups}
    # A scheduler saved over the checkpoint's groups is read through the same map (remap_scheduler_state).
    inner.__dict__["_unsloth_portable_groups"] = (origin, len(named["param_groups"]))
    if bnb:
        return _bnb_import(inner, loaded)
    # The class method: a Trainer load in progress may have routed the instance's load_state_dict here.
    return type(inner).load_state_dict(inner, loaded)


def remap_scheduler_state(optimizer, state):
    """An LR scheduler state saved over the checkpoint's optimizer groups, re-indexed to this optimizer's groups
    after a portable load (a group split in two by tensor parallel / native FSDP2 takes its origin's entries)."""
    inner = _inner(optimizer) if optimizer is not None else None
    mapping = getattr(inner, "__dict__", {}).get("_unsloth_portable_groups")
    if mapping is None or not isinstance(state, dict):
        return state
    origin, saved = mapping
    if origin == list(range(saved)):
        return state
    return {
        k: ([v[i] for i in origin] if isinstance(v, list) and len(v) == saved else v)
        for k, v in state.items()
    }


def portable_load_optimizer(model, optimizer, state_dict):
    """Collective: a 1-GPU optimizer state (name-keyed from a parallel checkpoint, or one GPU's int-keyed one)
    into this rank's optimizer, each rank keeping its slices."""
    inner = _inner(optimizer)
    if not _is_named(state_dict):
        state_dict = _named_from_positional(model, inner, state_dict)
    return _load_named(model, inner, state_dict, _slice_maps(model))


class _swap_load_state_dict:
    """Route `optimizer.load_state_dict` (the inner torch optimizer's) through `fn` for one Trainer load."""

    def __init__(self, optimizer, fn):
        self.inner, self.fn = _inner(optimizer), fn

    def __enter__(self):
        self.prior = self.inner.__dict__.get("load_state_dict")
        self.inner.load_state_dict = self.fn
        return self

    def __exit__(self, *exc):
        if self.prior is None:
            self.inner.__dict__.pop("load_state_dict", None)
        else:
            self.inner.load_state_dict = self.prior
        return False


# --------------------------------------------------------------------------------------------------------------
# One GPU / DDP: the Trainer's own load, made name-aware
# --------------------------------------------------------------------------------------------------------------


def _sharded_trainer(trainer):
    if (
        bool(trainer.__dict__.get("_unsloth_tensor_parallel"))
        or bool(getattr(trainer, "is_deepspeed_enabled", False))
        or bool(getattr(trainer, "is_fsdp_enabled", False))
    ):
        return True
    model = getattr(trainer, "model", None)
    if model is None or not isinstance(model, torch.nn.Module):
        return True
    import sys

    # A module that was never imported sharded nothing (and one GPU need not import it).
    for module, check in (
        ("ep_native", "is_expert_parallel"),
        ("tp_native", "is_tensor_parallel"),
        ("pp_native", "is_pipeline_parallel"),
        ("fsdp2_native", "is_native_fsdp2"),
    ):
        loaded = sys.modules.get(f"{__package__}.{module}")
        if loaded is not None and getattr(loaded, check)(model):
            return True
    return False


def install_portable_resume(trainer):
    """After Trainer.__init__ (models/_utils.py), for a model this process holds whole (one GPU, DDP): a
    name-keyed optimizer.pt (a tensor / pipeline / expert parallel or native FSDP2 checkpoint) is loaded by
    parameter name, torch Adam or bitsandbytes Adam alike; a parallel checkpoint without it is refused."""
    if _sharded_trainer(trainer) or trainer.__dict__.get("_unsloth_portable_resume"):
        return False
    prior = trainer.__dict__.get("_load_optimizer_and_scheduler")

    def original(*args, **kwargs):
        # Resolved per call: native FSDP2 patches the class when it shards, after Trainer.__init__.
        fn = prior or type(trainer)._load_optimizer_and_scheduler.__get__(trainer)
        return fn(*args, **kwargs)

    def _load_optimizer_and_scheduler(checkpoint, *args, **kwargs):
        if (
            checkpoint is None
            or getattr(trainer, "optimizer", None) is None
            or _sharded_trainer(trainer)
        ):
            return original(checkpoint, *args, **kwargs)
        has_opt = os.path.isfile(os.path.join(checkpoint, PORTABLE_OPTIMIZER))
        tp_dir = os.path.isdir(os.path.join(checkpoint, "unsloth_tp"))
        ep_dir = os.path.isdir(os.path.join(checkpoint, "unsloth_ep"))
        if (tp_dir or ep_dir) and not has_opt:
            raise RuntimeError(
                f"Unsloth: {checkpoint} was written under tensor / pipeline / expert parallel without the 1-GPU "
                f"{PORTABLE_OPTIMIZER} (an older Unsloth, or an optimizer whose state cannot be converted), so its "
                "optimizer state only resumes on the layout it was saved on."
            )
        inner = _inner(trainer.optimizer)
        own = inner.__dict__.get("load_state_dict")

        def load(state_dict, *a, **k):
            if _is_named(state_dict):
                names = {n for n, _ in trainer.model.named_parameters()}
                if names & {n for g in state_dict["param_groups"] for n in g["params"]}:
                    return _load_named(trainer.model, inner, state_dict)
            elif ep_dir:
                raise RuntimeError(
                    f"Unsloth: {checkpoint}'s {PORTABLE_OPTIMIZER} holds expert parallel rank 0's experts only "
                    "(written by an older Unsloth); its optimizer state only resumes on the layout it was saved on."
                )
            return (own or type(inner).load_state_dict.__get__(inner))(state_dict, *a, **k)

        scheduler = getattr(trainer, "lr_scheduler", None)
        scheduler_load = (
            scheduler.__dict__.get("load_state_dict") if scheduler is not None else None
        )

        def load_scheduler(state, *a, **k):
            state = remap_scheduler_state(trainer.optimizer, state)
            return (scheduler_load or type(scheduler).load_state_dict.__get__(scheduler))(
                state, *a, **k
            )

        if scheduler is not None:
            scheduler.load_state_dict = load_scheduler
        try:
            with _swap_load_state_dict(trainer.optimizer, load):
                return original(checkpoint, *args, **kwargs)
        finally:
            if scheduler is not None:
                if scheduler_load is None:
                    scheduler.__dict__.pop("load_state_dict", None)
                else:
                    scheduler.load_state_dict = scheduler_load

    trainer.__dict__["_load_optimizer_and_scheduler"] = _load_optimizer_and_scheduler
    trainer.__dict__["_unsloth_portable_resume"] = True
    return True
