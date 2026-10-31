# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Post-training calls on a TP model, every rank must make them; saves gather to rank 0 while other
ranks wait (no collective timeout); generate copies tp rank 0 RNG."""

from __future__ import annotations

import datetime
import gc
import os
import pickle
import types

import torch
import torch.nn as nn

from . import tp_native as tpn

__all__ = [
    "tp_root",
    "gather_replica",
    "release_replica",
    "replica_active",
    "tp_for_inference",
    "tp_for_training",
    "tp_generate",
    "tp_collective",
    "tp_world_save",
    "mark_world_save",
    "tp_world_push",
]

_REPLICA = "_unsloth_tp_replica"
_ACTIVE = []  # rank 0 inside a collective save: nested saves write directly
_GENERATE_DEPTH = [0]
_COUNTERS = {}
_MiB = 1 << 20


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def _rank():
    dist = _dist()
    return dist.get_rank() if dist is not None else 0


def tp_root(model):
    """The module apply_tensor_parallel was called on (any wrapper level of it resolves there)."""
    for level in tpn._tp_chain(model):
        root = level.__dict__.get("_unsloth_tp_root")
        if root is not None:
            return root
    return model


def replica_active(model):
    return tp_root(model).__dict__.get(_REPLICA) is not None


def _device(root):
    for param in root.parameters():
        if tpn._is_dtensor(param):
            return param.to_local().device
        return param.device
    return torch.device("cpu")


def _gib(n):
    return f"{n / (1 << 30):.2f} GiB" if n >= (1 << 30) // 10 else f"{n / _MiB:.1f} MiB"


def _marked(model, flag):
    return any(level.__dict__.get(flag) is not None for level in tpn._tp_chain(model))


def _fsdp2_units(model):
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        try:
            from torch.distributed._composable.fsdp import FSDPModule
        except ImportError:
            return []
    return [m for m in model.modules() if isinstance(m, FSDPModule)]


def _refusal(root, what):
    """None when `what` can run on this model, else why not."""
    if _marked(root, "_unsloth_pp"):
        return (
            f"Unsloth: `{what}` is not supported on a pipeline parallel model (each rank holds one stage); save "
            "the adapter (PipelineParallel.save_adapter) and run it on one GPU."
        )
    if _fsdp2_units(root):
        return (
            f"Unsloth: `{what}` is not supported on a tensor parallel x native FSDP2 model yet; save the adapter "
            "(model.save_pretrained or save_tp_adapter, on every rank) and run it on one GPU."
        )
    if any(tpn._is_dtensor(p) and p.device_mesh.ndim != 1 for p in root.parameters()):
        return f"Unsloth: `{what}` supports 1-D tensor parallel meshes only."
    return None


def collective_applies(model):
    """True when the save / generate of `model` goes through this module (tensor parallel, tp x DDP)."""
    return _marked(model, tpn._TP_FLAG) and _refusal(tp_root(model), "save") is None


def _items(root, select):
    """(module, name, param) of the sharded parameters a replica replaces, in the same order on every rank."""
    owners, order, seen = {}, [], set()
    for module in root.modules():
        for name, param in module._parameters.items():
            if param is None:
                continue
            owners.setdefault(id(param), []).append((module, name))
            if id(param) not in seen:
                seen.add(id(param))
                order.append(param)
    names = {id(p): n for n, p in root.named_parameters(remove_duplicate = False)}
    out = []
    for param in order:
        name = names.get(id(param), "")
        four_bit = getattr(param, "_unsloth_tp_layout", None) is not None
        if not (tpn._is_dtensor(param) or four_bit):
            continue
        if select != "all" and (four_bit or not (param.requires_grad or "lora_" in name)):
            continue
        out.append((param, owners[id(param)]))
    return out


def _bytes(param):
    if tpn._is_dtensor(param):
        return param.numel() * param.element_size()  # global shape
    tp = param._unsloth_tp_layout[3]
    full_state = getattr(param, "_unsloth_tp_quant_state", None)
    absmax = (full_state or param.quant_state).absmax
    extra = absmax.numel() * absmax.element_size() * (1 if full_state is not None else tp)
    return param.data.numel() * param.data.element_size() * tp + extra


def _workspace(items):
    """Peak transient gather bytes: pieces, concatenation and reorder of the largest tensor."""
    return 3 * max((_bytes(p) for p, _ in items), default = 0)


def _merge_workspace(items):
    """Peak GPU bytes to merge one CPU-staged 4-bit projection (save.py _merge_lora)."""
    peak = 0
    for p, _ in items:
        if not tpn._is_dtensor(p):
            numel = p.data.numel() * p.data.element_size() * 2 * p._unsloth_tp_layout[3]
            peak = max(peak, _bytes(p) + numel * (2 + 4 + 4))
    return max(_workspace(items), peak)


def _free_bytes(device):
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        return free, "free on its GPU"
    return _host_free(), "of available host RAM"


def _host_free():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else 0
        return pages * os.sysconf("SC_PAGE_SIZE") if pages else 1 << 62


def _margin(need):
    return max(512 * _MiB, need // 10)


def _comm_device(group, device):
    dist = _dist()
    backend = str(dist.get_backend(group)) if dist is not None else "gloo"
    if "nccl" in backend or "xccl" in backend or "hccl" in backend:
        return device
    return torch.device("cpu")


def _gather_cat(local, dim, group, tp):
    import torch.distributed as dist

    local = local.contiguous()
    pieces = [torch.empty_like(local) for _ in range(tp)]
    dist.all_gather(pieces, local, group = group)
    return torch.cat(pieces, dim = dim)


def _full_params4bit(weight, base, group, device, keep):
    """The whole 4-bit weight: the shards' packed codes concatenated back in the 1-GPU order (bitwise the 1-GPU
    codes) with the 1-GPU QuantState apply_tensor_parallel kept on the CPU (double quantized absmax included)."""
    import bitsandbytes as bnb

    style, parts, rank, tp = weight._unsloth_tp_layout
    state = weight.quant_state
    blocksize = state.blocksize
    out_local, in_local = state.shape
    codes = weight.data.reshape(-1).view(torch.uint8).view(out_local, in_local // 2)
    kv = style == tpn.KV_REPLICATED
    col = style == tpn.COLWISE or kv
    dim = 0 if col else 1
    codes = _gather_cat(codes, dim, group, tp)
    full_state = getattr(weight, "_unsloth_tp_quant_state", None)
    absmax = None
    if full_state is None:  # no 1-GPU state kept: the shards' float32 absmax, gathered the same way
        absmax = _gather_cat(
            state.absmax.float().view(out_local, in_local // blocksize), dim, group, tp
        )
    if kv:
        # Replicated key / value heads: tp / num_key_value_heads consecutive ranks hold the same head's rows.
        step = tp // (sum(parts) // out_local)
        codes = torch.cat(codes.split(out_local)[::step])
        absmax = None if absmax is None else torch.cat(absmax.split(out_local)[::step])
        tp = 1
    if not keep:
        return None
    if col:
        codes = tpn._unpermute(codes, 0, parts, tp)
        absmax = None if absmax is None else tpn._unpermute(absmax, 0, parts, tp)
        shape = (sum(parts), in_local)
    else:
        shape = (out_local, in_local * tp)
    storage = weight.data.dtype
    packed = codes.contiguous().view(storage).reshape(-1, 1).to(device)
    if full_state is not None:
        full_state = tpn.quant_state_copy(full_state, device)
    else:
        full_state = bnb.functional.QuantState(
            absmax = absmax.contiguous().to(device),
            shape = torch.Size(shape),
            code = state.code.to(device),
            blocksize = blocksize,
            quant_type = state.quant_type,
            dtype = state.dtype,
        )
    full = bnb.nn.Params4bit(
        packed,
        requires_grad = False,
        quant_state = full_state,
        blocksize = blocksize,
        compress_statistics = full_state.nested,
        quant_type = state.quant_type,
        quant_storage = storage,
        module = base,
        bnb_quantized = True,
    )
    if device.type == "cpu" and weight.device.type != "cpu":
        # CPU-staged export replica: save.py _merge_lora dequantizes it on its original GPU.
        full._unsloth_tp_stage_device = weight.device
    return full


class _Replica:
    def __init__(self, select, device):
        self.select = select
        self.device = device
        self.persistent = False  # for_inference: kept until for_training / a gradient forward
        self.swaps = []  # (module, name, sharded parameter)
        self.attrs = []  # (module __dict__, key, previous value or _MISSING)
        self.hooks = []

    def set(self, obj, key, value):
        d = obj.__dict__
        self.attrs.append((d, key, d.get(key, tpn._MISSING)))
        if value is tpn._MISSING:
            d.pop(key, None)
        else:
            d[key] = value


def _room(need, free, device):
    """Every rank's (need, free) in bytes, rank order (all ranks run the gather or none does)."""
    dist = _dist()
    mine = torch.tensor(
        [float(need), float(free)], dtype = torch.float64, device = _comm_device(None, device)
    )
    if dist is None:
        return [(need, free)]
    rows = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(rows, mine)
    return [(int(r[0].item()), int(r[1].item())) for r in rows]


def _short(room):
    """The first rank without room, or None."""
    return next(
        (r for r, (need, free) in enumerate(room) if need and need + _margin(need) > free), None
    )


def _plan(root, select, keep, what, allow_cpu):
    """(items, device) once every rank agreed the gather fits; raises with the numbers otherwise."""
    items = _items(root, select)
    kept = sum(_bytes(p) for p, _ in items) if keep else 0
    workspace = _workspace(items)
    device = _device(root)
    free, where = _free_bytes(device)
    room = _room(kept + workspace, free, device)
    short = _short(room)
    if short is None:
        return items, device
    if allow_cpu and device.type != "cpu":
        staging = _room(_merge_workspace(items), free, device)
        cpu_room = _room(kept, _host_free(), device)
        if _short(staging) is None and _short(cpu_room) is None:
            if keep:
                print(
                    f"Unsloth: tensor parallel `{what}`: the whole model ({_gib(kept)}) does not fit "
                    f"{device} ({_gib(free)} free); gathering it to the CPU instead."
                )
            return items, torch.device("cpu")
        where = f"{where}, {_gib(cpu_room[short][1])} of available host RAM"
    need_s, free_s = room[short]
    detail = (
        f"{_gib(need_s)} on its device for it ({_gib(need_s - workspace)} for the replica it keeps, "
        f"{_gib(workspace)} of gather workspace)"
    )
    raise RuntimeError(
        f"Unsloth: tensor parallel `{what}` gathers a whole single-GPU replica of the model, but rank {short} "
        f"needs {detail} (+ {_gib(_margin(need_s))} headroom) and has {_gib(free_s)} {where}. "
        "Free memory (e.g. del trainer; gc.collect(); torch.cuda.empty_cache()), or save the adapter with "
        "model.save_pretrained(...) and run this on one GPU."
    )


def gather_replica(
    model,
    select = "all",
    keep = True,
    what = "generate",
    allow_cpu = False,
):
    """Collective: replace the sharded parameters by whole tensors. select "all" or "adapters"; keep = False
    joins the gather without keeping anything. Idempotent; release_replica undoes it."""
    root = tp_root(model)
    current = root.__dict__.get(_REPLICA)
    if current is not None and (current.select == "all" or current.select == select):
        return current
    if current is not None:
        release_replica(root)
    refusal = _refusal(root, what)
    if refusal is not None:
        raise RuntimeError(refusal)
    mesh = tpn._tp_meshes(root)[0]
    if mesh is None:
        raise RuntimeError("Unsloth: this tensor parallel model has no tp mesh.")
    group, tp = mesh.get_group(), mesh.size()
    items, device = _plan(root, select, keep, what, allow_cpu)
    layouts = tpn._param_layouts(root)
    replica = _Replica(select, device) if keep else None
    with torch.no_grad():
        for param, owners in items:
            module = owners[0][0]
            if tpn._is_dtensor(param):
                full = tpn._full(param, param, layouts.get(id(param)))
                if not keep:
                    del full
                    continue
                new = nn.Parameter(full.detach().to(device), requires_grad = param.requires_grad)
            else:
                new = _full_params4bit(param, module, group, device, keep)
                if not keep:
                    continue
                if "quant_state" in module.__dict__:
                    replica.set(module, "quant_state", new.quant_state)
            for owner, name in owners:
                replica.swaps.append((owner, name, param))
                owner._parameters[name] = new
    if not keep:
        return None
    if select == "all":
        _switch_to_one_gpu(root, replica)
    root.__dict__[_REPLICA] = replica
    return replica


def _switch_to_one_gpu(root, replica):
    """The module state apply_tensor_parallel changed, back to the one-GPU model while the replica is in."""
    plan = root.__dict__.get("_unsloth_tp_plan")
    for region in plan.regions if plan is not None else ():
        module = region.module
        replica.set(module, "_unsloth_tp_gathered", True)  # the region hooks pass through
        saved = module.__dict__.get("_unsloth_tp_saved_attrs")
        if saved is not None:
            for key, value in saved.items():
                replica.set(module, key, value)
        for key in ("_unsloth_tp_heads", "_unsloth_tp_kv_heads"):
            if key in module.__dict__:
                replica.set(module, key, tpn._MISSING)
        for proj in region.projs:
            p = proj.module
            if "_unsloth_tp_prior_forward" in p.__dict__:
                replica.set(p, "forward", p.__dict__["_unsloth_tp_prior_forward"])
            if "_unsloth_tp_local" in p.__dict__:
                replica.set(p, "_unsloth_tp_local", tpn._MISSING)
    for block in plan.moe_blocks if plan is not None else ():
        replica.set(block, "_unsloth_tp_gathered", True)  # replicated MoE: no tp average
    for level in tpn._tp_chain(root):
        forward = level.__dict__.get("forward")
        original = getattr(forward, "__wrapped__", None)
        if getattr(forward, "__name__", "") != "forward_without_cache" or original is None:
            continue
        bound_class = getattr(original, "__func__", None) is getattr(type(level), "forward", None)
        same = getattr(original, "__self__", None) is level
        replica.set(level, "forward", tpn._MISSING if bound_class and same else original)
    replica.hooks.append(root.register_forward_pre_hook(_release_before_training))


def _release_before_training(module, args):
    # A grad-recording forward (also in eval mode) must run on the optimizer's shards, never the replica.
    if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
        release_replica(module)


def release_replica(model):
    """Put the sharded parameters back (local, no collective). True when a replica was in."""
    root = tp_root(model)
    replica = root.__dict__.pop(_REPLICA, None)
    if replica is None:
        return False
    for owner, name, param in reversed(replica.swaps):
        owner._parameters[name] = param
    for d, key, value in reversed(replica.attrs):
        if value is tpn._MISSING:
            d.pop(key, None)
        else:
            d[key] = value
    for hook in replica.hooks:
        hook.remove()
    replica.swaps.clear()
    replica.attrs.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def tp_for_inference(model):
    if _marked(model, tpn._TP_FLAG):
        replica = gather_replica(model, "all", keep = True, what = "for_inference")
        if (
            _GENERATE_DEPTH[0] == 0
        ):  # the user's call (generate's own for_inference does not keep it)
            replica.persistent = True
    elif _marked(model, "_unsloth_pp"):
        raise RuntimeError(_refusal(model, "for_inference"))


def tp_for_training(model):
    if _marked(model, tpn._TP_FLAG):
        release_replica(model)


def sync_generation_rng(model):
    """tp rank 0's CPU and accelerator RNG states to its tp group: sampled tokens agree across tp ranks."""
    dist = _dist()
    root = tp_root(model)
    mesh = tpn._tp_meshes(root)[0]
    if dist is None or mesh is None or mesh.size() == 1:
        return
    group = mesh.get_group()
    src = dist.get_global_rank(group, 0)
    device = _device(root)
    comm = _comm_device(group, device)
    cpu = torch.get_rng_state().to(comm)
    dist.broadcast(cpu, src = src, group = group)
    torch.set_rng_state(cpu.cpu())
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device).to(comm)
        dist.broadcast(state, src = src, group = group)
        torch.cuda.set_rng_state(state.cpu(), device)


def tp_generate(level, prior, args, kwargs):
    """Gather the replica (collective), sync sampling, generate; a replica gathered here is released after."""
    transient = False
    if _GENERATE_DEPTH[0] == 0:
        transient = not replica_active(level)
        gather_replica(level, "all", keep = True, what = "generate")
        sync_generation_rng(level)
    if prior is not None:
        fn = prior
    elif hasattr(type(level), "generate"):
        fn = types.MethodType(type(level).generate, level)
    else:  # a forwarding wrapper (PEFT LoraModel): what its __getattr__ resolves, the level below
        fn = type(level).__getattr__(level, "generate")
    _GENERATE_DEPTH[0] += 1
    try:
        return fn(*args, **kwargs)
    finally:
        _GENERATE_DEPTH[0] -= 1
        replica = tp_root(level).__dict__.get(_REPLICA)
        if transient and replica is not None and not replica.persistent:
            release_replica(level)


def _timeout():
    try:
        return datetime.timedelta(
            seconds = float(os.environ.get("UNSLOTH_TP_SAVE_TIMEOUT", 7 * 86400))
        )
    except ValueError:
        return datetime.timedelta(days = 7)


def _key(namespace):
    _COUNTERS[namespace] = _COUNTERS.get(namespace, 0) + 1
    return f"unsloth_tp_{namespace}_{_COUNTERS[namespace]}"


def _store():
    dist = _dist()
    try:
        return dist.distributed_c10d._get_default_store()
    except Exception:
        return None


def _publish(key, payload):
    store = _store()
    if store is None:
        _dist().barrier()
        return
    try:
        data = pickle.dumps(payload)
    except Exception:
        data = pickle.dumps((payload[0], None) + tuple(payload[2:]))
    store.set(key, data)


def _wait(key):
    store = _store()
    if store is None:
        _dist().barrier()
        return ("ok", None)
    store.wait([key], _timeout())
    return pickle.loads(store.get(key))


def _keep_tree(root):
    """Every module's children, parameters and PEFT merge flags: a writer that unloads or merges in place
    (merge_and_unload) must leave rank 0 with the module tree the other ranks still have."""
    if not isinstance(root, nn.Module):
        return []
    kept = []
    for module in root.modules():
        merged = getattr(module, "merged_adapters", None)
        kept.append(
            (
                module,
                list(module._modules.items()),
                list(module._parameters.items()),
                list(merged) if isinstance(merged, list) else None,
            )
        )
    return kept


def _restore_tree(kept):
    for module, children, params, merged in kept:
        if list(module._modules.items()) != children:
            module._modules.clear()
            module._modules.update(children)
        if list(module._parameters.items()) != params:
            module._parameters.clear()
            module._parameters.update(params)
        if merged is not None and getattr(module, "merged_adapters", None) is not None:
            module.merged_adapters[:] = merged


def _on_rank0(
    key,
    what,
    call,
    model = None,
):
    """Rank 0 runs `call`; the others wait for it on the store and return its result or raise its error."""
    if _rank() == 0:
        _ACTIVE.append(what)
        payload = ("error", "rank 0 did not finish")
        kept = _keep_tree(model)
        try:
            result = call()
            payload = ("ok", result)
            return result
        except BaseException as exc:
            payload = ("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _restore_tree(kept)
            _ACTIVE.pop()
            _publish(key, payload)
    status, value = _wait(key)[:2]
    if status != "ok":
        raise RuntimeError(f"Unsloth: tensor parallel `{what}` failed on rank 0: {value}")
    return value


def tp_collective(
    model,
    what,
    call,
    select = "all",
):
    """Every rank calls; the tensors `select` needs are gathered to rank 0, which runs `call` (the unchanged
    single-process save) on the one-GPU-looking model; the other ranks wait and return rank 0's result."""
    if _ACTIVE or _dist() is None:
        return call()
    root = tp_root(model)
    refusal = _refusal(root, what)
    if refusal is not None:
        raise RuntimeError(refusal)
    key = _key("save")
    writer = _rank() == 0
    current = root.__dict__.get(_REPLICA)
    gathered = False
    if current is None or (current.select != "all" and select == "all"):
        if _rank() == 0 and select == "all":
            print(
                f"Unsloth: tensor parallel `{what}`: gathering the whole model to rank 0, which saves."
            )
        gather_replica(root, select, keep = writer, what = what, allow_cpu = select == "all")
        gathered = writer
    try:
        return _on_rank0(key, what, call, root)
    finally:
        if gathered:
            release_replica(root)


def mark_world_save(obj):
    """Route a tokenizer / processor's save_pretrained through tp_world_save (nested tokenizer included)."""
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict) or d.get("_unsloth_tp_world_save"):
        return obj
    d["_unsloth_tp_world_save"] = True
    save = getattr(obj, "save_pretrained", None)
    if callable(save) and getattr(save, "__name__", "") != "unsloth_tokenizer_save_pretrained":
        from ..save import _TokenizerBoundMethod
        if "save_pretrained" in d:
            d["_unsloth_tp_inner_save"] = d["save_pretrained"]
        # A picklable bound wrapper, not a closure: spawn-based DataLoader workers pickle the processor.
        d["save_pretrained"] = _TokenizerBoundMethod(
            unsloth_tp_world_save_pretrained, obj, "save_pretrained"
        )
    nested = d.get("tokenizer")
    if nested is not None and nested is not obj:
        mark_world_save(nested)
    return obj


def unsloth_tp_world_save_pretrained(
    obj,
    save_directory,
    *args,
    push_to_hub = False,
    **kwargs,
):
    """mark_world_save's save_pretrained: the object's own save, through tp_world_save."""
    save = obj.__dict__.get("_unsloth_tp_inner_save") or type(obj).save_pretrained.__get__(obj)
    what = f"{type(obj).__name__}.save_pretrained"
    result = tp_world_save(what, lambda path: save(path, *args, **kwargs), save_directory)
    if push_to_hub:
        repo_id = kwargs.pop(
            "repo_id", os.path.basename(os.path.normpath(os.fspath(save_directory)))
        )
        push = {
            k: kwargs[k] for k in ("commit_message", "token", "private", "revision") if k in kwargs
        }
        tp_world_push(f"{type(obj).__name__}.push_to_hub", lambda: obj.push_to_hub(repo_id, **push))
    return result


def _relocated(value, private, target):
    """The save's return value with paths under `private` pointing into `target` instead."""
    if isinstance(value, (tuple, list)):
        return type(value)(_relocated(v, private, target) for v in value)
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value)
        if os.path.abspath(path).startswith(private):
            moved = os.path.join(target, os.path.relpath(os.path.abspath(path), private))
            return type(value)(moved) if not isinstance(value, str) else moved
    return value


def tp_world_save(what, write, directory):
    """Each rank writes a private copy and os.replace's every file in; no rank waits, so nothing can desync."""
    dist = _dist()
    try:
        target = os.fspath(directory)
    except TypeError:
        target = None
    if _ACTIVE or dist is None or dist.get_world_size() == 1 or not target:
        return write(directory)
    import shutil
    import tempfile

    absolute = os.path.abspath(target)
    parent = os.path.dirname(absolute)
    os.makedirs(absolute, exist_ok = True)
    private = tempfile.mkdtemp(
        prefix = f".{os.path.basename(absolute)}.unsloth_tp_rank{_rank()}_", dir = parent
    )
    try:
        result = write(private)
        for root, _, files in os.walk(private):
            rel = os.path.relpath(root, private)
            destination = absolute if rel == "." else os.path.join(absolute, rel)
            os.makedirs(destination, exist_ok = True)
            for name in files:
                os.replace(os.path.join(root, name), os.path.join(destination, name))
    finally:
        shutil.rmtree(private, ignore_errors = True)
    return _relocated(result, private, target)


def tp_world_push(what, call):
    """A Hub upload of a tokenizer / processor: global rank 0 uploads, the other ranks return None (no
    rendezvous, so an upload rank 0 makes alone desyncs nothing)."""
    dist = _dist()
    if _ACTIVE or dist is None or dist.get_world_size() == 1:
        return call()
    if _rank() != 0:
        return None
    _ACTIVE.append(what)
    try:
        return call()
    finally:
        _ACTIVE.pop()
