# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expert parallel: data parallel ranks that each keep only E / ep of every MoE block's routed experts.

    UNSLOTH_PARALLEL="ep=2" torchrun --nproc-per-node 2 train.py

    with expert_parallel_loading(mesh):              # optional: 4-bit quantize-on-load keeps this rank's experts
        model, tokenizer = FastModel.from_pretrained(name, load_in_4bit = True)
    model = FastModel.get_peft_model(model, ...)
    apply_expert_parallel(model, mesh)               # the 1-D mesh over every rank (ep = data parallel size)

Every rank reads its own batches (DDP). Everything but the routed experts stays whole and data parallel: DDP
averages its gradients. Rank r owns the contiguous experts [r E / ep, (r + 1) E / ep) and their LoRA factors
(sliced from the 1-GPU init, so the adapter equals one GPU's). Each experts call sends every token once to each
rank owning one of its top-k experts (all-to-all), that rank runs its experts on the (token, slot) pairs it got
and the weighted outputs come back the same way, summed per token. The backward is the same exchange reversed.
The expert parameters are kept out of DDP; their gradients (the sum over every rank's tokens of every rank's
loss) are divided by ep, which is DDP's average. Clipping counts each expert once (a sum over ranks), and a
non-finite expert gradient on one rank makes every rank skip the fp16 step, so the replicas never drift.

The 4-bit expert stacks reuse the tensor parallel expert-index slicing (tp_native): bitwise the same bytes.
Collective: every rank must run the same number of MoE forwards (the Trainer's even batches do; generate on
one rank alone does not). save_model writes the 1-GPU adapter (expert LoRA gathered). Merged / GGUF exports and
Trainer checkpoint resume are refused under expert parallel."""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "apply_expert_parallel",
    "expert_parallel_loading",
    "is_expert_parallel",
    "ep_full_adapter_state_dict",
    "ep_clip_grad_norm_",
    "ep_env_size",
    "ep_env_mesh",
    "ep_env_load",
    "ep_env_apply",
    "ep_trainer_setup",
]

_EP = "_unsloth_ep"  # on each expert parameter: its gather layout; on the model: the mesh
_EP_ENV = "UNSLOTH_EP_SIZE"


def ep_env_size():
    try:
        return int(os.environ.get(_EP_ENV, "1") or 1)
    except ValueError:
        raise ValueError(f"Unsloth: {_EP_ENV}={os.environ.get(_EP_ENV)!r} is not an integer.")


_ENV_MESH = []


def ep_env_mesh():
    """UNSLOTH_EP_SIZE / UNSLOTH_PARALLEL="ep=N": the 1-D mesh over every rank (made once)."""
    if not _ENV_MESH:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            from .tp_native import _world_mesh
            _world_mesh()  # raises: no process group
        ep = ep_env_size()
        if dist.get_world_size() != ep:
            raise ValueError(
                f"Unsloth: {_EP_ENV}={ep} needs WORLD_SIZE = {ep}, got {dist.get_world_size()}."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ENV_MESH.append(init_device_mesh(device, (ep,), mesh_dim_names = ("ep",))["ep"])
    return _ENV_MESH[0]


def ep_env_load(fn, args, kwargs, four_bit):
    """from_pretrained under UNSLOTH_EP_SIZE: a 4-bit quantize-on-load keeps only this rank's experts."""
    if not four_bit:
        return fn(*args, **kwargs)
    try:
        with expert_parallel_loading(ep_env_mesh()):
            return fn(*args, **kwargs)
    except ValueError as exc:
        if not str(exc).startswith("Unsloth:") or "sharded loading" not in str(exc):
            raise
        import torch.distributed as dist

        if dist.get_rank() == 0:
            print(
                f"{exc} Loading in full on every rank instead; get_peft_model shards the experts."
            )
        return fn(*args, **kwargs)


def ep_env_apply(model):
    """get_peft_model / Trainer.__init__ under UNSLOTH_EP_SIZE."""
    if ep_env_size() > 1 and isinstance(model, nn.Module) and not is_expert_parallel(model):
        apply_expert_parallel(model, ep_env_mesh())
    return model


def _tpn():
    from . import tp_native
    return tp_native


@contextlib.contextmanager
def expert_parallel_loading(ep_mesh = None):
    """A bitsandbytes 4-bit quantize-on-load (transformers 5) keeps only this rank's routed expert stacks, cut as
    transformers quantizes them (the tensor parallel cut); everything else loads whole. Other loads are untouched."""
    tpn = _tpn()
    with tpn._sharded_from_pretrained(
        {
            "mesh": ep_mesh,
            "plan": None,
            "quantize_on_load": tpn._tp_load_quantizing,
            "experts_only": True,
            "label": "expert parallel",
        }
    ) as state:
        yield state


def _a2a(x, send, recv, group):
    import torch.distributed as dist

    out = x.new_empty((sum(recv),) + tuple(x.shape[1:]))
    dist.all_to_all_single(
        out, x.contiguous(), output_split_sizes = recv, input_split_sizes = send, group = group
    )
    return out


class _Exchange(torch.autograd.Function):
    """All-to-all of rows (send[i] rows to rank i); the backward sends the gradients back the same way. Several
    tensors in one node, so every rank runs their backward exchanges in the same order."""

    @staticmethod
    def forward(ctx, send, recv, group, *tensors):
        ctx.split = (send, recv, group)
        return tuple(_a2a(t, send, recv, group) for t in tensors)

    @staticmethod
    def backward(ctx, *grads):
        send, recv, group = ctx.split
        return (None, None, None) + tuple(_a2a(g, recv, send, group) for g in grads)


def _ep_experts(shard, prior, hidden, index, weights):
    import torch.distributed as dist

    tpn = _tpn()
    routing = tpn._moe_checkpoint_routing()
    if routing is not None:
        # Gradient checkpointing replay must select the first pass's (token, slot) pairs.
        index, weights = routing(shard.leaf, index, weights)
    group, ep = shard.group, shard.ep
    shape = hidden.shape
    x = hidden.reshape(-1, shape[-1])
    tokens = x.shape[0]
    index = index.reshape(tokens, -1)
    if (
        weights.dim() == 2
        and weights.shape[-1] == shard.experts
        and index.shape[-1] != shard.experts
    ):
        weights = weights.gather(1, index)  # dense [tokens, E] routing weights
    weights = weights.reshape(tokens, -1)
    # Each token goes once to every rank owning one of its experts.
    owner = torch.div(index, shard.per_rank, rounding_mode = "floor").clamp_(0, ep - 1)
    need = F.one_hot(owner, ep).amax(1).bool()  # [tokens, ep]
    dest, token = need.t().nonzero(as_tuple = True)  # grouped by destination rank
    send = torch.bincount(dest, minlength = ep)
    counts = torch.empty_like(send)
    dist.all_to_all_single(counts, send, group = group)
    send, recv = send.tolist(), counts.tolist()
    (got_index,) = _Exchange.apply(send, recv, group, index.index_select(0, token))
    got_x, got_w = _Exchange.apply(
        send, recv, group, x.index_select(0, token), weights.index_select(0, token)
    )
    lo, hi = shard.lo, shard.hi
    row, slot = ((got_index >= lo) & (got_index < hi)).nonzero(as_tuple = True)
    if row.numel() == 0:
        # Nothing routed here: one zero row through the experts keeps their gradients defined (zeros).
        xc = got_x.new_zeros((1, x.shape[-1]))
        ic = got_index.new_zeros((1, 1))
        wc = got_w.new_zeros((1, 1))
    else:
        xc = got_x.index_select(0, row)
        ic = (got_index[row, slot] - lo).unsqueeze(1)
        wc = got_w[row, slot].unsqueeze(1)
    with tpn._expert_locals(shard):
        yc = prior(xc, ic, wc)
    y_got = yc.new_zeros((got_x.shape[0], yc.shape[-1]))
    if row.numel():
        y_got = y_got.index_add(0, row, yc)
    else:
        y_got = y_got + yc.sum() * 0
    (y_back,) = _Exchange.apply(recv, send, group, y_got)
    y = y_back.new_zeros((tokens, y_back.shape[-1])).index_add(0, token, y_back)
    return y.reshape(*shape[:-1], y.shape[-1])


def _ep_experts_forward(shard, prior):
    tpn = _tpn()

    def forward(*args, **kwargs):
        call = tpn._expert_call_args(args, kwargs)
        if call is None:
            raise RuntimeError(
                f"Unsloth: expert parallel experts of {type(shard.block).__name__} were called with "
                f"{len(args)} positional / {sorted(kwargs)} keyword arguments, not {tpn._EXPERT_CALL}."
            )
        return _ep_experts(shard, prior, *call)

    forward._unsloth_ep_experts = True
    return forward


def _localize(shard, ep):
    """The DTensor shards tp_native made -> plain local parameters (DDP-ignored, any optimizer steps them),
    each tagged with how to gather it back: (dim, groups, per, E, E_pad)."""
    tpn = _tpn()
    for owner, name in shard.swaps:
        dt = owner._parameters[name]
        layout = owner.__dict__.get("_unsloth_tp_layouts", {}).pop(name, None)
        dim, _parts, _tp, (groups, E, E_pad, per) = layout
        local = nn.Parameter(dt.to_local().detach().contiguous(), requires_grad = dt.requires_grad)
        setattr(local, _EP, (dim, groups, per, E, E_pad, ep))
        owner._parameters[name] = local
    shard.swaps = []  # nothing left to swap: the parameters are local
    for name in tpn._EXPERT_STACKS:
        p = shard.leaf._parameters.get(name)
        if tpn._is_params4bit(p):
            p.__dict__.pop("_unsloth_tp_layout", None)
            setattr(p, _EP, ("4bit", shard.experts, ep))


def is_expert_parallel(model):
    return any(m.__dict__.get(_EP) is not None for m in _levels(model))


def _levels(model):
    inner = model.__dict__.get("_modules", {}).get("module")  # a DDP wrapper
    return _tpn()._tp_chain(inner if isinstance(inner, nn.Module) else model)


def _ep_params(model):
    return [(n, p) for n, p in model.named_parameters() if getattr(p, _EP, None) is not None]


def apply_expert_parallel(model, ep_mesh = None):
    """Shard every MoE block's routed experts by index over `ep_mesh` (1-D, every rank: ep is the data
    parallel size) once LoRA is attached. Blocks whose experts cannot be cut exactly stay whole (data parallel)."""
    import torch.distributed as dist

    tpn = _tpn()
    if ep_mesh is None:
        ep_mesh = tpn._world_mesh()
    if is_expert_parallel(model):
        return model
    if ep_mesh.ndim != 1:
        raise ValueError(
            "Unsloth: apply_expert_parallel takes a 1-D mesh over the expert parallel ranks."
        )
    ep, rank = ep_mesh.size(), ep_mesh.get_local_rank()
    if dist.get_world_size() != ep:
        raise NotImplementedError(
            f"Unsloth: expert parallel needs ep = the world size ({dist.get_world_size()}), got ep {ep}; "
            "ep with tensor / pipeline / context parallel or replicated expert groups is not supported."
        )
    if any(
        m.__dict__.get("_unsloth_tp_plan") is not None or m.__dict__.get("_unsloth_fsdp2_native")
        for m in _levels(model)
    ):
        raise NotImplementedError(
            "Unsloth: expert parallel cannot be combined with tensor parallel / FSDP2."
        )
    blocks = tpn.moe_blocks_with_experts(model)
    sharded, kept = [], []
    for block in blocks:
        problem = tpn.expert_shard_problem(block, ep)
        if problem is None and ep > 1:
            sharded.append(block)
        else:
            kept.append((block, problem))
    for block, problem in kept:
        chain = tpn._experts_chain(block)
        if chain and any(
            getattr(chain[-1]._parameters.get(n), tpn._EXPERT_PRELOAD, None) is not None
            for n in tpn._EXPERT_STACKS
        ):
            raise ValueError(
                f"Unsloth: expert_parallel_loading kept only this rank's 4-bit experts of "
                f"{type(block).__name__}, but they cannot be sharded now ({problem}); load in full."
            )
    if not sharded:
        reasons = sorted({str(p) for _, p in kept})[:3]
        raise ValueError(f"Unsloth: expert parallel found no shardable MoE block ({reasons}).")
    for block in sharded:
        shard = tpn._shard_moe_experts(block, ep_mesh, rank, ep, make_forward = _ep_experts_forward)
        shard.ep = ep
        _localize(shard, ep)
    for _, p in _ep_params(model):
        if p.requires_grad:
            # DDP averages the other gradients over ep; an expert's is the sum over every rank's loss.
            p.register_hook(lambda g, ep = ep: g / ep)
    for level in _levels(model):
        level.__dict__[_EP] = ep_mesh
        names = [n for n, _ in _ep_params(level)]
        ignored = list(getattr(level, "_ddp_params_and_buffers_to_ignore", None) or [])
        level._ddp_params_and_buffers_to_ignore = ignored + [
            n for n in names if n not in set(ignored)
        ]
    if rank == 0:
        print(
            f"Unsloth: expert parallel ep={ep}: {len(sharded)} mixture-of-experts blocks keep "
            f"{sharded[0].__dict__['_unsloth_tp_expert_shard'].per_rank} of "
            f"{sharded[0].__dict__['_unsloth_tp_expert_shard'].experts} routed experts per rank"
            + (f", {len(kept)} kept whole" if kept else "")
            + "."
        )
    return model


def _mesh(model):
    return next((m.__dict__[_EP] for m in _levels(model) if m.__dict__.get(_EP) is not None), None)


def _gather_expert(
    p,
    group,
    value = None,
):
    """The 1-GPU tensor of expert parameter `p` (or of `value`, its gradient) gathered over the ep ranks."""
    import torch.distributed as dist

    dim, groups, per, E, E_pad, ep = getattr(p, _EP)
    local = (p if value is None else value).detach().contiguous()
    parts = [torch.empty_like(local) for _ in range(ep)]
    dist.all_gather(parts, local, group = group)
    per_rank = E_pad // ep
    pieces = [t.unflatten(dim, (groups, per_rank, per)) for t in parts]
    full = torch.cat(pieces, dim = dim + 1).flatten(dim, dim + 2)
    return _tpn()._trim_expert_pad(full, dim, groups, E, E_pad, per)


def ep_full_adapter_state_dict(model):
    """Collective: the model's state dict with every expert LoRA factor gathered to its 1-GPU shape."""
    group = _mesh(model).get_group()
    state = dict(model.state_dict())
    for n, p in model.named_parameters():
        layout = getattr(p, _EP, None)
        if (
            layout is not None and layout[0] != "4bit"
        ):  # the frozen 4-bit stacks are not in an adapter
            state[n] = _gather_expert(p, group)
    return state


def ep_clip_grad_norm_(
    parameters,
    max_norm,
    norm_type = 2.0,
    group = None,
):
    """clip_grad_norm_ over the whole model: the replicated gradients counted once, the expert shards summed
    over the expert parallel ranks. Returns the same total norm on every rank."""
    import torch.distributed as dist

    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    norm_type = float(norm_type)
    device = params[0].grad.device
    local = [p.grad for p in params if getattr(p, _EP, None) is not None]
    shared = [p.grad for p in params if getattr(p, _EP, None) is None]

    def norm(gs):
        if not gs:
            return torch.zeros((), device = device, dtype = torch.float32)
        return torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.float(), norm_type).to(device) for g in gs]),
            norm_type,
        )

    a, b = norm(shared), norm(local)
    if norm_type == float("inf"):
        dist.all_reduce(b, op = dist.ReduceOp.MAX, group = group)
        total = torch.maximum(a, b)
    else:
        b = b**norm_type
        dist.all_reduce(b, group = group)
        total = (a**norm_type + b) ** (1.0 / norm_type)
    coef = torch.clamp(max_norm / (total + 1e-6), max = 1.0)
    for p in params:
        p.grad.mul_(coef.to(p.grad.device, p.grad.dtype))
    return total


def _sync_nonfinite(model, group):
    """One rank's non-finite expert gradient poisons every rank's (GradScaler then skips the step everywhere)."""
    import torch.distributed as dist

    grads = [p.grad for _, p in _ep_params(model) if p.grad is not None]
    if not grads:
        return
    bad = torch.stack([~torch.isfinite(g).all() for g in grads]).any().to(torch.int32).reshape(1)
    dist.all_reduce(bad, op = dist.ReduceOp.MAX, group = group)
    if bad.item():
        grads[0].view(-1)[0] = float("inf")


def _refuse(what):
    def refused(*args, **kwargs):
        raise NotImplementedError(f"Unsloth: {what} is not supported under expert parallel.")

    return refused


def ep_trainer_setup(trainer):
    """Trainer.__init__ (after accelerate exists): EP-aware clipping, non-finite sync, 1-GPU adapter saves."""
    model = getattr(trainer, "model", None)
    if not isinstance(model, nn.Module) or not is_expert_parallel(model):
        return False
    mesh = _mesh(model)
    group = mesh.get_group()
    accelerator = trainer.accelerator
    backward = accelerator.backward

    def ep_backward(*args, **kwargs):
        out = backward(*args, **kwargs)
        if accelerator.sync_gradients:
            _sync_nonfinite(model, group)
        return out

    accelerator.backward = ep_backward

    def clip_grad_norm_(
        parameters,
        max_norm,
        norm_type = 2,
    ):
        accelerator.unscale_gradients()
        return ep_clip_grad_norm_(parameters, max_norm, norm_type, group)

    accelerator.clip_grad_norm_ = clip_grad_norm_
    save_model = trainer.save_model

    def ep_save_model(output_dir = None, _internal_call = False):
        # Every rank gathers the expert LoRA (collective), then the Trainer's own save on the main process.
        state = ep_full_adapter_state_dict(model)
        directory = output_dir or trainer.args.output_dir
        if trainer.args.should_save:
            model.save_pretrained(directory, state_dict = state)
            tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
            if tok is not None and hasattr(tok, "save_pretrained"):
                tok.save_pretrained(directory)
            torch.save(trainer.args, os.path.join(directory, "training_args.bin"))
        import torch.distributed as dist

        dist.barrier(group = group)

    trainer.save_model = ep_save_model
    train = trainer.train

    def ep_train(
        *args,
        resume_from_checkpoint = None,
        **kwargs,
    ):
        if resume_from_checkpoint:
            raise NotImplementedError(
                "Unsloth: resume_from_checkpoint is not supported under expert parallel (each rank's expert "
                "optimizer state is its own)."
            )
        return train(*args, **kwargs)

    trainer.train = ep_train
    for level in _levels(model):
        for attr in (
            "save_pretrained_merged",
            "save_pretrained_gguf",
            "push_to_hub_merged",
            "push_to_hub_gguf",
        ):
            if callable(getattr(level, attr, None)):
                setattr(level, attr, _refuse(attr))
    return True
