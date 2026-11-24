# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expert parallel: data parallel ranks that each keep only E / ep of every MoE block's routed experts.

    UNSLOTH_PARALLEL="ep=2" torchrun --nproc-per-node 2 train.py      # ep = world
    UNSLOTH_PARALLEL="ep=4" torchrun --nproc-per-node 8 train.py      # 2 replica groups of 4 expert ranks

    with expert_parallel_loading(ep_mesh):           # optional: 4-bit quantize-on-load keeps this rank's experts
        model, tokenizer = FastModel.from_pretrained(name, load_in_4bit = True)
    model = FastModel.get_peft_model(model, ...)
    apply_expert_parallel(model, mesh)               # 1-D ep mesh, or 2-D (replicas, "ep"), ep innermost

Every rank reads its own batches (DDP). Everything but the routed experts stays whole and data parallel: DDP
averages its gradients. Within an ep group rank r owns the contiguous experts [r E / ep, (r + 1) E / ep) and their
LoRA factors (sliced from the 1-GPU init, so the adapter equals one GPU's). Each experts call sends every token
once to each rank of its group owning one of its top-k experts (all-to-all), that rank runs its experts on the
(token, slot) pairs it got and the weighted outputs come back the same way, summed per token; the backward is the
same exchange reversed. The expert parameters are kept out of DDP: their gradients are divided by the world size
(DDP's average) and, with replica groups, summed over the groups holding the same experts. Clipping counts each
expert once, and a non-finite expert gradient on one rank makes every rank skip the fp16 step.

Any block the tensor parallel expert shard accepts (fused stacks with the expert on dim 0, biases included; the
experts call (hidden_states, top_k_index, top_k_weights) or gpt-oss' (router_indices, routing_weights)). A block
forward that computes its experts without that call (a fused / decode-only path) raises instead of silently using
E / ep experts. The 4-bit expert stacks reuse the tensor parallel expert-index slicing: bitwise the same bytes.
Collective: every rank must run the same number of MoE forwards; generate defaults to synced_gpus = True, so ranks
that finish first keep stepping. save_model writes the 1-GPU adapter (expert LoRA gathered). Trainer checkpoints
add each rank's trainable tensors and optimizer state (unsloth_ep/rankNNNNN.pt): resume_from_checkpoint restores
them on the same layout. Merged / GGUF exports are refused under expert parallel."""

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
    "ep_sync_expert_grads",
]

_EP = "_unsloth_ep"  # on each expert parameter: its gather layout; on the model: the mesh
_EP_ENV = "UNSLOTH_EP_SIZE"
_EP_REP = "_unsloth_ep_replicas"  # on the model: the replica mesh (ep < world), or None


def ep_env_size():
    try:
        return int(os.environ.get(_EP_ENV, "1") or 1)
    except ValueError:
        raise ValueError(f"Unsloth: {_EP_ENV}={os.environ.get(_EP_ENV)!r} is not an integer.")


_ENV_MESH = []


def ep_env_mesh():
    """UNSLOTH_EP_SIZE / UNSLOTH_PARALLEL="ep=N": the 1-D ep mesh over every rank, or the 2-D (ep_rep, ep) mesh
    when ep divides the world (ep innermost: adjacent ranks share an expert group). Made once."""
    if not _ENV_MESH:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not dist.is_initialized():
            # As parallel_env.ParallelLayout.mesh (torchrun's environment); the Trainer's accelerate reuses it.
            dist.init_process_group("nccl" if cuda else "gloo")
        ep, world = ep_env_size(), dist.get_world_size()
        if world % ep:
            raise ValueError(f"Unsloth: {_EP_ENV}={ep} must divide WORLD_SIZE = {world}.")
        device = "cuda" if cuda else "cpu"
        if world == ep:
            mesh = init_device_mesh(device, (ep,), mesh_dim_names = ("ep",))["ep"]
        else:
            mesh = init_device_mesh(device, (world // ep, ep), mesh_dim_names = ("ep_rep", "ep"))
        _ENV_MESH.append(mesh)
    return _ENV_MESH[0]


def ep_env_load(fn, args, kwargs, four_bit):
    """from_pretrained under UNSLOTH_EP_SIZE: a 4-bit quantize-on-load keeps only this rank's experts."""
    if not four_bit:
        return fn(*args, **kwargs)
    try:
        with expert_parallel_loading(_split_mesh(ep_env_mesh())[0]):
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
    shard.calls += 1
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
        # Nothing routed here: one zero row through the experts keeps their gradients defined (zeros). It is tied
        # to the received tensors so this rank still runs the exchange's backward that every other rank runs.
        xc = got_x.new_zeros((1, x.shape[-1])) + got_x[:0].sum()
        ic = got_index.new_zeros((1, 1))
        wc = got_w.new_zeros((1, 1)) + got_w[:0].sum()
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
        y_got = y_got + yc[:0].sum()
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


def _split_mesh(mesh):
    """(ep mesh, replica mesh or None) of a 1-D ep mesh or a 2-D (ep_rep, ep) mesh."""
    if mesh.ndim == 1:
        return mesh, None
    names = mesh.mesh_dim_names
    if mesh.ndim != 2 or names is None or names[1] != "ep":
        raise ValueError(
            "Unsloth: apply_expert_parallel takes a 1-D ep mesh or a 2-D (replicas, 'ep') mesh, ep innermost."
        )
    return mesh["ep"], mesh[names[0]]


def apply_expert_parallel(model, mesh = None):
    """Shard every MoE block's routed experts by index over the ep ranks once LoRA is attached. `mesh`: a 1-D ep
    mesh over every rank, or a 2-D (replicas, "ep") mesh: each group of ep ranks holds all experts once and the
    groups are replicas (experts' gradients summed over them). Every rank is data parallel. Blocks whose experts
    cannot be cut exactly stay whole."""
    import torch.distributed as dist

    tpn = _tpn()
    if mesh is None:
        mesh = tpn._world_mesh()
    if is_expert_parallel(model):
        return model
    ep_mesh, rep_mesh = _split_mesh(mesh)
    ep, rank = ep_mesh.size(), ep_mesh.get_local_rank()
    world = dist.get_world_size()
    if ep * (rep_mesh.size() if rep_mesh is not None else 1) != world:
        raise NotImplementedError(
            f"Unsloth: expert parallel needs ep x replicas = the world size ({world}), got ep {ep}; "
            "ep with tensor / pipeline / context parallel is not supported."
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
        shard.ep, shard.calls = ep, 0
        _localize(shard, ep)
        plain = _BLOCK_FORWARDS.get(type(block).__name__)
        if plain is not None:
            block.forward = plain(block)
        _guard_bypass(block, shard)
    for _, p in _ep_params(model):
        if p.requires_grad:
            # DDP averages the other gradients over the world; an expert's is the sum over its ep group's losses
            # (then over the replicas, ep_sync_expert_grads): / world is DDP's average.
            p.register_hook(lambda g, world = world: g / world)
    for level in _levels(model):
        level.__dict__[_EP] = ep_mesh
        level.__dict__[_EP_REP] = rep_mesh
        if callable(level.__dict__.get("generate")) or hasattr(type(level), "generate"):
            _sync_generate(level)
        _ep_save_entrypoints(level)
        names = [n for n, _ in _ep_params(level)]
        ignored = list(getattr(level, "_ddp_params_and_buffers_to_ignore", None) or [])
        level._ddp_params_and_buffers_to_ignore = ignored + [
            n for n in names if n not in set(ignored)
        ]
    if rank == 0:
        print(
            f"Unsloth: expert parallel ep={ep}"
            + (f" x {rep_mesh.size()} replicas" if rep_mesh is not None else "")
            + f": {len(sharded)} mixture-of-experts blocks keep "
            f"{sharded[0].__dict__['_unsloth_tp_expert_shard'].per_rank} of "
            f"{sharded[0].__dict__['_unsloth_tp_expert_shard'].experts} routed experts per rank"
            + (f", {len(kept)} kept whole" if kept else "")
            + "."
        )
    return model


def _mesh(model):
    return next((m.__dict__[_EP] for m in _levels(model) if m.__dict__.get(_EP) is not None), None)


def _replicas(model):
    return next(
        (m.__dict__[_EP_REP] for m in _levels(model) if m.__dict__.get(_EP_REP) is not None), None
    )


def _gpt_oss_block_forward(block):
    """GptOssMLP under expert parallel: always router -> experts call. unsloth_zoo's GptOssMLP takes
    moe_forward_inference for a 1-token decode in eval, which reads the stacks directly (all E experts, so this
    rank's E / ep would be wrong); transformers' own forward is the same as this."""

    def forward(hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_out = block.router(flat)
        if isinstance(router_out, tuple) and len(router_out) == 3:
            _, router_scores, router_indices = router_out
        else:
            router_scores, router_indices = router_out
        out = block.experts(flat, router_indices = router_indices, routing_weights = router_scores)
        return out.reshape(batch_size, sequence_length, hidden_dim), router_scores

    return forward


# Blocks whose own forward has a path around the experts call: replaced by the plain router -> experts forward.
_BLOCK_FORWARDS = {"GptOssMLP": _gpt_oss_block_forward}


def _bypass_message(block, shard):
    return (
        f"Unsloth: {type(block).__name__}.forward computed its experts without calling "
        f"{type(shard.leaf).__name__} (a fused or decode-only path), which under expert parallel holds only "
        f"{shard.per_rank} of {shard.experts} experts. Disable that path (for example generate in "
        "training mode or the zoo inference MoE path) or train without expert parallel."
    )


def _guard_bypass(block, shard):
    """A block forward that never reached its experts call (a fused / decode path computing all E experts from
    this rank's E / ep) would silently be wrong under expert parallel: refuse it. Such a path reading the local
    stacks with a global expert id >= E / ep raises IndexError before the post hook can run: that is refused the
    same way (the IndexError chained), so the outcome does not depend on which experts the router picked."""

    def pre(module, args):
        module.__dict__["_unsloth_ep_calls"] = shard.calls

    def post(module, args, output):
        if shard.calls == module.__dict__.get("_unsloth_ep_calls"):
            raise RuntimeError(_bypass_message(module, shard))

    inner = block.forward

    def forward(*args, **kwargs):
        before = shard.calls
        try:
            return inner(*args, **kwargs)
        except IndexError as exc:
            if shard.calls == before:
                raise RuntimeError(_bypass_message(block, shard)) from exc
            raise

    block.forward = forward
    block.register_forward_pre_hook(pre)
    block.register_forward_hook(post)


def _sync_generate(level):
    """generate under expert parallel: every experts call is collective, so a rank that finishes first keeps
    running forwards until every rank is done (transformers' synced_gpus)."""
    prior = level.generate

    def generate(*args, **kwargs):
        kwargs.setdefault("synced_gpus", True)
        return prior(*args, **kwargs)

    generate._unsloth_ep_generate = True
    level.generate = generate


_SAVING = []  # inside an expert parallel save_pretrained: inner saves write what they are given


def _ep_save_entrypoints(level):
    """save_pretrained / push_to_hub under expert parallel: each rank holds only its experts' tensors. Every
    rank calls save_pretrained: the expert tensors are gathered (collective) and global rank 0 writes the 1-GPU
    files. push_to_hub is refused on every rank (save, then upload that directory from one process)."""
    import torch.distributed as dist

    prior_save = level.__dict__.get("save_pretrained") or getattr(level, "save_pretrained", None)
    if callable(prior_save) and not getattr(prior_save, "_unsloth_ep_save", False):

        def save_pretrained(*args, **kwargs):
            if _SAVING or kwargs.get("state_dict") is not None:
                return prior_save(*args, **kwargs)
            if kwargs.get("push_to_hub"):
                raise NotImplementedError(_EP_PUSH_REFUSAL)
            from peft import PeftModel

            if not isinstance(level, PeftModel) and any(
                getattr(p, _EP)[0] == "4bit" for _, p in _ep_params(level)
            ):
                raise NotImplementedError(
                    "Unsloth: save_pretrained of the whole model under expert parallel would write only this "
                    "rank's 4-bit expert stacks. Save the adapter (the PEFT model's save_pretrained or "
                    "trainer.save_model), or save from one process."
                )
            state = ep_full_adapter_state_dict(level)
            _SAVING.append(True)
            try:
                out = (
                    prior_save(*args, state_dict = state, **kwargs) if dist.get_rank() == 0 else None
                )
            finally:
                _SAVING.pop()
            dist.barrier()
            return out

        save_pretrained._unsloth_ep_save = True
        level.save_pretrained = save_pretrained
    if callable(getattr(level, "push_to_hub", None)):

        def push_to_hub(*args, **kwargs):
            raise NotImplementedError(_EP_PUSH_REFUSAL)

        level.push_to_hub = push_to_hub


_EP_PUSH_REFUSAL = (
    "Unsloth: push_to_hub is not supported under expert parallel (each rank holds only its experts). Call "
    "model.save_pretrained(directory) or trainer.save_model(directory) on every rank, which writes the 1-GPU "
    "adapter, then upload that directory from one process (huggingface_hub upload_folder)."
)


def ep_refuse_prebuilt_optimizer(
    model,
    optimizer,
    kwargs_params = False,
):
    """Trainer.__init__ before UNSLOTH_EP_SIZE shards a model handed in unsharded: sharding replaces the expert
    parameters, so an optimizer built before it would step the old ones (the experts never train)."""
    if ep_env_size() <= 1 or not isinstance(model, nn.Module) or is_expert_parallel(model):
        return
    tpn = _tpn()
    replaced = {
        id(p)
        for block in tpn.moe_blocks_with_experts(model)
        for chain in (tpn._experts_chain(block),)
        if chain
        for p in chain[0].parameters()
    }
    held = set()
    for group in getattr(optimizer, "param_groups", None) or []:
        held.update(id(p) for p in group.get("params", []))
    if kwargs_params or held & replaced:
        raise NotImplementedError(
            f"Unsloth: {_EP_ENV} shards this model's experts in Trainer.__init__, which replaces the expert "
            "parameters, but Trainer(optimizers = (optimizer, ...)) or optimizer_cls_and_kwargs = (cls, "
            "{'params': ...}) already holds them, so the experts would never update. Let the Trainer build the "
            "optimizer (TrainingArguments optim / learning_rate), override create_optimizer, or build the "
            "optimizer after get_peft_model (which shards first)."
        )


def ep_sync_expert_grads(model):
    """ep < world: sum the expert gradients over the replica groups (each holds the same experts). A no-op for
    ep = world. Call once per optimizer step, after the last backward (ep_trainer_setup does)."""
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    rep = _replicas(model)
    if rep is None:
        return
    grads = [p.grad for _, p in _ep_params(model) if p.requires_grad]
    if any(g is None for g in grads):
        grads = []
        for _, p in _ep_params(model):
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                grads.append(p.grad)
    buckets = {}
    for g in grads:
        buckets.setdefault((g.device, g.dtype), []).append(g)
    with torch.no_grad():
        for tensors in buckets.values():
            flat = _flatten_dense_tensors(tensors)
            dist.all_reduce(flat, group = rep.get_group())
            for t, v in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
                t.copy_(v)


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


_CKPT_DIR = "unsloth_ep"


def _layout(model):
    import torch.distributed as dist
    rep = _replicas(model)
    return {
        "world": dist.get_world_size(),
        "ep": _mesh(model).size(),
        "replicas": rep.size() if rep is not None else 1,
    }


def _rank_path(directory):
    import torch.distributed as dist
    return os.path.join(directory, _CKPT_DIR, f"rank{dist.get_rank():05d}.pt")


def _save_rank_state(model, optimizer, directory):
    """Every rank: its trainable tensors as held (expert LoRA shards local) and its optimizer state."""
    import torch.distributed as dist

    path = _rank_path(directory)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    optim = optimizer.state_dict() if optimizer is not None else None
    torch.save({"layout": _layout(model), "weights": weights, "optimizer": optim}, path)
    dist.barrier()


def _read_rank_state(model, directory):
    path = _rank_path(directory)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Unsloth: {directory} has no expert parallel rank state ({path}); it was not written under expert "
            "parallel, so its expert optimizer state cannot be restored."
        )
    payload = torch.load(path, map_location = "cpu", weights_only = False)
    if payload["layout"] != _layout(model):
        raise ValueError(
            f"Unsloth: the checkpoint {directory} was written with expert parallel layout {payload['layout']}, "
            f"not this run's {_layout(model)}; resume with the same world size and ep."
        )
    return payload


def _load_rank_weights(model, directory):
    payload = _read_rank_state(model, directory)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["weights"].items():
            if name not in params or tuple(params[name].shape) != tuple(value.shape):
                raise ValueError(
                    f"Unsloth: expert parallel checkpoint tensor {name} does not match this model."
                )
            params[name].copy_(value.to(params[name].device))


def _load_rank_optimizer(model, optimizer, directory):
    payload = _read_rank_state(model, directory)
    if payload["optimizer"] is not None and optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])


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
            ep_sync_expert_grads(model)
            _sync_nonfinite(model, None)  # every rank (replicas included) must skip the same steps
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
    # Checkpoints: the adapter is the 1-GPU one (save_model above); each rank also writes its own trainable
    # tensors (local expert shards) and its optimizer state, and a resume loads them back on the same layout.
    save_optim = trainer._save_optimizer_and_scheduler
    load_optim = trainer._load_optimizer_and_scheduler
    load_weights = trainer._load_from_checkpoint

    def ep_save_optimizer_and_scheduler(output_dir):
        out = save_optim(output_dir)
        _save_rank_state(model, trainer.optimizer, output_dir)
        return out

    def ep_load_from_checkpoint(resume_from_checkpoint, model_ = None):
        _load_rank_weights(model, resume_from_checkpoint)

    def ep_load_optimizer_and_scheduler(checkpoint):
        out = load_optim(checkpoint)
        if checkpoint is not None:
            _load_rank_optimizer(model, trainer.optimizer, checkpoint)
        return out

    if getattr(trainer.args, "load_best_model_at_end", False) and getattr(
        trainer.args, "save_only_model", False
    ):
        # The best checkpoint's rank-local tensors are written beside the optimizer state, which
        # save_only_model skips (transformers refuses the same pair under DeepSpeed / FSDP).
        raise NotImplementedError(
            "Unsloth: load_best_model_at_end with save_only_model = True is not supported under expert "
            "parallel; unset save_only_model."
        )

    def ep_load_best_model():
        # The checkpoint's adapter is the gathered 1-GPU one; this rank restores its own (rank-local) tensors.
        _load_rank_weights(model, trainer.state.best_model_checkpoint)

    trainer._save_optimizer_and_scheduler = ep_save_optimizer_and_scheduler
    trainer._load_from_checkpoint = ep_load_from_checkpoint
    trainer._load_best_model = ep_load_best_model
    trainer._load_optimizer_and_scheduler = ep_load_optimizer_and_scheduler
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
