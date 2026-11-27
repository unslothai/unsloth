# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expert parallel: data parallel ranks that each keep only E / ep of every MoE block's routed experts.

    UNSLOTH_PARALLEL="ep=2" torchrun --nproc-per-node 2 train.py      # ep = world
    UNSLOTH_PARALLEL="ep=4" torchrun --nproc-per-node 8 train.py      # 2 replica groups of 4 expert ranks

    with expert_parallel_loading(ep_mesh):           # optional: 4-bit quantize-on-load keeps this rank's experts
        model, tokenizer = FastModel.from_pretrained(name, load_in_4bit = True)
    model = FastModel.get_peft_model(model, ...)
    apply_expert_parallel(model, mesh)               # 1-D ep mesh, or 2-D (replicas, "ep"), ep innermost

Every rank reads its own batches (DDP), or with Ulysses context parallel (UNSLOTH_PARALLEL="ep=2,cp=2") its shard
of each row's sequence: the experts are token-local, so a cp shard's tokens dispatch like a data parallel row's.
Everything but the routed experts stays whole and data parallel: DDP averages its gradients. Within an ep group rank r owns the contiguous experts [r E / ep, (r + 1) E / ep) and their
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
add each rank's trainable tensors and optimizer state (unsloth_ep/rankNNNNN.pt), which resume_from_checkpoint restores
on the same layout, and the 1-GPU optimizer.pt, which resumes on another layout or one GPU (portable_resume.py).
Merged / GGUF exports are refused under expert parallel."""

from __future__ import annotations

import contextlib
import functools
import os

import torch
import torch.nn as nn

__all__ = [
    "apply_expert_parallel",
    "expert_parallel_loading",
    "is_expert_parallel",
    "ep_full_adapter_state_dict",
    "ep_clip_grad_norm_",
    "ep_env_size",
    "ep_env_mesh",
    "ep_env_load",
    "ep_refuse_fast_inference",
    "ep_refuse_composition",
    "ep_refuse_per_expert_lists",
    "ep_refuse_trainer_fsdp",
    "ep_env_apply",
    "ep_trainer_setup",
    "ep_sync_expert_grads",
]

_EP = "_unsloth_ep"  # on each expert parameter: its gather layout; on the model: the mesh
_EP_ENV = "UNSLOTH_EP_SIZE"
_EP_REP = "_unsloth_ep_replicas"  # on the model: the replica mesh (ep < world), or None
_EP_LORA_B = "_unsloth_ep_lora_b_layout"  # on each sharded expert lora_B: the UNSLOTH_MOE_LORA_B_LAYOUT it was cut by


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
        if cuda and not dist.is_initialized() and not dist.is_nccl_available():
            raise RuntimeError(
                f"Unsloth: {_EP_ENV}={ep_env_size()} (expert parallel) runs its collectives over NCCL on CUDA GPUs, "
                "but this PyTorch build has no NCCL backend (torch.distributed.is_nccl_available() is False; "
                "native Windows builds ship without it). Use a PyTorch build with NCCL (Linux or WSL), or "
                "train on one GPU."
            )
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


def ep_env_load(fn, args, kwargs, sharded):
    """from_pretrained under UNSLOTH_EP_SIZE; `sharded` False (adapter or QAT load) loads in full."""
    _lists().keep_zoo_expert_loops()
    if not sharded:
        return fn(*args, **kwargs)
    try:
        with expert_parallel_loading(_split_mesh(ep_env_mesh())[0]):
            return fn(*args, **kwargs)
    except _tpn().HFDistributedKwargsError:
        raise  # a whole load would forward the same kwarg to transformers
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


def ep_refuse_composition(model):
    """Before apply_expert_parallel changes anything: a pipeline stage, a tensor parallel model or a live vLLM
    engine (its LoRA sync would send one rank's expert slice)."""
    levels = _levels(model)
    if any(
        m.__dict__.get("_unsloth_pp") is not None or m.__dict__.get("_unsloth_pp_stage")
        for m in levels
    ):
        raise NotImplementedError(
            "Unsloth: expert parallel cannot be applied to a pipeline parallel stage (apply_pipeline_parallel): "
            "ep with pipeline parallel is not supported."
        )
    if _tpn().is_tensor_parallel(model):
        raise NotImplementedError(
            "Unsloth: expert parallel cannot be applied to a tensor parallel model (apply_tensor_parallel): "
            "ep with tensor parallel is not supported."
        )
    if any(getattr(m, "vllm_engine", None) is not None for m in levels):
        raise NotImplementedError(
            "Unsloth: expert parallel cannot be applied to a model with a live vLLM engine (fast_inference = "
            "True): each rank would hold only its own experts' LoRA, and the vLLM LoRA sync would send one "
            "rank's slice. Load with fast_inference = False."
        )


_PER_EXPERT_LISTS = "per-expert Linear lists"


def _per_expert_lists_refusal(problems):
    lists = sorted({p for p in problems if isinstance(p, str) and p.startswith(_PER_EXPERT_LISTS)})
    if not lists or any(p is None for p in problems):
        return None
    return (
        f"Unsloth: expert parallel cannot shard this model's experts yet: {lists[0]}. Load with "
        f"load_in_4bit = False, or unset {_EP_ENV} (or ep in UNSLOTH_PARALLEL) and run DDP or UNSLOTH_FSDP2=1."
    )


def ep_refuse_per_expert_lists(model):
    """Right after from_pretrained under UNSLOTH_EP_SIZE, before get_peft_model: no MoE block shards because its
    experts are per-expert Linear lists (zoo's GptOssExpertsBnb4bit for a load_in_4bit gpt-oss)."""
    if not isinstance(model, nn.Module) or is_expert_parallel(model):
        return
    tpn, lists = _tpn(), _lists()
    problems = []
    for block in tpn.moe_blocks_with_experts(model):
        why = lists.fused_list_problem(block, ep_env_size())
        problem = None if why is None else tpn.expert_shard_problem(block, ep_env_size())
        if isinstance(problem, str) and problem.startswith(_PER_EXPERT_LISTS) and why != problem:
            problem = f"{problem}; not shardable as per-expert lists either: {why}"
        problems.append(problem)
    message = _per_expert_lists_refusal(problems)
    if message is not None:
        raise NotImplementedError(message)


def ep_refuse_trainer_fsdp(trainer):
    """Trainer.__init__: accelerate's FSDP (an `accelerate launch` FSDP config / fsdp args) would wrap and shard
    the per-rank expert slices at train() as if they were one tensor; DeepSpeed ZeRO would partition them and
    average their gradients over every rank the same way."""
    model = getattr(trainer, "model", None)
    if not isinstance(model, nn.Module) or not is_expert_parallel(model):
        return
    if getattr(trainer, "is_fsdp_enabled", False):
        raise NotImplementedError(
            "Unsloth: expert parallel cannot be combined with accelerate's FSDP (an accelerate FSDP config or "
            "TrainingArguments fsdp): every rank holds different experts. Launch with torchrun and DDP, or drop ep."
        )
    if getattr(trainer, "is_deepspeed_enabled", False):
        raise NotImplementedError(
            "Unsloth: expert parallel cannot be combined with DeepSpeed (an accelerate DeepSpeed config or "
            "TrainingArguments deepspeed): every rank holds different experts, and DeepSpeed would partition "
            "and average them as one tensor. Launch with torchrun and DDP, or drop ep."
        )


def ep_refuse_fast_inference(values):
    """vLLM's LoRA sync gathers only DTensor adapters; ep keeps expert LoRA as local parameters."""
    if values.get("fast_inference"):
        raise NotImplementedError(
            f"Unsloth: {_EP_ENV}={ep_env_size()} (expert parallel) does not support vLLM (fast_inference = True): "
            "each rank holds only its own experts' LoRA, and the vLLM LoRA sync would send one rank's slice. "
            f"Unset {_EP_ENV} (or ep in UNSLOTH_PARALLEL) to run this script with DDP or UNSLOTH_FSDP2=1, "
            "or load with fast_inference = False."
        )


def _tpn():
    from . import tp_native
    return tp_native


@contextlib.contextmanager
def expert_parallel_loading(ep_mesh = None):
    """16-bit, bitsandbytes 4-bit or block-FP8 loads (transformers 5) keep only this rank's routed expert stacks, and
    pre-quantized packed NVFP4 / MXFP8 per-expert checkpoints read only this rank's experts; others load whole."""
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


def _compiler_disable(fn):
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:  # torch < 2.1
        import torch._dynamo as dynamo
        disable = dynamo.disable
    return disable(fn)


def _route_width(route, tokens):
    """reshape(tokens, -1) cannot infer the width of an empty tensor."""
    if tokens:
        return -1
    return route.shape[-1] if route.dim() > 1 else 1


def _destinations(owner, ep):
    return owner.new_zeros((owner.shape[0], ep), dtype = torch.bool).scatter_(1, owner, True)


_EP_ROWS_ENV = "UNSLOTH_EP_EXPERT_ROWS"


def _ep_expert_rows():
    """N rows per checkpointed experts call on this rank (no per-chunk collective); unset or 0 = one call."""
    value = os.environ.get(_EP_ROWS_ENV, "").strip()
    if not value:
        return None
    try:
        rows = int(value)
    except ValueError:
        rows = -1
    if rows < 0:
        raise ValueError(f"Unsloth: {_EP_ROWS_ENV}={value!r} must be a row count >= 0 (0 = off).")
    return rows or None


def _ep_expert_chunk(shard, prior, dense, got_x, row, ic, wc):
    xc = got_x.index_select(0, row)
    if dense:
        # 4.x gpt-oss experts take E from routing_weights' width: rebuild it dense over the local experts.
        wc = wc.new_zeros((wc.shape[0], shard.per_rank)).scatter(1, ic, wc)
    with _tpn()._expert_locals(shard):
        yc = prior(xc, ic, wc)
    return yc.reshape(-1, yc.shape[-1])


def _ep_experts_chunked(shard, prior, dense, got_x, row, ic, wc, rows):
    # torch's own: smart GC's replacement would offload each chunk and drop closed-over expert LoRA grads.
    checkpoint = _tpn()._torch_checkpoint()

    y_got = None
    for start in range(0, row.numel(), rows):
        part = slice(start, start + rows)
        args = (shard, prior, dense, got_x, row[part], ic[part], wc[part])
        if torch.is_grad_enabled():
            yc = checkpoint(_ep_expert_chunk, *args, use_reentrant = False)
        else:
            yc = _ep_expert_chunk(*args)
        if y_got is None:
            y_got = yc.new_zeros((got_x.shape[0], yc.shape[-1]))
        y_got.index_add_(0, row[part], yc)
    shard.chunked_calls = getattr(shard, "chunked_calls", 0) + 1
    return y_got


# Data-dependent splits and all_to_all: kept out of Dynamo graphs.
@_compiler_disable
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
    index = index.reshape(tokens, _route_width(index, tokens))
    dense = (
        weights.dim() == 2
        and weights.shape[-1] == shard.experts
        and index.shape[-1] != shard.experts
    )
    if dense:
        weights = weights.gather(1, index)
    weights = weights.reshape(tokens, _route_width(weights, tokens))
    # Each token goes once to every rank owning one of its experts.
    owner = torch.div(index, shard.per_rank, rounding_mode = "floor").clamp_(0, ep - 1)
    need = _destinations(owner, ep)
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
    chunk = _ep_expert_rows()
    if chunk is not None and row.numel() > chunk:
        y_got = _ep_experts_chunked(
            shard,
            prior,
            dense,
            got_x,
            row,
            (got_index[row, slot] - lo).unsqueeze(1),
            got_w[row, slot].unsqueeze(1),
            chunk,
        )
        (y_back,) = _Exchange.apply(recv, send, group, y_got)
        y = y_back.new_zeros((tokens, y_back.shape[-1])).index_add(0, token, y_back)
        return y.reshape(*shape[:-1], y.shape[-1])
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
    if dense:
        # 4.x gpt-oss experts take E from routing_weights' width: rebuild it dense over the local experts.
        wc = wc.new_zeros((wc.shape[0], shard.per_rank)).scatter(1, ic, wc)
    with tpn._expert_locals(shard):
        yc = prior(xc, ic, wc)
    yc = yc.reshape(-1, yc.shape[-1])  # transformers 4.x gpt-oss returns [rows, 1, hidden]
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
    for p in shard.leaf._parameters.values():
        # Frozen quantized stacks (bitsandbytes 4-bit, zoo's packed NVFP4 / MXFP8 / MXFP4): already this rank's
        # experts, never in an adapter, never gathered.
        if tpn._is_params4bit(p) or tpn._is_packed_expert(p):
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
    ep_refuse_composition(model)
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
    _lists().check_remote_layout(model, ep, rank, ep_mesh.mesh.tolist())
    blocks = tpn.moe_blocks_with_experts(model)
    sharded, kept, listed, fused_lists = [], [], [], []
    for block in blocks:
        problem = tpn.expert_shard_problem(block, ep)
        if problem is None and ep > 1:
            sharded.append(block)
        elif ep > 1 and _lists().list_shard_problem(block, ep) is None:
            # transformers 4.x per-expert ModuleList: the block loops its experts itself
            listed.append(block)
        elif ep > 1 and _lists().fused_list_problem(block, ep) is None:
            # one experts call over per-expert Linear lists (zoo's load_in_4bit gpt-oss)
            fused_lists.append(block)
        else:
            why = _lists().fused_list_problem(block, ep) if ep > 1 else None
            if (
                isinstance(problem, str)
                and problem.startswith(_PER_EXPERT_LISTS)
                and why != problem
            ):
                problem = f"{problem}; not shardable as per-expert lists either: {why}"
            kept.append((block, problem))
    for block, problem in kept:
        chain = tpn._experts_chain(block)
        if (
            chain
            and any(
                getattr(p, tpn._EXPERT_PRELOAD, None) is not None
                for p in chain[-1]._parameters.values()
            )
        ) or any(_lists().is_remote(p) for p in block.parameters()):
            raise ValueError(
                f"Unsloth: expert_parallel_loading kept only this rank's experts of "
                f"{type(block).__name__}, but they cannot be sharded now ({problem}); load in full."
            )
    if not sharded and not listed and not fused_lists:
        message = _per_expert_lists_refusal([p for _, p in kept]) if ep > 1 else None
        if message is not None:
            raise NotImplementedError(message)
        reasons = sorted({str(p) for _, p in kept})[:3]
        raise ValueError(f"Unsloth: expert parallel found no shardable MoE block ({reasons}).")
    for block in sharded:
        for wrapper in tpn._experts_chain(block)[
            :-1
        ]:  # the lora_B column slicing the shard is about to use
            for adapter in getattr(wrapper, "lora_A", {}):
                wrapper.lora_B[adapter].__dict__[_EP_LORA_B] = tpn._lora_b_layout(wrapper, adapter)
        shard = tpn._shard_moe_experts(block, ep_mesh, rank, ep, make_forward = _ep_experts_forward)
        shard.ep, shard.calls = ep, 0
        _localize(shard, ep)
        plain = _BLOCK_FORWARDS.get(type(block).__name__)
        if plain is not None:
            block.forward = plain(block)
        _guard_bypass(block, shard)
    for block in listed:
        _lists().shard_expert_list(block, ep_mesh, rank, ep)
    for block in fused_lists:
        shard = _lists().shard_fused_lists(block, ep_mesh, rank, ep)
        plain = _BLOCK_FORWARDS.get(type(block).__name__)
        if plain is not None:
            block.forward = plain(block)
        _guard_bypass(block, shard)
    for _, p in _ep_params(model):
        if p.requires_grad:
            # DDP averages the other gradients over the world; an expert's is the sum over its ep group's losses
            # (then over the replicas, ep_sync_expert_grads): / world is DDP's average.
            p.register_hook(lambda g, world = world: g / world)
    tpn._drop_remote_device_map(model, ep, rank)
    for level in _levels(model):
        level.__dict__[_EP] = ep_mesh
        level.__dict__[_EP_REP] = rep_mesh
        if callable(level.__dict__.get("generate")) or hasattr(type(level), "generate"):
            _sync_generate(level)
        _ep_save_entrypoints(level)
        _ep_refuse_exports(level)  # a custom training loop never reaches ep_trainer_setup
        names = [n for n, _ in _ep_params(level)]
        ignored = list(getattr(level, "_ddp_params_and_buffers_to_ignore", None) or [])
        level._ddp_params_and_buffers_to_ignore = ignored + [
            n for n in names if n not in set(ignored)
        ]
    if rank == 0:
        first = (
            sharded[0].__dict__["_unsloth_tp_expert_shard"]
            if sharded
            else (listed or fused_lists)[0].__dict__[_lists()._SHARD]
        )
        print(
            f"Unsloth: expert parallel ep={ep}"
            + (f" x {rep_mesh.size()} replicas" if rep_mesh is not None else "")
            + f": {len(sharded) + len(listed) + len(fused_lists)} mixture-of-experts blocks keep "
            f"{first.per_rank} of {first.experts} routed experts per rank"
            + (
                f" ({len(listed) + len(fused_lists)} as per-expert modules)"
                if listed or fused_lists
                else ""
            )
            + (f", {len(kept)} kept whole" if kept else "")
            + "."
        )
        hint = _list_exchange_hint(model, len(listed), ep)
        if hint is not None:
            print(hint)
    return model


# The whole model's parameters within this share of one GPU: data parallel holds it per GPU with room for training.
_DDP_FIT_FRACTION = 0.6


def _device_total_bytes():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory


def _list_exchange_hint(model, listed, ep):
    """The rank-0 note when transformers 4.x per-expert ModuleList blocks were sharded and the whole model would fit
    one GPU: their exchange runs one small all-to-all and matmul per expert from the host (OLMoE on 2x T4: 3.4x
    slower per step than data parallel), so data parallel or transformers 5's fused experts is faster."""
    if not listed:
        return None
    device = _device_total_bytes()
    if not device:
        return None
    whole = 0
    for _, p in model.named_parameters():
        nbytes = p.numel() * p.element_size()
        whole += (
            nbytes * ep if getattr(p, _EP, None) is not None else nbytes
        )  # each rank holds 1 / ep of them
    if whole > _DDP_FIT_FRACTION * device:
        return None
    return (
        f"Unsloth: {listed} mixture-of-experts blocks are transformers 4.x per-expert modules, whose expert "
        "parallel exchange is bound by the host (one small exchange and matmul per expert). The whole model "
        f"({whole / 2**30:.1f} GiB of parameters) fits one GPU ({device / 2**30:.0f} GiB): data parallel (unset "
        "UNSLOTH_EP_SIZE) is usually faster, as is transformers 5, whose fused experts shard without it."
    )


def _lists():
    from . import ep_expert_lists
    return ep_expert_lists


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

    @functools.wraps(inner)
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

    @functools.wraps(prior)
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

        @functools.wraps(prior_save)
        def save_pretrained(*args, **kwargs):
            if _SAVING or kwargs.get("state_dict") is not None:
                # a gathered state dict keeps every expert
                with _lists().remote_experts_for_save(level), _whole_expert_counts(level):
                    return prior_save(*args, **kwargs)
            if kwargs.get("push_to_hub"):
                raise NotImplementedError(_EP_PUSH_REFUSAL)
            from peft import PeftModel

            if not isinstance(level, PeftModel) and any(
                getattr(p, _EP)[0] == "4bit" for _, p in _ep_params(level)
            ):
                raise NotImplementedError(
                    "Unsloth: save_pretrained of the whole model under expert parallel would write only this "
                    "rank's quantized (4-bit or packed) expert stacks. Save the adapter (the PEFT model's save_pretrained or "
                    "trainer.save_model), or save from one process."
                )
            # Only global rank 0 writes: only it receives the gathered tensors (the adapter's alone for a PEFT
            # model), so no rank holds the whole expert set.
            state = ep_full_adapter_state_dict(level, dst = 0)

            def write():
                if dist.get_rank() != 0:
                    return None
                _SAVING.append(True)
                try:
                    with _lists().remote_experts_for_save(level), _whole_expert_counts(level):
                        return prior_save(*args, state_dict = state, **kwargs)
                finally:
                    _SAVING.pop()

            return _tpn()._agree_write(write)

        save_pretrained._unsloth_ep_save = True
        level.save_pretrained = save_pretrained
    if callable(getattr(level, "push_to_hub", None)):

        def push_to_hub(*args, **kwargs):
            raise NotImplementedError(_EP_PUSH_REFUSAL)

        level.push_to_hub = push_to_hub


@contextlib.contextmanager
def _whole_expert_counts(model):
    """While the writing rank saves the gathered state dict: the experts modules and their PEFT wrappers report
    the whole expert count (num_experts, LoRA factor sizes) again. unsloth_zoo's save rewrite of swapped expert
    LoRA factors (Llama 4, Qwen3-VL-MoE) and of the legacy grouped lora_B packing reshapes by num_experts, which
    the shard set to this rank's count."""
    restore = []
    try:
        for block in _tpn().moe_blocks_with_experts(model):
            shard = block.__dict__.get("_unsloth_tp_expert_shard")
            for module, key, value in getattr(shard, "full_attrs", None) or ():
                if key in module.__dict__:
                    restore.append((module, key, module.__dict__[key]))
                    module.__dict__[key] = value
        yield
    finally:
        for module, key, value in reversed(restore):
            module.__dict__[key] = value


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
    group = rep.get_group()
    with torch.no_grad():
        for tensors in _replica_buckets(grads, _REPLICA_BUCKET_BYTES):
            if len(tensors) == 1 and tensors[0].is_contiguous():
                dist.all_reduce(tensors[0], group = group)
                continue
            flat = _flatten_dense_tensors(tensors)
            dist.all_reduce(flat, group = group)
            for t, v in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
                t.copy_(v)
            del flat, v  # a view of flat in v would keep it alive beside the next bucket's buffer


# One flat copy of every expert gradient would double their memory.
_REPLICA_BUCKET_BYTES = 256 << 20


def _replica_buckets(grads, cap):
    """Shape-only split, so every replica reduces the same buckets."""
    open_buckets = {}
    for g in grads:
        nbytes = g.numel() * g.element_size()
        if nbytes >= cap:
            yield [g]
            continue
        key = (g.device, g.dtype)
        bucket = open_buckets.setdefault(key, [[], 0])
        if bucket[1] + nbytes > cap:
            yield bucket[0]
            bucket[0], bucket[1] = [], 0
        bucket[0].append(g)
        bucket[1] += nbytes
    for tensors, _ in open_buckets.values():
        if tensors:
            yield tensors


def _gather_expert(
    p,
    group,
    value = None,
    dst = None,
):
    """The 1-GPU tensor of expert parameter `p` (or of `value`, its gradient) gathered over the ep ranks. `dst` (a
    global rank): only that rank receives the parts (the others send theirs and get None)."""
    import torch.distributed as dist

    dim, groups, per, E, E_pad, ep = getattr(p, _EP)
    local = (p if value is None else value).detach().contiguous()
    if dst is None:
        parts = [torch.empty_like(local) for _ in range(ep)]
        dist.all_gather(parts, local, group = group)
    else:
        parts = [torch.empty_like(local) for _ in range(ep)] if dist.get_rank() == dst else None
        dist.gather(local, parts, dst = dst, group = group)
        if parts is None:
            return None
    per_rank = E_pad // ep
    pieces = [t.unflatten(dim, (groups, per_rank, per)) for t in parts]
    full = torch.cat(pieces, dim = dim + 1).flatten(dim, dim + 2)
    return _tpn()._trim_expert_pad(full, dim, groups, E, E_pad, per)


def _is_peft(model):
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


def ep_full_adapter_state_dict(
    model,
    adapter = None,
    dst = None,
):
    """Collective over the ep ranks: the model's state dict with its expert tensors gathered to the 1-GPU shape.
    `adapter` (default: `model` is a PEFT model): only the expert tensors an adapter keeps (the LoRA factors and
    any trainable stack) are gathered; the frozen base stacks PEFT drops stay this rank's (gathering them would
    put every expert on every rank). `dst` (a global rank, the one that writes): only it receives the gathered
    tensors, each moved to the CPU as it arrives (one stack on the GPU at a time); every other rank gets None,
    and the ranks of other replica groups (they hold the same experts) take no part."""
    import torch.distributed as dist

    if adapter is None:
        adapter = _is_peft(model)
    group = _mesh(model).get_group()
    if dst is not None and dst not in dist.get_process_group_ranks(group):
        return None
    receives = dst is None or dist.get_rank() == dst
    state = dict(model.state_dict()) if receives else None
    lists = bool(_lists().list_blocks(model))  # every rank, also one holding none of the experts
    for n, p in model.named_parameters():
        layout = getattr(p, _EP, None)
        if layout is not None and layout[0] == _lists().LIST:
            continue  # whole per-expert modules: gathered by name below
        if layout is None or layout[0] == "4bit":  # the frozen 4-bit stacks are not in an adapter
            continue
        if adapter and not (p.requires_grad or "lora_" in n):
            continue  # a frozen base stack: not in the adapter
        full = _gather_expert(p, group, dst = dst)
        if receives:
            state[n] = full if dst is None else full.cpu()
    if lists:
        gathered = _lists().gather_list_state(model, group, dst = dst, adapter = adapter)
        if receives:
            if dst is None:  # every rank keeps it: back on the device the parameters live on
                device = next(p.device for p in model.parameters())
                gathered = {k: v.to(device) for k, v in gathered.items()}
            state.update(gathered)
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

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    params = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    # A rank without gradients still enters the all_reduce its peers enter, with the identity.
    if params:
        device = params[0].grad.device
    else:
        device = parameters[0].device if parameters else _group_device(group)
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
    _tpn()._scale_grads_(params, total, max_norm, 1e-6)
    return total


def _group_device(group):
    import torch.distributed as dist
    if dist.get_backend(group) == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _sync_nonfinite(model, group):
    """One rank's non-finite expert gradient poisons every rank's (GradScaler then skips the step everywhere)."""
    import torch.distributed as dist

    params = [p for _, p in _ep_params(model)]
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        device = grads[0].device
    else:
        device = params[0].device if params else _group_device(group)
    bad = torch.zeros(1, dtype = torch.int32, device = device)
    if grads:
        bad = (
            torch.stack([~torch.isfinite(g).all() for g in grads]).any().to(torch.int32).reshape(1)
        )
    dist.all_reduce(bad, op = dist.ReduceOp.MAX, group = group)
    # A rank with no expert gradients (none of its experts was routed to) poisons another trainable gradient:
    # the scaler reads every gradient it unscales, so this rank skips the step with its peers.
    target = grads[0] if grads else None
    if target is None:
        target = next(
            (
                p.grad
                for p in model.parameters()
                if p.grad is not None and p.grad.is_floating_point() and p.grad.numel()
            ),
            None,
        )
    if target is not None:
        target.view(-1)[:1].masked_fill_(bad.to(target.device).bool(), float("inf"))


_CKPT_DIR = "unsloth_ep"


def _layout(model):
    import torch.distributed as dist

    rep = _replicas(model)
    group = _mesh(model).get_group()
    return {
        "world": dist.get_world_size(),
        "ep": _mesh(model).size(),
        "ep_rank": dist.get_rank(group),
        "ep_members": list(dist.get_process_group_ranks(group)),
        "replicas": rep.size() if rep is not None else 1,
        # rank_major and grouped_by_expert cut lora_B into same-shaped but different column sets per rank.
        "lora_b": {
            n: m.__dict__[_EP_LORA_B] for n, m in model.named_modules() if _EP_LORA_B in m.__dict__
        },
    }


def _rank_path(directory):
    import torch.distributed as dist
    return os.path.join(directory, _CKPT_DIR, f"rank{dist.get_rank():05d}.pt")


def _save_rank_state(
    model,
    optimizer,
    directory,
    scheduler = None,
):
    """Every rank: its trainable tensors as held (expert LoRA shards local) and its optimizer state. `scheduler`
    (the writer rank's, or None): its state is saved beside, in the same agreed write."""
    path = _rank_path(directory)
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    optim = optimizer.state_dict() if optimizer is not None else None
    payload = {"layout": _layout(model), "weights": weights, "optimizer": optim}
    tpn = _tpn()

    def write():
        os.makedirs(os.path.dirname(path), exist_ok = True)
        tpn._atomic_torch_save(payload, path)
        if scheduler is not None:
            from transformers.trainer import SCHEDULER_NAME
            tpn._atomic_torch_save(scheduler.state_dict(), os.path.join(directory, SCHEDULER_NAME))

    tpn._agree_write(write)


class EPLayoutChanged(ValueError):
    """The checkpoint's rank state was written on another expert parallel layout (or this rank has none)."""


class _EPMissingRankState(FileNotFoundError, EPLayoutChanged):
    pass


def _read_rank_state(model, directory):
    path = _rank_path(directory)
    if not os.path.exists(path):
        raise _EPMissingRankState(
            f"Unsloth: {directory} has no expert parallel rank state ({path}); it was not written under expert "
            "parallel, so its expert optimizer state cannot be restored."
        )
    # Tensors, ints, strings and an optimizer state dict: weights_only reads it, as tp_native's resume does.
    payload = _tpn()._resume_torch_load(path, map_location = "cpu")
    saved, want = dict(payload["layout"]), _layout(model)
    saved_b, want_b = saved.pop("lora_b", None), want.pop("lora_b")
    # Older checkpoints lack these keys and still load.
    member = {k: want.pop(k) for k in ("ep_rank", "ep_members")}
    saved_member = {k: saved.pop(k) for k in ("ep_rank", "ep_members") if k in saved}
    if saved != want:
        raise EPLayoutChanged(
            f"Unsloth: the checkpoint {directory} was written with expert parallel layout {saved}, "
            f"not this run's {want}; resume with the same world size and ep."
        )
    if saved_member and saved_member != member:
        raise EPLayoutChanged(
            f"Unsloth: {path} was written by expert parallel rank {saved_member.get('ep_rank')} of the group "
            f"{saved_member.get('ep_members')}, but this rank is {member['ep_rank']} of {member['ep_members']}: "
            "the expert shards and their optimizer state belong to other experts. Resume on the same rank "
            "layout (the same ep mesh over the same global ranks)."
        )
    if (
        saved_b is None
    ):  # written before the layout was recorded: only the default rank_major existed then
        saved_b = {n: "rank_major" for n in want_b}
    if saved_b != want_b:
        diff = sorted(
            (n, saved_b.get(n), want_b.get(n))
            for n in set(saved_b) | set(want_b)
            if saved_b.get(n) != want_b.get(n)
        )
        was = sorted({b for _, b, _ in diff if b})
        raise ValueError(
            f"Unsloth: the checkpoint {directory} holds MoE expert lora_B shards cut with "
            f"UNSLOTH_MOE_LORA_B_LAYOUT={'/'.join(was) or 'none'}, but this run cuts them as "
            f"{'/'.join(sorted({w for _, _, w in diff if w})) or 'none'} (e.g. {diff[0][0]}). Both have the "
            "same shape, so loading would pair columns with the wrong experts; resume with the "
            "UNSLOTH_MOE_LORA_B_LAYOUT the checkpoint was trained with."
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


_EP_EXPORTS = (
    "save_pretrained_merged",
    "save_pretrained_gguf",
    "push_to_hub_merged",
    "push_to_hub_gguf",
    "save_pretrained_torchao",
)


def _ep_refuse_exports(level):
    """Merged / GGUF / torchao exports read every weight in this process, and each rank holds only its experts:
    refused (also by unsloth/save.py's model parallel export guard)."""
    for attr in _EP_EXPORTS:
        prior = getattr(level, attr, None)
        if callable(prior) and not getattr(prior, "_unsloth_ep_refused", False):
            refused = _refuse(attr)
            refused._unsloth_ep_refused = True
            setattr(level, attr, refused)


_MODEL_INIT_REFUSAL = (
    "Unsloth: Trainer(model_init = ...) is not supported under expert parallel: train() rebuilds the model "
    "from model_init, while the expert parallel saves, gradient sync and checkpoint restore stay bound to the "
    "first one. Build the model first and pass Trainer(model = model)."
)


def _ddp_reduces_any(model):
    """A trainable parameter DDP would reduce (not in `_ddp_params_and_buffers_to_ignore`)."""
    ignored = set(getattr(model, "_ddp_params_and_buffers_to_ignore", None) or ())
    return any(p.requires_grad and n not in ignored for n, p in model.named_parameters())


def ep_trainer_setup(trainer):
    """Trainer.__init__ (after accelerate exists): EP-aware clipping, non-finite sync, 1-GPU adapter saves."""
    model = getattr(trainer, "model", None)
    if not isinstance(model, nn.Module) or not is_expert_parallel(model):
        return False
    if getattr(trainer, "model_init", None) is not None:
        raise NotImplementedError(_MODEL_INIT_REFUSAL)
    mesh = _mesh(model)
    group = mesh.get_group()
    accelerator = trainer.accelerator
    backward = accelerator.backward
    prepare_model = accelerator.prepare_model

    def ep_prepare_model(
        target,
        device_placement = None,
        evaluation_mode = False,
        **kwargs,
    ):
        # Expert-only LoRA: every trainable tensor is DDP-ignored (expert shards), and DDP refuses a module
        # with nothing to reduce. Not wrapped: the expert gradients never went through DDP (ep_backward syncs
        # the replicas, the hooks average), and placement / autocast / compile still apply.
        if not evaluation_mode and is_expert_parallel(target) and not _ddp_reduces_any(target):
            evaluation_mode = True
        return prepare_model(target, device_placement, evaluation_mode, **kwargs)

    accelerator.prepare_model = ep_prepare_model

    def ep_backward(*args, **kwargs):
        out = backward(*args, **kwargs)
        # UNSLOTH_TOKEN_BUDGET's earlier chunks of a step: the replicas are summed once, after the last one.
        if accelerator.sync_gradients and not trainer.__dict__.get("_unsloth_token_budget_partial"):
            ep_sync_expert_grads(model)
            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None and scaler.is_enabled():
                _sync_nonfinite(model, None)
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
        # Every rank takes part in gathering the expert LoRA onto rank 0 (each node's writer under
        # save_on_each_node), then the Trainer's own save there.
        import torch.distributed as dist

        if trainer.args.save_on_each_node:
            flags = torch.zeros(
                dist.get_world_size(), dtype = torch.int32, device = _tpn()._flag_device()
            )
            flags[dist.get_rank()] = int(bool(trainer.args.should_save))
            dist.all_reduce(flags)
            state = None
            for writer in flags.nonzero().flatten().tolist():
                part = ep_full_adapter_state_dict(model, dst = writer)
                if writer == dist.get_rank():
                    state = part
        else:
            state = ep_full_adapter_state_dict(model, dst = 0)
        directory = output_dir or trainer.args.output_dir

        def write():
            if trainer.args.should_save:
                model.save_pretrained(directory, state_dict = state)
                tok = getattr(trainer, "processing_class", None) or getattr(
                    trainer, "tokenizer", None
                )
                if tok is not None and hasattr(tok, "save_pretrained"):
                    tok.save_pretrained(directory)
                torch.save(trainer.args, os.path.join(directory, "training_args.bin"))

        # World, not group: other replica groups must not run ahead of the write.
        _tpn()._agree_write(write)

    trainer.save_model = ep_save_model

    # Checkpoints: the adapter is the 1-GPU one (save_model above); each rank also writes its own trainable
    # tensors (local expert shards) and its optimizer state, and a resume loads them back on the same layout;
    # on another layout the 1-GPU adapter and optimizer.pt are resharded (portable_resume.py).
    # The stock optimizer.pt (rank 0's optimizer state) is not written: rank 0 writes the 1-GPU one instead, in
    # an agreed write (a failed write leaves no peer waiting).
    from . import portable_resume

    def ep_save_optimizer_and_scheduler(output_dir):
        scheduler = getattr(trainer, "lr_scheduler", None) if trainer.args.should_save else None
        _save_rank_state(model, trainer.optimizer, output_dir, scheduler)
        if trainer.optimizer is None:
            return
        # Collective: the optimizer state in its 1-GPU form, which also resumes on another layout or one GPU
        # (a stale file is removed when this optimizer's state cannot be converted).
        portable, why = portable_resume.portable_optimizer_state(model, trainer.optimizer)
        _tpn()._agree_write(
            lambda: portable_resume.write_portable_optimizer(output_dir, portable, why)
        )

    resume_mode = {}

    def mode_of(checkpoint):
        """ "rank" (the saved layout: per-rank state) or "portable" (another layout: the 1-GPU adapter and
        optimizer.pt resharded); collective, every rank decides alike."""
        if checkpoint not in resume_mode:
            error = None
            try:
                _read_rank_state(model, checkpoint)
            except (FileNotFoundError, ValueError) as exc:
                error = exc
            if portable_resume._agree(error is None):
                resume_mode[checkpoint] = "rank"
            elif portable_resume._agree(
                error is None or isinstance(error, EPLayoutChanged)
            ) and portable_resume.portable_available(checkpoint, model):
                _tpn()._print_rank0(
                    f"Unsloth: {checkpoint} was written on another parallel layout; resuming from its 1-GPU "
                    "adapter and optimizer state, resharded."
                )
                resume_mode[checkpoint] = "portable"
            else:
                raise error or ValueError(
                    f"Unsloth: another rank cannot resume its expert parallel state from {checkpoint} (see that "
                    "rank's error)."
                )
        return resume_mode[checkpoint]

    def ep_load_from_checkpoint(resume_from_checkpoint, model_ = None):
        if mode_of(resume_from_checkpoint) == "portable":
            portable_resume.portable_load_weights(model, resume_from_checkpoint)
        else:
            _load_rank_weights(model, resume_from_checkpoint)

    def ep_load_optimizer_and_scheduler(checkpoint):
        # The rank state is authoritative on the saved layout; an older checkpoint's optimizer.pt (rank 0's
        # state) is not read. On another layout the 1-GPU optimizer.pt is resharded.
        if checkpoint is None:
            return None
        if mode_of(checkpoint) == "portable":
            state = _tpn()._resume_torch_load(
                os.path.join(checkpoint, portable_resume.PORTABLE_OPTIMIZER), map_location = "cpu"
            )
            portable_resume.portable_load_optimizer(model, trainer.optimizer, state)
        else:
            _load_rank_optimizer(model, trainer.optimizer, checkpoint)
        from transformers.trainer import SCHEDULER_NAME

        path = os.path.join(checkpoint, SCHEDULER_NAME)
        scheduler = getattr(trainer, "lr_scheduler", None)
        if scheduler is not None and os.path.isfile(path):
            scheduler.load_state_dict(
                portable_resume.remap_scheduler_state(
                    trainer.optimizer, _tpn()._resume_torch_load(path)
                )
            )
        return None

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
        # The checkpoint's adapter is the gathered 1-GPU one; this rank restores its own (rank-local) tensors, or
        # its slices of the adapter when the best checkpoint was written on another layout (before a resume).
        best = trainer.state.best_model_checkpoint
        if mode_of(best) == "portable":
            portable_resume.portable_load_weights(model, best)
        else:
            _load_rank_weights(model, best)

    trainer._save_optimizer_and_scheduler = ep_save_optimizer_and_scheduler
    trainer._load_from_checkpoint = ep_load_from_checkpoint
    trainer._load_best_model = ep_load_best_model
    trainer._load_optimizer_and_scheduler = ep_load_optimizer_and_scheduler
    for level in _levels(model):
        _ep_refuse_exports(level)
    return True
