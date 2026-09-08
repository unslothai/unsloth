# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-split (pipeline-parallel) finetuning across two or more NVIDIA DGX Sparks.

Why layer splitting: CAPACITY. `device_count()` is 1 on each Spark and a 70B does not fit on
one, so it cannot be trained on a single machine at all. Stage `s` of `W` owns a contiguous
slice of decoder layers; stage 0 also owns the embedding, the last stage the loss.

`--pp-backend torch` (default) is not a preference: the hand-written `interleaved` schedule
below deadlocked AND computed wrong gradients, because a single `loss.backward()` cannot
span the rank cut -- there is no autograd edge across a `dist.irecv`. Upstream does an
explicit grad send/recv with a locally rooted backward, so the defect cannot occur.
`legacy` keeps the hand-written schedules as a control arm. Upstream APIs are reached by
feature detection, never a version compare: torch 2.11 has no `get_mesh=` and later does.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import time
from typing import List, Optional, Sequence

# Architectures nest the decoder stack differently. Getting this wrong is silent -- the split
# would "work" and train the wrong parameters -- so it raises rather than guessing.
_LAYER_PATHS = (
    ("model", "layers"),  # Llama, Qwen, Mistral, Gemma...
    ("model", "decoder", "layers"),  # OPT-style
    ("transformer", "h"),  # GPT-2/NeoX-style
    ("gpt_neox", "layers"),
)


def _resolve(root, path: Sequence[str]):
    node = root
    for attr in path:
        if not hasattr(node, attr):
            return None
        node = getattr(node, attr)
    return node


def find_layers(model):
    for path in _LAYER_PATHS:
        layers = _resolve(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers):
            owner = _resolve(model, path[:-1])
            return owner, layers
    raise RuntimeError(
        "could not locate the decoder layer list on this architecture; "
        f"tried {_LAYER_PATHS}. Layer splitting needs an explicit layer container."
    )


def interleaved_layers(
    n_layers: int,
    rank: int,
    world: int,
    virtual: int = 2,
) -> List[List[int]]:
    """Layer chunks for INTERLEAVED pipeline parallelism (Megatron's schedule). A plain 2-stage
    pipeline cannot beat `world * M/(M+1)`; `virtual` NON-CONTIGUOUS chunks per device shrink
    the bubble to `1/(virtual*M + 1)`, at `2*virtual - 1` wire crossings instead of one."""
    total_chunks = world * virtual
    if total_chunks > n_layers:
        raise RuntimeError(
            f"{world} stages x {virtual} virtual = {total_chunks} chunks but the model has "
            f"only {n_layers} layers; lower --virtual-stages"
        )
    base, extra = divmod(n_layers, total_chunks)
    bounds, start = [], 0
    for c in range(total_chunks):
        size = base + (1 if c < extra else 0)
        bounds.append(list(range(start, start + size)))
        start += size
    # Chunk c lives on rank c % world, so consecutive chunks alternate devices.
    return [bounds[c] for c in range(total_chunks) if c % world == rank]


def stage_layers(n_layers: int, rank: int, world: int) -> List[int]:
    """Contiguous slice of layer indices, balanced by COUNT: right only for homogeneous
    layers, where a mixed dense/sparse MoE would want a cost-weighted split."""
    base, extra = divmod(n_layers, world)
    start = rank * base + min(rank, extra)
    return list(range(start, start + base + (1 if rank < extra else 0)))


def _ranges(ids: Sequence[int]) -> str:
    """`[0,1,2,7,8]` -> `"0-2,7-8"`; layer sets are not contiguous once a rank owns two."""
    out, ids = [], sorted(ids)
    for i in ids:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


# `loop` maps stage i to rank i % world, so every hop changes rank. `v` walks out and back
# (pp=2, 4 stages: {0:0, 1:1, 2:1, 3:0}), leaving two boundaries co-located, and upstream
# skips send/recv entirely for those: the whole reason DualPipeV is here.
# This duplicates upstream's `generate_stage_to_rank_mapping` rather than importing it, so the
# plan is testable with no torch. `torch_pp_plan` cross-checks them: a silent disagreement
# would place layers on the wrong node.


def stage_to_rank_map(
    world: int,
    num_stages: int,
    style: str = "loop",
) -> dict:
    if world < 1 or num_stages < 1:
        raise RuntimeError(f"bad pipeline shape: world={world} num_stages={num_stages}")
    if style == "loop":
        return {i: i % world for i in range(num_stages)}
    if style == "v":
        if num_stages % world:
            raise RuntimeError(
                f"a V-layout needs num_stages ({num_stages}) divisible by the number of "
                f"ranks ({world})"
            )
        mapping, r = {}, 0
        for i in range(num_stages):
            mapping[i] = r
            if (i + 1) % world == 0:
                continue  # at the fold, stay put: that is what makes the V
            r += 1 if (i // world) % 2 == 0 else -1
        return mapping
    raise RuntimeError(f"unknown pipeline layout {style!r}")


# --schedule -> (upstream class, stages per rank or None = --virtual-stages, layout)
TORCH_PP_SCHEDULES = {
    "gpipe": ("GPipe", 1, "loop"),
    "1f1b": ("1F1B", 1, "loop"),
    "loopedbfs": ("LoopedBFS", None, "loop"),
    "interleaved": ("Interleaved1F1B", None, "loop"),
    "zerobubble": ("InterleavedZeroBubble", None, "loop"),
    "zbv": ("ZBVZeroBubble", 2, "v"),
    "dualpipev": ("DualPipeV", 2, "v"),
}


def torch_pp_plan(
    schedule: str, world: int, microbatches: int, virtual_stages: int, n_layers: int
) -> dict:
    """Resolve `--schedule` into a concrete stage/layer/rank assignment. Pure. Everything
    refusable is refused before a tensor is allocated, because the alternative here is a
    300 s silence indistinguishable from broken hardware."""
    if schedule not in TORCH_PP_SCHEDULES:
        raise RuntimeError(
            f"--schedule {schedule!r} has no torch.distributed.pipelining equivalent; "
            f"available: {sorted(TORCH_PP_SCHEDULES)}. Use --pp-backend legacy for the "
            f"hand-written schedules."
        )
    class_name, fixed_v, style = TORCH_PP_SCHEDULES[schedule]
    v = fixed_v if fixed_v is not None else virtual_stages
    if fixed_v is not None and virtual_stages != fixed_v and virtual_stages != 2:
        # Defaults to 2, so only complain when the user actually chose an impossible value.
        raise RuntimeError(
            f"--schedule {schedule} requires exactly {fixed_v} stage(s) per rank; "
            f"--virtual-stages {virtual_stages} cannot be satisfied."
        )
    num_stages = world * v
    if n_layers < num_stages:
        raise RuntimeError(
            f"--schedule {schedule} wants {num_stages} stages ({v} per rank x {world} "
            f"ranks) but the model has only {n_layers} decoder layers; lower "
            f"--virtual-stages or pick a single-stage schedule."
        )
    if style == "v" and v != 2:
        raise RuntimeError(f"the V layout requires exactly 2 stages per rank, got {v}")
    if schedule == "dualpipev" and microbatches < num_stages:
        # Enforced by ScheduleDualPipeV too; caught here so the message beats the model load.
        raise RuntimeError(
            f"--schedule dualpipev requires --microbatches >= num_stages "
            f"({microbatches} < {num_stages})."
        )
    if microbatches < 1:
        raise RuntimeError(f"--microbatches must be >= 1 (got {microbatches})")

    mapping = stage_to_rank_map(world, num_stages, style)
    layers_of = {i: stage_layers(n_layers, i, num_stages) for i in range(num_stages)}
    return {
        "schedule": schedule,
        "class_name": class_name,
        "num_stages": num_stages,
        "stages_per_rank": v,
        "style": style,
        "stage_to_rank": mapping,
        "stage_layers": layers_of,
        "loss_rank": mapping[num_stages - 1],
        "first_rank": mapping[0],
    }


def plan_for_rank(plan: dict, rank: int) -> dict:
    mine = [i for i, r in sorted(plan["stage_to_rank"].items()) if r == rank]
    if not mine:
        raise RuntimeError(f"rank {rank} owns no pipeline stage under {plan['schedule']!r}")
    return {
        "stages": mine,
        "layers": sorted(i for s in mine for i in plan["stage_layers"][s]),
        "keep_embed": 0 in mine,
        "keep_head": (plan["num_stages"] - 1) in mine,
    }


def build_stage_model(
    model_name: str,
    rank: int,
    world: int,
    device,
    *,
    shard_load: bool,
    dtype,
    log = print,
    keep_all_layers: bool = False,
    keep_layers: Optional[Sequence[int]] = None,
    keep_embed: Optional[bool] = None,
    keep_head: Optional[bool] = None,
):
    """Build only this stage's slice of the model. With `shard_load` the skeleton is built on
    meta and only the owned tensors are read out of the shards, because materialising the
    whole model and dropping half needs more memory than the node has.
    `keep_layers`/`keep_embed`/`keep_head` override the contiguous slice: under DualPipeV rank
    0 owns the first AND last stage, so it needs both embedding and head, and rank 1 neither."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if not shard_load:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype = dtype)
        model.config.use_cache = False
        cfg = model.config
    else:
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.use_cache = False
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, dtype = dtype)

    owner, layers = find_layers(model)
    n = len(layers)
    if keep_layers is None and world > n:
        raise RuntimeError(f"{world} stages requested but the model has only {n} layers")
    mine = sorted(set(keep_layers)) if keep_layers is not None else stage_layers(n, rank, world)
    if not mine:
        raise RuntimeError(f"rank {rank} was assigned no decoder layers at all")
    if mine[-1] >= n or mine[0] < 0:
        raise RuntimeError(f"layer ids {mine[:2]}..{mine[-2:]} out of range for {n} layers")
    want_embed = (rank == 0) if keep_embed is None else bool(keep_embed)
    want_head = (rank == world - 1) if keep_head is None else bool(keep_head)

    keep = set(mine)
    if not keep_all_layers:
        for i in range(n):
            if i not in keep:
                layers[i] = torch.nn.Identity()
    if not keep_all_layers and not want_embed and hasattr(owner, "embed_tokens"):
        owner.embed_tokens = torch.nn.Identity()
    if not keep_all_layers and not want_head:
        if hasattr(owner, "norm"):
            owner.norm = torch.nn.Identity()
        if hasattr(model, "lm_head"):
            model.lm_head = torch.nn.Identity()

    log(
        f"{n} decoder layers; this stage owns {_ranges(mine)}"
        f"{' +embed' if want_embed else ''}{' +head' if want_head else ''}"
    )

    if shard_load:
        _materialise(model, model_name, cfg, device, dtype, log)

    return model, cfg, mine


def _materialise(model, model_name, cfg, device, dtype, log):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    wanted = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    snap = snapshot_download(model_name, allow_patterns = ["*.safetensors", "*.json"])

    loaded, seen = {}, 0
    for f in sorted(glob.glob(osp.join(snap, "*.safetensors"))):
        with safe_open(f, framework = "pt", device = "cpu") as sf:
            for k in sf.keys():
                if k in wanted:
                    # One tensor at a time: reading the half into host memory first needs TWO
                    # copies of ~70 GiB, and the OOM killer leaves no Python traceback.
                    loaded[k] = sf.get_tensor(k).to(dtype).to(device, non_blocking = False)
                    seen += 1
    model.load_state_dict(loaded, strict = False, assign = True)

    # Non-persistent buffers (rotary inv_freq, causal masks) are never in safetensors, so
    # load_state_dict leaves them on meta and only the first real use says so.
    meta_bufs = [n for n, b in model.named_buffers() if b.is_meta]
    if meta_bufs:
        log(f"shard-load: rebuilding {len(meta_bufs)} meta buffers ({meta_bufs[:3]})")
        inner = getattr(model, "model", model)
        rot = getattr(inner, "rotary_emb", None)
        if rot is not None:
            inner.rotary_emb = type(rot)(config = cfg, device = device)
        for _, mod in model.named_modules():
            if any(b.is_meta for _, b in mod.named_buffers(recurse = False)):
                mod.to_empty(device = device, recurse = False)

    still = [k for k, v in model.named_parameters() if v.is_meta] + [
        k for k, v in model.named_buffers() if v.is_meta
    ]
    log(f"shard-load: materialised {seen} tensors; {len(still)} still on meta")
    if still:
        raise RuntimeError(f"unmaterialised tensors remain: {still[:4]}")


# Everything down to `run_zerobubble` is the hand-written legacy backend, retained unchanged
# as a control arm: `--pp-backend legacy --schedule gpipe` reproduces the old behaviour.


class _Stage:

    def __init__(
        self,
        model,
        cfg,
        rank,
        world,
        device,
        dtype,
        microbatches,
        chunks = None,
        n_chunks = None,
    ):
        self.model, self.cfg = model, cfg
        self.rank, self.world = rank, world
        self.device, self.dtype = device, dtype
        self.microbatches = microbatches
        # Interleaved mode only: global chunk indices this rank owns, and the layers in each.
        # Contiguous mode leaves both None and takes the single-slice path.
        self.chunks = chunks or []
        self.n_chunks = n_chunks or world
        self.chunk_layers = {}
        self.is_first = rank == 0
        self.is_last = rank == world - 1
        base = getattr(model, "base_model", None)
        self.base = base.model if base is not None else model  # unwrap PEFT
        self.inner = getattr(self.base, "model", self.base)
        self.hidden = self.base.config.hidden_size

    def forward_chunk(self, ids, hidden, posid, chunk):
        import torch

        layers = self.chunk_layers[chunk]
        h = self.inner.embed_tokens(ids) if chunk == 0 else hidden
        pos = self.inner.rotary_emb(h, posid)
        for i in layers:
            out = self.inner.layers[i](h, position_embeddings = pos)
            h = out[0] if isinstance(out, tuple) else out
        if chunk != self.n_chunks - 1:
            return h, None
        import torch.nn.functional as F

        logits = self.base.lm_head(self.inner.norm(h))
        loss = (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                ids[:, 1:].reshape(-1),
            )
            / self.microbatches
        )
        return h, loss

    def forward(self, ids, hidden, posid):
        import torch

        h = self.inner.embed_tokens(ids) if self.is_first else hidden
        pos = self.inner.rotary_emb(h, posid)
        for layer in self.inner.layers:
            if isinstance(layer, torch.nn.Identity):
                continue
            out = layer(h, position_embeddings = pos)
            h = out[0] if isinstance(out, tuple) else out
        if not self.is_last:
            return h, None
        import torch.nn.functional as F

        logits = self.base.lm_head(self.inner.norm(h))
        loss = (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                ids[:, 1:].reshape(-1),
            )
            / self.microbatches
        )
        return h, loss


def warmup_p2p(stage, dist, torch):
    """Mirrored one-element exchange in each direction, which pins the ordered per-pair p2p
    channel open (see the root-cause note above `_p2p_group`). The ORDER is load-bearing at
    any `world`: DOWNSTREAM pair first (send then recv with rank+1), then UPSTREAM (recv then
    send with rank-1), so pairs resolve from the tail backwards and no rank blocks on a peer
    that is blocked on it. Swapping the blocks reintroduces the circular wait."""
    tiny = torch.zeros(1, dtype = stage.dtype, device = stage.device)
    if not stage.is_last:
        dist.send(tiny, dst = stage.rank + 1)
        dist.recv(tiny, src = stage.rank + 1)
    if not stage.is_first:
        dist.recv(tiny, src = stage.rank - 1)
        dist.send(tiny, dst = stage.rank - 1)


# ROOT CAUSE of the first-step hangs: un-batched `isend`/`irecv` between one pair of ranks
# share a single ORDERED p2p stream, so an op enqueued behind a receive with no matching send
# yet never launches. PyTorch promises order-independence only for `batch_isend_irecv`.
# gpipe survives because its two ranks mirror, sends then receives, never interleaving
# directions; zerobubble, 1f1b and interleaved all post a receive inside the forward loop.
#
# A batched group is ATOMIC: it rendezvouses as a unit, so if A issues {send->B, recv<-B}
# then B must issue the mirror {recv<-A, send->A} as ONE group at the same step. Splitting
# either side deadlocks even though the per-direction op order still matches.
#
# gloo has no such ordering constraint (independent buffers, a progress thread), so a green
# CPU run is NOT evidence that a p2p ordering change is safe. Two offline models and one CPU
# gradient suite have each certified code that hangs here, so: a simulation may REJECT a
# schedule, never certify one.
#
# What the hardware supports, stated as weakly as the evidence allows: gpipe and zerobubble
# work and never need a send and a receive in flight at once; 1f1b and interleaved need
# concurrent send+recv on one pair and have hung on every attempt. There is ZERO hardware
# evidence that batched p2p groups work on this stack, so treat `_p2p_group` as unproven and
# do not build a schedule on it. `--schedule interleaved` is refused outright.


def _p2p_group(dist, ops):
    """Issue point-to-point ops as one batched group and wait. Across groups only
    SAME-DIRECTION order matters, and every schedule here keeps that ascending in
    (microbatch, chunk). An empty list is a no-op, so callers can accumulate and flush late."""
    if not ops:
        return
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def _check_schedule(name, stage, batches):
    """Refuse a configuration the schedule cannot honour, loudly and early: a 1-stage
    "pipeline" that sends to itself, an empty microbatch list, or a chunk map for another
    rank all otherwise produce a run that completes and trains the wrong thing."""
    world, M = stage.world, len(batches)
    if world < 2:
        raise RuntimeError(
            f"schedule {name!r} needs at least two pipeline stages (WORLD_SIZE >= 2); got "
            f"world={world}. A one-stage pipeline would send to itself; run the model "
            f"single-node instead."
        )
    if M < 1:
        raise RuntimeError(f"schedule {name!r} needs at least one microbatch; got {M}.")
    if stage.rank < 0 or stage.rank >= world:
        raise RuntimeError(f"rank {stage.rank} out of range for world {world}")


def run_gpipe(stage, batches, posid, mb_rows, dist, torch):
    """All forwards, then all backwards. Blocking `send`/`recv` match in issue order, so the
    stages must mirror: interleaving on one side only deadlocks the communicator."""
    _check_schedule("gpipe", stage, batches)
    acts, held = [], []
    total = torch.zeros((), device = stage.device)
    # Pre-posted: a blocking `recv` issued only when the value is wanted makes the sender wait
    # for the receiver, serialising the stages. Pre-posting changes when the buffer is
    # available, not the order ops are matched in, so gpipe's mirroring still holds.
    hidden_bufs, hreq = {}, {}
    if not stage.is_first:
        for m in range(len(batches)):
            hidden_bufs[m] = torch.empty(
                mb_rows, posid.shape[1], stage.hidden, dtype = stage.dtype, device = stage.device
            )
            hreq[m] = dist.irecv(hidden_bufs[m], src = stage.rank - 1)

    for m, ids in enumerate(batches):
        hidden = None
        if not stage.is_first:
            hreq[m].wait()
            hidden = hidden_bufs[m]
            hidden.requires_grad_(True)
        h, loss = stage.forward(ids, hidden, posid)
        if not stage.is_last:
            dist.send(h.detach().contiguous(), dst = stage.rank + 1)
        acts.append((h, hidden))
        held.append(loss)

    # Pre-posted for the whole backward pass, so the neighbour's sends land as produced.
    gbufs, gr = {}, {}
    if not stage.is_last:
        for m in range(len(batches)):
            gbufs[m] = torch.empty_like(acts[m][0])
            gr[m] = dist.irecv(gbufs[m], src = stage.rank + 1)

    for m in range(len(batches)):
        h, hidden = acts[m]
        if stage.is_last:
            held[m].backward()
            # On the DEVICE: `.item()` here syncs once per microbatch, stalling the pipeline
            # it is meant to be measuring.
            total = total + held[m].detach()
        else:
            gr[m].wait()
            h.backward(gbufs[m])
            gbufs[m] = None
        if not stage.is_first:
            dist.send(hidden.grad.contiguous(), dst = stage.rank - 1)
    return total


def run_1f1b(stage, batches, posid, mb_rows, dist, torch):
    """One forward, one backward, with a per-rank warmup depth. CURRENTLY REFUSED: it
    DEADLOCKS on hardware (ONEF1B_REFUSAL); SPARK_PP_DIAGNOSE=1 runs it anyway and
    SPARK_PP_TRACE=1 shows where it stops. Stage `r` holds `world - 1 - r` forwards in flight
    before its first backward, because microbatch m's gradient cannot return until m has
    crossed the remaining stages both ways.

    Ruled out, so nobody re-runs the chase: p2p posting order (real, but that was
    zerobubble's bug), atomic group boundaries (honoured here), and `batch_isend_irecv` itself
    (a standalone probe replays this exact group sequence, at real shapes and the M that
    deadlocks, in 0.17 s). So the defect is in the NON-communication logic. A simulation may
    reject a schedule, never certify one, so the next step is a stack from a live hang.
    """
    _check_schedule("1f1b", stage, batches)
    if not _ALLOW_REFUSED:
        raise RuntimeError(ONEF1B_REFUSAL)
    M = len(batches)
    world, rank = stage.world, stage.rank
    warm = min(world - 1 - rank, M)  # forwards in flight before this stage's first B
    rem = M - warm
    up = rank - 1 if not stage.is_first else None
    dn = rank + 1 if not stage.is_last else None
    seq_len = posid.shape[1]

    # Group BOUNDARIES must correspond across the pair, not merely op order (see _p2p_group).
    def new_hidden():
        return torch.empty(mb_rows, seq_len, stage.hidden, dtype = stage.dtype, device = stage.device)

    def recv_forward():
        if up is None:
            return None
        buf = new_hidden()
        _trace(stage, "p2p recv_forward enter")
        _p2p_group(dist, [dist.P2POp(dist.irecv, buf, up)])
        _trace(stage, "p2p recv_forward done")
        buf.requires_grad_(True)
        return buf

    def send_forward(t):
        if dn is None:
            return
        _trace(stage, "p2p send_forward enter")
        _p2p_group(dist, [dist.P2POp(dist.isend, t, dn)])
        _trace(stage, "p2p send_forward done")

    def send_forward_recv_backward(t):
        if dn is None:
            return None
        g = torch.empty_like(t)
        _trace(stage, "p2p send_forward_recv_backward enter")
        _p2p_group(dist, [dist.P2POp(dist.isend, t, dn), dist.P2POp(dist.irecv, g, dn)])
        _trace(stage, "p2p send_forward_recv_backward done")
        return g

    def send_backward_recv_forward(t):
        if up is None:
            return None
        buf = new_hidden()
        _trace(stage, "p2p send_backward_recv_forward enter")
        _p2p_group(dist, [dist.P2POp(dist.isend, t, up), dist.P2POp(dist.irecv, buf, up)])
        _trace(stage, "p2p send_backward_recv_forward done")
        buf.requires_grad_(True)
        return buf

    def send_backward(t):
        if up is None:
            return
        _trace(stage, "p2p send_backward enter")
        _p2p_group(dist, [dist.P2POp(dist.isend, t, up)])
        _trace(stage, "p2p send_backward done")

    def recv_backward(like):
        if dn is None:
            return None
        g = torch.empty_like(like)
        _trace(stage, "p2p recv_backward enter")
        _p2p_group(dist, [dist.P2POp(dist.irecv, g, dn)])
        _trace(stage, "p2p recv_backward done")
        return g

    acts = []  # FIFO of (h, hidden, loss) awaiting backward
    total = torch.zeros((), device = stage.device)

    def do_backward(gout):
        h, hidden, loss = acts.pop(0)
        _trace(stage, f"compute backward enter (acts left {len(acts)})")
        contribution = 0.0
        if stage.is_last:
            _phase("bwd", lambda: loss.backward(), torch)
            contribution = loss.detach()  # device-side; no per-microbatch sync
        else:
            _phase("bwd", lambda: h.backward(gout), torch)
        _trace(stage, "compute backward done")
        if hidden is not None and hidden.grad is None:
            # An autograd wait and a missing gradient look identical from outside.
            raise RuntimeError(
                "1f1b: the received activation got no gradient from backward. Its graph "
                "does not reach this stage's input, so the upstream stage would train on "
                "nothing. This is the interleaved bug in blocking form."
            )
        grad = hidden.grad.contiguous() if hidden is not None else None
        return contribution, grad

    _trace(stage, f"START warm={warm} rem={rem} M={M}")
    for i in range(warm):
        _trace(stage, f"warmup F({i})")
        hidden = recv_forward()
        h, loss = stage.forward(batches[i], hidden, posid)
        send_forward(h.detach().contiguous())
        acts.append((h, hidden, loss))

    hidden = recv_forward() if rem > 0 else None
    for i in range(rem):
        m = warm + i
        _trace(stage, f"steady F({m}) [i={i}/{rem}]")
        h, loss = stage.forward(batches[m], hidden, posid)
        gout = send_forward_recv_backward(h.detach().contiguous())
        acts.append((h, hidden, loss))
        contribution, grad = do_backward(gout)
        total = total + contribution
        if i == rem - 1:
            send_backward(grad)
            hidden = None
        else:
            hidden = send_backward_recv_forward(grad)

    for c in range(warm):
        _trace(stage, f"cooldown B [{c}/{warm}]")
        gout = _phase("wait", lambda: recv_backward(acts[0][0]), torch)
        contribution, grad = do_backward(gout)
        total = total + contribution
        send_backward(grad)
    _trace(stage, "DONE step")
    return total


# SPARK_PP_TIME=1 cuda-syncs, which serialises the overlap a pipeline exists to create: an
# instrumented run is NOT a valid throughput measurement. SPARK_PP_TRACE=1 does not sync; its
# only job is to make a hang name itself, so the last line says which rank blocked where.
_TRACE = os.environ.get("SPARK_PP_TRACE", "0") == "1"


def _trace(stage, msg):
    if _TRACE:
        import sys
        print(f"[pp-trace {stage.rank}] {msg}", file = sys.stderr, flush = True)


PHASE = {"wait": 0.0, "bwd": 0.0, "fwd": 0.0}
_TIME_PHASES = os.environ.get("SPARK_PP_TIME", "0") == "1"


def _phase(name, fn, torch):
    if not _TIME_PHASES:
        return fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    PHASE[name] += time.perf_counter() - t0
    return out


def _backward_one(stage, acts, held, grads, greq, m, inflight, dist, torch):
    h, hidden = acts[m]
    contribution = 0.0
    if stage.is_last:
        _phase("bwd", lambda: held[m].backward(), torch)
        contribution = held[m].detach()  # device-side; no per-microbatch sync
    else:
        _phase("wait", lambda: greq[m].wait(), torch)
        _phase("bwd", lambda: h.backward(grads[m]), torch)
        grads[m] = None
    if not stage.is_first:
        grad = hidden.grad.contiguous()  # keep the reference; see run_1f1b
        inflight.append((grad, dist.isend(grad, dst = stage.rank - 1)))
    # Dropping the activation here rather than at end of step is what bounds 1F1B's memory to
    # the pipeline depth instead of the microbatch count: the whole argument over GPipe.
    acts[m] = (None, None)
    held[m] = None
    return contribution


def run_interleaved(stage, batches, posid, mb_rows, dist, torch):
    """Interleaved (virtual-stage) pipeline parallelism. Each rank owns `v` non-contiguous
    chunks; chunk `c` lives on rank `c % world`.

    Two silent bugs were fixed here, both of which completed the run with a falling loss.
    (1) `loss.backward()` only walks THIS process's autograd graph, which ends at the received
    activation: there is no autograd edge across a `dist.irecv`, so only the last chunk on the
    last rank got gradients. An explicit backward phase is required.
    (2) p2p ops match in POSTING ORDER per direction and NCCL ignores tags, so a data-driven
    execution order leaked onto the wire and one microbatch's activations landed in another's
    buffer. The fix is a canonical order both sides derive without communicating.

    At `world == 2`, `(rank+1) % 2 == (rank-1) % 2`, so forward activations and backward
    gradients share ONE FIFO. That is safe only because every rank finishes its whole forward
    phase before starting backward: do not merge the loops without re-deriving this. All
    `M * v` local graphs stay alive between the phases, so this is GPipe-shaped in memory.

    Three rules for anyone editing, all of which gloo absorbs and NCCL hangs on: never block
    on a receive that is not the EARLIEST outstanding one on its direction; never let
    execution order leak onto the wire; never leave a send and a receive un-batched.
    """
    _check_schedule("interleaved", stage, batches)
    if not _ALLOW_REFUSED:
        raise RuntimeError(INTERLEAVED_REFUSAL)
    world, n_chunks = stage.world, stage.n_chunks
    chunks = sorted(stage.chunks)
    M = len(batches)
    seq = posid.shape[1]

    # Validate the chunk map: getting this wrong is the silent-wrong-parameters case.
    v, rem = divmod(n_chunks, world)
    if rem or v < 1:
        raise RuntimeError(
            f"interleaved: n_chunks={n_chunks} is not a positive multiple of world={world}; "
            f"chunk c must live on rank c % world for the round-robin to close."
        )
    expected = [stage.rank + k * world for k in range(v)]
    if chunks != expected:
        raise RuntimeError(
            f"interleaved: rank {stage.rank} was given chunks {chunks} but chunk c lives on "
            f"rank c % world, so it must own exactly {expected}. Mismatched chunk maps "
            f"train the wrong layers silently."
        )
    missing = [c for c in chunks if not stage.chunk_layers.get(c)]
    if missing:
        raise RuntimeError(f"interleaved: no layers assigned to chunk(s) {missing}")

    def prev_rank(c):
        return (c - 1) % world

    def next_rank(c):
        return (c + 1) % world

    def new_buf():
        return torch.empty(mb_rows, seq, stage.hidden, dtype = stage.dtype, device = stage.device)

    # A statically-derived, globally-agreed op order. Chunk c here corresponds to chunk c+1 on
    # the successor and that mapping is order-preserving, so both ranks' same-direction
    # sequences agree element for element without either knowing the other's state.
    fwd_tasks = [(m, c) for m in range(M) for c in chunks]
    bwd_tasks = [(m, c) for m in range(M) for c in reversed(chunks)]

    pending, keep = [], []
    losses = torch.zeros((), device = stage.device)

    def exchange(recv_ops):
        _p2p_group(dist, pending + recv_ops)
        pending.clear()

    def queue_send(t, dst):
        keep.append(t)
        pending.append(dist.P2POp(dist.isend, t, dst))

    fwd = {}
    for m, c in fwd_tasks:
        inp = None
        if c != 0:
            inp = new_buf()
            # One group, so neither the send for c-1 nor the receive for c queues behind the other.
            exchange([dist.P2POp(dist.irecv, inp, prev_rank(c))])
            inp.requires_grad_(True)
        h, loss = stage.forward_chunk(batches[m], inp, posid, c)
        fwd[(m, c)] = (h, inp, loss)
        if loss is not None:
            losses = losses + loss.detach()
        if c != n_chunks - 1:
            queue_send(h.detach().contiguous(), next_rank(c))
    exchange([])

    # Without this phase the run trains only the final chunk -- see bug (1) above.
    for m, c in bwd_tasks:
        h, inp, loss = fwd[(m, c)]
        if c == n_chunks - 1:
            loss.backward()
        else:
            gbuf = torch.empty_like(h)
            exchange([dist.P2POp(dist.irecv, gbuf, next_rank(c))])
            h.backward(gbuf)
        if c != 0:
            if inp is None or inp.grad is None:
                raise RuntimeError(
                    f"interleaved: chunk {c} produced no gradient for its input "
                    f"(microbatch {m}). The chunk's forward did not consume the received "
                    f"activation, so the upstream stage would train on nothing -- "
                    f"refusing rather than sending a wrong gradient."
                )
            queue_send(inp.grad.contiguous(), prev_rank(c))
        fwd[(m, c)] = (None, None, None)  # release the graph as soon as it is spent
    exchange([])
    return losses


def run_zerobubble(stage, batches, posid, mb_rows, dist, torch):
    """Zero-bubble pipeline parallelism (Qi et al., 2023): the only schedule whose ceiling is
    the device count rather than `world * M/(M+1)`. A backward pass is two separable
    computations, B (w.r.t. the stage INPUT, which the neighbour is waiting on) and W (w.r.t.
    the WEIGHTS, which nothing waits on and can be deferred into the bubble).
    `torch.autograd.grad` splits them directly, and `retain_graph=True` is both what makes the
    deferral legal and its cost, since activation memory is held longer than in 1F1B."""
    _check_schedule("zerobubble", stage, batches)
    M = len(batches)
    seq = posid.shape[1]

    # Pre-posted: a receive posted on demand blocks its sender. Both directions and both
    # sides order by ascending microbatch, which is what makes untagged FIFO matching correct
    # for any `world`.
    hbuf, hreq = {}, {}
    if not stage.is_first:
        for m in range(M):
            hbuf[m] = torch.empty(
                mb_rows, seq, stage.hidden, dtype = stage.dtype, device = stage.device
            )
            hreq[m] = dist.irecv(hbuf[m], src = stage.rank - 1)
    gbuf, greq = {}, {}
    inflight = []
    acts, held = {}, {}
    total = torch.zeros((), device = stage.device)
    params = [q for q in stage.model.parameters() if q.requires_grad]

    deferred_w = []  # (output_tensor, grad_output) pairs awaiting their W pass

    def run_w(budget = 1):
        """Spend idle time on deferred weight gradients. This is the whole trick."""
        done = 0
        while deferred_w and done < budget:
            out, gout = deferred_w.pop(0)
            grads = torch.autograd.grad(
                outputs = out, inputs = params, grad_outputs = gout, retain_graph = False, allow_unused = True
            )
            for q, g in zip(params, grads):
                if g is None:
                    continue
                q.grad = g if q.grad is None else q.grad + g
            done += 1

    for m, ids in enumerate(batches):
        hidden = None
        if not stage.is_first:
            hreq[m].wait()
            hidden = hbuf[m]
            hidden.requires_grad_(True)
        h, loss = stage.forward(ids, hidden, posid)
        acts[m], held[m] = (h, hidden), loss
        if not stage.is_last:
            payload = h.detach().contiguous()
            inflight.append((payload, dist.isend(payload, dst = stage.rank + 1)))

    # The gradient receives are posted HERE, after every forward send, not inside the loop
    # above: posting `irecv(g_m)` between `isend(h_m)` and `isend(h_{m+1})` puts an
    # unmatchable receive in front of a send the peer is waiting for, and hung this schedule
    # before step 1. Out here the op order per direction is GPipe's mirrored all-sends-then-
    # all-receives, the one shape measured to work.
    if not stage.is_last:
        for m in range(M):
            gbuf[m] = torch.empty_like(acts[m][0])
            greq[m] = dist.irecv(gbuf[m], src = stage.rank + 1)

    for m in range(M):
        h, hidden = acts[m]
        if stage.is_last:
            gout = torch.ones_like(held[m])
            total = total + held[m].detach()
            out_for_w, grad_for_w = held[m], gout
            if not stage.is_first:
                gin = torch.autograd.grad(
                    outputs = held[m], inputs = hidden, grad_outputs = gout, retain_graph = True
                )[0]
        else:
            # Spend the wait on deferred W work rather than idling: this is where the bubble goes.
            while not greq[m].is_completed() and deferred_w:
                run_w(1)
            greq[m].wait()
            gout = gbuf[m]
            out_for_w, grad_for_w = h, gout
            if not stage.is_first:
                gin = torch.autograd.grad(
                    outputs = h, inputs = hidden, grad_outputs = gout, retain_graph = True
                )[0]
        if not stage.is_first:
            gsend = gin.contiguous()
            inflight.append((gsend, dist.isend(gsend, dst = stage.rank - 1)))
        deferred_w.append((out_for_w, grad_for_w))

    run_w(budget = len(deferred_w))  # drain any W still outstanding
    for _, r in inflight:
        r.wait()
    return total


# Why `torch.distributed.pipelining` is the default. Both defects in the hand-written
# schedules above have one root cause: a single `loss.backward()` cannot span the rank cut,
# because there is no autograd edge across a `dist.irecv`. Upstream solves that structurally
# rather than by being more careful. ORDERING: every rank runs the SAME deterministic
# simulation and a rank never schedules its own receive -- the sender emits it -- so an
# unschedulable order raises "Malformed compute schedule" at construction instead of hanging.
# GRADIENTS: backward is an explicit grad send/recv with a locally-rooted
# `torch.autograd.backward(stage_output, grad_tensors=received_grad)`, so nothing is asked to
# cross the process boundary.
#
# Gloo cannot certify a schedule against NCCL (no p2p ordering constraint), so gloo runs are
# correctness evidence only. On hardware every arm lands on the same loss and none falls
# FASTER: a faster-falling loss is a gradient defect, which is how the broken hand-written
# interleaved was caught. GPipe is the arm to avoid -- it holds every microbatch's
# activations to end of step, ~100 GiB against 1F1B's 7, which on a 121.69 GiB node is the
# difference between fitting and not.
#
# GOTCHA: the V layouts CANNOT be validated on gloo. At pp=2 with 4 stages the map is
# {0:0, 1:1, 2:1, 3:0}, so `_get_init_p2p_neighbors_ops` emits a send and a recv from rank 1
# to ITSELF; gloo fails that with "Pair is not connected" while NCCL handles it as a local
# copy. A gloo failure of a V schedule is not a defect in this file.

_STAGE_MODULE_CLS = None


def _dist_pipelining_available() -> bool:
    """Is `torch.distributed.pipelining` importable here? Feature detection, never a version
    string, and never at module scope: `unsloth run` imports this file on macOS and Windows,
    where `torch.distributed` may be absent and importing torch at all is a pinned regression."""
    try:
        import torch
        if not torch.distributed.is_available():
            return False
        import torch.distributed.pipelining  # noqa: F401
    except Exception:
        return False
    return True


def config_num_layers(cfg) -> int:
    """Decoder depth, before any model is built: the layer->stage assignment must exist before
    `build_stage_model` is told what to keep, and building the model to count its layers would
    defeat shard loading. Raises rather than guessing an unknown attribute name."""
    for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
        n = getattr(cfg, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    text = getattr(cfg, "text_config", None)
    if text is not None:
        return config_num_layers(text)
    raise RuntimeError(
        "could not read the decoder depth off this config; the pipeline plan needs it "
        "before the model is built."
    )


def unwrap_stack(model):
    """Return `(causal_lm, decoder_stack)` through PEFT and the HF wrappers. `base_model.model`
    is right only under PEFT: on a bare `LlamaForCausalLM`, `base_model` already returns the
    decoder stack. `get_base_model()` is absent off PEFT, so it is a safe discriminator."""
    top = model.get_base_model() if hasattr(model, "get_base_model") else model
    owner, _ = find_layers(top)
    return top, owner


def stage_module_cls():
    """The `nn.Module` one pipeline stage runs. Built lazily: subclassing `nn.Module` at
    module scope would import torch on a Mac, which a test forbids."""
    global _STAGE_MODULE_CLS
    if _STAGE_MODULE_CLS is not None:
        return _STAGE_MODULE_CLS
    import torch

    class _PPStageModule(torch.nn.Module):
        """A contiguous run of decoder layers, plus the embedding on the first stage and the
        norm + lm_head on the last. `PipelineStage` takes a MANUALLY split module -- no tracer,
        which is what makes this work where a symbolic trace of an HF model would not -- and
        position ids are recomputed, so a ragged final microbatch cannot desync the stage."""

        def __init__(self, top, owner, layer_ids, *, is_first, is_last, grad_checkpoint):
            super().__init__()
            self.is_first, self.is_last = bool(is_first), bool(is_last)
            self.grad_checkpoint = bool(grad_checkpoint)
            self.layers = torch.nn.ModuleList([owner.layers[i] for i in layer_ids])
            self.rotary_emb = getattr(owner, "rotary_emb", None)
            self.embed_tokens = getattr(owner, "embed_tokens", None) if is_first else None
            self.norm = getattr(owner, "norm", None) if is_last else None
            self.lm_head = getattr(top, "lm_head", None) if is_last else None
            if self.is_first and not isinstance(self.embed_tokens, torch.nn.Module):
                raise RuntimeError("the first pipeline stage has no embedding to run")
            if self.is_last and self.lm_head is None:
                raise RuntimeError("the last pipeline stage has no lm_head to run")

        @staticmethod
        def _call_layer(layer, h, pos):
            out = layer(h, position_embeddings = pos)
            return out[0] if isinstance(out, tuple) else out

        def forward(self, x):
            h = self.embed_tokens(x) if self.is_first else x
            pos = None
            if self.rotary_emb is not None:
                ids = torch.arange(h.shape[1], device = h.device)
                pos = self.rotary_emb(h, ids.unsqueeze(0).expand(h.shape[0], -1))
            ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
            for layer in self.layers:
                if ckpt:
                    # use_reentrant=False: the reentrant path drops the grad_fn the stage's
                    # activation-gradient handoff needs.
                    h = torch.utils.checkpoint.checkpoint(
                        self._call_layer, layer, h, pos, use_reentrant = False
                    )
                else:
                    h = self._call_layer(layer, h, pos)
            if self.is_last:
                h = self.lm_head(self.norm(h) if self.norm is not None else h)
            return h

    _STAGE_MODULE_CLS = _PPStageModule
    return _STAGE_MODULE_CLS


def pp_loss_fn(logits, target):
    """Next-token cross entropy, MEAN-reduced. This is the legacy schedules' loss minus their
    `/ microbatches`, which upstream does itself via `scale_grads=True`; keeping `/M` here as
    well would scale every gradient by `1/M^2`."""
    import torch.nn.functional as F
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        target[:, 1:].reshape(-1),
    )


# `pp_loss_fn` mean-reduces, so gradients must be divided by the microbatch count. Set
# explicitly, not left to the upstream default: sum-reducing the loss means False here.
PP_SCALE_GRADS = True


def build_torch_schedule(
    model,
    plan,
    my,
    *,
    microbatches,
    device,
    grad_checkpoint,
    log = print,
):
    """Assemble upstream `PipelineStage`s and the requested schedule for this rank.

    Every upstream API touched here is feature-detected with `hasattr`/`inspect`, never a
    version string: torch 2.11 has no `get_mesh=` on `PipelineStage` and later torch does, so
    passing or omitting it unconditionally breaks one of them. Ask the signature.
    """
    import inspect
    import torch
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining import schedules as _schedules
    from torch.distributed.pipelining.microbatch import TensorChunkSpec

    # Cross-check our pure layout against upstream's, which the runtime actually uses for
    # send/recv: a disagreement trains the wrong parameters with no error at all.
    gen = getattr(_schedules, "generate_stage_to_rank_mapping", None)
    if gen is not None:
        theirs = gen(dist.get_world_size(), plan["num_stages"], style = plan["style"])
        if dict(theirs) != dict(plan["stage_to_rank"]):
            raise RuntimeError(
                f"stage->rank layout disagrees with torch.distributed.pipelining: "
                f"ours={plan['stage_to_rank']} theirs={dict(theirs)}. Refusing to run "
                f"rather than place layers on the wrong node."
            )

    get_cls = getattr(_schedules, "get_schedule_class", None)
    if get_cls is not None:
        sched_cls = get_cls(plan["class_name"])
    else:  # older/newer layout: fall back to the name
        sched_cls = getattr(_schedules, "Schedule" + plan["class_name"], None)
        if sched_cls is None:
            raise RuntimeError(
                f"this torch has no schedule {plan['class_name']!r} and no "
                f"get_schedule_class(); use --pp-backend legacy"
            )

    top, owner = unwrap_stack(model)
    cls = stage_module_cls()
    stage_kwargs = {}
    stage_params = inspect.signature(PipelineStage.__init__).parameters
    if "group" in stage_params:
        stage_kwargs["group"] = None  # default process group

    stages, mods = [], []
    for idx in my["stages"]:
        mod = cls(
            top,
            owner,
            plan["stage_layers"][idx],
            is_first = (idx == 0),
            is_last = (idx == plan["num_stages"] - 1),
            grad_checkpoint = grad_checkpoint,
        ).to(device)
        mods.append(mod)
        # input_args=None: the stage infers the boundary shape by propagating stage 0's real
        # output. Hand-specifying it is how a seq-length change turns into a hang.
        stages.append(PipelineStage(mod, idx, plan["num_stages"], device, **stage_kwargs))

    sched_params = inspect.signature(sched_cls.__init__).parameters
    kw = {"loss_fn": pp_loss_fn}
    if "args_chunk_spec" in sched_params:
        kw["args_chunk_spec"] = (TensorChunkSpec(0),)
    if "scale_grads" in sched_params:
        kw["scale_grads"] = PP_SCALE_GRADS
    multi = len(stages) > 1 or "stages" in sched_params
    schedule = sched_cls(stages if multi else stages[0], microbatches, **kw)

    step_params = inspect.signature(schedule.step).parameters
    step_kw = {}
    if "return_outputs" in step_params:
        # Merged logits are `batch x seq x vocab` in fp32, tens of GiB on the loss rank for a
        # tensor nobody reads. Only newer torch can decline it.
        step_kw["return_outputs"] = False

    log(
        f"torch.distributed.pipelining: {sched_cls.__name__} "
        f"{plan['num_stages']} stages ({plan['stages_per_rank']}/rank, "
        f"{plan['style']}-layout), M={microbatches}, scale_grads={PP_SCALE_GRADS}"
    )
    log(
        f"  this rank runs stage(s) {my['stages']} = layers {_ranges(my['layers'])}; "
        f"loss lands on rank {plan['loss_rank']}"
    )
    if plan["style"] == "v":
        colocated = sum(
            1
            for s in range(plan["num_stages"] - 1)
            if plan["stage_to_rank"][s] == plan["stage_to_rank"][s + 1]
        )
        log(
            f"  V layout: {colocated} of {plan['num_stages'] - 1} stage boundaries are "
            f"co-located and skip send/recv entirely"
        )
    return schedule, mods, step_kw


# Re-enables the refused (deadlocking) schedules so a stack can be taken without editing this
# file mid-measurement. Never set it in anything a user runs.
_ALLOW_REFUSED = os.environ.get("SPARK_PP_DIAGNOSE", "0") == "1"


ONEF1B_REFUSAL = (
    "--schedule 1f1b is DISABLED on --pp-backend legacy. Use the default backend:\n"
    "  `--schedule 1f1b` with --pp-backend torch runs upstream's Schedule1F1B, which does\n"
    "  not have this defect. What follows is the record of the hand-written one.\n"
    "  Measured 2026-09-03 on two DGX Sparks: M=8, zero steps in 300 s, killed by timeout,\n"
    "  against gpipe completing 20 steps in 25.3 s on the same pair minutes earlier.\n"
    "  Two rewrites were tried. Both passed every CPU (gloo) gradient check and both passed\n"
    "  an offline op-ordering model; both still hung. 1F1B is the only schedule here that\n"
    "  REQUIRES a send and a receive in flight on the same rank pair at once -- that is what\n"
    "  the schedule is -- so it is the only one that depends on batched point-to-point\n"
    "  groups, and that path has never once been observed to work on this stack.\n"
    "  Use --schedule gpipe (3234 tok/s at M=8) or --schedule zerobubble (2577 tok/s,\n"
    "  measured working). See the root-cause note above `_p2p_group`."
)


INTERLEAVED_REFUSAL = (
    "--schedule interleaved is DISABLED on --pp-backend legacy. Use the default backend:\n"
    "  `--schedule interleaved` with --pp-backend torch runs upstream's\n"
    "  ScheduleInterleaved1F1B, which neither deadlocks nor drops gradients.\n"
    "  What follows is the record of the hand-written one.\n"
    "  Measured 2026-09-03 on two DGX Sparks: v=2, M=8, zero steps in 300 s, killed by\n"
    "  timeout. Two separate attempts to fix the point-to-point op ordering both passed\n"
    "  every CPU (gloo) test and both still hung on NCCL, so the schedule is refused here\n"
    "  rather than left to hang -- a 300 s silence is indistinguishable from broken\n"
    "  hardware, and that is a worse failure than a missing feature.\n"
    "  Use --schedule gpipe. It is measured working: 3234 tok/s at M=8 on the same pair.\n"
    "  See the root-cause note above `_p2p_group` in this file for what is still unknown."
)


SCHEDULES = {
    "gpipe": run_gpipe,
    "1f1b": run_1f1b,
    "interleaved": run_interleaved,
    "zerobubble": run_zerobubble,
}

# The union of both backends' names; an upstream-only layout under `--pp-backend legacy` is
# refused by name rather than ignored.
SCHEDULE_CHOICES = sorted(set(SCHEDULES) | set(TORCH_PP_SCHEDULES))
PP_BACKENDS = ("torch", "legacy")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog = "spark_pipeline", description = "Layer-split finetuning across DGX Sparks."
    )
    p.add_argument("--model", required = True)
    p.add_argument("--steps", type = int, default = 20)
    p.add_argument("--batch", type = int, default = 8, help = "global batch per step")
    p.add_argument("--microbatches", type = int, default = 4)
    p.add_argument("--seq", type = int, default = 512)
    p.add_argument("--lr", type = float, default = 1e-4)
    p.add_argument("--schedule", choices = SCHEDULE_CHOICES, default = "gpipe")
    p.add_argument(
        "--pp-backend",
        choices = PP_BACKENDS,
        default = "torch",
        help = "'torch' drives torch.distributed.pipelining (default): it fixes "
        "the interleaved deadlock and the missing gradients, because its "
        "send/recv order is derived by a simulation both ranks run and "
        "its backward is explicit rather than one loss.backward() "
        "spanning the rank cut. 'legacy' runs the hand-written schedules "
        "in this file, so a regression is one flag away from isolation.",
    )
    p.add_argument(
        "--virtual-stages",
        type = int,
        default = 2,
        help = "pipeline stages per device for the interleaved/looped "
        "schedules. Shrinks the fill/drain bubble as 1/(v*M+1) instead "
        "of 1/(M+1), at the cost of 2v-1 wire crossings per microbatch "
        "(cheap on this link). Forced to 2 for the V layouts (zbv, "
        "dualpipev), which require exactly two stages per rank.",
    )
    p.add_argument(
        "--full-finetune",
        action = "store_true",
        help = "train every parameter instead of LoRA adapters",
    )
    p.add_argument("--lora-r", type = int, default = 16)
    p.add_argument(
        "--grad-checkpoint",
        action = "store_true",
        help = "recompute activations in the backward pass. Trades ~30% step "
        "time for a large activation-memory saving, which is what lets "
        "microbatches be big enough to stay compute-bound",
    )
    p.add_argument(
        "--shard-load",
        action = "store_true",
        help = "load only this stage's tensors; required for models larger than one Spark",
    )
    p.add_argument(
        "--data", default = None, help = "jsonl with {q, a} rows; random token ids if omitted"
    )
    p.add_argument("--save", default = None, help = "directory to save this stage into")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # Refuse impossible combinations before a model loads or a collective is issued;
    # otherwise they surface as a hang on the wire.
    if world < 2:
        raise SystemExit(
            f"spark_pipeline needs WORLD_SIZE >= 2 (got {world}); a one-stage pipeline is "
            f"just a single-node run. Use `unsloth train` instead."
        )
    if args.microbatches < 1:
        raise SystemExit(f"--microbatches must be >= 1 (got {args.microbatches})")
    if args.virtual_stages < 1:
        raise SystemExit(f"--virtual-stages must be >= 1 (got {args.virtual_stages})")
    use_torch_pp = args.pp_backend == "torch"
    # Fail before the tokenizer and model load, so the reason appears in a second instead of
    # a silent process. The refusals apply to the LEGACY backend only: the same schedule
    # names work under the torch backend, so refusing them outright would refuse a working
    # configuration.
    if not use_torch_pp:
        if args.schedule not in SCHEDULES:
            raise SystemExit(
                f"--schedule {args.schedule} has no hand-written implementation here; it "
                f"exists only on --pp-backend torch. Legacy schedules: "
                f"{sorted(SCHEDULES)}"
            )
        if args.schedule == "interleaved" and not _ALLOW_REFUSED:
            raise SystemExit(INTERLEAVED_REFUSAL)
        if args.schedule == "1f1b" and not _ALLOW_REFUSED:
            raise SystemExit(ONEF1B_REFUSAL)
        if _ALLOW_REFUSED and args.schedule in ("1f1b", "interleaved"):
            # print, not log(): `log` is defined below, and calling it here raised
            # UnboundLocalError, killing the diagnostic run this warning announces.
            print(
                f"[spark-pp] SPARK_PP_DIAGNOSE=1: running the REFUSED legacy schedule "
                f"{args.schedule!r}. This deadlocks on hardware; it is enabled only for "
                f"diagnosis. Pair it with SPARK_PP_TRACE=1 to see where it stops.",
                flush = True,
            )
    elif not _dist_pipelining_available():
        raise SystemExit(
            "--pp-backend torch needs torch.distributed.pipelining, which this torch does "
            "not provide (or torch.distributed is unavailable, as on a default macOS "
            "build). Re-run with --pp-backend legacy --schedule gpipe."
        )
    # NCCL cannot do point-to-point between two processes on the SAME device, so a one-box
    # functional test is impossible on CUDA; this gloo path exists for that.
    use_cpu = os.environ.get("SPARK_PP_CPU", "0") == "1"
    if use_cpu:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
        device = torch.device("cuda")
        dtype = torch.bfloat16

    def log(msg):
        print(f"[spark-pp {rank}/{world}] {msg}", flush = True)

    log(f"host={os.uname().nodename} schedule={args.schedule} backend={args.pp_backend}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    plan = my_plan = None
    if use_torch_pp:
        # Every rank runs the same pure function on the same arguments, so the layout agrees
        # across the cluster without a collective and nothing is negotiated on the wire.
        from transformers import AutoConfig

        n_layers = config_num_layers(AutoConfig.from_pretrained(args.model))
        try:
            plan = torch_pp_plan(
                args.schedule, world, args.microbatches, args.virtual_stages, n_layers
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        my_plan = plan_for_rank(plan, rank)

    # Multi-stage layouts own non-contiguous chunks, so the contiguous drop-to-Identity would
    # remove layers this rank needs; the legacy interleaved path has no such set and keeps
    # the whole stack instead.
    model, cfg, _ = build_stage_model(
        args.model,
        rank,
        world,
        device,
        shard_load = args.shard_load,
        dtype = dtype,
        log = log,
        keep_all_layers = (not use_torch_pp and args.schedule == "interleaved"),
        keep_layers = my_plan["layers"] if my_plan else None,
        keep_embed = my_plan["keep_embed"] if my_plan else None,
        keep_head = my_plan["keep_head"] if my_plan else None,
    )
    stage_model_ref = [model]

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(
            model,
            LoraConfig(
                r = args.lora_r,
                lora_alpha = args.lora_r,
                lora_dropout = 0.0,
                bias = "none",
                task_type = "CAUSAL_LM",
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    if args.grad_checkpoint and use_torch_pp:
        # `_PPStageModule` calls the decoder layers directly, so transformers'
        # `gradient_checkpointing_enable()` is consulted in a `forward` never reached here
        # and would be inert; the stage module wraps each layer itself.
        log("gradient checkpointing enabled per decoder layer (use_reentrant=False)")
    elif args.grad_checkpoint:
        base_model = getattr(model, "base_model", model)
        inner_model = getattr(base_model, "model", base_model)
        if hasattr(inner_model, "gradient_checkpointing_enable"):
            inner_model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        # use_reentrant=False: the reentrant version drops the grad_fn the p2p-boundary
        # activation-gradient handoff depends on.
        if hasattr(inner_model, "gradient_checkpointing_kwargs"):
            inner_model.gradient_checkpointing_kwargs = {"use_reentrant": False}
        log("gradient checkpointing enabled (use_reentrant=False)")

    model.to(device)  # shard-load already placed the base; this catches new adapters
    if not use_cpu:
        torch.cuda.empty_cache()

    trainable = [p for p in model.parameters() if p.requires_grad]
    # `torch.cuda.memory_allocated()` raises without a CUDA device, and the gloo path is how
    # the schedules are unit-tested.
    resident = f"{torch.cuda.memory_allocated()/2**30:.2f} GiB" if not use_cpu else "cpu"
    log(
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f} B params resident "
        f"({resident}), "
        f"{sum(p.numel() for p in trainable)/1e6:.1f} M trainable"
    )

    if args.batch % args.microbatches:
        raise SystemExit("--batch must be divisible by --microbatches")
    mb_rows = args.batch // args.microbatches

    # A split only pays while the work is COMPUTE-bound. Below the GB10 roofline crossover a
    # step is limited by weight traffic, and the two stages read their halves sequentially for
    # the same microbatch, so total bytes per step is unchanged and the split cannot help.
    # Too many microbatches therefore fills the pipeline while starving each one. Warn rather
    # than override: a user may be trading throughput for memory deliberately.
    ROOFLINE_CROSSOVER_TOKENS = 436
    mb_tokens = mb_rows * args.seq
    if mb_tokens < ROOFLINE_CROSSOVER_TOKENS:
        log(
            f"WARNING: each microbatch is {mb_tokens} tokens, below the ~"
            f"{ROOFLINE_CROSSOVER_TOKENS}-token compute/bandwidth crossover on this "
            f"hardware."
        )
        log(
            f"         At this size the step is memory-bound and the layer split cannot "
            f"speed it up."
        )
        log(
            f"         Raise --batch, raise --seq, or lower --microbatches "
            f"(currently {args.microbatches})."
        )

    # Fewer microbatches than stages never fills the pipeline: correct, just wasteful.
    if args.microbatches < world:
        log(
            f"WARNING: --microbatches ({args.microbatches}) is below the pipeline depth "
            f"({world}); the pipeline never fills and at best "
            f"{args.microbatches/(args.microbatches + world - 1):.0%} of the devices are "
            f"busy. Use at least {world} microbatches, ideally {4*world}."
        )

    pp_schedule = pp_step_kw = None
    if use_torch_pp:
        pp_schedule, _pp_mods, pp_step_kw = build_torch_schedule(
            model,
            plan,
            my_plan,
            microbatches = args.microbatches,
            device = device,
            grad_checkpoint = args.grad_checkpoint,
            log = log,
        )
        stage = None
        is_loss_rank = rank == plan["loss_rank"]
    elif args.schedule == "interleaved":
        owner, layers_mod = find_layers(stage_model_ref[0])
        n_layers = len(layers_mod)
        v = args.virtual_stages
        my_chunks_layers = interleaved_layers(n_layers, rank, world, v)
        my_chunk_ids = [rank + k * world for k in range(v)]
        stage = _Stage(
            model,
            cfg,
            rank,
            world,
            device,
            dtype,
            args.microbatches,
            chunks = my_chunk_ids,
            n_chunks = world * v,
        )
        stage.chunk_layers = dict(zip(my_chunk_ids, my_chunks_layers))
        log(
            f"interleaved: v={v}, {world * v} chunks, this rank owns "
            f"{[(c, (l[0], l[-1])) for c, l in stage.chunk_layers.items()]}"
        )
        log(
            f"bubble ~1/({v}*M+1); ideal speedup at M={args.microbatches} is "
            f"{world * (1 - 1/(v * args.microbatches + 1)):.2f}x"
        )
    else:
        stage = _Stage(model, cfg, rank, world, device, dtype, args.microbatches)
    if not use_torch_pp:
        is_loss_rank = stage.is_last
    opt = torch.optim.AdamW(trainable, lr = args.lr)

    torch.manual_seed(3407)
    need = args.batch * args.steps
    if args.data:
        rows = [json.loads(l) for l in open(args.data)]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}],
                tokenize = False,
            )
            for r in rows
        ]
        enc = tok(
            texts, return_tensors = "pt", padding = "max_length", truncation = True, max_length = args.seq
        ).input_ids
        ids_all = enc.repeat((need + len(enc) - 1) // len(enc), 1)[:need].to(device)
    else:
        ids_all = torch.randint(0, tok.vocab_size, (need, args.seq), device = device)

    posid = torch.arange(args.seq, device = device).unsqueeze(0).expand(mb_rows, -1)
    schedule = SCHEDULES[args.schedule] if not use_torch_pp else None

    dist.barrier()
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none = True)
        if use_torch_pp:
            # Upstream chunks the whole batch; that chunking must agree with `target`'s, and
            # one implementation owning both is how they stay agreed.
            whole = ids_all[step * args.batch : (step + 1) * args.batch]
            losses = [] if is_loss_rank else None
            # Only the rank holding stage 0 may supply positional inputs; every other stage's
            # input is the wire. `target` goes to every rank but is read only by the one
            # computing the loss, which under a V layout is rank 0, not the last rank.
            step_args = (whole,) if rank == plan["first_rank"] else ()
            pp_schedule.step(*step_args, target = whole, losses = losses, **pp_step_kw)
            # One mean-reduced loss per microbatch, so the step loss is their mean. That
            # equals what the legacy schedules return, keeping the two backends comparable.
            loss = (sum(losses) / len(losses)) if losses else None
        else:
            batches = [
                ids_all[step * args.batch + m * mb_rows :][:mb_rows]
                for m in range(args.microbatches)
            ]
            if step == 0:
                warmup_p2p(stage, dist, torch)
            loss = schedule(stage, batches, posid, mb_rows, dist, torch)
        opt.step()
        # `is_loss_rank`, not just "loss is not None": the legacy schedules return a zeroed
        # accumulator on every rank, which printed a confident `loss=0.0000` next to the real one.
        if is_loss_rank and loss is not None and ((step + 1) % 5 == 0 or args.steps <= 10):
            # The ONE sync per step, and only when a line is actually printed.
            value = loss.item() if hasattr(loss, "item") else float(loss)
            log(f"step {step+1}/{args.steps} loss={value:.4f}")

    dist.barrier()
    elapsed = time.perf_counter() - t0
    if is_loss_rank:
        # Under a V layout this lands on rank 0, not the last rank; keying off `stage.is_last`
        # printed nothing at all for DualPipeV, which reads exactly like a hang.
        toks = args.batch * args.seq * args.steps
        log(
            f"DONE {args.steps} steps in {elapsed:.1f}s | "
            f"{elapsed/args.steps:.2f}s/step | {toks/elapsed:.0f} tok/s"
        )
    if not use_cpu:
        log(f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    if _TIME_PHASES:
        log(
            f"phases: wait {PHASE['wait']:.1f}s | backward {PHASE['bwd']:.1f}s "
            f"| forward {PHASE['fwd']:.1f}s"
        )

    if args.save:
        out = osp.join(args.save, f"stage{rank}")
        os.makedirs(out, exist_ok = True)
        model.save_pretrained(out)
        log(f"saved stage {rank} to {out}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
