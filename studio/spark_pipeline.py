# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-split (pipeline-parallel) finetuning across two or more NVIDIA DGX Sparks.

Two Sparks are two machines, not one two-GPU machine: `torch.cuda.device_count()` is 1 on
each and NVLink-C2C never leaves the package. So "use both" means two processes over the
ConnectX RoCE link, and the question is only what crosses it.

Why layer splitting: **capacity**. Llama-3.3-70B is 132 GiB in bf16 against 121.69 GiB of
usable memory on one Spark, so it cannot be trained on a single machine at all. Split across
two it trains, using ~73 GiB per node. No amount of interconnect bandwidth changes that --
the weights simply do not fit.

What crosses the wire here is only the hidden state at the stage boundary,
`hidden_size * 2` bytes per token (order 10-16 KiB), measured at ~0.2 s of a 27 s run. So
layer splitting is also indifferent to link speed, which is a useful property but is no
longer the reason to choose it.

A correction worth recording, because it was load-bearing in earlier versions of this file:
these Sparks were measured at ~3 GB/s of NCCL bandwidth and that was used to argue FSDP was
hopeless (~65 GiB of shards per step for a 70B). **That 3 GB/s was a hardware fault**, not a
platform limit -- a full power cycle (a reboot is not enough) restored 21.6 GB/s, 88% of the
24.5 GB/s raw RDMA ceiling. `unsloth spark doctor` detects it. FSDP across two Sparks is
therefore viable, and should be compared on its merits rather than dismissed.

Stage `s` of `W` owns a contiguous slice of decoder layers. Stage 0 additionally owns the
embedding; the last stage owns the final norm and lm_head and computes the loss.
Activations flow forward, activation gradients flow back.

WHAT DRIVES THE PIPELINE. Two backends, selected with `--pp-backend`:

  torch   (default) `torch.distributed.pipelining`, which ships inside the torch we already
          depend on (2.11.0+cu130). It owns the send/recv order and the backward pass. This
          is not a preference: the hand-written `interleaved` schedule below deadlocks AND
          computed wrong gradients, for one reason -- we tried to make a single
          `loss.backward()` span the rank cut, and there is no autograd edge across a
          `dist.irecv`. Upstream does an explicit grad send/recv with a locally-rooted
          backward instead, so the defect cannot occur. See the long note above
          `build_torch_schedule`.

  legacy  the four hand-written schedules in this file. Kept reachable so that a regression
          in the upstream path is one flag away from being isolated, not a git bisect.

This module is imported only when someone actually asks for a layer-split run, so it costs
nothing on any other platform. Everything here is plain PyTorch + transformers + peft, and
every upstream API is reached through `hasattr`/`inspect` feature detection rather than a
version comparison -- torch 2.11 has no `get_mesh=` on `PipelineStage`, later torch does,
and both must work.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import time
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Layer-container discovery
# ---------------------------------------------------------------------------

# Different architectures nest the decoder stack differently. Rather than hardcode
# `model.model.layers`, walk the small set of shapes transformers actually uses. Getting
# this wrong is silent -- the split would "work" and train the wrong parameters -- so it
# raises instead of guessing.
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
    """Return `(container_owner, layers_module_list)` for the decoder stack."""
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
    """Layer chunks for INTERLEAVED pipeline parallelism (Megatron's schedule).

    A plain 2-stage pipeline cannot beat `world * M/(M+1)` -- the fill/drain bubble costs one
    stage-time no matter how many microbatches you use, which caps two Sparks near 1.8x in
    practice. Interleaving fixes that by giving each device `virtual` NON-CONTIGUOUS chunks,
    so the bubble shrinks as `1/(virtual*M + 1)` instead of `1/(M + 1)`:

        v=1, M=8  -> 1.78x        v=2, M=8  -> 1.88x
        v=1, M=16 -> 1.88x        v=2, M=16 -> 1.94x        v=4, M=8 -> 1.94x

    The cost is that activations cross the wire `2*virtual - 1` times per microbatch instead
    of once. That is a bad trade on a slow link and a good one here: each crossing is
    `hidden_size * 2` bytes per token (10-16 KiB), against 21.6 GB/s.

    With `world=2, virtual=2` and 24 layers the chunks are
    `rank0 -> [[0..5], [12..17]]`, `rank1 -> [[6..11], [18..23]]`, and a microbatch visits
    chunk 0 (r0) -> 1 (r1) -> 2 (r0) -> 3 (r1).
    """
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
    """Contiguous, balanced slice of layer indices for this stage.

    Balanced by *count*, which is the right proxy only when layers are homogeneous. It is
    for every dense decoder we target; an MoE model with some dense and some sparse layers
    would want a cost-weighted split instead.
    """
    base, extra = divmod(n_layers, world)
    start = rank * base + min(rank, extra)
    return list(range(start, start + base + (1 if rank < extra else 0)))


def _ranges(ids: Sequence[int]) -> str:
    """`[0,1,2,7,8]` -> `"0-2,7-8"`. Layer sets stopped being contiguous when a rank
    started owning more than one stage, and printing 40 integers helps nobody."""
    out, ids = [], sorted(ids)
    for i in ids:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


# ---------------------------------------------------------------------------
# Where each pipeline stage lives
# ---------------------------------------------------------------------------
# Two layouts, and the difference between them is the whole reason DualPipeV is here.
#
#   loop   stage i -> rank i % world.   pp=2, 4 stages: {0:0, 1:1, 2:0, 3:1}
#          Every hop 0->1->2->3 changes rank, so a microbatch crosses the wire 3 times
#          forward and 3 times back.
#
#   v      the stage index walks out to the last rank and back.
#          pp=2, 4 stages: {0:0, 1:1, 2:1, 3:0}
#          Hop 1->2 is rank1->rank1 and hop 3->(loss) never leaves rank 0. Upstream skips
#          send/recv entirely for a co-located hop (see the "[Note: V-schedule special
#          case]" comments in torch/distributed/pipelining/schedules.py), so only 2 of the
#          4 boundaries cross the ConnectX link instead of 3 of 4 -- roughly half the
#          inter-node traffic, which matters on a 21 GB/s link with no GPUDirect RDMA.
#
# This is a pure duplicate of upstream's `generate_stage_to_rank_mapping`. It is duplicated
# rather than imported so the plan can be computed and unit-tested with no torch present
# (this module must stay importable on a Mac). `torch_pp_plan` cross-checks the two at
# runtime and refuses to run if they ever disagree -- a silent disagreement would place
# layers on the wrong node and train the wrong parameters.


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
                continue  # at the fold, stay put -- that is what makes the V
            r += 1 if (i // world) % 2 == 0 else -1
        return mapping
    raise RuntimeError(f"unknown pipeline layout {style!r}")


# our --schedule name -> (upstream class name for get_schedule_class,
#                         stages per rank or None = take it from --virtual-stages,
#                         layout)
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
    """Resolve `--schedule` into a concrete stage/layer/rank assignment.

    Pure: no torch, no process group, no model. Everything that can be refused is refused
    here, before a single tensor is allocated, because the alternative on this cluster is a
    300 s silence that is indistinguishable from broken hardware.
    """
    if schedule not in TORCH_PP_SCHEDULES:
        raise RuntimeError(
            f"--schedule {schedule!r} has no torch.distributed.pipelining equivalent; "
            f"available: {sorted(TORCH_PP_SCHEDULES)}. Use --pp-backend legacy for the "
            f"hand-written schedules."
        )
    class_name, fixed_v, style = TORCH_PP_SCHEDULES[schedule]
    v = fixed_v if fixed_v is not None else virtual_stages
    if fixed_v is not None and virtual_stages != fixed_v and virtual_stages != 2:
        # --virtual-stages defaults to 2, so only complain when the user actually chose a
        # value this schedule cannot honour.
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
        # Enforced by ScheduleDualPipeV itself; caught here so the message arrives in a
        # second rather than after the model loads.
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
    """The slice of a plan that one rank has to act on."""
    mine = [i for i, r in sorted(plan["stage_to_rank"].items()) if r == rank]
    if not mine:
        raise RuntimeError(f"rank {rank} owns no pipeline stage under {plan['schedule']!r}")
    return {
        "stages": mine,
        "layers": sorted(i for s in mine for i in plan["stage_layers"][s]),
        "keep_embed": 0 in mine,
        "keep_head": (plan["num_stages"] - 1) in mine,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


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
    """Build only this stage's slice of the model, on `device`.

    With `shard_load` the skeleton is created on the meta device (which allocates nothing),
    every layer this stage does not own is replaced by Identity, and then only the
    remaining tensors are read out of the safetensors shards. That is what makes a model
    larger than one Spark loadable: materialising the whole thing and dropping half needs
    more memory than the node has.

    `keep_layers` overrides the contiguous one-stage-per-rank slice. It exists for the
    multi-stage layouts (interleaved and V), where a rank owns two NON-CONTIGUOUS chunks
    and the contiguous slice would drop layers this rank needs. `keep_embed`/`keep_head`
    likewise override "embedding on rank 0, head on the last rank": under DualPipeV's V
    layout rank 0 owns both the first and the last stage, so it needs BOTH, and rank 1
    needs neither. Defaulting them to None reproduces the contiguous behaviour exactly, so
    the single-stage path is unchanged.
    """
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
    """Read this stage's tensors straight from the safetensors shards onto the GPU."""
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
                    # One tensor at a time, straight to the device. Reading the whole half
                    # into host memory first and then calling .to(cuda) needs TWO copies of
                    # ~70 GiB for a 70B, and the OOM killer ends the process with no Python
                    # traceback -- the peer just sees a broken pipe.
                    loaded[k] = sf.get_tensor(k).to(dtype).to(device, non_blocking = False)
                    seen += 1
    model.load_state_dict(loaded, strict = False, assign = True)

    # Non-persistent buffers (rotary inv_freq, causal masks) are computed in __init__ and
    # never stored in safetensors, so load_state_dict leaves them on meta. Checking only
    # named_parameters() hides this; it resurfaces much later as
    # "NotImplementedError: Cannot copy out of meta tensor" on the first real use.
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


# ---------------------------------------------------------------------------
# Pipeline schedules -- LEGACY (--pp-backend legacy)
# ---------------------------------------------------------------------------
# Everything from here down to `run_zerobubble` is the hand-written implementation. It is
# no longer the default; `torch.distributed.pipelining` is (see `build_torch_schedule`).
# It is retained, unchanged, as the control arm: if the upstream backend ever regresses,
# `--pp-backend legacy --schedule gpipe` reproduces the previously measured behaviour
# without reverting anything.


class _Stage:
    """One pipeline stage's forward/backward, independent of the schedule driving it."""

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
        # Interleaved mode: `chunks` are the global chunk indices this rank owns, and
        # `chunk_layers[c]` the layer indices in chunk c. Contiguous mode leaves both None
        # and uses the single-slice path.
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
        """Run one interleaved chunk. Only the FIRST chunk embeds and only the LAST
        produces logits and a loss; everything between is pure residual-stream work."""
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
    """Force both point-to-point NCCL communicators to exist before the schedule runs.

    NCCL builds a separate 2-rank communicator for each *direction* on first use, and that
    construction is COLLECTIVE -- `irecv` blocks, spinning, until the peer posts a matching
    op on the same direction. Despite the name, an unmatched `irecv` is not asynchronous at
    all the first time a direction is used.

    That deadlocks 1F1B, and this is the exact failure it caused (located with py-spy):

        rank0: isend(h0) -> ok
               irecv(g0) -> BLOCKS building the 1->0 communicator
        rank1: recv h0, forward, then wait for h1 -- which rank0 never sends,
               because rank0 is stuck, and rank1's first 1->0 send only happens
               after that forward completes. Circular wait.

    GPipe survives by accident: it issues every send before any gradient receive, so both
    ranks reach the 1->0 direction at about the same time.

    A one-element exchange in each direction, ordered so the ranks mirror each other, builds
    both communicators up front and costs microseconds.

    CORRECTION (see the root-cause note above `_p2p_group`): the "separate communicator
    per direction" explanation above is wrong. There is one p2p channel per rank PAIR and
    it is ordered, so the original deadlock was an op-ordering deadlock, not a communicator
    construction one. This function survives because its exchange happens to be mirrored --
    which is the property that actually matters -- and it is retained because a mirrored
    warmup is harmless and pins the channel open.

    This is correct for any `world`, not just two, and the ordering is load-bearing. Every
    rank does its DOWNSTREAM pair first (send then recv with rank+1), then its UPSTREAM pair
    (recv then send with rank-1). Walk world=3: rank 2 posts recv<-1 before anything, so
    rank 1's send->2 matches; rank 1 then recv<-2 matches rank 2's send->1; only then does
    rank 1 reach recv<-0, which matches rank 0's send->1 that has been waiting. No rank ever
    blocks on a peer that is itself blocked on it, because the pairs are resolved from the
    tail of the pipeline backwards. Swapping the two blocks, or making either pair
    send-then-send, reintroduces the circular wait.
    """
    tiny = torch.zeros(1, dtype = stage.dtype, device = stage.device)
    if not stage.is_last:
        dist.send(tiny, dst = stage.rank + 1)
        dist.recv(tiny, src = stage.rank + 1)
    if not stage.is_first:
        dist.recv(tiny, src = stage.rank - 1)
        dist.send(tiny, dst = stage.rank - 1)


# ---------------------------------------------------------------------------
# ROOT CAUSE of the interleaved/zerobubble first-step hangs and (probably) 1F1B's 26x
# ---------------------------------------------------------------------------
# Three schedules failed and one worked, and the property that separates them is the
# ORDER each rank issues its point-to-point ops in relative to its peer.
#
# PyTorch documents the rule directly, on `batch_isend_irecv`:
#
#     "All operations in the list are treated as a single batch -- the relative ordering
#      of sends vs. receives in the list does not matter and WILL NOT CAUSE DEADLOCKS."
#
# The promise is only made for the batched form. Un-batched `isend`/`irecv` between the
# same pair of ranks are issued onto one ordered p2p stream, so an op enqueued behind a
# receive that has no matching send yet never launches. Concretely, at world=2:
#
#   gpipe        rank0: [S h0 .. S h(M-1)] then [R g0 .. R g(M-1)]
#                rank1: [R h0 .. R h(M-1)] then [S g0 .. S g(M-1)]
#                Directions never interleave; the two sequences mirror. WORKS, every M.
#
#   zerobubble   rank0: [S h0, R g0, S h1, R g1, ...]   (grad recv posted INSIDE the
#                forward loop). `R g0` cannot complete until rank1 has consumed every
#                h, but `S h1` sits behind `R g0` and never launches. HANGS before step 1.
#
#   interleaved  every receive pre-posted BEFORE any send: maximally interleaved, so the
#                first send is queued behind M*v receives. HANGS before step 1.
#
#   1f1b         [S h_m, R g_m, S h_{m+1}, ...] -- the same interleave as zerobubble.
#
# This also corrects the story in `warmup_p2p`. There is no separate communicator per
# direction to build; the tiny mirrored exchange works because it *is* mirrored, which is
# the property that actually matters. The docstring there has been amended.
#
# And it explains why none of this reproduces on CPU: gloo gives every op an independent
# unbound buffer and a progress thread, so it has no such ordering constraint. A schedule
# can pass every gloo test and hang instantly on NCCL. Do not treat a green CPU run as
# evidence that a p2p ordering change is safe.
#
# The fix: wherever a rank needs a send and a receive in flight at the same time, issue
# them as ONE batched group -- AND make the group boundaries correspond across the pair.
#
# THE SECOND CONSTRAINT WAS LEARNED THE HARD WAY, by being wrong on hardware.
# A first fix batched sends with "whatever receive comes next" (accumulate, then flush).
# It passed every gloo test with exact gradients and passed a 180-config ordering
# simulation, and it STILL deadlocked on NCCL: interleaved v=2 M=8, zero steps in 300 s.
#
# What that simulation was missing: it treated a group as a bag of ops that could each be
# matched independently, so rank A's group could be satisfied by ops spread across TWO
# different groups on rank B. That silently assumes a group's send becomes visible to the
# peer while the same group's receive is still outstanding. It does not. A batched group
# is ATOMIC: it rendezvouses as a unit, so if A issues {send->B, recv<-B} then B must issue
# the mirror {recv<-A, send->A} as ONE group at the same step. Split either side into two
# groups and it deadlocks even though the per-direction op order still matches perfectly.
#
# The corrected model (a standalone rendezvous simulator, not shipped) requires MUTUAL readiness
# at a fixed point -- a group completes only if every group it rendezvouses with also
# completes -- and it reproduces the hardware:
#
#     schedule              world  M   result
#     gpipe                     2  8   completes     <- measured working, 3234 tok/s
#     zerobubble  (fixed)       2  8   completes
#     1f1b        (fixed)       2  8   completes
#     interleaved (1st fix)     2  8   DEADLOCK      <- measured hanging, 300 s, 0 steps
#     1f1b        (1st fix)     2  8   DEADLOCK
#
# THAT MODEL IS ALSO FALSIFIED. It passed a Megatron-structured 1F1B across 105
# configurations; that exact code deadlocked at world=2, M=8 on hardware. Two offline
# models and one CPU gradient suite have now each certified code that hangs on this fabric.
# The standing rule is therefore: a simulation may REJECT a schedule, never certify one.
#
# What the hardware actually supports, stated as weakly as the evidence allows:
#
#   WORKS  gpipe      (measured 3234 tok/s, M=8)   never needs a send and a receive in
#   WORKS  zerobubble (measured 2577 tok/s, M=8)   flight at once: every group is a lone
#                                                  op, phase-separated per direction.
#   HANGS  1f1b       (2 attempts)                 requires concurrent send+recv on one
#   HANGS  interleaved(2 attempts)                 pair, so depends on batched groups.
#
# Zero-bubble is NOT evidence that carefully-aligned batched groups work. It is evidence
# that AVOIDING concurrent bidirectional traffic works -- its comms are gpipe's, only its
# compute differs. Every schedule satisfying that weaker rule runs; every schedule relying
# on `batch_isend_irecv` has hung, four times, however the group boundaries were aligned.
# There is currently ZERO hardware evidence that batched p2p groups function on this stack
# at all. Until a standalone probe shows otherwise, treat `_p2p_group` as unproven and do
# not build a schedule on it.
#
# `--schedule interleaved` is refused outright (see INTERLEAVED_REFUSAL). Two attempts to
# order its exchanges correctly both passed CPU and both hung on NCCL; a schedule whose
# failure mode is a 300 s silence is worse than a schedule that is missing.


def _p2p_group(dist, ops):
    """Issue point-to-point ops as a single batched group, and wait for them.

    `ops` is a list of `dist.P2POp`. Sends and receives inside one group cannot deadlock
    each other; across groups only SAME-DIRECTION order matters, and every schedule here
    keeps that ascending in (microbatch, chunk). Passing an empty list is a no-op, which
    is why the callers can accumulate sends and flush them with whatever receive comes
    next instead of flushing eagerly.
    """
    if not ops:
        return
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def _check_schedule(name, stage, batches):
    """Refuse a configuration the schedule cannot honour, loudly and early.

    The failure modes these guard against are the quiet ones: a 1-stage "pipeline" that
    sends to itself, an empty microbatch list that makes every loop body dead code, or an
    interleaved chunk map that does not match the rank it is running on. All of those
    otherwise produce a run that completes and trains the wrong thing.
    """
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
    """All forwards, then all backwards.

    Blocking `send`/`recv` match in the order each rank issues them, so the stages must
    mirror each other: a stage that sends M times then receives M times requires its
    neighbour to receive M times then send M times. Interleaving on one side only deadlocks
    the communicator.
    """
    _check_schedule("gpipe", stage, batches)
    acts, held = [], []
    total = torch.zeros((), device = stage.device)
    # Pre-post the hidden-state receives here too, for the same reason as 1F1B: a blocking
    # `recv` issued only when the value is wanted forces the sender to wait for the receiver
    # to arrive, which serialises the two stages. GPipe's ordering constraint (all sends then
    # all receives, mirrored between neighbours) is preserved -- pre-posting changes when the
    # buffer is made available, not the order operations are matched in.
    hidden_bufs, hreq = {}, {}
    if not stage.is_first:
        for m in range(len(batches)):
            hidden_bufs[m] = torch.empty(
                mb_rows, posid.shape[1], stage.hidden, dtype = stage.dtype, device = stage.device
            )
            hreq[m] = dist.irecv(hidden_bufs[m], src = stage.rank - 1)

    for m, ids in enumerate(batches):  # forwards
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

    # Gradient receives are pre-posted for the whole backward pass before any of it runs, so
    # the neighbour's sends land as they are produced rather than waiting for us to ask.
    gbufs, gr = {}, {}
    if not stage.is_last:
        for m in range(len(batches)):
            gbufs[m] = torch.empty_like(acts[m][0])
            gr[m] = dist.irecv(gbufs[m], src = stage.rank + 1)

    for m in range(len(batches)):  # backwards
        h, hidden = acts[m]
        if stage.is_last:
            held[m].backward()
            # Accumulate on the DEVICE. `.item()` here would sync once per microbatch,
            # stalling the pipeline it is supposed to be measuring -- 8 syncs per step at
            # 8 microbatches. Read it once, after the step.
            total = total + held[m].detach()
        else:
            gr[m].wait()
            h.backward(gbufs[m])
            gbufs[m] = None
        if not stage.is_first:
            dist.send(hidden.grad.contiguous(), dst = stage.rank - 1)
    return total


def run_1f1b(stage, batches, posid, mb_rows, dist, torch):
    """One forward, one backward, with a per-rank warmup depth. CURRENTLY REFUSED.

    STATUS: this schedule DEADLOCKS on hardware and is disabled (see ONEF1B_REFUSAL).
    Set SPARK_PP_DIAGNOSE=1 to run it anyway, and SPARK_PP_TRACE=1 to see where it stops.

    Structure: stage `r` holds `world - 1 - r` forwards in flight before its first backward,
    because microbatch m's activation gradient cannot return until m has crossed the
    remaining `world - 1 - r` stages forward and back. Warmup and cooldown issue lone p2p
    ops; the steady state issues `send_forward_recv_backward` on stage r against the
    group-for-group mirror `send_backward_recv_forward` on stage r+1.

    ------------------------------------------------------------------------------------
    WHAT IS KNOWN, AND WHAT IS STILL OPEN. Read this before proposing a fix.
    ------------------------------------------------------------------------------------
    Four hypotheses have been chased and the first three were each a real property of the
    fabric that turned out NOT to be this bug. Recording them so nobody re-runs the chase:

      FIXED    The warmup depth used to be hardcoded to 1 on every rank, making the
               schedule cost a full pipeline round trip per microbatch. Real, and fixed.
      REAL BUT NOT THIS BUG. Point-to-point ops between a pair are matched in posting
               order, so a schedule whose two ranks disagree on op order deadlocks. This
               is what broke zerobubble, and moving its gradient receives out of the
               forward loop FIXED it on hardware (measured: 20 steps, 2577 tok/s).
      REAL BUT NOT THIS BUG. A batched group is atomic and its boundaries must correspond
               across the pair. True, and this schedule now honours it.
      FALSIFIED `batch_isend_irecv` is broken on this stack. It is not: a standalone probe
               runs the batched pattern in 80 ms.
      FALSIFIED The op sequence is wrong. It is not. A standalone two-node probe (not shipped),
               CASE5/6/7 replay THIS function's exact group sequence at real tensor shapes,
               with matmul and with autograd between groups, at M=2 and M=8 -- the same M
               that deadlocks in the trainer -- and completes in 0.17 s.

    So the defect is in the NON-COMMUNICATION logic: how activations are retained between
    forward and backward, what `do_backward` is handed, or an autograd wait rather than a
    network wait. Two offline models and one CPU gradient suite have each certified this
    code as correct while it hung, so the standing rule is: a simulation may reject a
    schedule, never certify one. The next step is a stack from a live hang, not more
    reading -- which is why SPARK_PP_TRACE exists.

    RETRACTED, do not cite: an earlier claim that pre-posting receives bought 1.47x
    (withdrawn as noise), and the "26x slower than gpipe" figure, which was measured on a
    version several rewrites old and has never been reproduced on the current one.
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

    # ---- the four exchange primitives, named as in Megatron ---------------------------
    # Group BOUNDARIES must correspond across a pair, not merely op order: a batched group
    # is atomic, so if rank r issues {send down, recv down} its downstream neighbour must
    # issue the mirror {send up, recv up} as ONE group at the same step. Splitting either
    # side into two groups deadlocks even though the per-direction op order still matches.
    # That is the constraint the first two attempts at this file missed.
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
        """Mirror of `send_backward_recv_forward` on the downstream stage."""
        if dn is None:
            return None
        g = torch.empty_like(t)
        _trace(stage, "p2p send_forward_recv_backward enter")
        _p2p_group(dist, [dist.P2POp(dist.isend, t, dn), dist.P2POp(dist.irecv, g, dn)])
        _trace(stage, "p2p send_forward_recv_backward done")
        return g

    def send_backward_recv_forward(t):
        """Mirror of `send_forward_recv_backward` on the upstream stage."""
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

    # ---- schedule ---------------------------------------------------------------------
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
            # An autograd wait and a missing gradient look identical from outside: both end
            # the step with no progress. Say which it is.
            raise RuntimeError(
                "1f1b: the received activation got no gradient from backward. Its graph "
                "does not reach this stage's input, so the upstream stage would train on "
                "nothing. This is the interleaved bug in blocking form."
            )
        grad = hidden.grad.contiguous() if hidden is not None else None
        return contribution, grad

    _trace(stage, f"START warm={warm} rem={rem} M={M}")
    for i in range(warm):  # warmup: forwards only
        _trace(stage, f"warmup F({i})")
        hidden = recv_forward()
        h, loss = stage.forward(batches[i], hidden, posid)
        send_forward(h.detach().contiguous())
        acts.append((h, hidden, loss))

    hidden = recv_forward() if rem > 0 else None
    for i in range(rem):  # steady state: 1F then 1B
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

    for c in range(warm):  # cooldown: backwards only
        _trace(stage, f"cooldown B [{c}/{warm}]")
        gout = _phase("wait", lambda: recv_backward(acts[0][0]), torch)
        contribution, grad = do_backward(gout)
        total = total + contribution
        send_backward(grad)
    _trace(stage, "DONE step")
    return total


# Per-phase timing, opt-in via SPARK_PP_TIME=1. Off by default because it cuda-syncs, which
# serialises exactly the overlap a pipeline exists to create -- an instrumented run is not a
# valid throughput measurement. It exists to answer "where does the time go", which is the
# open question for 1F1B: functionally correct, loss falls, but 7+ minutes against GPipe's
# 16 s for identical work, sitting in `_engine_run_backward`.
# Opt-in progress trace: SPARK_PP_TRACE=1. Prints to stderr, flushed, with NO cuda sync,
# so it does not perturb the overlap the way the phase timer does. Its only job is to make
# a hang name itself: the last line printed says which rank was in which operation. That is
# the difference between "0 steps in 300 s" and "rank 0 blocked entering B(3)".
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
    # The activation is dead once its backward has run. Dropping it here rather than at end
    # of step is what bounds 1F1B's memory to the pipeline depth instead of the microbatch
    # count -- the whole memory argument for 1F1B over GPipe.
    acts[m] = (None, None)
    held[m] = None
    return contribution


def run_interleaved(stage, batches, posid, mb_rows, dist, torch):
    """Interleaved (virtual-stage) pipeline parallelism.

    Each rank owns `v` non-contiguous chunks; chunk `c` lives on rank `c % world`, so a
    microbatch walks 0 -> 1 -> ... -> world*v-1, crossing the wire `2v-1` times forward and
    the same coming back. On this pair a crossing is 10-16 KiB against 21.6 GB/s, so the
    extra traffic is free and the shrunken fill/drain bubble is the gain.

    ------------------------------------------------------------------------------------
    TWO BUGS FIXED HERE. Both were silent -- the run completed and the loss fell.
    ------------------------------------------------------------------------------------
    (1) NO BACKWARD ACROSS THE CHUNK BOUNDARY. The previous version called
        `loss.backward()` on the final chunk and stopped. `backward()` only walks the
        autograd graph that exists in THIS process, and that graph ends at the received
        activation tensor -- there is no autograd edge across a `dist.irecv`. So the only
        parameters that got a gradient were those of the LAST chunk on the LAST rank.
        Every other chunk, and on `world=2, v=2` the whole of rank 0, stepped its optimiser
        on `None`/stale gradients. The old docstring's claim that "the backward pass ... is
        driven by the autograd graph the forwards built, so no separate ordering is needed"
        was false: the graph is per-process. A backward phase that receives the output
        gradient, runs `h.backward(gout)`, and sends `inp.grad` upstream is required, and is
        implemented below.

    (2) NONDETERMINISTIC RECEIVE MATCHING. Point-to-point ops are matched in POSTING ORDER
        per (src, dst) direction -- NCCL ignores tags entirely, so a tag cannot rescue this.
        Every chunk this rank sends goes to the same peer `(rank+1) % world`, and every
        chunk it receives comes from `(rank-1) % world`. The old code posted its receives
        chunk-major (`for c: for m:`) but emitted its sends in whatever order the
        data-driven loop happened to complete tasks -- roughly microbatch-major, and
        genuinely nondeterministic because a task was skipped when its input had not landed.
        At `world=2, v=2, M>=2` the second posted receive is `(m=1, chunk 1)` while the
        second send is `(m=0, chunk 2)`: one microbatch's activations land in another
        microbatch's buffer. Wrong numbers, no error.

        The fix is a canonical order that both sides agree on without communicating:
        forward traffic in ascending `(microbatch, chunk)`, backward traffic in ascending
        `(microbatch, -chunk)`. Sends are held in a queue and released only in that order,
        so the data-driven execution order no longer leaks onto the wire. Chunk `c` on a
        rank corresponds to chunk `c+1` on its successor, and that correspondence is
        order-preserving, so the two sides' orderings agree element for element.

        Note for `world == 2`: `(rank+1) % 2 == (rank-1) % 2`, so forward activations and
        backward gradients share ONE direction and therefore one FIFO. That is safe only
        because every rank completes its entire forward phase before starting its backward
        phase -- do not merge the two loops without re-deriving this.

    No deadlock: forward task `(m, c)` depends only on `(m, c-1)`, which is strictly earlier
    in the canonical forward order, and backward task `(m, c)` only on `(m, c+1)`, strictly
    earlier in the canonical backward order. Both dependency graphs are acyclic and the send
    queues drain in an order consistent with them.

    Cost of correctness: all `M * v` local graphs stay alive between the two phases, so this
    is GPipe-shaped in memory, not 1F1B-shaped. The `world*(1 - 1/(v*M+1))` bubble figure
    printed at startup is the ceiling for a true interleaved 1F1B ordering; this
    all-forward-then-all-backward variant will land below it.

    ------------------------------------------------------------------------------------
    WHAT WAS ACTUALLY REPRODUCED, and the CPU/GPU asymmetry that hides these bugs
    ------------------------------------------------------------------------------------
    Bug (1) was reproduced directly on 2 gloo processes against a single-process reference:
    the old executor left 12 of 16 parameter tensors with `grad is None` -- every parameter
    on rank 0, and rank 1's non-final chunk. The four that did get a gradient matched the
    reference exactly, which is why the loss curve looked plausible.

    Bug (2) was NOT reproducible on gloo and that is the point. Gloo gives each `irecv` its
    own unbound buffer and can satisfy receive k while k-1 is still unmatched; NCCL enqueues
    every p2p op for a direction on ONE stream and completes them strictly in posting order.
    So an ordering defect that gloo silently absorbs is a hard hang on NCCL. Two consequences
    for anyone editing this function:

      * never block on a receive that is not the EARLIEST outstanding one on its direction;
        waiting on the k-th parks us where we can never consume k-1;
      * never let execution order leak onto the wire -- the schedule is static here;
      * never leave a send and a receive un-batched when both must be in flight.

    A draft of this rewrite violated the first rule -- the work list was ordered by
    ascending chunk while backward receives were posted by descending chunk -- and it
    deadlocked the backward phase on gloo in seconds. That is the same shape of defect the
    cluster saw as a 420 s timeout before step 1, and it is what pointed at the ordering
    rule that the root-cause note above `_p2p_group` now states in full.
    """
    _check_schedule("interleaved", stage, batches)
    if not _ALLOW_REFUSED:
        raise RuntimeError(INTERLEAVED_REFUSAL)
    world, n_chunks = stage.world, stage.n_chunks
    chunks = sorted(stage.chunks)
    M = len(batches)
    seq = posid.shape[1]

    # --- validate the chunk map. Getting this wrong is the silent-wrong-parameters case. --
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

    # A single, statically-derived, globally-agreed op order -- the property that makes
    # GPipe the only schedule that ever worked. Forward visits (microbatch, chunk)
    # ascending; backward visits (microbatch, chunk) DESCENDING in chunk. Chunk c on this
    # rank corresponds to chunk c+1 on its successor and that mapping is order-preserving,
    # so the two ranks' same-direction sequences agree element for element without either
    # rank knowing the other's state. Nothing is data-driven any more: the previous
    # executor ran whichever task's input had landed, which leaked a nondeterministic order
    # onto the wire.
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

    # --- forward phase -----------------------------------------------------------------
    fwd = {}
    for m, c in fwd_tasks:
        inp = None
        if c != 0:
            inp = new_buf()
            # Flush the outputs produced so far together with the input we now need. The
            # send for chunk c-1 and the receive for chunk c are in one group, so neither
            # can be queued behind the other.
            exchange([dist.P2POp(dist.irecv, inp, prev_rank(c))])
            inp.requires_grad_(True)
        h, loss = stage.forward_chunk(batches[m], inp, posid, c)
        fwd[(m, c)] = (h, inp, loss)
        if loss is not None:
            losses = losses + loss.detach()
        if c != n_chunks - 1:
            queue_send(h.detach().contiguous(), next_rank(c))
    exchange([])

    # --- backward phase ----------------------------------------------------------------
    # The mirror image: the last chunk seeds from the loss, every other chunk waits for the
    # gradient of its output and hands its input gradient to the chunk before it. Without
    # this phase the run trains only the final chunk -- see bug (1) in the docstring.
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
    """Zero-bubble pipeline parallelism (Qi et al., 2023) -- the only schedule whose ceiling
    is the device count rather than `world * M/(M+1)`.

    The insight: a backward pass is two separable computations.

      B = gradient w.r.t. the stage INPUT   -- must be ordered, the neighbour is waiting
      W = gradient w.r.t. the stage WEIGHTS -- nothing waits on it; it can be deferred

    GPipe and 1F1B treat backward as atomic, so a stage with nothing to receive simply
    idles. Splitting it lets that idle time be spent on deferred W work, and the fill/drain
    bubble stops costing anything. The ceiling becomes 2.00x on two devices instead of
    1.78x at M=8 or 1.94x at M=32.

    Mechanically, `torch.autograd.grad` gives the split directly: ask for grads w.r.t. the
    input tensor with `retain_graph=True` (that is B, sent onward immediately), then later
    ask for grads w.r.t. the parameters (that is W, accumulated into `.grad` by hand).
    `retain_graph` is what makes the deferral legal, and it is also the cost -- the graph
    stays alive until its W runs, so activation memory is held longer than in 1F1B.
    """
    _check_schedule("zerobubble", stage, batches)
    M = len(batches)
    seq = posid.shape[1]

    # Pre-post everything, as elsewhere: a receive posted on demand blocks its sender.
    # Ordering is ascending-microbatch on both directions and on both sides (the forward
    # loop sends in `m` order, the B loop sends gradients in `m` order), which is what makes
    # the untagged FIFO matching correct for any `world`, not just two.
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

    # ---- forwards ----------------------------------------------------------------
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

    # The gradient receives are posted HERE, after every forward send, and not inside the
    # loop above. That single line of placement is what hung this schedule before step 1:
    # posting `irecv(g_m)` between `isend(h_m)` and `isend(h_{m+1})` puts an unmatchable
    # receive in front of a send the peer is waiting for, on an ordered p2p channel. With
    # the posts moved out, this rank's op order on each direction is `all sends, then all
    # receives` -- exactly GPipe's mirrored shape, the one shape measured to work -- for
    # any `world`, since a middle stage's two neighbours are different pairs.
    if not stage.is_last:
        for m in range(M):
            gbuf[m] = torch.empty_like(acts[m][0])
            greq[m] = dist.irecv(gbuf[m], src = stage.rank + 1)

    # ---- B passes, with W filling the gaps ---------------------------------------
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
            # Wait for the downstream gradient; spend the wait on deferred W work rather
            # than blocking idle -- this is where the bubble goes.
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


# ---------------------------------------------------------------------------
# The upstream backend: torch.distributed.pipelining
# ---------------------------------------------------------------------------
# WHY THIS EXISTS, AND WHY IT IS THE DEFAULT.
#
# Everything above is hand-written, and two of the four schedules do not work: zerobubble is
# correct but slow, and interleaved deadlocks -- and before it deadlocked it computed WRONG
# GRADIENTS (12 of 16 parameter tensors ended a step with `grad is None`). Both defects have
# one root cause, stated plainly: we tried to make a single `loss.backward()` span the rank
# cut, and there is no autograd edge across a `dist.irecv`. Every attempt to patch around
# that produced either a missing gradient or an op-ordering deadlock.
#
# `torch.distributed.pipelining` -- already inside the torch we ship, 2.11.0+cu130 -- solves
# both structurally rather than by being more careful:
#
#   ORDERING.  `_add_send_recv` (schedules.py) has every rank run the SAME deterministic
#              simulation of the schedule, and a rank never schedules its own receive: the
#              sender emits it. So the two ranks' op sequences correspond by construction
#              rather than by us matching them by hand. An order that cannot be scheduled
#              raises "Malformed compute schedule" at CONSTRUCTION time instead of hanging.
#
#   GRADIENTS. Backward is an explicit grad send/recv with a locally-rooted
#              `torch.autograd.backward(stage_output, grad_tensors=received_grad)`
#              (_backward.py). Nothing is asked to cross the process boundary, so nothing
#              can silently fail to.
#
# Verified on 2 gloo ranks at our shape, including the config that deadlocks for us:
#   Interleaved1F1B 4 stages (2/rank), 8 mb   -> grad_is_None = 0, no deadlock
#   Interleaved1F1B 8 stages (4/rank), 32 mb  -> grad_is_None = 0, no deadlock
#   ZBVZeroBubble / DualPipeV                 -> grad_is_None = 0
# Gloo cannot certify a schedule against NCCL (it has no p2p ordering constraint, so it
# passes code that deadlocks on NCCL) -- those runs are correctness evidence only.
#
# MEASURED, two DGX Sparks, NCCL, unsloth/Qwen3.5-2B seq 512 batch 64 M=32, 20 steps, LoRA
# r=16, seed 3407, file frozen at md5 de88aa45f4630051e544817ad4efcc0f, 2026-09-03.
# Denominator is the best single-Spark configuration re-measured in the same session:
# 2149 tok/s (batch 8, seq 512).
#
#     arm                                        tok/s   speedup  peak GiB (r0/r1)  loss@20
#     torch  dualpipev   (V, 2 stages/rank)       4204     1.96x    9.58 /  5.84    12.5764
#     torch  1f1b                                 4174     1.94x    5.52 /  7.34    12.5789
#     torch  zbv         (V, 2 stages/rank)       4166     1.94x    8.97 /  5.59    12.5774
#     torch  interleaved (v=2)  <- USED TO HANG   4148     1.93x    6.46 /  8.40    12.5793
#     torch  gpipe                                3997     1.86x   50.93 / 99.92    12.5778
#     torch  zerobubble  (v=2)                    3702     1.72x    6.23 /  9.82    12.5771
#     legacy gpipe       (control arm)            3898     1.81x   50.91 / 84.86    12.5767
#
# Read three things off that table.
#
#  1. `interleaved` RUNS. It is the configuration that produced zero steps in 300 s twice,
#     and it is now within 1.3% of the best arm. That is the port paying for itself.
#  2. Every arm lands on the same loss to ~0.03%, and no arm falls FASTER than the others.
#     A loss that falls faster is a gradient defect, not a win -- that exact symptom is how
#     the broken hand-written interleaved was caught -- so the agreement is the point.
#  3. GPipe's memory is the reason not to use it: 99.92 GiB against 1F1B's 7.34 GiB for the
#     same work, because GPipe holds every microbatch's activations to the end of the step.
#     On a 121.69 GiB node that is the difference between fitting and not.
#
# DualPipeV wins, but by 0.7% over 1F1B, not by the ~half-the-traffic its co-located hop
# suggests -- because at this size the link was never the constraint (a boundary crossing is
# hidden_size*2 bytes per token, ~10-16 KiB, against 21.6 GB/s). Its argument is capacity and
# larger models, and it should be re-measured at 70B before being recommended on that basis.
#
# GOTCHA, found the hard way: the V layouts CANNOT be validated on gloo. At pp=2 with 4
# stages the map is {0:0, 1:1, 2:1, 3:0}, so stages 1 and 2 are both on rank 1 and
# `_get_init_p2p_neighbors_ops` (stage.py) emits a send and a recv from rank 1 to ITSELF
# without checking for the co-located case. Gloo cannot do self p2p -- it fails with
# "Pair is not connected" -- while NCCL handles it inside the group as a local copy. So zbv
# and dualpipev fail instantly on the CPU path and run correctly on hardware. Do not read a
# gloo failure of a V schedule as a defect in this file.
#
# `--pp-backend legacy` keeps the hand-written schedules reachable, so a regression here is
# one flag away from being isolated rather than a git bisect.

_STAGE_MODULE_CLS = None


def _dist_pipelining_available() -> bool:
    """Is `torch.distributed.pipelining` importable here?

    Feature detection, never a version string, and never at module scope: this file is
    imported by `unsloth run` on Windows, macOS and AMD boxes where `torch.distributed` may
    be absent entirely (`torch.distributed.is_available()` is False on a default macOS
    build) and where importing torch at all would be a regression a test pins.
    """
    try:
        import torch
        if not torch.distributed.is_available():
            return False
        import torch.distributed.pipelining  # noqa: F401
    except Exception:
        return False
    return True


def config_num_layers(cfg) -> int:
    """Decoder depth, before any model is built.

    Needed because the layer->stage assignment has to exist before `build_stage_model` is
    told which layers to keep, and building the model to count its layers would defeat the
    point of shard loading. Architectures disagree on the attribute name, so try the small
    set transformers actually uses and raise rather than guess.
    """
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
    """Return `(causal_lm, decoder_stack)` through PEFT and the HF wrappers.

    `model.base_model.model` (what the legacy `_Stage` does) is only right for PEFT: on a
    bare `LlamaForCausalLM`, `base_model` is the HF property that already returns the
    decoder stack, and `.model` on that raises. `get_base_model()` is PEFT's own accessor
    and is absent off PEFT, which makes it a safe discriminator.
    """
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
        """A contiguous run of decoder layers, plus the embedding on the first stage and
        the norm + lm_head on the last.

        `PipelineStage` takes a MANUALLY split module -- there is no tracer involved, which
        is what makes this work on arbitrary transformers versions where a symbolic trace
        of a HF model would not. With `input_args=None` the stage infers the boundary
        tensor's shape and dtype at runtime by propagating stage 0's real output, so no
        activation shape is hand-specified here.

        Position ids are recomputed from the hidden state rather than passed in, so this
        module is shape-agnostic and a ragged final microbatch cannot desync it.
        """

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
                    # use_reentrant=False is required: the reentrant implementation drops
                    # the grad_fn that the stage's activation-gradient handoff needs.
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
    """Next-token cross entropy, MEAN-reduced over tokens.

    This is exactly the legacy schedules' loss minus their `/ microbatches`: upstream does
    that division itself via `scale_grads=True`. Keeping the `/M` here as well would scale
    every gradient by `1/M^2`.
    """
    import torch.nn.functional as F
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        target[:, 1:].reshape(-1),
    )


# `pp_loss_fn` mean-reduces (F.cross_entropy defaults to reduction="mean"), so gradients
# must be divided by the microbatch count -- which is what scale_grads=True does. Set
# explicitly rather than left to the upstream default so that flipping the loss to a sum
# reduction cannot silently leave gradients off by 1/M. Sum-reducing => False.
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
    version string. The concrete case this guards: torch 2.11 has no `get_mesh=` kwarg on
    `PipelineStage` (it was added later), so passing one unconditionally would break the
    torch we actually ship, and refusing to pass one unconditionally would break a later
    torch that needs it. The rule is: ask the signature.

    Both sides of that split were exercised, on 2 gloo ranks, same seed, same model:

        torch 2.11.0+cu130  transformers 4.57.6   LoRA          step-1 loss 12.4108
        torch 2.11.0+cu130  transformers 5.5.0    LoRA          step-1 loss 12.4108
        torch 2.13.0+cu130  transformers 5.16.1   full finetune step-1 loss 12.4108

    `inspect.signature(PipelineStage.__init__)` lists `get_mesh` on 2.13 and does not on
    2.11; `step()` gained `loss_kwargs` on 2.13. Nothing here passes either, and `group=`
    and `return_outputs=` are passed only when the signature admits them.
    """
    import inspect
    import torch
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining import schedules as _schedules
    from torch.distributed.pipelining.microbatch import TensorChunkSpec

    # Cross-check our pure layout against upstream's, which is what the runtime actually
    # uses for send/recv. A disagreement would put layers on the wrong node and train the
    # wrong parameters with no error at all.
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
        # input_args=None: let the stage infer the boundary shape at runtime by propagating
        # stage 0's real output. Hand-specifying it is how a seq-length or batch change
        # turns into a hang.
        stages.append(PipelineStage(mod, idx, plan["num_stages"], device, **stage_kwargs))

    sched_params = inspect.signature(sched_cls.__init__).parameters
    kw = {"loss_fn": pp_loss_fn}
    if "args_chunk_spec" in sched_params:
        # Split the token-id batch along dim 0. Upstream's
        # `split_args_kwargs_into_chunks` does the splitting, replacing our slicer.
        kw["args_chunk_spec"] = (TensorChunkSpec(0),)
    if "scale_grads" in sched_params:
        kw["scale_grads"] = PP_SCALE_GRADS
    multi = len(stages) > 1 or "stages" in sched_params
    schedule = sched_cls(stages if multi else stages[0], microbatches, **kw)

    step_params = inspect.signature(schedule.step).parameters
    step_kw = {}
    if "return_outputs" in step_params:
        # The merged logits for a whole batch are `batch x seq x vocab` in fp32 -- for a 2B
        # at batch 64 seq 512 that is tens of GiB on the loss rank alone, for a tensor
        # nobody reads. Only newer torch can decline it.
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


# Diagnosis escape hatch. The refused schedules deadlock on hardware, so they are off by
# default; SPARK_PP_DIAGNOSE=1 re-enables them so a stack or a trace can be taken WITHOUT
# editing this file mid-measurement. Never set it in anything a user runs.
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

# `--schedule` accepts the union: the four legacy names (which both backends understand)
# plus the layouts only upstream can drive. Asking for one of the latter under
# `--pp-backend legacy` is refused by name rather than ignored.
SCHEDULE_CHOICES = sorted(set(SCHEDULES) | set(TORCH_PP_SCHEDULES))
PP_BACKENDS = ("torch", "legacy")


# ---------------------------------------------------------------------------
# Entry point (run under torchrun)
# ---------------------------------------------------------------------------


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
    p.add_argument(
        "--data-parallel",
        action = "store_true",
        help = "one FULL model per rank, gradients averaged by DDP (or parameters "
        "sharded with --fsdp). Buys throughput, not capacity: the model must fit "
        "on one Spark. Same data, loss and LoRA setup as the layer split, so the "
        "two are directly comparable. WORLD_SIZE=1 is the single-Spark control.",
    )
    p.add_argument(
        "--fsdp",
        action = "store_true",
        help = "with --data-parallel: shard the base weights across the ranks "
        "(torch.distributed.fsdp.fully_shard) instead of replicating them",
    )
    return p


LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def apply_lora(model, r: int):
    """The one LoRA configuration every arm trains, so that a layer split and a data
    parallel replica of the same model train the same adapters."""
    from peft import LoraConfig, get_peft_model
    return get_peft_model(
        model,
        LoraConfig(
            r = r,
            lora_alpha = r,
            lora_dropout = 0.0,
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules = list(LORA_TARGET_MODULES),
        ),
    )


def make_token_batches(tok, args, device):
    """The training rows for a run, identical on every rank.

    Seeded at 3407 and drawn on every rank rather than broadcast: it is what the layer
    split already relies on (stage 0 draws the inputs, the loss stage draws the targets,
    and they have to agree), and it keeps the data parallel arm on the very same rows.
    """
    import torch

    torch.manual_seed(3407)
    need = args.batch * args.steps
    if args.data:
        rows = [json.loads(l) for l in open(args.data, encoding = "utf-8")]
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
        return enc.repeat((need + len(enc) - 1) // len(enc), 1)[:need].to(device)
    return torch.randint(0, tok.vocab_size, (need, args.seq), device = device)


def _main_data_parallel(args) -> int:
    """`--data-parallel`: one whole model per rank; the ranks average gradients.

    The comparison the layer split has always lacked. A split of a model that FITS on
    one Spark buys nothing by construction (both nodes still read every weight once per
    step), so the honest question for such a model is data parallel against pipeline
    parallel, measured on the same rows with the same loss. With LoRA the all-reduce
    carries only the adapters (tens of MB), so the link is never the limit; with
    `--fsdp` the base weights are sharded instead and gathered per layer as needed,
    which also halves the resident weights.

    WORLD_SIZE=1 runs the identical code with no wrapper, and is the single-Spark
    control every two-Spark number is divided by.
    """
    import contextlib

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.shard_load:
        raise SystemExit(
            "--shard-load is a layer-split option; a data-parallel replica holds the "
            "whole model on every rank."
        )
    if args.batch % args.microbatches:
        raise SystemExit("--batch must be divisible by --microbatches")
    if args.batch % world or args.microbatches % world:
        raise SystemExit(
            f"--batch ({args.batch}) and --microbatches ({args.microbatches}) must both "
            f"be divisible by the world size ({world}) so every rank gets equal rows."
        )
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
        print(f"[spark-dp {rank}/{world}] {msg}", flush = True)

    mode = "fsdp" if (args.fsdp and world > 1) else ("ddp" if world > 1 else "single")
    log(f"host={os.uname().nodename} data-parallel mode={mode}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # rank 0 of a world of 1: the whole stack, embedding and head, on this device.
    model, cfg, _ = build_stage_model(
        args.model, 0, 1, device, shard_load = False, dtype = dtype, log = log
    )
    if not args.full_finetune:
        model = apply_lora(model, args.lora_r)
    if args.grad_checkpoint:
        # The HF forward is what runs here, so transformers' own switch is honoured.
        base_model = getattr(model, "base_model", model)
        inner_model = getattr(base_model, "model", base_model)
        target = inner_model if hasattr(inner_model, "gradient_checkpointing_enable") else model
        target.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        log("gradient checkpointing enabled (use_reentrant=False)")
    model.to(device)
    if not use_cpu:
        torch.cuda.empty_cache()

    unwrapped = model
    no_sync = None
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids = None if use_cpu else [device.index or 0])
        no_sync = model.no_sync
    elif mode == "fsdp":
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError as exc:
            raise SystemExit(f"--fsdp needs torch.distributed.fsdp.fully_shard: {exc}")
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu" if use_cpu else "cuda", (world,))
        _, owner = unwrap_stack(model)
        for layer in owner.layers:
            if isinstance(layer, torch.nn.Module) and any(True for _ in layer.parameters()):
                fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)

        def _sync(on):
            model.set_requires_gradient_sync(on)

        @contextlib.contextmanager
        def _no_sync():
            _sync(False)
            try:
                yield
            finally:
                _sync(True)

        no_sync = _no_sync

    trainable = [p for p in model.parameters() if p.requires_grad]
    resident = f"{torch.cuda.memory_allocated()/2**30:.2f} GiB" if not use_cpu else "cpu"
    log(
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f} B params resident "
        f"({resident}), {sum(p.numel() for p in trainable)/1e6:.1f} M trainable"
    )
    opt = torch.optim.AdamW(trainable, lr = args.lr)

    ids_all = make_token_batches(tok, args, device)
    per_rank = args.batch // world
    mb_per_rank = args.microbatches // world
    mb_rows = per_rank // mb_per_rank
    mb_tokens = mb_rows * args.seq
    if mb_tokens < 436:
        log(
            f"WARNING: each microbatch is {mb_tokens} tokens, below the ~436-token "
            f"compute/bandwidth crossover; raise --batch or --seq, or lower --microbatches."
        )
    log(
        f"global batch {args.batch} = {world} rank(s) x {mb_per_rank} microbatch(es) "
        f"x {mb_rows} rows x {args.seq} tokens"
    )

    dist.barrier()
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none = True)
        whole = ids_all[step * args.batch : (step + 1) * args.batch]
        mine = whole[rank * per_rank : (rank + 1) * per_rank]
        acc = torch.zeros((), device = device, dtype = torch.float32)
        for m in range(mb_per_rank):
            x = mine[m * mb_rows : (m + 1) * mb_rows]
            last = m == mb_per_rank - 1
            ctx = contextlib.nullcontext() if (last or no_sync is None) else no_sync()
            with ctx:
                logits = model(input_ids = x, use_cache = False).logits
                # Same mean-reduced next-token loss as the pipeline, and the same 1/M
                # scaling that scale_grads=True applies there.
                loss = pp_loss_fn(logits, x) / mb_per_rank
                loss.backward()
            acc += loss.detach().float()
        opt.step()
        if (step + 1) % 5 == 0 or args.steps <= 10:
            if world > 1:
                dist.all_reduce(acc, op = dist.ReduceOp.AVG)
            if rank == 0:
                log(f"step {step+1}/{args.steps} loss={acc.item():.4f}")

    dist.barrier()
    elapsed = time.perf_counter() - t0
    if rank == 0:
        toks = args.batch * args.seq * args.steps
        log(
            f"DONE {args.steps} steps in {elapsed:.1f}s | "
            f"{elapsed/args.steps:.2f}s/step | {toks/elapsed:.0f} tok/s"
        )
    if not use_cpu:
        log(f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    if args.save and rank == 0 and mode != "fsdp":
        os.makedirs(args.save, exist_ok = True)
        unwrapped.save_pretrained(args.save)
        log(f"saved to {args.save}")
    elif args.save and mode == "fsdp":
        log("--save is not implemented for --fsdp (sharded parameters); skipped")

    dist.destroy_process_group()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_parallel:
        return _main_data_parallel(args)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # Refuse impossible combinations here, before a model is loaded and before any
    # collective is issued -- a bad combination otherwise surfaces as a hang on the wire.
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
    # Fail here, before the tokenizer and the model load, so the user sees the reason in
    # under a second instead of watching a silent process for five minutes. Measured: the
    # interleaved refusal exits in 11 s end to end against a 300 s hang.
    #
    # The refusals apply to the LEGACY backend only. They record what the hand-written
    # schedules in this file do on this hardware, and that has not changed; what changed is
    # that there is now a backend which does not have those defects, so refusing the name
    # outright would refuse a working configuration.
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
            # print, not log(): `log` is defined ~20 lines below, after the process group is
            # up, so calling it here raised UnboundLocalError and killed the very diagnostic
            # run this warning exists to announce.
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
    # CPU mode exists so the pipeline protocol can be exercised without two GPUs. NCCL
    # cannot do point-to-point between two processes on the SAME device, so a one-box
    # functional test is impossible on CUDA -- and a scheduling bug that serialises the
    # pipeline is visible on CPU just as clearly, because it is a protocol defect rather
    # than a hardware one.
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
        # Resolve the whole layout before anything is allocated. Every rank runs the same
        # pure function on the same arguments, so the assignment agrees across the cluster
        # without a collective -- which is the same property upstream relies on for its
        # send/recv order, and the reason nothing here has to be negotiated on the wire.
        from transformers import AutoConfig

        n_layers = config_num_layers(AutoConfig.from_pretrained(args.model))
        try:
            plan = torch_pp_plan(
                args.schedule, world, args.microbatches, args.virtual_stages, n_layers
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        my_plan = plan_for_rank(plan, rank)

    # Multi-stage layouts own non-contiguous chunks, so the contiguous drop-to-Identity in
    # build_stage_model would remove layers this rank needs; pass the real set instead. The
    # legacy interleaved path has no such set, so it keeps the whole stack -- a smaller
    # memory saving, which is the honest trade there.
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
        model = apply_lora(model, args.lora_r)
    if args.grad_checkpoint and use_torch_pp:
        # The torch backend calls the decoder layers directly out of `_PPStageModule`, so
        # transformers' own `gradient_checkpointing_enable()` -- which is consulted inside
        # `LlamaModel.forward`, a function that is never reached here -- would be inert. The
        # stage module wraps each layer in `torch.utils.checkpoint` itself instead; the flag
        # is passed to `build_torch_schedule` below.
        log("gradient checkpointing enabled per decoder layer (use_reentrant=False)")
    elif args.grad_checkpoint:
        # Pipeline parallelism only pays above the ~436-token compute/bandwidth crossover,
        # and reaching that means larger microbatches, which costs activation memory on a
        # node already holding half a 70B. Recomputing activations is the standard trade:
        # ~30% more step time for a large memory saving, which is worth it because a
        # microbatch below the crossover cannot benefit from the split at all.
        base_model = getattr(model, "base_model", model)
        inner_model = getattr(base_model, "model", base_model)
        if hasattr(inner_model, "gradient_checkpointing_enable"):
            inner_model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        # use_reentrant=False is required for checkpointing to coexist with the autograd
        # graph that spans the p2p boundary; the reentrant version drops the grad_fn that
        # the activation-gradient handoff depends on.
        if hasattr(inner_model, "gradient_checkpointing_kwargs"):
            inner_model.gradient_checkpointing_kwargs = {"use_reentrant": False}
        log("gradient checkpointing enabled (use_reentrant=False)")

    model.to(device)  # shard-load already placed the base; this catches new adapters
    if not use_cpu:
        torch.cuda.empty_cache()

    trainable = [p for p in model.parameters() if p.requires_grad]
    # `torch.cuda.memory_allocated()` raises without a CUDA device, and this module has to
    # stay runnable on the CPU/gloo path (that is how the schedules are unit-tested).
    resident = f"{torch.cuda.memory_allocated()/2**30:.2f} GiB" if not use_cpu else "cpu"
    log(
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f} B params resident "
        f"({resident}), "
        f"{sum(p.numel() for p in trainable)/1e6:.1f} M trainable"
    )

    if args.batch % args.microbatches:
        raise SystemExit("--batch must be divisible by --microbatches")
    mb_rows = args.batch // args.microbatches

    # Pipeline parallelism only pays while the work is COMPUTE-bound. Measured on GB10 the
    # roofline crossover sits at M ~ 436 tokens: below it a step is limited by weight
    # traffic, and splitting layers across nodes cannot help, because the two stages read
    # their halves sequentially for the same microbatch -- total bytes per step is
    # unchanged. Above it the step is limited by FLOPs, which DO add across nodes.
    #
    # Slicing a batch into too many microbatches is therefore self-defeating: it fills the
    # pipeline but drops each microbatch under the crossover. Warn rather than override,
    # since a user may be deliberately trading throughput for memory.
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

    # A pipeline that is shallower than it is deep never fills: stage `world-1` cannot
    # start until `world-1` microbatches have been issued, so the last (world-1) of every
    # step's stage-slots are bubble. Correct, just wasteful -- say so rather than silently
    # reporting a bad number.
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
        # Global chunk index for each of this rank's chunks: chunk c lives on rank c % world,
        # so this rank owns rank, rank+world, rank+2*world, ...
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

    ids_all = make_token_batches(tok, args, device)

    posid = torch.arange(args.seq, device = device).unsqueeze(0).expand(mb_rows, -1)
    schedule = SCHEDULES[args.schedule] if not use_torch_pp else None

    dist.barrier()
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none = True)
        if use_torch_pp:
            # Hand the WHOLE step batch to the schedule and let upstream's
            # `split_args_kwargs_into_chunks` cut it into microbatches, rather than slicing
            # it ourselves: the chunking has to agree with the chunking of `target`, and
            # letting one implementation own both is how they stay agreed.
            whole = ids_all[step * args.batch : (step + 1) * args.batch]
            losses = [] if is_loss_rank else None
            # Only the rank holding stage 0 may supply positional inputs -- upstream
            # asserts "Can't supply input args for shape inference on non-first stage",
            # because every other stage's input is the wire. `target` is handed to every
            # rank; only the one computing the loss reads it, and under a V layout that is
            # rank 0 rather than the last rank.
            step_args = (whole,) if rank == plan["first_rank"] else ()
            pp_schedule.step(*step_args, target = whole, losses = losses, **pp_step_kw)
            # `losses` holds one MEAN-reduced cross entropy per microbatch, so the step loss
            # is their mean. That equals what the legacy schedules return (they divide each
            # microbatch loss by M and sum), which is what makes the two backends' loss
            # curves directly comparable at a fixed seed.
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
        # accumulator on every rank, so gating on the value alone made rank 0 print a
        # confident `loss=0.0000` beside the real number from rank 1.
        if is_loss_rank and loss is not None and ((step + 1) % 5 == 0 or args.steps <= 10):
            # The ONE sync per step, and only when a line is actually printed.
            value = loss.item() if hasattr(loss, "item") else float(loss)
            log(f"step {step+1}/{args.steps} loss={value:.4f}")

    dist.barrier()
    elapsed = time.perf_counter() - t0
    if is_loss_rank:
        # Under a V layout the loss -- and therefore this line -- lands on rank 0, not on
        # the last rank. Keying the report off `stage.is_last` printed nothing at all for
        # DualPipeV, which reads exactly like a hang.
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
        # Each stage holds a different slice, so each writes its own directory. Merging
        # them back into one checkpoint is a separate step.
        out = osp.join(args.save, f"stage{rank}")
        os.makedirs(out, exist_ok = True)
        model.save_pretrained(out)
        log(f"saved stage {rank} to {out}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
