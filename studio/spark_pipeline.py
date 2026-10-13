# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-split (pipeline-parallel) finetuning across two or more NVIDIA DGX Sparks."""

from __future__ import annotations

import argparse
import contextlib
import glob
import inspect
import json
import os
import os.path as osp
import time
from typing import List, Optional, Sequence

# Architectures nest the decoder stack differently.
_LAYER_PATHS = (
    ("model", "layers"),  # Llama, Qwen, Mistral, Gemma...
    ("model", "decoder", "layers"),  # OPT-style
    ("transformer", "h"),  # GPT-2/NeoX-style
    ("gpt_neox", "layers"),
)


# find_layers accepts OPT, GPT-2 and GPT-NeoX, so the stage wrapper must not assume Llama's names.
TRAIN_SEED = 3407
_EMBED_NAMES = ("embed_tokens", "wte", "embed_in")
_FINAL_NORM_NAMES = ("norm", "ln_f", "final_layer_norm")
_LAYER_CONTAINER_NAMES = ("layers", "h")


def _first_attr(owner, names: Sequence[str]):
    return _first_named(owner, names)[1]


def _embed_scale(config, embed) -> Optional[float]:
    """sqrt(hidden_size) for Gemma 1/2 on transformers 4.x, which scale in the model forward a stage
    never calls; newer releases scale inside the embedding (embed_scale) and need nothing here."""
    if getattr(config, "model_type", None) not in ("gemma", "gemma2") or hasattr(
        embed, "embed_scale"
    ):
        return None
    return float(config.hidden_size) ** 0.5


def _scale_embeddings(h, scale):
    if scale is None:
        return h
    import torch
    return h * torch.tensor(scale, dtype = h.dtype, device = h.device)


def _softcap_logits(config, logits):
    # Gemma 2/3 cap logits in ForCausalLM.forward, which a stage never calls.
    cap = getattr(config, "final_logit_softcapping", None)
    if not cap:
        return logits
    import torch

    return torch.tanh(logits / cap) * cap


def _first_named(owner, names: Sequence[str]):
    for name in names:
        found = getattr(owner, name, None)
        if found is not None:
            return name, found
    return None, None


def _forward_params(module) -> set:
    """Parameter names of a module's `forward`, empty when it cannot be introspected."""
    try:
        return set(inspect.signature(type(module).forward).parameters)
    except (TypeError, ValueError):
        return set()


def _unrun_parameters(owner, run_names: Sequence[str]) -> list[str]:
    """Direct children of the decoder stack that carry weights and that a stage never runs."""
    kept = set(run_names)
    return [
        name
        for name, child in owner.named_children()
        if name not in kept and any(p.numel() for p in child.parameters(recurse = True))
    ]


def _resolve(root, path: Sequence[str]):
    node = root
    for attr in path:
        if not hasattr(node, attr):
            return None
        node = getattr(node, attr)
    return node


def dataset_problem(path: str) -> Optional[str]:
    """Why `--data path` cannot be trained on, or None."""
    try:
        with open(path, encoding = "utf-8") as handle:
            for line in handle:
                if line.strip():
                    return None
    except OSError as exc:
        return f"--data {path} could not be read: {exc}"
    return f"--data {path} has no rows; nothing to train on"


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


LORA_TARGETS_LLAMA = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_target_modules(model) -> List[str]:
    """The projection names to attach LoRA to, read off THIS model's decoder layers."""
    import torch.nn as nn

    _, layers = find_layers(model)
    names = set()
    for layer in layers:
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                names.add(name.rsplit(".", 1)[-1])
    keep = [n for n in LORA_TARGETS_LLAMA if n in names]
    if keep:
        return keep
    if names:
        return sorted(names)
    raise RuntimeError(
        "no nn.Linear modules were found inside this model's decoder layers, so there is "
        "nothing for LoRA to attach to. Train with --full-finetune, or use a model whose "
        "layers hold ordinary linear projections."
    )


def interleaved_layers(
    n_layers: int,
    rank: int,
    world: int,
    virtual: int = 2,
) -> List[List[int]]:
    """Layer chunks for INTERLEAVED pipeline parallelism (Megatron's schedule)."""
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
    return [bounds[c] for c in range(total_chunks) if c % world == rank]


def stage_layers(n_layers: int, rank: int, world: int) -> List[int]:
    """Contiguous slice of layer indices, balanced by COUNT: right only for homogeneous layers, where a
    mixed dense/sparse MoE would want a cost-weighted split."""
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


# `loop` maps stage i to rank i % world, so every hop changes rank.


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
    """Resolve `--schedule` into a concrete stage/layer/rank assignment."""
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


def _dtype_kwargs(dtype) -> dict:
    """transformers 4.56 renamed torch_dtype to dtype; the 4.52.4 floor rejects `dtype`."""
    import transformers
    from packaging.version import Version

    new = Version(transformers.__version__) >= Version("4.56.0")
    return {"dtype" if new else "torch_dtype": dtype}


@contextlib.contextmanager
def _parameters_on_meta():
    """Parameters go to meta as they register, buffers stay real: non-persistent buffers (every
    rotary inv_freq, causal masks) are never in safetensors, so their own __init__ must compute them."""
    import torch

    register = torch.nn.Module.register_parameter

    def to_meta(module, name, param):
        register(module, name, param)
        if param is not None:
            p = module._parameters[name]
            module._parameters[name] = type(p)(p.to("meta"), requires_grad = p.requires_grad)

    torch.nn.Module.register_parameter = to_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register


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
    """Build only this stage's slice of the model."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if not shard_load:
        model = AutoModelForCausalLM.from_pretrained(model_name, **_dtype_kwargs(dtype))
        model.config.use_cache = False
        cfg = model.config
    else:
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.use_cache = False
        with _parameters_on_meta():
            model = AutoModelForCausalLM.from_config(cfg, **_dtype_kwargs(dtype))

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
    if not keep_all_layers and not want_embed:
        # By the architecture's own name, not `embed_tokens`: `find_layers` accepts GPT-NeoX, whose
        # table is `embed_in`, so every rank kept and materialised the whole embedding while
        # believing it had dropped it.
        embed_name, _ = _first_named(owner, _EMBED_NAMES)
        if embed_name:
            setattr(owner, embed_name, torch.nn.Identity())
    if not keep_all_layers and not want_head:
        # GPT-NeoX names these final_layer_norm and embed_out, as _PPStageModule resolves them.
        norm_name, _ = _first_named(owner, _FINAL_NORM_NAMES)
        if norm_name:
            setattr(owner, norm_name, torch.nn.Identity())
        head_name, _ = _first_named(model, ("lm_head", "embed_out"))
        if head_name:
            setattr(model, head_name, torch.nn.Identity())

    log(
        f"{n} decoder layers; this stage owns {_ranges(mine)}"
        f"{' +embed' if want_embed else ''}{' +head' if want_head else ''}"
    )

    if shard_load:
        _materialise(model, model_name, cfg, device, dtype, log)

    return model, cfg, mine


def _has_parameter(model, name: str) -> bool:
    """Whether `name` names a parameter this stage still owns; a dropped module does not."""
    node = model
    parts = name.split(".")
    for part in parts[:-1]:
        node = getattr(node, part, None)
        if node is None:
            return False
    return getattr(node, parts[-1], None) is not None


def tied_split_problem(
    tied: bool,
    full_finetune: bool,
    world: int,
    stage_to_rank: Optional[dict] = None,
) -> Optional[str]:
    """Why a tied embedding must not be split across ranks, or None when it is not."""
    if not tied or not full_finetune or world < 2:
        return None
    if stage_to_rank:
        first = stage_to_rank[min(stage_to_rank)]
        last = stage_to_rank[max(stage_to_rank)]
    else:
        first, last = 0, world - 1
    if first == last:
        return None
    return (
        f"this checkpoint ties its input embedding to its lm_head, and --full-finetune would "
        f"put them on different ranks ({first} and {last}) as two independently optimized "
        f"copies, which drift apart with nothing to keep them equal. Use --schedule zbv or "
        f"dualpipev, which keep the first and last stage on one rank, or drop --full-finetune, "
        f"where the tied weights are frozen"
    )


def full_finetune_save_problem(full_finetune: bool, save: str, world: int) -> Optional[str]:
    """Why `--full-finetune --save` cannot produce a usable model across ranks, or None."""
    if not full_finetune or not save or world < 2:
        return None
    return (
        "--full-finetune with --save has no way to produce a loadable model across "
        f"{world} ranks: each stage saves only the layers it owns, and `unsloth spark merge` "
        "joins LoRA adapters, not base weights. Train with LoRA and merge the adapters, or "
        "run --full-finetune on a single node where the checkpoint is complete."
    )


def legacy_attention_problem(cfg, seq: int) -> Optional[str]:
    """Why the legacy backend cannot reproduce this model's attention, or None."""
    impl = getattr(cfg, "_attn_implementation", "sdpa")
    if impl not in ("sdpa", "flash_attention_2", "flash_attention_3"):
        return (
            f"--pp-backend legacy calls decoder blocks without an attention mask, which is "
            f"causal only under sdpa or flash. This model is loaded with {impl!r}, where "
            f"attention with no mask is bidirectional: the run would train on future tokens "
            f"at a flattering loss. Use --pp-backend torch, which builds the mask."
        )
    window = getattr(cfg, "sliding_window", None)
    if isinstance(window, int) and window > 0 and seq > window:
        return (
            f"--pp-backend legacy does not reproduce sliding-window attention, and --seq "
            f"{seq} is past this model's {window}-token window, so the run would train under "
            f"full causal attention instead. Use --pp-backend torch, or --seq {window} or "
            f"fewer."
        )
    return None


def _tied_aliases(model) -> dict:
    """`{parameter name saved under another name: the name it is saved under}`."""
    declared = getattr(type(model), "_tied_weights_keys", None)
    if declared is None:
        declared = getattr(model, "_tied_weights_keys", None)
    if not declared:
        return {}
    if isinstance(declared, dict):
        return dict(declared)

    source = None
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None:
        for name, module in model.named_modules():
            if module is embeddings:
                source = f"{name}.weight" if name else "weight"
                break
    return {alias: source for alias in declared}


def _materialise(model, model_name, cfg, device, dtype, log):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    wanted = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    # A tied head is not saved under its own name: a checkpoint whose lm_head is the embedding
    # stores only the embedding key, so an exact-name filter looked for `lm_head.weight`, found
    # nothing, and the meta check below refused the load.
    aliases = {alias: src for alias, src in _tied_aliases(model).items() if src}
    wanted |= set(aliases.values())
    snap = (
        model_name
        if osp.isdir(model_name)
        else snapshot_download(model_name, allow_patterns = ["*.safetensors", "*.json"])
    )

    shards = sorted(glob.glob(osp.join(snap, "*.safetensors")))
    if not shards:
        # Said plainly, and before the tensors are read.
        legacy = glob.glob(osp.join(snap, "*.bin"))
        raise RuntimeError(
            f"--shard-load reads safetensors, and {snap} has none"
            + (
                f" ({len(legacy)} .bin shard(s) instead). Convert the checkpoint to "
                f"safetensors, or load it without --shard-load."
                if legacy
                else "."
            )
        )

    loaded, seen = {}, 0
    for f in shards:
        with safe_open(f, framework = "pt", device = "cpu") as sf:
            for k in sf.keys():
                if k in wanted:
                    loaded[k] = sf.get_tensor(k).to(dtype).to(device, non_blocking = False)
                    seen += 1
    for alias, src in aliases.items():
        # The SAME tensor object, not a copy: where a stage keeps both the embedding and the head,
        # they must stay one parameter or training would update two halves of a weight the model
        # requires to be shared.
        if alias not in loaded and src in loaded and _has_parameter(model, alias):
            loaded[alias] = loaded[src]
    model.load_state_dict(loaded, strict = False, assign = True)
    # assign=True installs a separate Parameter per name even for one tensor, so the optimizer would
    # step a tied weight twice: put the source's Parameter object back under the alias.
    for alias, src in aliases.items():
        if alias in loaded and _has_parameter(model, alias) and _has_parameter(model, src):
            owner_name, _, leaf = alias.rpartition(".")
            source = model.get_parameter(src)
            setattr(model.get_submodule(owner_name) if owner_name else model, leaf, source)

    # Buffers were computed on the CPU by _parameters_on_meta; move them beside the weights.
    for mod in model.modules():
        for name, buf in mod._buffers.items():
            if buf is not None and not buf.is_meta:
                mod._buffers[name] = buf.to(device)

    still = [k for k, v in model.named_parameters() if v.is_meta] + [
        k for k, v in model.named_buffers() if v.is_meta
    ]
    log(f"shard-load: materialised {seen} tensors; {len(still)} still on meta")
    if still:
        raise RuntimeError(f"unmaterialised tensors remain: {still[:4]}")


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
        self.chunks = chunks or []
        self.n_chunks = n_chunks or world
        self.chunk_layers = {}
        self.is_first = rank == 0
        self.is_last = rank == world - 1
        # unwrap_stack, not `base_model.model`: on a bare causal LM `base_model` is already the
        # decoder stack, so `.model` raised AttributeError before the first step of any
        # --full-finetune run on this backend. get_base_model() is the PEFT discriminator.
        self.base, self.inner = unwrap_stack(model)
        self.embed_scale = _embed_scale(self.base.config, self.inner.embed_tokens)
        self.hidden = self.base.config.hidden_size
        # Set by the caller.
        self.grad_checkpoint = False

    def _run_layer(self, layer, h, pos):
        import torch

        def call(layer, h, pos):
            out = layer(h, position_embeddings = pos)
            return out[0] if isinstance(out, tuple) else out

        if self.grad_checkpoint and self.model.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(call, layer, h, pos, use_reentrant = False)
        return call(layer, h, pos)

    def forward_chunk(self, ids, hidden, posid, chunk):
        import torch

        layers = self.chunk_layers[chunk]
        h = (
            _scale_embeddings(self.inner.embed_tokens(ids), self.embed_scale)
            if chunk == 0
            else hidden
        )
        pos = self.inner.rotary_emb(h, posid)
        for i in layers:
            h = self._run_layer(self.inner.layers[i], h, pos)
        if chunk != self.n_chunks - 1:
            return h, None
        import torch.nn.functional as F

        logits = _softcap_logits(self.base.config, self.base.lm_head(self.inner.norm(h)))
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

        h = (
            _scale_embeddings(self.inner.embed_tokens(ids), self.embed_scale)
            if self.is_first
            else hidden
        )
        pos = self.inner.rotary_emb(h, posid)
        for layer in self.inner.layers:
            if isinstance(layer, torch.nn.Identity):
                continue
            h = self._run_layer(layer, h, pos)
        if not self.is_last:
            return h, None
        import torch.nn.functional as F

        logits = _softcap_logits(self.base.config, self.base.lm_head(self.inner.norm(h)))
        loss = (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                ids[:, 1:].reshape(-1),
            )
            / self.microbatches
        )
        return h, loss


def warmup_p2p(stage, dist, torch):
    """Mirrored one-element exchange in each direction, which pins the ordered per-pair p2p channel
    open (see the root-cause note above `_p2p_group`)."""
    tiny = torch.zeros(1, dtype = stage.dtype, device = stage.device)
    if not stage.is_last:
        dist.send(tiny, dst = stage.rank + 1)
        dist.recv(tiny, src = stage.rank + 1)
    if not stage.is_first:
        dist.recv(tiny, src = stage.rank - 1)
        dist.send(tiny, dst = stage.rank - 1)


# ROOT CAUSE of the first-step hangs: un-batched `isend`/`irecv` between one pair of ranks share a
# single ORDERED p2p stream, so an op enqueued behind a receive with no matching send yet never
# launches.


def _p2p_group(dist, ops):
    """Issue point-to-point ops as one batched group and wait."""
    if not ops:
        return
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def _check_schedule(name, stage, batches):
    """Refuse a configuration the schedule cannot honour, loudly and early: a 1-stage "pipeline" that
    sends to itself, an empty microbatch list, or a chunk map for another rank all otherwise produce
    a run that completes and trains the wrong thing."""
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
    """All forwards, then all backwards."""
    _check_schedule("gpipe", stage, batches)
    acts, held = [], []
    total = torch.zeros((), device = stage.device)
    # Pre-posted: a blocking `recv` issued only when the value is wanted makes the sender wait for
    # the receiver, serialising the stages.
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

    gbufs, gr = {}, {}
    if not stage.is_last:
        for m in range(len(batches)):
            gbufs[m] = torch.empty_like(acts[m][0])
            gr[m] = dist.irecv(gbufs[m], src = stage.rank + 1)

    for m in range(len(batches)):
        h, hidden = acts[m]
        if stage.is_last:
            held[m].backward()
            total = total + held[m].detach()
        else:
            gr[m].wait()
            h.backward(gbufs[m])
            gbufs[m] = None
        if not stage.is_first:
            dist.send(hidden.grad.contiguous(), dst = stage.rank - 1)
    return total


def run_1f1b(stage, batches, posid, mb_rows, dist, torch):
    """One forward, one backward, with a per-rank warmup depth."""
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
# instrumented run is NOT a valid throughput measurement.
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
    acts[m] = (None, None)
    held[m] = None
    return contribution


def run_interleaved(stage, batches, posid, mb_rows, dist, torch):
    """Interleaved (virtual-stage) pipeline parallelism."""
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
    """Zero-bubble pipeline parallelism (Qi et al., 2023): the only schedule whose ceiling is the
    device count rather than `world * M/(M+1)`."""
    _check_schedule("zerobubble", stage, batches)
    M = len(batches)
    seq = posid.shape[1]

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

    # The gradient receives are posted HERE, after every forward send, not inside the loop above:
    # posting `irecv(g_m)` between `isend(h_m)` and `isend(h_{m+1})` puts an unmatchable receive in
    # front of a send the peer is waiting for, and hung this schedule before step 1.
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


# Why `torch.distributed.pipelining` is the default.

_STAGE_MODULE_CLS = None


def _dist_pipelining_available() -> bool:
    """Is `torch.distributed.pipelining` importable here?"""
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
    defeat shard loading."""
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
    """Return `(causal_lm, decoder_stack)` through PEFT and the HF wrappers."""
    top = model.get_base_model() if hasattr(model, "get_base_model") else model
    owner, _ = find_layers(top)
    return top, owner


def stage_module_cls():
    """The `nn.Module` one pipeline stage runs."""
    global _STAGE_MODULE_CLS
    if _STAGE_MODULE_CLS is not None:
        return _STAGE_MODULE_CLS
    import torch

    class _PPStageModule(torch.nn.Module):
        """A contiguous run of decoder layers, plus the embedding on the first stage and the norm +
        lm_head on the last."""

        def __init__(self, top, owner, layer_ids, *, is_first, is_last, grad_checkpoint):
            super().__init__()
            self.is_first, self.is_last = bool(is_first), bool(is_last)
            self.grad_checkpoint = bool(grad_checkpoint)
            container_name, container = _first_named(owner, _LAYER_CONTAINER_NAMES)
            if container is None:
                raise RuntimeError(
                    f"no decoder layer container on {type(owner).__name__}; tried "
                    f"{_LAYER_CONTAINER_NAMES}"
                )
            self.layers = torch.nn.ModuleList([container[i] for i in layer_ids])
            self.rotary_emb = getattr(owner, "rotary_emb", None)
            # GPT-J (and GPT-NeoX before transformers 4.48) rotate inside the block from position_ids;
            # Gemma 2/3 below transformers 4.53 slice a sliding-window mask by cache_position.
            block_params = _forward_params(self.layers[0]) if len(self.layers) else set()
            self.pass_position_ids = "position_ids" in block_params
            self.pass_cache_position = "cache_position" in block_params
            # How a block wants its rope tables is read off its own signature, because it is not one
            # shape.
            self.position_params = _forward_params(self.layers[0]) if len(self.layers) else set()
            self.position_params = sorted(
                p for p in self.position_params if p.startswith("position_embeddings")
            )
            self.rotary_for = {}
            for name in self.position_params:
                suffix = name[len("position_embeddings") :].lstrip("_")
                found = (
                    self.rotary_emb
                    if suffix in ("", "global")
                    else getattr(owner, f"rotary_emb_{suffix}", None)
                )
                self.rotary_for[name] = self.rotary_emb if found is None else found
            self.rotary_wants_layer_type = self.rotary_emb is not None and (
                "layer_type" in _forward_params(self.rotary_emb)
            )
            # Sliding-window attention is not reproduced here, and below the window it does not need
            # to be: every query already reaches every earlier token, so the two masks are the same
            # matrix.
            cfg = getattr(top, "config", None)
            window = getattr(cfg, "sliding_window", None)
            self.sliding_window = int(window) if isinstance(window, int) and window > 0 else None
            # sdpa and flash derive causality from is_causal when attention_mask is None, but eager
            # only masks what it is given: transformers' eager_attention_forward adds the mask under
            # `if attention_mask is not None`, so passing None there trains the model
            # bidirectionally, at a flattering loss, and the checkpoint is not a causal LM.
            impl = getattr(getattr(top, "config", None), "_attn_implementation", "sdpa")
            self.needs_causal_mask = impl not in (
                "sdpa",
                "flash_attention_2",
                "flash_attention_3",
            )
            embed_name, embed = _first_named(owner, _EMBED_NAMES)
            norm_name, norm = _first_named(owner, _FINAL_NORM_NAMES)
            # Every stage checks the whole stack, not just the part it runs: a dropped module is
            # wrong for the model however the layers happen to be divided up.
            skipped = _unrun_parameters(
                owner, (embed_name, norm_name, container_name, "rotary_emb")
            )
            if skipped:
                # GPT-2 keeps learned positions in `wpe` and OPT in `embed_positions`, neither of
                # which lives inside a decoder layer, so running the layers alone gives the model no
                # position information at all.
                raise RuntimeError(
                    f"{type(owner).__name__} carries {sorted(skipped)}, which the pipeline stage "
                    f"does not run, so a split would train a different model than the "
                    f"checkpoint. Train this architecture on a single node. NOT "
                    f"`--pp-backend legacy`: that stage hard-codes `embed_tokens`, "
                    f"`rotary_emb`, `layers` and `norm`, so it raises AttributeError here "
                    f"rather than supporting these models."
                )
            self.embed_tokens = embed if is_first else None
            # GPT-NeoX (`emb_dropout`) and GPT-J (`drop`) apply dropout between the embedding and the
            # first layer.
            dropout = _first_attr(owner, ("emb_dropout", "drop"))
            self.embed_dropout = (
                dropout if is_first and isinstance(dropout, torch.nn.Dropout) else None
            )
            self.norm = norm if is_last else None
            self.lm_head = _first_attr(top, ("lm_head", "embed_out")) if is_last else None
            self.config = getattr(top, "config", None)
            self.embed_scale = _embed_scale(self.config, embed) if is_first else None
            self.pass_position_embeddings = True
            if len(self.layers):
                try:
                    self.pass_position_embeddings = (
                        "position_embeddings"
                        in inspect.signature(type(self.layers[0]).forward).parameters
                    )
                except (TypeError, ValueError):
                    pass
            if self.is_first and not isinstance(self.embed_tokens, torch.nn.Module):
                raise RuntimeError(
                    f"the first pipeline stage has no embedding to run; tried {_EMBED_NAMES} "
                    f"on {type(owner).__name__}"
                )
            if self.is_last and not isinstance(self.norm, torch.nn.Module):
                # Not optional, and silence here is the dangerous outcome: skipping the final
                # normalisation trains and saves a model whose last stage is subtly wrong.
                raise RuntimeError(
                    f"the last pipeline stage has no final normalisation to run; tried "
                    f"{_FINAL_NORM_NAMES} on {type(owner).__name__}"
                )
            if self.is_last and self.lm_head is None:
                raise RuntimeError("the last pipeline stage has no lm_head to run")

        @staticmethod
        def _layer_type(layer):
            for holder in (
                layer,
                getattr(layer, "self_attn", None),
                getattr(layer, "attention", None),
            ):
                for name in ("layer_type", "attention_type"):
                    found = getattr(holder, name, None) if holder is not None else None
                    if isinstance(found, str):
                        return found
            return None

        def _rotary(self, layer, h, ids):
            """The rope tables this block asks for by name, empty when it asks for none."""
            out = {}
            for name, rotary in self.rotary_for.items():
                if rotary is None:
                    continue
                out[name] = (
                    rotary(h, ids, layer_type = self._layer_type(layer))
                    if self.rotary_wants_layer_type
                    else rotary(h, ids)
                )
            return out

        def _call_layer(
            self,
            layer,
            h,
            pos,
            mask,
            ids = None,
        ):
            if self.pass_position_ids:
                pos = {**pos, "position_ids": ids}
            if self.pass_cache_position:
                pos = {**pos, "cache_position": ids[0]}
            out = layer(h, attention_mask = mask, **pos)
            return out[0] if isinstance(out, tuple) else out

        def forward(self, x):
            h = _scale_embeddings(self.embed_tokens(x), self.embed_scale) if self.is_first else x
            if self.embed_dropout is not None:
                h = self.embed_dropout(h)
            if self.sliding_window is not None and h.shape[1] > self.sliding_window:
                raise RuntimeError(
                    f"this model attends over a {self.sliding_window}-token sliding window and "
                    f"the batch is {h.shape[1]} tokens, which a pipeline stage does not "
                    f"reproduce; train at a sequence length of {self.sliding_window} or fewer"
                )
            ids = None
            if self.rotary_emb is not None or self.pass_position_ids or self.pass_cache_position:
                ids = torch.arange(h.shape[1], device = h.device)
                ids = ids.unsqueeze(0).expand(h.shape[0], -1)
            # Only recomputed per layer when the tables actually differ per layer.
            pos = {} if self.rotary_wants_layer_type else self._rotary(None, h, ids)
            mask = None
            if self.needs_causal_mask:
                # Additive, upper triangle excluding the diagonal, broadcast over batch and heads.
                length = h.shape[1]
                mask = torch.full(
                    (length, length), torch.finfo(h.dtype).min, device = h.device, dtype = h.dtype
                ).triu(1)[None, None]
            ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
            for layer in self.layers:
                if self.rotary_wants_layer_type:
                    pos = self._rotary(layer, h, ids)
                if ckpt:
                    h = torch.utils.checkpoint.checkpoint(
                        self._call_layer, layer, h, pos, mask, ids, use_reentrant = False
                    )
                else:
                    h = self._call_layer(layer, h, pos, mask, ids)
            if self.is_last:
                h = _softcap_logits(self.config, self.lm_head(self.norm(h)))
            return h

    _STAGE_MODULE_CLS = _PPStageModule
    return _STAGE_MODULE_CLS


# Valid target tokens in this step's whole batch over the microbatch count, set by the training
# loop. Each microbatch then divides its summed loss by it, so after the 1/M grad scaling the step
# optimises the full-batch token mean: a per-microbatch mean weighs a short padded row like a long
# one (the gradient-accumulation bug). None falls back to a plain mean.
_PP_TOKENS_PER_MICROBATCH = [None]


def set_pp_batch_tokens(labels, microbatches: int) -> None:
    _PP_TOKENS_PER_MICROBATCH[0] = max(1, int((labels[:, 1:] != -100).sum())) / microbatches


def pp_loss_fn(logits, target):
    """Next-token cross entropy, token-mean over the whole step's batch."""
    import torch.nn.functional as F

    logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
    target = target[:, 1:].reshape(-1)
    per_microbatch = _PP_TOKENS_PER_MICROBATCH[0]
    if per_microbatch is None:
        return F.cross_entropy(logits, target, ignore_index = -100)
    return F.cross_entropy(logits, target, ignore_index = -100, reduction = "sum") / per_microbatch


# `pp_loss_fn` returns M times a microbatch's share, so gradients must be divided by M.
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
    """Assemble upstream `PipelineStage`s and the requested schedule for this rank."""
    import inspect
    import torch
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining import schedules as _schedules
    from torch.distributed.pipelining.microbatch import TensorChunkSpec

    # Cross-check our pure layout against upstream's, which the runtime actually uses for send/recv:
    # a disagreement trains the wrong parameters with no error at all.
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
        stages.append(PipelineStage(mod, idx, plan["num_stages"], device, **stage_kwargs))

    sched_params = inspect.signature(sched_cls.__init__).parameters
    kw = {"loss_fn": pp_loss_fn}
    if "args_chunk_spec" in sched_params:
        kw["args_chunk_spec"] = (TensorChunkSpec(0),)
    # `scale_grads` does not exist before torch 2.7, and 2.6 does no scaling of its own: it was
    # added in 2.7 together with `PipelineStage.scale_grads`.
    upstream_scales_grads = "scale_grads" in sched_params
    if upstream_scales_grads:
        kw["scale_grads"] = PP_SCALE_GRADS
    multi = len(stages) > 1 or "stages" in sched_params
    schedule = sched_cls(stages if multi else stages[0], microbatches, **kw)

    step_params = inspect.signature(schedule.step).parameters
    step_kw = {}
    if "return_outputs" in step_params:
        # Merged logits are `batch x seq x vocab` in fp32, tens of GiB on the loss rank for a tensor
        # nobody reads.
        step_kw["return_outputs"] = False

    def scale_grads_after_step():
        """Upstream's `PipelineStage.scale_grads` for the versions that do not have it."""
        if upstream_scales_grads or not PP_SCALE_GRADS or microbatches == 1:
            return
        # By identity, not per module.
        seen = set()
        for mod in mods:
            for p in mod.parameters():
                if p.grad is not None and id(p) not in seen:
                    seen.add(id(p))
                    p.grad.div_(microbatches)

    where = "upstream" if upstream_scales_grads else "here (this torch has no scale_grads)"
    log(
        f"torch.distributed.pipelining: {sched_cls.__name__} "
        f"{plan['num_stages']} stages ({plan['stages_per_rank']}/rank, "
        f"{plan['style']}-layout), M={microbatches}, "
        f"scale_grads={PP_SCALE_GRADS} applied by {where}"
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
    return schedule, mods, step_kw, scale_grads_after_step


# Re-enables the refused (deadlocking) schedules so a stack can be taken without editing this file
# mid-measurement.
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

# The union of both backends' names; an upstream-only layout under `--pp-backend legacy` is refused
# by name rather than ignored.
SCHEDULE_CHOICES = sorted(set(SCHEDULES) | set(TORCH_PP_SCHEDULES))
PP_BACKENDS = ("torch", "legacy")

# Per-backend default for --schedule. gpipe is the arm to avoid on this hardware: it holds every
# microbatch's activations at once and peaked at 99.92 GiB against a 121.69 GiB node, where 1f1b
# peaked at 7.34 GiB for identical work. legacy keeps gpipe because its hand-written 1f1b deadlocks
# and is refused, so 1f1b there would be an error, not a default.
DEFAULT_SCHEDULE = {"torch": "1f1b", "legacy": "gpipe"}


def default_schedule(pp_backend: str) -> str:
    return DEFAULT_SCHEDULE.get(pp_backend, "1f1b")


# Hybrid-attention models (Qwen3.5 and friends) run fused linear-attention kernels when
# flash-linear-attention and causal-conv1d are importable and a slow torch fallback when they are
# not.
FAST_PATH_PENALTY = "2.1x slower (877 -> 418 tok/s measured on Qwen3.5-9B, two Sparks)"
FAST_PATH_INSTALL = "pip install flash-linear-attention causal-conv1d"


def fast_path_warning(
    model_type: Optional[str], fast_path_available: Optional[bool], missing: Sequence[str]
) -> Optional[str]:
    """The line to print when a model that HAS a fused attention fast path is about to run without it."""
    if fast_path_available is not False:
        return None
    named = ", ".join(missing) if missing else "its fused kernels"
    return (
        f"WARNING: {model_type or 'this model'} has a fused attention fast path and it is "
        f"NOT available here (missing: {named}). This run will be about {FAST_PATH_PENALTY}, "
        f"and the only other sign is one line from transformers during the weight load. "
        f"Install it on BOTH nodes, or the two ranks will not even be slow in the same way: "
        f"{FAST_PATH_INSTALL}"
    )


def check_fast_path(
    model_name: str,
    log = print,
    use_cpu: bool = False,
) -> Optional[str]:
    """Advisory only: warn when this interpreter will take the slow attention path."""
    if use_cpu:
        return None  # the fused kernels are CUDA-only; the gloo path has nothing to lose
    try:
        import importlib

        from transformers import AutoConfig

        model_type = getattr(AutoConfig.from_pretrained(model_name), "model_type", None)
        if not model_type:
            return None
        module = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        available = getattr(module, "is_fast_path_available", None)
        if available is not None:
            available = bool(available)
        from transformers.utils import import_utils

        missing = [
            name
            for name, probe in (
                ("flash-linear-attention", "is_flash_linear_attention_available"),
                ("causal-conv1d", "is_causal_conv1d_available"),
            )
            if not getattr(import_utils, probe, lambda: True)()
        ]
        message = fast_path_warning(model_type, available, missing)
    except Exception:
        return None
    if message:
        log(message)
    return message


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
    # Deliberately None, resolved by `default_schedule()` once --pp-backend is known: the right
    # default differs per backend and argparse cannot express that.
    p.add_argument(
        "--schedule",
        choices = SCHEDULE_CHOICES,
        default = None,
        help = "default 1f1b on --pp-backend torch. Measured on two Sparks vs one: 1f1b "
        "1.94x, dualpipev 1.96x, zbv 1.94x, interleaved 1.93x, gpipe 1.86x, zerobubble "
        "1.72x. 1f1b wins over dualpipev because the 0.7%% gap is within noise while its "
        "peak memory is lower (7.34 vs 9.58 GiB). Avoid gpipe: it holds every "
        "microbatch's activations at once and peaked at 99.92 GiB for the same work, "
        "against a 121.69 GiB node. On --pp-backend legacy the default stays gpipe, "
        "because the hand-written 1f1b deadlocks and is refused.",
    )
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
    if args.schedule is None:
        args.schedule = default_schedule(args.pp_backend)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
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
    if args.data and not use_torch_pp:
        raise SystemExit("--data needs --pp-backend torch; legacy cannot mask padded labels")
    if args.data:
        problem = dataset_problem(args.data)
        if problem:
            raise SystemExit(problem)
    # Fail before the tokenizer and model load, so the reason appears in a second instead of a
    # silent process.
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
    if tok.pad_token is None:
        # Base decoder-only checkpoints ship without one, and padding then raises before the first
        # step.
        tok.pad_token = tok.eos_token
    # Right padding keeps every real token preceded only by real tokens, so a causal model needs no
    # padding mask for the representations; only the labels have to exclude pads.
    tok.padding_side = "right"

    # Before the weights load, so the sentence is not buried in the loading output.
    check_fast_path(args.model, log = log, use_cpu = use_cpu)

    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(args.model)
    n_layers_total = config_num_layers(base_config)

    plan = my_plan = None
    if use_torch_pp:
        n_layers = n_layers_total
        try:
            plan = torch_pp_plan(
                args.schedule, world, args.microbatches, args.virtual_stages, n_layers
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        my_plan = plan_for_rank(plan, rank)

    # Checked before the model is built, so a run that cannot be correct stops in seconds rather
    # than after a 70B load.
    tied_problem = tied_split_problem(
        bool(getattr(base_config, "tie_word_embeddings", False)),
        bool(args.full_finetune),
        world,
        plan["stage_to_rank"] if plan else None,
    )
    if tied_problem:
        raise SystemExit(tied_problem)
    if not use_torch_pp:
        legacy_problem = legacy_attention_problem(base_config, int(args.seq))
        if legacy_problem:
            raise SystemExit(legacy_problem)
    if args.data and getattr(tok, "chat_template", None) is None:
        # `apply_chat_template` raises on a base tokenizer, and it did so only after both ranks had
        # loaded and materialised the model.
        raise SystemExit(
            f"--data formats each row with the tokenizer's chat template, and {args.model} "
            f"has none (it is a base checkpoint). Point --model at an instruction-tuned "
            f"checkpoint, or drop --data to train on synthetic ids."
        )
    save_problem = full_finetune_save_problem(bool(args.full_finetune), args.save or "", world)
    if save_problem:
        raise SystemExit(save_problem)

    # Multi-stage layouts own non-contiguous chunks, so the contiguous drop-to-Identity would remove
    # layers this rank needs; the legacy interleaved path has no such set and keeps the whole stack
    # instead.
    torch.manual_seed(TRAIN_SEED)
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
                target_modules = lora_target_modules(model),
            ),
        )
    model.train()
    if args.grad_checkpoint and use_torch_pp:
        # `_PPStageModule` calls the decoder layers directly, so transformers'
        # `gradient_checkpointing_enable()` is consulted in a `forward` never reached here and would
        # be inert; the stage module wraps each layer itself.
        log("gradient checkpointing enabled per decoder layer (use_reentrant=False)")
    elif args.grad_checkpoint:
        # The legacy stage also calls decoder layers directly, so it wraps each one itself for the
        # same reason.
        base_model = getattr(model, "base_model", model)
        inner_model = getattr(base_model, "model", base_model)
        if hasattr(inner_model, "gradient_checkpointing_enable"):
            inner_model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(inner_model, "gradient_checkpointing_kwargs"):
            inner_model.gradient_checkpointing_kwargs = {"use_reentrant": False}
        log(
            "gradient checkpointing enabled per decoder layer and on the model (use_reentrant=False)"
        )

    model.to(device)  # shard-load already placed the base; this catches new adapters
    if not use_cpu:
        torch.cuda.empty_cache()

    trainable = [p for p in model.parameters() if p.requires_grad]
    resident = f"{torch.cuda.memory_allocated()/2**30:.2f} GiB" if not use_cpu else "cpu"
    log(
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f} B params resident "
        f"({resident}), "
        f"{sum(p.numel() for p in trainable)/1e6:.1f} M trainable"
    )

    if args.batch % args.microbatches:
        raise SystemExit("--batch must be divisible by --microbatches")
    mb_rows = args.batch // args.microbatches

    # A split only pays while the work is COMPUTE-bound.
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
        pp_schedule, _pp_mods, pp_step_kw, pp_scale_grads = build_torch_schedule(
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
        stage.grad_checkpoint = bool(args.grad_checkpoint)
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
        stage.grad_checkpoint = bool(args.grad_checkpoint)
    if not use_torch_pp:
        is_loss_rank = stage.is_last
    opt = torch.optim.AdamW(trainable, lr = args.lr)

    # Again, so the synthetic data does not depend on how many draws the model construction above
    # happened to take.
    torch.manual_seed(TRAIN_SEED)
    need = args.batch * args.steps
    if args.data:
        # `dataset_problem()` accepts a file with a trailing blank line, since it only needs one
        # nonblank row; this used to hand every raw line to `json.loads`, so the JSONDecodeError
        # arrived after both ranks had materialised the model.
        rows = [json.loads(line) for line in open(args.data, encoding = "utf-8") if line.strip()]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}],
                tokenize = False,
            )
            for r in rows
        ]
        enc = tok(
            texts,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = args.seq,
            # The template already rendered BOS/EOS into the text.
            add_special_tokens = False,
        )
        ids = enc.input_ids
        # Padded positions are not text.
        labels = ids.masked_fill(enc.attention_mask == 0, -100)
        reps = (need + len(ids) - 1) // len(ids)
        ids_all = ids.repeat(reps, 1)[:need]
        labels_all = labels.repeat(reps, 1)[:need]
    else:
        ids_all = torch.randint(0, tok.vocab_size, (need, args.seq))
        labels_all = ids_all
    # Kept on the CPU and moved one step at a time: batch x steps x seq on every GPU grows with
    # --steps, which `spark estimate` has no term for.

    posid = torch.arange(args.seq, device = device).unsqueeze(0).expand(mb_rows, -1)
    schedule = SCHEDULES[args.schedule] if not use_torch_pp else None

    dist.barrier()
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none = True)
        if use_torch_pp:
            # Upstream chunks the whole batch; that chunking must agree with `target`'s, and one
            # implementation owning both is how they stay agreed.
            whole = ids_all[step * args.batch : (step + 1) * args.batch].to(device)
            whole_labels = labels_all[step * args.batch : (step + 1) * args.batch].to(device)
            losses = [] if is_loss_rank else None
            # Only the rank holding stage 0 may supply positional inputs; every other stage's input
            # is the wire.
            step_args = (whole,) if rank == plan["first_rank"] else ()
            set_pp_batch_tokens(whole_labels, args.microbatches)
            pp_schedule.step(*step_args, target = whole_labels, losses = losses, **pp_step_kw)
            pp_scale_grads()
            loss = (sum(losses) / len(losses)) if losses else None
        else:
            batches = [
                ids_all[step * args.batch + m * mb_rows :][:mb_rows].to(device)
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
        # How many stages there were meant to be.
        with open(osp.join(out, "unsloth_stage.json"), "w", encoding = "utf-8") as handle:
            json.dump({"rank": rank, "world": world, "n_layers": n_layers_total}, handle)
        log(f"saved stage {rank} to {out}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
