# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which parallel compositions run, in one table, consulted before anything touches the model.

Three tables, the only place features meet:

- `FEATURES`: every feature (parallel axes, precision, compile, attention backend, packing, GC / offload,
  kernels, fused losses) declares what it takes from the root mesh (parallel_env.ParallelLayout), the batch
  metadata it consumes and produces (positions, packed lengths / cu_seqlens, BlockMask, the CP permutation,
  shifted labels, the sequence shard) and the hooks it installs (and whether it must own a hook alone).
  `check_features` refuses a set of active features whose meshes are missing, whose consumed metadata nobody
  produces, or that claim one exclusive hook twice. Features never patch each other.
- `RULES`: known refusals; each names its axis and what to run instead.
- `CELLS`: the composed layouts that are supported, each with its three kinds of evidence (numeric parity vs
  one process, engagement: the feature really ran, compile: graph breaks and recompiles not above one process),
  as test ids. A composition M1 opened (context parallel with tensor / pipeline parallel) runs only when a
  cell with all three covers it; anything else is refused with the evidence that is missing. Layouts that
  shipped before the registry (tp, pp, dp alone or together) report their status (`cell_status`) and run;
  UNSLOTH_PARALLEL_STRICT=1 refuses them too until their cells carry the evidence.

A request is a `Query` over every axis: trainer, model family (hybrid, multimodal, windowed), mode (LoRA, QLoRA,
full, FP8, NVFP4, MXFP8), packing, attention backend, compile, memory (GC / offload), kernels, loss, the CP
backend, torch / sm, and the layout. `check` runs `check_features`, `RULES` and the cell gate, in that order.
`markdown_table` prints the refusals and the cells (docs, launcher README).

Rules about one axis alone stay where that axis is implemented (context_parallel.py, models/pp_env.py,
models/tp_env.py); the rows here that repeat them run earlier, before the split, so a refused launch leaves the
model untouched.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

__all__ = [
    "Query",
    "Rule",
    "RULES",
    "Feature",
    "FEATURES",
    "Cell",
    "CELLS",
    "METADATA",
    "HOOKS",
    "active_features",
    "check_features",
    "cell_status",
    "check",
    "check_layout",
    "query_from",
    "markdown_table",
]


@dataclasses.dataclass(frozen = True)
class Query:
    trainer: str = "trainer"  # trainer, sft, dpo, grpo, other_rl, other
    family: str = ""  # config.model_type
    hybrid: bool = False  # linear attention / state space layers beside attention
    multimodal: bool = False
    windowed: bool = False  # sliding window, chunked attention or logit softcapping
    mode: str = "lora"  # lora, qlora, full
    packing: bool = False
    cp_backend: Optional[str] = None  # ulysses, ring (resolved)
    torch: tuple = (0, 0)
    sm: Optional[int] = None  # CUDA compute capability major, None off CUDA
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    fsdp2: bool = False
    vllm: bool = False
    pp_schedule: Optional[str] = None  # gpipe, 1f1b, interleaved
    attention: str = "sdpa"  # sdpa, eager, flash, flash_varlen, fa3, fa4, flex, flex_sparse
    compile: str = "eager"  # eager, block (Unsloth's per decoder layer torch.compile)
    memory: str = "none"  # none, gc (torch checkpoint), unsloth_gc, unsloth_offload, tiled_mlp
    kernels: bool = False  # Unsloth's fused RoPE / RMSNorm / SwiGLU / LoRA kernels
    loss: str = "standard"  # standard, fused_ce, chunked_ce, fused_rl


@dataclasses.dataclass(frozen = True)
class Rule:
    axis: str
    when: str  # the cell, in words (the table's first column)
    applies: Callable[[Query], bool]
    reason: str
    alternative: str


def _rl(q):
    return q.trainer in ("dpo", "grpo", "other_rl")


RULES = (
    Rule(
        "cp x pp",
        "context parallel inside pipeline stages",
        lambda q: q.cp > 1 and q.pp > 1,
        "context parallel inside pipeline stages is not composed yet (stage boundaries would carry sequence "
        "shards the schedule does not know about)",
        "drop cp, or drop pp (cp composes with tp and with DDP / FSDP2)",
    ),
    Rule(
        "cp x tp",
        "context parallel x tensor parallel with the ring backend",
        lambda q: q.cp > 1 and q.tp > 1 and q.cp_backend == "ring",
        "ring attention on tensor parallel heads is not verified",
        "cp_backend=ulysses (or auto)",
    ),
    Rule(
        "cp x tp",
        "context parallel x tensor parallel for DPO / GRPO",
        lambda q: q.cp > 1 and q.tp > 1 and _rl(q),
        "the DPO / GRPO context parallel hooks build their own mesh",
        "run DPO / GRPO with cp alone (DDP), or with tp and cp = 1",
    ),
    Rule(
        "cp x tp",
        "context parallel x tensor parallel on hybrid (linear attention) or multimodal models",
        lambda q: q.cp > 1 and q.tp > 1 and (q.hybrid or q.multimodal),
        "tensor parallel keeps the linear attention layers replicated, and that path is not verified with "
        "Ulysses",
        "drop tp (Ulysses cp runs Qwen3-Next / Qwen3.5 hybrids alone or with DDP / FSDP2)",
    ),
    Rule(
        "cp x tp",
        "context parallel x tensor parallel with packed rows",
        lambda q: q.cp > 1 and q.tp > 1 and q.packing,
        "packed rows under cp x tp are not verified",
        "packing = False, or drop tp",
    ),
    Rule(
        "cp x tp",
        "context parallel x tensor parallel on full finetuning",
        lambda q: q.cp > 1 and q.tp > 1 and q.mode == "full",
        "tensor parallel trains LoRA adapters (full finetuning is refused by UNSLOTH_TP_SIZE)",
        "use LoRA, or UNSLOTH_FSDP2=1 with cp for full finetuning",
    ),
    Rule(
        "cp backend",
        "ring context parallel with packed rows",
        lambda q: q.cp > 1 and q.cp_backend == "ring" and q.packing,
        "ring attention cannot see document boundaries in a packed row",
        "cp_backend=ulysses (or auto), or packing = False",
    ),
    Rule(
        "cp backend",
        "ring context parallel on hybrid (linear attention) models",
        lambda q: q.cp > 1 and q.cp_backend == "ring" and q.hybrid,
        "ring attention cannot split the sequential scans of linear attention layers",
        "cp_backend=ulysses (or auto)",
    ),
    Rule(
        "cp backend",
        "ring context parallel for DPO / GRPO",
        lambda q: q.cp > 1 and q.cp_backend == "ring" and _rl(q),
        "the DPO / GRPO context parallel hooks shard decoder inputs for Ulysses only",
        "cp_backend=ulysses (or auto)",
    ),
    Rule(
        "cp backend",
        "ring context parallel on GPUs below compute capability 8.0",
        lambda q: q.cp > 1 and q.cp_backend == "ring" and q.sm is not None and q.sm < 8,
        "torch ring attention fails in backward without flash attention (T4, V100)",
        "cp_backend=ulysses (or auto)",
    ),
    Rule(
        "cp x attention",
        "context parallel on sliding window / chunked / softcapped attention",
        lambda q: q.cp > 1 and q.windowed,
        "Unsloth's context parallel attention drops the window / chunk / softcap",
        "drop cp for this model",
    ),
    Rule(
        "cp x torch",
        "context parallel on torch < 2.7",
        lambda q: q.cp > 1 and q.torch < (2, 7),
        "torch's context parallel API arrived in 2.7",
        "upgrade torch, or drop cp",
    ),
    Rule(
        "pp x trainer",
        "pipeline parallel for DPO / GRPO / other RL trainers",
        lambda q: q.pp > 1 and (_rl(q) or q.trainer == "other"),
        "pipeline parallel v1 runs the Trainer / SFTTrainer causal LM loss only",
        "drop pp (tp, cp or FSDP2 run the RL trainers' whole model per rank)",
    ),
    Rule(
        "pp x packing",
        "pipeline parallel with packed / padding-free rows",
        lambda q: q.pp > 1 and q.packing,
        "packed rows are not carried between pipeline stages yet",
        "packing = False, or drop pp",
    ),
    Rule(
        "pp / tp x mode",
        "pipeline / tensor parallel full finetuning",
        lambda q: (q.pp > 1 or q.tp > 1) and q.mode == "full",
        "pipeline and tensor parallel train LoRA adapters",
        "use LoRA, or UNSLOTH_FSDP2=1 (dp) for full finetuning",
    ),
    Rule(
        "pp x schedule",
        "interleaved pipeline schedules",
        lambda q: q.pp > 1 and q.pp_schedule == "interleaved",
        "one stage per rank in v1",
        "pp_schedule=gpipe or 1f1b",
    ),
    Rule(
        "cp x attention",
        "context parallel with flex attention block masks",
        lambda q: q.cp > 1 and q.attention in ("flex", "flex_sparse"),
        "the BlockMask is built for the whole sequence, not for the rank-local (permuted) shard",
        "attention sdpa / flash, or drop cp",
    ),
    Rule(
        "cp x precision",
        "context parallel with FP8 / NVFP4 / MXFP8 training",
        lambda q: q.cp > 1 and q.mode in ("fp8", "nvfp4", "mxfp8"),
        "low precision training is not composed with the sequence exchange",
        "LoRA / QLoRA / full in 16-bit, or drop cp",
    ),
    Rule(
        "tp / pp x precision",
        "tensor / pipeline parallel with FP8 / NVFP4 / MXFP8 training",
        lambda q: (q.tp > 1 or q.pp > 1) and q.mode in ("fp8", "nvfp4", "mxfp8"),
        "the low precision linear layers are not sharded by tensor parallel or split by pipeline parallel",
        "UNSLOTH_FSDP2=1 (dp) for FP8 / NVFP4, or 16-bit LoRA with tp / pp",
    ),
    Rule(
        "pp / tp x vLLM",
        "pipeline / tensor parallel with vLLM (fast_inference)",
        lambda q: (q.pp > 1 or q.tp > 1) and q.vllm,
        "vLLM needs the whole model on each rank",
        "fast_inference = False, or drop tp / pp",
    ),
)


# Batch metadata a feature can consume or produce. "batch" means the collator's own keys.
METADATA = (
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",  # per token positions (global ones once the sequence is sharded)
    "packed_seq_lengths",  # document lengths of a packed row (Unsloth's packing)
    "cu_seqlens",  # cumulative lengths for varlen kernels, derived from packed_seq_lengths
    "block_mask",  # flex attention BlockMask
    "cp_permutation",  # ring CP load balancing: the order tokens were sharded in
    "seq_shard",  # this rank holds a slice of the sequence
    "shift_labels",  # next-token labels built before sharding (a shard cannot see its neighbour)
    "num_items_in_batch",  # the accumulation window's token count (loss normaliser)
    "stage_hidden",  # hidden states sent between pipeline stages
)
_BATCH = ("input_ids", "labels", "attention_mask", "position_ids", "num_items_in_batch")

# Where a feature hooks in. Exclusive hooks run the step themselves: two owners cannot compose.
HOOKS = (
    "from_pretrained",
    "get_peft_model",
    "Trainer.__init__",
    "Trainer.training_step",
    "Trainer.prediction_step",
    "Trainer.save_model",
    "Accelerator.prepare_model",
    "Accelerator.prepare_data_loader",
    "Accelerator.clip_grad_norm_",
    "optimizer.step",
    "collator",
    "decoder_layer.forward",
    "attention_function",
    "loss_function",
    "torch.utils.checkpoint",
)


@dataclasses.dataclass(frozen = True)
class Feature:
    name: str
    kind: str  # parallel, precision, compile, attention, packing, memory, kernels, loss
    module: str  # where it is implemented
    active: Callable[[Query], bool]
    mesh: tuple = ()  # root mesh axes it takes (parallel_env.ParallelLayout.dims names; "dp" = the dp axis)
    consumes: tuple = ()  # required: someone must produce it
    uses: tuple = ()  # read when present (e.g. packed lengths for Ulysses)
    produces: tuple = ()
    hooks: tuple = ()
    exclusive: tuple = ()  # hooks it must own alone
    scope: str = "instance"  # instance: hooks per trainer / model; class: still patches a class (debt)


def _gc(q):
    return q.memory in ("gc", "unsloth_gc", "unsloth_offload")


FEATURES = (
    # Parallel axes.
    Feature(
        "ddp", "parallel", "transformers / accelerate DDP; tp_native._dp_average under tp / pp",
        lambda q: q.dp > 1 and not q.fsdp2, mesh = ("dp", "cp"),
        consumes = ("num_items_in_batch",), hooks = ("Accelerator.prepare_model", "Accelerator.clip_grad_norm_"),
    ),
    Feature(
        "fsdp2", "parallel", "models/fsdp2_native.py",
        lambda q: q.dp > 1 and q.fsdp2, mesh = ("dp", "cp"),
        consumes = ("num_items_in_batch",),
        hooks = ("Trainer.__init__", "Accelerator.prepare_model", "Accelerator.clip_grad_norm_", "optimizer.step"),
    ),
    Feature(
        "tp", "parallel", "models/tp_native.py, models/tp_env.py",
        lambda q: q.tp > 1, mesh = ("tp",),
        hooks = (
            "from_pretrained", "get_peft_model", "Trainer.__init__", "Accelerator.prepare_model",
            "Accelerator.prepare_data_loader", "Accelerator.clip_grad_norm_", "Trainer.save_model",
        ),
    ),
    Feature(
        "pp", "parallel", "models/pp_native.py, models/pp_env.py",
        lambda q: q.pp > 1, mesh = ("pp",),
        consumes = ("input_ids", "labels", "position_ids", "num_items_in_batch"),
        produces = ("stage_hidden",),
        hooks = ("Trainer.__init__", "Accelerator.prepare_data_loader", "Accelerator.clip_grad_norm_",
                 "Trainer.save_model"),
        exclusive = ("Trainer.training_step", "Trainer.prediction_step"),
    ),
    Feature(
        "cp.ulysses", "parallel", "context_parallel.py, utils/ulysses.py, utils/ulysses_hybrid.py",
        lambda q: q.cp > 1 and q.cp_backend != "ring", mesh = ("cp",),
        consumes = ("input_ids", "labels", "position_ids", "num_items_in_batch"),
        uses = ("packed_seq_lengths",),
        produces = ("seq_shard", "shift_labels", "position_ids"),
        hooks = ("Trainer.__init__", "attention_function"),
        exclusive = ("Trainer.training_step", "Trainer.prediction_step"),
        scope = "class",
    ),
    Feature(
        "cp.ring", "parallel", "context_parallel.py (torch context_parallel)",
        lambda q: q.cp > 1 and q.cp_backend == "ring", mesh = ("cp",),
        consumes = ("input_ids", "labels", "position_ids", "num_items_in_batch"),
        produces = ("seq_shard", "shift_labels", "cp_permutation", "position_ids"),
        hooks = ("Trainer.__init__", "attention_function"),
        exclusive = ("Trainer.training_step", "Trainer.prediction_step"),
        scope = "class",
    ),
    # Precision.
    Feature("lora", "precision", "peft + models/lora_init.py", lambda q: q.mode == "lora", hooks = ("get_peft_model",)),
    Feature(
        "qlora", "precision", "bitsandbytes 4-bit + peft", lambda q: q.mode == "qlora",
        hooks = ("from_pretrained", "get_peft_model"),
    ),
    Feature("full", "precision", "full finetuning", lambda q: q.mode == "full", hooks = ("from_pretrained",)),
    Feature(
        "low_precision", "precision", "models/float8_training.py (FP8 / MXFP8 / NVFP4 via torchao)",
        lambda q: q.mode in ("fp8", "nvfp4", "mxfp8"), hooks = ("from_pretrained", "Trainer.__init__"),
    ),
    # Compile.
    Feature(
        "block_compile", "compile", "models/_block_compile.py (torch.compile per decoder layer)",
        lambda q: q.compile == "block", hooks = ("decoder_layer.forward",),
    ),
    # Attention backends.
    Feature("sdpa", "attention", "torch SDPA", lambda q: q.attention in ("sdpa", "eager"), hooks = ("attention_function",)),
    Feature(
        "flash", "attention", "utils/attention_dispatch.py (flash dense / xformers)",
        lambda q: q.attention == "flash", uses = ("attention_mask",), hooks = ("attention_function",),
    ),
    Feature(
        "flash_varlen", "attention", "utils/attention_dispatch.py (flash / FA3 / FA4 varlen)",
        lambda q: q.attention in ("flash_varlen", "fa3", "fa4"),
        consumes = ("cu_seqlens",), hooks = ("attention_function",),
    ),
    Feature(
        "flex", "attention", "utils/attention_dispatch.py (flex attention)",
        lambda q: q.attention in ("flex", "flex_sparse"),
        consumes = ("position_ids",), produces = ("block_mask",), hooks = ("attention_function",),
    ),
    # Packing.
    Feature(
        "packing", "packing", "utils/packing.py (sample packing / padding-free)",
        lambda q: q.packing, produces = ("packed_seq_lengths", "cu_seqlens", "position_ids"), hooks = ("collator",),
    ),
    # Memory.
    Feature(
        "gradient_checkpointing", "memory", "torch.utils.checkpoint / unsloth_zoo gradient_checkpointing",
        _gc, hooks = ("torch.utils.checkpoint",),
    ),
    Feature(
        "tiled_mlp", "memory", "models/_tiled_mlp_ddp.py + unsloth_zoo tiled MLP",
        lambda q: q.memory == "tiled_mlp", hooks = ("decoder_layer.forward",),
    ),
    # Kernels and losses.
    Feature(
        "unsloth_kernels", "kernels", "kernels/ (RoPE, RMSNorm, SwiGLU, fast LoRA)",
        lambda q: q.kernels, hooks = ("decoder_layer.forward", "attention_function"),
    ),
    Feature(
        "fused_loss", "loss", "unsloth_zoo fused / chunked CE, models/_fused_rl_logprobs.py",
        lambda q: q.loss in ("fused_ce", "chunked_ce", "fused_rl"),
        consumes = ("labels",), hooks = ("loss_function",),
    ),
)


def active_features(query: Query) -> list:
    return [f for f in FEATURES if f.active(query)]


def check_features(query: Query) -> list:
    """Composition from the declarations alone: mesh axes, metadata and exclusive hooks. Returns the active
    feature names; NotImplementedError on the first conflict."""
    active = active_features(query)
    sizes = {"dp": query.dp, "tp": query.tp, "pp": query.pp, "cp": query.cp}
    for feature in active:
        # A data parallel feature reduces over (dp, cp); it needs at least one of them.
        if feature.mesh and feature.kind == "parallel" and not any(sizes.get(a, 1) > 1 for a in feature.mesh):
            raise NotImplementedError(
                f"Unsloth: {feature.name}: needs the {' / '.join(feature.mesh)} mesh axis. ({_describe(query)})"
            )
    produced = set(_BATCH)
    for feature in active:
        produced.update(feature.produces)
    for feature in active:
        missing = [m for m in feature.consumes if m not in produced]
        if missing:
            raise NotImplementedError(
                f"Unsloth: {feature.kind} x metadata: {feature.name} consumes {', '.join(missing)}, which no "
                f"active feature produces. ({_describe(query)})"
            )
    owners = {}
    for feature in active:
        for hook in feature.exclusive:
            if hook in owners:
                raise NotImplementedError(
                    f"Unsloth: {owners[hook]} x {feature.name}: both run {hook} themselves, so neither would "
                    f"see the other's step. ({_describe(query)})"
                )
            owners[hook] = feature.name
    return [f.name for f in active]


@dataclasses.dataclass(frozen = True)
class Cell:
    """A supported composition: `match` holds the allowed values per Query field (a tuple), and the three
    kinds of evidence as test ids. A missing kind is None."""

    name: str
    match: dict
    parity: Optional[str]
    engagement: Optional[str]
    compile: Optional[str]

    def covers(self, query: Query) -> bool:
        return all(getattr(query, k) in v for k, v in self.match.items())

    def missing(self) -> list:
        return [k for k in ("parity", "engagement", "compile") if not getattr(self, k)]


_GLOO = "tests/test_parallel_cp_compose_gloo.py::"
_CP_TP = {
    "trainer": ("sft",),
    "family": ("llama",),
    "hybrid": (False,),
    "multimodal": (False,),
    "windowed": (False,),
    "mode": ("lora",),
    "packing": (False,),
    "cp_backend": ("ulysses",),
    "pp": (1,),
    "attention": ("sdpa", "eager"),
    "memory": ("none", "gc"),
    "kernels": (False,),
    "loss": ("standard",),
    "vllm": (False,),
}
_PARITY = _GLOO + "test_cp_x_tp_from_unsloth_parallel_matches_one_process"
CELLS = (
    Cell(
        "cp x tp (DDP dp), eager",
        {**_CP_TP, "fsdp2": (False,), "compile": ("eager",), "dp": (1, 2), "tp": (2,), "cp": (2,)},
        parity = _PARITY + "[cp2_tp2|dp2_cp2_tp2]",
        engagement = _PARITY + " (seq / cp tokens per rank, one Ulysses exchange per layer, cp group != tp group)",
        compile = "eager: no compiled region",
    ),
    Cell(
        "cp x tp (FSDP2 dp), eager",
        {**_CP_TP, "fsdp2": (True,), "compile": ("eager",), "dp": (2,), "tp": (2,), "cp": (2,)},
        parity = _PARITY + "[fsdp2_cp2_tp2]",
        engagement = _PARITY + "[fsdp2_cp2_tp2] (as above, FSDP2 over the flattened (dp_shard, cp) mesh)",
        compile = "eager: no compiled region",
    ),
    Cell(
        "cp x tp, Unsloth block compile",
        {**_CP_TP, "fsdp2": (False, True), "compile": ("block",), "dp": (1, 2), "tp": (2,), "cp": (2,)},
        parity = _GLOO + "test_block_compile_composes_like_one_process",
        engagement = _GLOO + "test_block_compile_composes_like_one_process (compiled classes, seq / cp tokens)",
        compile = _GLOO + "test_block_compile_composes_like_one_process (graph breaks / recompiles <= one process)",
    ),
)


def _strict() -> bool:
    import os

    return os.environ.get("UNSLOTH_PARALLEL_STRICT", "0").strip().lower() in ("1", "true", "yes", "on")


def _gated(query: Query) -> bool:
    """Compositions opened by the one-flag work: they run only with a cell."""
    return query.cp > 1 and (query.tp > 1 or query.pp > 1)


def cell_status(query: Query) -> tuple:
    """("supported", cell) | ("unverified", reason) | ("single", None) for one axis alone or none."""
    axes = sum(1 for n in (query.tp, query.pp, query.cp) if n > 1) + (query.dp > 1)
    for cell in CELLS:
        if cell.covers(query):
            missing = cell.missing()
            if not missing:
                return ("supported", cell)
            return ("unverified", f"cell {cell.name!r} lacks {', '.join(missing)} evidence")
    if axes <= 1 and not _gated(query):
        return ("single", None)
    differ = []
    for cell in CELLS:
        off = [f"{k}={getattr(query, k)!r}" for k, v in cell.match.items() if getattr(query, k) not in v]
        differ.append(f"{cell.name!r} differs in {', '.join(off)}")
    return ("unverified", "no supported cell covers it" + (f" ({'; '.join(differ[:3])})" if differ else ""))


def check(query: Query, cells: bool = True) -> None:
    for rule in RULES:
        if rule.applies(query):
            raise NotImplementedError(
                f"Unsloth: {rule.axis}: {rule.when} is not supported: {rule.reason}. Instead: "
                f"{rule.alternative}. ({_describe(query)})"
            )
    check_features(query)
    if not cells:
        return  # the layout alone (read at from_pretrained): the trainer checks the cell once it knows the rest
    status, detail = cell_status(query)
    if status == "unverified" and (_gated(query) or _strict()):
        raise NotImplementedError(
            f"Unsloth: composition: this cell has no numeric parity / engagement / compile evidence yet: "
            f"{detail}. Instead: a supported cell (see unsloth.parallel_capabilities.CELLS), or drop an axis. "
            f"({_describe(query)})"
        )


def _describe(q: Query) -> str:
    parts = [f"{name} {getattr(q, name)}" for name in ("pp", "dp", "cp", "tp") if getattr(q, name) > 1]
    if q.cp > 1 and q.cp_backend:
        parts.append(f"cp_backend {q.cp_backend}")
    if q.fsdp2:
        parts.append("FSDP2")
    return "layout " + (" x ".join(parts) or "single rank") + f", trainer {q.trainer}" + (
        f", model {q.family}" if q.family else ""
    )


def _torch_version():
    import torch
    from packaging.version import Version

    v = Version(Version(torch.__version__).base_version)
    return (v.major, v.minor)


def _sm():
    try:
        import torch

        if torch.cuda.is_available() and getattr(torch.version, "cuda", None):
            return torch.cuda.get_device_capability()[0]
    except Exception:
        pass
    return None


def _trainer_kind(trainer_cls) -> str:
    if trainer_cls is None:
        return "trainer"
    names = [c.__name__ for c in getattr(trainer_cls, "__mro__", ())]
    for kind, markers in (
        ("grpo", ("GRPOTrainer",)),
        ("dpo", ("DPOTrainer",)),
        ("sft", ("SFTTrainer",)),
    ):
        if any(n in markers or n == "Unsloth" + markers[0] for n in names):
            return kind
    rl = ("ORPOTrainer", "KTOTrainer", "PPOTrainer", "RLOOTrainer", "CPOTrainer", "OnlineDPOTrainer", "RewardTrainer")
    if any(n in rl or n.removeprefix("Unsloth") in rl for n in names):
        return "other_rl"
    return "trainer" if "Trainer" in names else "other"


def _model_traits(model):
    config = getattr(model, "config", None)
    if config is None and isinstance(model, str):
        return "", False, False, False
    text = getattr(config, "text_config", None) or config
    family = str(getattr(text, "model_type", "") or getattr(config, "model_type", "") or "")
    layer_types = list(getattr(text, "layer_types", None) or [])
    hybrid = any(t not in ("full_attention", "sliding_attention", "chunked_attention") for t in layer_types)
    multimodal = getattr(config, "vision_config", None) is not None
    window = getattr(text, "sliding_window", None)
    windowed = (
        (isinstance(window, int) and window > 0)
        or getattr(text, "use_sliding_window", None) is True
        or any(t in ("sliding_attention", "chunked_attention") for t in layer_types)
        or bool(getattr(text, "attention_chunk_size", None))
        or bool(getattr(text, "attn_logit_softcapping", None))
    )
    return family, hybrid, multimodal, windowed


def _mode(model):
    if model is None or isinstance(model, str):
        return "lora"
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        getattr(model, "config", None), "quantization_config", None
    ):
        return "qlora"
    if hasattr(model, "peft_config") and getattr(model, "peft_config", None):
        return "lora"
    try:
        trainable = any(p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
    except Exception:
        trainable = False
    return "full" if trainable else "lora"


def _compile_mode(model):
    if model is None or isinstance(model, str):
        return "eager"
    try:
        modules = list(model.modules())
    except Exception:
        return "eager"
    seen = set()
    for m in modules:
        cls = type(m)
        if cls in seen:
            continue
        seen.add(cls)
        if getattr(cls.forward, "_unsloth_block_compiled", False):
            return "block"
    return "eager"


def _attention(model, packing):
    config = getattr(model, "config", None)
    impl = str(getattr(config, "_attn_implementation", None) or "sdpa")
    if impl.startswith("flex"):
        return "flex"
    if "flash" in impl or "kernels-community" in impl:
        return "flash_varlen" if packing else "flash"
    return "eager" if impl == "eager" else "sdpa"


def _memory(model, args):
    try:
        import torch.utils.checkpoint as checkpoint

        name = getattr(checkpoint.checkpoint, "__name__", "")
    except Exception:
        name = ""
    gc = bool(getattr(args, "gradient_checkpointing", False)) or bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    if not gc:
        return "none"
    if name == "unsloth_offloaded_gradient_checkpoint":
        return "unsloth_offload"
    if name == "unsloth_gradient_checkpoint":
        return "unsloth_gc"
    return "gc"


def _kernels(model):
    if model is None or isinstance(model, str):
        return False
    try:
        for m in model.modules():
            forward = getattr(type(m), "forward", None)
            if getattr(forward, "__module__", "").startswith("unsloth.models.") and type(m).__name__.endswith(
                "Attention"
            ):
                return True
    except Exception:
        pass
    return False


def query_from(layout = None, trainer_cls = None, model = None, args = None, cp = None, cp_backend = None):
    family, hybrid, multimodal, windowed = _model_traits(model)
    packing = bool(getattr(args, "packing", False) or getattr(args, "padding_free", False))
    kernels = _kernels(model)
    return Query(
        trainer = _trainer_kind(trainer_cls),
        family = family,
        hybrid = hybrid,
        multimodal = multimodal,
        windowed = windowed,
        mode = _mode(model),
        packing = packing,
        cp_backend = cp_backend if cp_backend is not None else getattr(layout, "cp_backend", None),
        torch = _torch_version(),
        sm = _sm(),
        dp = getattr(layout, "dp", 1),
        tp = getattr(layout, "tp", 1),
        pp = getattr(layout, "pp", 1),
        cp = cp if cp is not None else getattr(layout, "cp", 1),
        fsdp2 = bool(getattr(layout, "fsdp2", False)),
        vllm = bool(getattr(model, "vllm_engine", None) is not None or getattr(args, "use_vllm", False)),
        pp_schedule = getattr(layout, "schedule", None) if getattr(layout, "pp", 1) > 1 else None,
        attention = _attention(model, packing),
        compile = _compile_mode(model),
        memory = _memory(model, args),
        kernels = kernels,
        # Unsloth's fast models compute the loss fused / chunked unless logits are asked for.
        loss = "fused_ce" if kernels and __import__("os").environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1" else "standard",
    )


def check_layout(layout, trainer_cls = None, model = None, args = None, cp = None, cp_backend = None, cells = True):
    """`check` for a ParallelLayout (and what is known of the trainer / model / arguments). `cells = False`:
    the rules and feature declarations only (the model and trainer are not known yet)."""
    size = cp if cp is not None else getattr(layout, "cp", 1)
    if size > 1 and cp_backend in (None, "auto"):
        from .context_parallel import resolve_cp_backend

        cp_backend = resolve_cp_backend("auto", size, model, tp = getattr(layout, "tp", 1))
    check(query_from(layout, trainer_cls, model, args, cp, cp_backend), cells = cells)


def markdown_table(kind = "rules") -> str:
    """kind: rules (refusals), features (declarations) or cells (supported compositions + evidence)."""
    if kind == "features":
        lines = [
            "| feature | kind | mesh | consumes | uses | produces | hooks | owns alone | scope | module |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for f in FEATURES:
            cols = [", ".join(x) or "-" for x in (f.mesh, f.consumes, f.uses, f.produces, f.hooks, f.exclusive)]
            lines.append(f"| {f.name} | {f.kind} | " + " | ".join(cols) + f" | {f.scope} | {f.module} |")
        return "\n".join(lines)
    if kind == "cells":
        lines = ["| cell | covers | parity | engagement | compile |", "|---|---|---|---|---|"]
        for c in CELLS:
            covers = "; ".join(f"{k} {'/'.join(map(str, v))}" for k, v in c.match.items())
            lines.append(f"| {c.name} | {covers} | {c.parity or 'missing'} | {c.engagement or 'missing'} | "
                         f"{c.compile or 'missing'} |")
        return "\n".join(lines)
    lines = ["| axis | refused cell | why | instead |", "|---|---|---|---|"]
    for rule in RULES:
        lines.append(f"| {rule.axis} | {rule.when} | {rule.reason} | {rule.alternative} |")
    return "\n".join(lines)
