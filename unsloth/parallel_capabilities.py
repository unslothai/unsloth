# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which parallel compositions run, in one table, consulted before anything touches the model.

A request is a `Query`: the trainer, the model family (and whether it is a hybrid, multimodal or windowed), the
mode (LoRA, QLoRA, full finetuning), packing, the context parallel backend, the torch version and the layout
(dp, tp, pp, cp, FSDP2). `check` walks `RULES` in order and raises on the first refusal; every refusal names the
axis it is about and what to run instead. `markdown_table` prints the table (docs, launcher README).

The table covers the cross-axis rules. Rules about one axis alone stay where that axis is implemented
(context_parallel.py, models/pp_env.py, models/tp_env.py); the rows here that repeat them run earlier, before
the split, so a refused launch leaves the model untouched.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

__all__ = ["Query", "Rule", "RULES", "check", "check_layout", "query_from", "markdown_table"]


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
        "pp / tp x vLLM",
        "pipeline / tensor parallel with vLLM (fast_inference)",
        lambda q: (q.pp > 1 or q.tp > 1) and q.vllm,
        "vLLM needs the whole model on each rank",
        "fast_inference = False, or drop tp / pp",
    ),
)


def check(query: Query) -> None:
    for rule in RULES:
        if rule.applies(query):
            raise NotImplementedError(
                f"Unsloth: {rule.axis}: {rule.when} is not supported: {rule.reason}. Instead: "
                f"{rule.alternative}. ({_describe(query)})"
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


def query_from(layout = None, trainer_cls = None, model = None, args = None, cp = None, cp_backend = None):
    family, hybrid, multimodal, windowed = _model_traits(model)
    packing = bool(getattr(args, "packing", False) or getattr(args, "padding_free", False))
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
    )


def check_layout(layout, trainer_cls = None, model = None, args = None, cp = None, cp_backend = None):
    """`check` for a ParallelLayout (and what is known of the trainer / model / arguments)."""
    check(query_from(layout, trainer_cls, model, args, cp, cp_backend))


def markdown_table() -> str:
    lines = ["| axis | refused cell | why | instead |", "|---|---|---|---|"]
    for rule in RULES:
        lines.append(f"| {rule.axis} | {rule.when} | {rule.reason} | {rule.alternative} |")
    return "\n".join(lines)
