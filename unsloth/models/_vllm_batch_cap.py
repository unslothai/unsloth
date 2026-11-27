# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""GRPO / RLOO with fast_inference: compare one vLLM generate call's rows against the engine's ``max_num_seqs``.

``from_pretrained(fast_inference = True)`` sizes ``max_num_seqs`` from free KV memory alone, before any trainer
exists, so a generation batch above it waits for free slots: the rows admitted last start late and, with GRPO's
long-tailed completion lengths, can set the step's makespan. The engine cannot grow its cap after it is built
(its persistent batch buffers and CUDA graphs are sized from it), so this only tells the user how to pass a larger
``max_num_seqs`` to ``from_pretrained``."""

import os

__all__ = ["grpo_generation_rows_per_engine", "grpo_check_max_num_seqs"]


def grpo_generation_rows_per_engine(args, world_size = 1):
    """Rows one colocated vLLM engine receives per generate call, or None when unknown."""
    if args is None:
        return None
    world_size = max(int(world_size or 1), 1)
    rows = getattr(args, "generation_batch_size", None)
    if rows:
        rows = int(rows) // world_size
    else:
        per_device = getattr(args, "per_device_train_batch_size", None)
        steps = getattr(args, "steps_per_generation", None) or getattr(
            args, "gradient_accumulation_steps", None
        )
        if not per_device or not steps:
            return None
        rows = int(per_device) * int(steps)
    # A tensor parallel group gathers its ranks' prompts into one engine.
    tp = int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)
    return rows * max(tp, 1)


def _engine_max_num_seqs(trainer):
    llm = getattr(getattr(trainer, "vllm_generation", None), "llm", None)
    if llm is None:
        llm = getattr(trainer, "llm", None)
    if llm is None:
        llm = getattr(getattr(trainer, "model", None), "vllm_engine", None)
    engine = getattr(llm, "llm_engine", None)
    config = getattr(engine, "vllm_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    cap = getattr(scheduler, "max_num_seqs", None)
    return int(cap) if isinstance(cap, int) and cap > 0 else None


def grpo_check_max_num_seqs(trainer):
    """Warn once per trainer when a generate call has more rows than vLLM runs at once. Returns (rows, cap) or None.
    UNSLOTH_GRPO_MAX_NUM_SEQS_WARN=0 silences it."""
    if os.environ.get("UNSLOTH_GRPO_MAX_NUM_SEQS_WARN", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    try:
        args = getattr(trainer, "args", None)
        if not getattr(args, "use_vllm", False):
            return None
        if str(getattr(args, "vllm_mode", "colocate") or "colocate") != "colocate":
            return None
        cap = _engine_max_num_seqs(trainer)
        if cap is None:
            return None
        accelerator = getattr(trainer, "accelerator", None)
        rows = grpo_generation_rows_per_engine(args, getattr(accelerator, "num_processes", 1))
        if rows is None or rows <= cap:
            return None
        if getattr(trainer, "_unsloth_max_num_seqs_warned", False):
            return rows, cap
        trainer._unsloth_max_num_seqs_warned = True
        print(
            f"Unsloth: each vLLM generate call has {rows} rows, but the engine runs at most "
            f"max_num_seqs = {cap} at once, so {rows - cap} rows wait for a free slot and the longest late "
            f"row can set the step time. If memory allows, pass max_num_seqs = {rows} to "
            f"FastLanguageModel.from_pretrained(..., fast_inference = True)."
        )
        return rows, cap
    except Exception:
        return None
