"""Qwen3-4B GRPO with transformers continuous-batching rollouts.

Unsloth's Qwen3Attention monkey-patch bypasses the functional attention
interface that `paged|<impl>` continuous batching relies on, so this script
loads a vanilla HF Qwen3 with PEFT LoRA instead. Training is slower than the
Unsloth path but the goal here is to evaluate transformers CB as a drop-in
replacement for vLLM rollouts. See benchmark_results.md for numbers.

Run:
    CUDA_VISIBLE_DEVICES=2 python scripts/qwen3_grpo_tpaged.py \
        --max_steps 61 --output_dir outputs/grpo_tpaged \
        --stats_path logs/tpaged_stats.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# `install_vllm_sampling_shim()` shims `vllm.sampling_params.GuidedDecodingParams`
# for newer vLLM releases so TRL's GRPOTrainer imports cleanly. We do NOT
# import `unsloth` here because that replaces TRL's GRPOTrainer with an
# Unsloth-compiled variant that assumes the model has `for_training()` /
# `for_inference()` hooks, which a vanilla HF model does not.
from unsloth_grpo_common import (  # noqa: E402
    StepTimer,
    apply_chat_template_to_tokenizer,
    build_dataset,
    build_grpo_kwargs,
    build_reward_funcs,
    install_vllm_sampling_shim,
    maybe_compile_trainer_forwards,
    write_stats,
)

install_vllm_sampling_shim()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import flash_attn_fa4_shim  # noqa: E402

flash_attn_fa4_shim.apply()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type = int, default = 2048)
    p.add_argument("--lora_rank", type = int, default = 32)
    p.add_argument("--max_steps", type = int, default = 61)
    p.add_argument("--num_generations", type = int, default = 4)
    p.add_argument("--per_device_train_batch_size", type = int, default = 1)
    p.add_argument("--gradient_accumulation_steps", type = int, default = 1)
    p.add_argument(
        "--attn_impl",
        default = "sdpa",
        help = "Base attention impl to compose with paged. 'sdpa' or 'flash_attention_2'.",
    )
    p.add_argument(
        "--max_batch_tokens",
        type = int,
        default = 8192,
        help = "PagedAttentionCache.max_batch_tokens. Default upper bound is 256 which is far too small.",
    )
    p.add_argument(
        "--num_blocks",
        type = int,
        default = 8192,
        help = "PagedAttentionCache.num_blocks (block_size=32). 8192*32 tokens of KV capacity.",
    )
    p.add_argument("--output_dir", default = "outputs/grpo_tpaged")
    p.add_argument("--stats_path", default = "logs/tpaged_stats.json")
    p.add_argument(
        "--persistent_cb",
        action = "store_true",
        help = "Reuse one ContinuousBatchingManager across every training step instead "
        "of letting TRL's generate_batch rebuild it (and the paged cache) each step.",
    )
    p.add_argument(
        "--compile_mode",
        default = None,
        choices = [None, "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
        help = "If set, torch.compile(model.forward, mode=...) after the trainer is built.",
    )
    p.add_argument("--compile_dynamic", action = "store_true", default = True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok = True)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok = True)

    # 1. Vanilla HF load (no Unsloth patches on the attention forward).
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype = torch.bfloat16,
        attn_implementation = args.attn_impl,
    )
    model.to("cuda")

    lora = LoraConfig(
        r = args.lora_rank,
        lora_alpha = args.lora_rank * 2,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    apply_chat_template_to_tokenizer(tokenizer)

    # 2. Dataset + rewards (identical to the vLLM script).
    dataset, maximum_length = build_dataset(
        tokenizer, max_seq_length = args.max_seq_length
    )
    print(f"[tpaged] Max prompt length (p90): {maximum_length}")
    reward_funcs = build_reward_funcs(tokenizer)

    # 3. Build GRPOConfig with transformers continuous batching enabled.
    from trl import GRPOConfig, GRPOTrainer

    shared = build_grpo_kwargs(
        tokenizer,
        maximum_length,
        max_seq_length = args.max_seq_length,
        max_steps = args.max_steps,
        num_generations = args.num_generations,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        output_dir = args.output_dir,
    )
    # transformers `TopKLogitsWarper` rejects -1. None skips the warper entirely.
    shared["top_k"] = None
    # The default PagedAttentionCache upper bounds
    # (`_upper_bound_max_batch_tokens=256`, `_upper_bound_num_blocks=4096`)
    # are extremely conservative and cause long decode loops. Raise them via
    # `generation_kwargs`, which TRL forwards to `GenerationConfig`, which the
    # CB manager then reads off when sizing the paged cache.
    training_args = GRPOConfig(
        use_vllm = False,
        use_transformers_paged = True,
        bf16 = True,
        generation_kwargs = {
            "max_batch_tokens": args.max_batch_tokens,
            "num_blocks": args.num_blocks,
        },
        **shared,
    )

    # 4. Timing callback.
    timer = StepTimer()

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
        callbacks = [timer],
    )

    maybe_compile_trainer_forwards(
        trainer, args.compile_mode, dynamic = args.compile_dynamic, tag = "tpaged"
    )

    if args.persistent_cb:
        # TRL constructs `self.generation_config` once in `__init__`; reuse
        # the same object so the persistent manager stays warm.
        from persistent_cb import install_for_model, teardown

        # TRL generates against the unwrapped base model; attach the patch
        # directly to it so every rollout picks up the persistent manager.
        base = (
            trainer.model_wrapped.base_model.model
            if hasattr(trainer.model_wrapped, "base_model")
            else trainer.model_wrapped
        )
        install_for_model(base, trainer.generation_config)
        # PEFT's wrapper chains `generate_batch` through `base_model.model` via
        # __getattr__, so installing on `base` is enough for TRL's call path.

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    try:
        trainer.train()
    finally:
        if args.persistent_cb:
            from persistent_cb import teardown

            teardown(base)
    t_train = time.perf_counter() - t_start

    peak = torch.cuda.max_memory_allocated() / 1024**3

    write_stats(
        args.stats_path,
        backend = "transformers_paged",
        timer = timer,
        train_wall_s = t_train,
        peak_memory_gb = peak,
        max_prompt_length = shared["max_prompt_length"],
        max_completion_length = shared["max_completion_length"],
        num_generations = args.num_generations,
        max_steps = args.max_steps,
        extra = {
            "attn_impl": args.attn_impl,
            "persistent_cb": args.persistent_cb,
        },
    )
    print(f"[tpaged] Wrote stats to {args.stats_path}")
    print(f"[tpaged] Total train wall: {t_train:.1f}s   Peak mem: {peak:.2f} GB")


if __name__ == "__main__":
    main()
