"""Qwen3-4B GRPO with naive TRL generation (no vLLM, no CB).

Mirrors the TRL example at https://huggingface.co/docs/trl/grpo_trainer: a
vanilla HF model + `GRPOTrainer` with the default rollout path, which calls
`model.generate(...)` per step. This is the honest "baseline" baseline — it
is what a user would get if they just followed the TRL docs without enabling
vLLM colocate or transformers continuous batching. Useful as a third column
in the benchmark table.

Hyperparameters, dataset, and reward functions are shared with the vLLM and
CB scripts via `unsloth_grpo_common.py` so numbers are apples-to-apples.

Run:
    CUDA_VISIBLE_DEVICES=2 python scripts/qwen3_grpo_naive.py \
        --max_steps 20 --num_generations 2 --per_device_train_batch_size 2 \
        --output_dir outputs/grpo_naive --stats_path logs/naive_stats.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type = int, default = 2048)
    p.add_argument("--lora_rank", type = int, default = 32)
    p.add_argument("--max_steps", type = int, default = 20)
    p.add_argument("--num_generations", type = int, default = 2)
    p.add_argument("--per_device_train_batch_size", type = int, default = 2)
    p.add_argument("--gradient_accumulation_steps", type = int, default = 1)
    p.add_argument(
        "--attn_impl",
        default = "sdpa",
        help = "Attention implementation: sdpa or flash_attention_2 (FA4 shim installed).",
    )
    p.add_argument("--output_dir", default = "outputs/grpo_naive")
    p.add_argument("--stats_path", default = "logs/naive_stats.json")
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

    # Install the FA4 shim only if the caller asked for flash_attention_2.
    # For sdpa we leave transformers untouched.
    if args.attn_impl == "flash_attention_2":
        import flash_attn_fa4_shim

        flash_attn_fa4_shim.apply()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype = torch.bfloat16,
        attn_implementation = args.attn_impl,
    ).to("cuda")

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

    dataset, maximum_length = build_dataset(
        tokenizer, max_seq_length = args.max_seq_length
    )
    print(f"[naive] Max prompt length (p90): {maximum_length}")
    reward_funcs = build_reward_funcs(tokenizer)

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
    # transformers' TopKLogitsWarper rejects -1. None skips the warper.
    shared["top_k"] = None

    training_args = GRPOConfig(
        use_vllm = False,
        use_transformers_paged = False,
        bf16 = True,
        **shared,
    )

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
        trainer, args.compile_mode, dynamic = args.compile_dynamic, tag = "naive"
    )

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    trainer.train()
    t_train = time.perf_counter() - t_start

    peak = torch.cuda.max_memory_allocated() / 1024**3

    write_stats(
        args.stats_path,
        backend = "naive_trl",
        timer = timer,
        train_wall_s = t_train,
        peak_memory_gb = peak,
        max_prompt_length = shared["max_prompt_length"],
        max_completion_length = shared["max_completion_length"],
        num_generations = args.num_generations,
        max_steps = args.max_steps,
        extra = {"attn_impl": args.attn_impl},
    )
    print(f"[naive] Wrote stats to {args.stats_path}")
    print(f"[naive] Total train wall: {t_train:.1f}s   Peak mem: {peak:.2f} GB")


if __name__ == "__main__":
    main()
