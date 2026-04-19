"""Qwen3-4B GRPO baseline (vLLM colocated) derived from the notebook.

Run:
    CUDA_VISIBLE_DEVICES=2 python scripts/qwen3_grpo_vllm.py \
        --max_steps 61 --output_dir outputs/grpo_vllm \
        --stats_path logs/vllm_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Unsloth must be imported before transformers / trl.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

# Allow sibling import of the common module.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402

from unsloth_grpo_common import (  # noqa: E402
    apply_chat_template_to_tokenizer,
    build_dataset,
    build_reward_funcs,
    build_grpo_kwargs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=61)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--output_dir", default="outputs/grpo_vllm")
    p.add_argument("--stats_path", default="logs/vllm_stats.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok=True)

    # 1. Load model with vLLM fast inference enabled.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    apply_chat_template_to_tokenizer(tokenizer)

    # 2. Dataset + rewards.
    dataset, maximum_length = build_dataset(tokenizer, max_seq_length=args.max_seq_length)
    print(f"[vllm] Max prompt length (p90): {maximum_length}")
    reward_funcs = build_reward_funcs(tokenizer)

    # 3. vLLM sampling params match the notebook.
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    # 4. Build GRPOConfig.
    from trl import GRPOConfig, GRPOTrainer
    shared = build_grpo_kwargs(
        tokenizer,
        maximum_length,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
    )
    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_sampling_params=vllm_sampling_params,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        **shared,
    )

    # 5. Timing callback.
    from transformers import TrainerCallback

    timings = {"step_wall": [], "loss": [], "reward": []}

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.t0 = None

        def on_step_begin(self, _args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if "loss" in logs:
                timings["loss"].append(float(logs["loss"]))
            if "reward" in logs:
                timings["reward"].append(float(logs["reward"]))

        def on_step_end(self, _args, state, control, **kwargs):
            if self.t0 is not None:
                torch.cuda.synchronize()
                timings["step_wall"].append(time.perf_counter() - self.t0)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[StepTimer()],
    )

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    trainer.train()
    t_train = time.perf_counter() - t_start

    peak = torch.cuda.max_memory_allocated() / 1024**3

    stats = {
        "backend": "vllm_colocated",
        "train_wall_s": t_train,
        "peak_memory_gb": peak,
        "step_wall_s": timings["step_wall"],
        "losses": timings["loss"],
        "rewards": timings["reward"],
        "max_prompt_length": shared["max_prompt_length"],
        "max_completion_length": shared["max_completion_length"],
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
    }
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[vllm] Wrote stats to {args.stats_path}")
    print(f"[vllm] Total train wall: {t_train:.1f}s   Peak mem: {peak:.2f} GB")


if __name__ == "__main__":
    main()
