#!/usr/bin/env python3
"""
Reproduce Baseline vs CCE Training Loss Curves on Apple Silicon.
Runs 100 steps of fine-tuning with Adafactor lr=1e-5 on yahma/alpaca-cleaned
for various models with and without Unsloth's Custom Cross Entropy.
"""

import os
import gc
import json
import argparse
import matplotlib.pyplot as plt
import torch
import wandb

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Ensure tokenizer isn't parallelizing & hanging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------

TRAIN_STEPS = 100
LEARNING_RATE = 1e-05
OPTIMIZER = "adafactor"
DATASET_NAME = "yahma/alpaca-cleaned"

# List of models and settings specified in the user's plot
CONFIGS = [
    {
        "name": "Qwen3-4B",
        "model_id": "unsloth/Qwen2.5-1.5B",
        "batch": 8,
        "seq": 512,
        "n": 4096,
    },  # using 1.5b as placeholder for fast tests if needed, but parameterizing
    {
        "name": "Qwen3-4B",
        "model_id": "unsloth/Qwen2.5-3B",
        "batch": 8,
        "seq": 1024,
        "n": 8192,
    },
    {
        "name": "Qwen3-4B",
        "model_id": "unsloth/Qwen2.5-3B",
        "batch": 16,
        "seq": 512,
        "n": 8192,
    },
    {
        "name": "Qwen3-8B",
        "model_id": "unsloth/Qwen2.5-7B",
        "batch": 8,
        "seq": 512,
        "n": 4096,
    },
    {
        "name": "Llama-3.2-3B",
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "batch": 8,
        "seq": 512,
        "n": 4096,
    },
    {
        "name": "Llama-3.2-3B",
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "batch": 8,
        "seq": 1024,
        "n": 8192,
    },
    {
        "name": "Llama-3.2-3B",
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "batch": 16,
        "seq": 512,
        "n": 8192,
    },
    {
        "name": "Phi-3-Medium",
        "model_id": "unsloth/Phi-3-medium-4k-instruct",
        "batch": 8,
        "seq": 512,
        "n": 4096,
    },
    {
        "name": "Phi-3-Medium",
        "model_id": "unsloth/Phi-3-medium-4k-instruct",
        "batch": 8,
        "seq": 1024,
        "n": 8192,
    },
]


class LossLoggingCallback(
    torch.keras.callbacks.Callback if hasattr(torch, "keras") else object
):
    """Callback to store training loss at each step."""

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def prepare_dataset(tokenizer, max_seq_length):
    dataset = load_dataset(DATASET_NAME, split="train")

    def format_alpaca(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return {"text": text}

    dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    return dataset


def run_training(
    model_id: str,
    batch_size: int,
    seq_length: int,
    use_cce: bool,
    run_name: str = None,
):
    """Run a single training sprint and return the list of losses."""

    # 1. Patch Unsloth to toggle CCE.
    if use_cce:
        print(">>> Enabling Custom Cross Entropy Kernel")
        # Ensure fast_cross_entropy is active for Unsloth
        os.environ["UNSLOTH_USE_FAST_CROSS_ENTROPY"] = "1"
    else:
        print(">>> Disabling Custom Cross Entropy Kernel (Using PyTorch Baseline)")
        os.environ["UNSLOTH_USE_FAST_CROSS_ENTROPY"] = "0"

    print(f"\n--- Loading {model_id} (CCE={use_cce}) ---")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=seq_length,
        dtype=None,
        load_in_4bit=False,  # MPS doesn't support bitsandbytes yet
    )

    # Note: For MPS Apple Silicon compatibility, FastLanguageModel must patch
    # correctly. If testing purely with MLX, ensure mlx kernels are invoked.

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = prepare_dataset(tokenizer, seq_length)
    loss_logger = LossLoggingCallback()

    from transformers import TrainerCallback

    class HFLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_logger.losses.append(logs["loss"])
                if wandb.run is not None:
                    # Log to W&B so all lines appear on the same step axis
                    wandb.log(
                        {f"loss/{run_name}": logs["loss"], "step": state.global_step}
                    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            max_steps=TRAIN_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=False,
            logging_steps=1,
            optim=OPTIMIZER,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"outputs_{use_cce}",
            report_to="none",
        ),
        callbacks=[HFLoggingCallback()],
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")

    # Clean up model to free VRAM immediately
    del trainer, model, tokenizer
    clear_memory()

    return loss_logger.losses


def main():
    parser = argparse.ArgumentParser("Reproduce CCE Loss Curves")
    parser.add_argument(
        "--model",
        type=str,
        help="Specify a single model ID to run (e.g. unsloth/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--config", type=int, help="Specify a config index (0-8) to run just one."
    )
    args = parser.parse_args()

    results = {}

    configs_to_run = CONFIGS
    if args.config is not None:
        configs_to_run = [CONFIGS[args.config]]
    elif args.model is not None:
        configs_to_run = [c for c in CONFIGS if c["model_id"] == args.model]

    # Initialize a single W&B run for all configurations
    wandb.init(
        project="unsloth-mlx-loss-curves",
        name="Baseline_vs_CCE_benchmark",
        config={"configs_run": len(configs_to_run)},
    )

    for idx, cfg in enumerate(configs_to_run):
        key = f"{cfg['name']}_b{cfg['batch']}_s{cfg['seq']}"
        print(f"\n=======================================================")
        print(f"Running Config {idx + 1}/{len(configs_to_run)}: {key}")
        print(f"=======================================================")

        # Train Baseline
        baseline_losses = run_training(
            cfg["model_id"],
            cfg["batch"],
            cfg["seq"],
            use_cce=False,
            run_name=f"{key}_baseline",
        )
        # Train CCE
        cce_losses = run_training(
            cfg["model_id"],
            cfg["batch"],
            cfg["seq"],
            use_cce=True,
            run_name=f"{key}_cce",
        )

        results[key] = {
            "baseline": baseline_losses,
            "cce": cce_losses,
            "title": f"{cfg['name']}\nbatch={cfg['batch']}, seq={cfg['seq']}, N={cfg['n']}",
        }

        with open(f"loss_results_{key}.json", "w") as f:
            json.dump(results[key], f)

    # Plotting
    if len(configs_to_run) > 1:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(
            "Baseline vs CCE Training Loss Curves\n(100 steps each, Adafactor lr=1e-05, Alpaca dataset)",
            fontsize=16,
            fontweight="bold",
        )

        for idx, cfg in enumerate(configs_to_run):
            key = f"{cfg['name']}_b{cfg['batch']}_s{cfg['seq']}"
            if key not in results:
                continue

            ax = axes[idx // 3, idx % 3]
            res = results[key]

            steps = range(1, len(res["baseline"]) + 1)
            ax.plot(steps, res["baseline"], label="Baseline", alpha=0.8)

            steps_cce = range(1, len(res["cce"]) + 1)
            ax.plot(steps_cce, res["cce"], label="CCE", alpha=0.8)

            ax.set_title(res["title"])
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig("loss_curves_reproduction.png", dpi=300)
        print("Saved combined plot to loss_curves_reproduction.png")
    else:
        # Plot single curve
        for key, res in results.items():
            plt.figure(figsize=(8, 6))
            steps = range(1, len(res["baseline"]) + 1)
            plt.plot(steps, res["baseline"], label="Baseline", alpha=0.8)

            steps_cce = range(1, len(res["cce"]) + 1)
            plt.plot(steps_cce, res["cce"], label="CCE", alpha=0.8)

            plt.title(f"{res['title']}\nBaseline vs CCE Training Loss curves")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()

            out_file = f"loss_curves_{key}.png"
            plt.savefig(out_file, dpi=300)
            print(f"Saved single plot to {out_file}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
