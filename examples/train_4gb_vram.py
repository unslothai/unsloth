#!/usr/bin/env python3
"""
Optimized QLoRA Fine-Tuning Script for 4GB VRAM Consumer GPUs (GTX 1650 / RTX 3050)
Author: Kunwar Satyam Singh (dante@5ingularity)

Compatible with: unsloth 2025+, trl 1.x (SFTConfig API), transformers 4.45+
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Prevents PyTorch from allocating fragmented blocks that cause premature OOM on Linux
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    print("🦥 Initializing Unsloth 4GB VRAM Optimization Pipeline...")

    # ── Configuration constants ───────────────────────────────────────────────
    max_seq_length = 1024   # Cap at 1024 tokens — attention memory scales O(N²)
    dtype          = None   # Auto-detect: Float16 on GTX 1650 (Turing), BF16 on Ampere+
    load_in_4bit   = True   # Mandatory NF4 4-bit quantization

    # ── 1. Load model & tokenizer ─────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = max_seq_length,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
    )

    # ── 2. Attach QLoRA adapters ──────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = 16,     # Rank 16: strong capacity, low memory footprint
        target_modules           = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
        lora_alpha               = 16,
        lora_dropout             = 0,      # 0 is Unsloth-optimized
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",  # CRITICAL: offloads activations, saves ~60% VRAM
        random_state             = 3407,
        use_rslora               = False,
        loftq_config             = None,
    )

    # ── 3. Load & format dataset ──────────────────────────────────────────────
    alpaca_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:\n{}"
    )

    EOS_TOKEN = tokenizer.eos_token

    def format_alpaca(examples):
        return {
            "text": [
                alpaca_prompt.format(inst, out) + EOS_TOKEN
                for inst, out in zip(examples["instruction"], examples["output"])
            ]
        }

    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")
    dataset = dataset.map(format_alpaca, batched=True)

    # ── 4. Training configuration locked for 4GB VRAM ────────────────────────
    # NOTE: In trl 1.x the SFT-specific params (dataset_text_field, max_length,
    # packing, dataset_num_proc) moved from SFTTrainer into SFTConfig.
    sft_config = SFTConfig(
        output_dir                  = "outputs_4gb_run",
        # Memory-critical hyperparameters
        per_device_train_batch_size = 1,   # Mandatory: keeps forward activation memory minimal
        gradient_accumulation_steps = 8,   # Effective batch size = 8, zero extra VRAM cost
        # Precision: GTX 1650 (Turing) is Float16; Ampere+ supports BF16
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        # Optimizer: 8-bit states + CPU paging absorbs momentary VRAM spikes
        optim                       = "paged_adamw_8bit",
        # Schedule
        warmup_steps                = 5,
        max_steps                   = 60,  # Short verification run (~15 min on GTX 1650)
        learning_rate               = 2e-4,
        lr_scheduler_type           = "linear",
        weight_decay                = 0.01,
        logging_steps               = 10,
        seed                        = 3407,
        report_to                   = "none",
        # SFT-specific (trl 1.x: these live in SFTConfig, not SFTTrainer)
        dataset_text_field          = "text",
        max_length                  = max_seq_length,
        dataset_num_proc            = 2,
        packing                     = False,  # True can spike VRAM on variable-length samples
    )

    # ── 5. Initialize trainer ─────────────────────────────────────────────────
    # NOTE: In trl 1.x 'tokenizer' param was renamed to 'processing_class'.
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = dataset,
        args             = sft_config,
    )

    # ── 6. Execute training & track GPU memory ────────────────────────────────
    print("🚀 Starting GPU Memory Tracking & Training...")
    gpu_stats        = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory       = round(gpu_stats.total_memory            / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max VRAM = {max_memory} GB.")
    print(f"Initial reserved memory = {start_gpu_memory} GB.")

    trainer_stats = trainer.train()

    # ── 7. Report peak memory usage ───────────────────────────────────────────
    used_memory          = round(torch.cuda.max_memory_reserved()  / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage      = round(used_memory          / max_memory * 100, 2)
    lora_percentage      = round(used_memory_for_lora / max_memory * 100, 2)
    print(f"\n✅ Training Complete!")
    print(f"Peak reserved memory     = {used_memory} GB ({used_percentage}% of total VRAM).")
    print(f"Memory used for LoRA     = {used_memory_for_lora} GB ({lora_percentage}%).")

    # ── 8. Save LoRA adapters ─────────────────────────────────────────────────
    model.save_pretrained("lora_model_4gb")
    tokenizer.save_pretrained("lora_model_4gb")
    print("💾 Adapters saved to `lora_model_4gb/`.")


if __name__ == "__main__":
    main()
