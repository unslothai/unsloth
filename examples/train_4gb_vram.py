#!/usr/bin/env python3
"""
Optimized QLoRA Fine-Tuning Script for 4GB VRAM Consumer GPUs (GTX 1650 / RTX 3050)
Author: Kunwar Satyam Singh (dante@5ingularity)

Compatible with: unsloth 2025+, trl>=0.18.2,<=0.24.0 (SFTConfig API), transformers>=4.51.3
"""

import os

# Must run before `import torch` / `import unsloth` — Unsloth's GPU init touches
# CUDA at import time, which lazily initializes the allocator before this can apply.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unsloth must be imported before `datasets`/`transformers` — its _gpu_init redirects
# a read-only HF cache, and Hub/Transformers modules freeze the cache constants on
# their own import, so importing them first silently defeats the redirect.
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    print("🦥 Initializing Unsloth 4GB VRAM Optimization Pipeline...")

    # ── Configuration constants ───────────────────────────────────────────────
    max_seq_length = 1024  # Cap at 1024 tokens — attention memory scales O(N²)
    dtype = None  # Auto-detect: Float16 on GTX 1650 (Turing), BF16 on Ampere+
    load_in_4bit = True  # Mandatory NF4 4-bit quantization

    # ── 1. Load model & tokenizer ─────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # ── 2. Attach QLoRA adapters ──────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,  # Rank 16: strong capacity, low memory footprint
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = 16,
        lora_dropout = 0,  # 0 is Unsloth-optimized
        bias = "none",
        use_gradient_checkpointing = "unsloth",  # CRITICAL: offloads activations, saves ~60% VRAM
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # ── 3. Load & format dataset ──────────────────────────────────────────────
    # yahma/alpaca-cleaned rows carry instruction/input/output; many rows have a
    # non-empty `input` holding context (a passage, table, etc.) the instruction
    # refers to. Dropping it teaches the model to answer without context it was
    # actually given, so we branch on whether `input` is present.
    prompt_with_input = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
    )
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:\n{}"
    )

    EOS_TOKEN = tokenizer.eos_token

    def format_alpaca(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            if inp.strip():
                text = prompt_with_input.format(inst, inp, out)
            else:
                text = prompt_no_input.format(inst, out)
            texts.append(text + EOS_TOKEN)
        return {"text": texts}

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:500]")
    dataset = dataset.map(format_alpaca, batched = True)

    # ── 4. Training configuration locked for 4GB VRAM ────────────────────────
    # NOTE: In trl>=0.18 the SFT-specific params (dataset_text_field, max_length,
    # packing, dataset_num_proc) moved from SFTTrainer into SFTConfig.
    sft_config = SFTConfig(
        output_dir = "outputs_4gb_run",
        # Memory-critical hyperparameters
        per_device_train_batch_size = 1,  # Mandatory: keeps forward activation memory minimal
        gradient_accumulation_steps = 8,  # Effective batch size = 8, zero extra VRAM cost
        # Precision: GTX 1650 (Turing) is Float16; Ampere+ supports BF16
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        # Optimizer: 8-bit states + CPU paging absorbs momentary VRAM spikes
        optim = "paged_adamw_8bit",
        # Schedule
        warmup_steps = 5,
        max_steps = 60,  # Short verification run (~15 min on GTX 1650)
        learning_rate = 2e-4,
        lr_scheduler_type = "linear",
        weight_decay = 0.01,
        logging_steps = 10,
        seed = 3407,
        report_to = "none",
        # SFT-specific (trl>=0.18: these live in SFTConfig, not SFTTrainer)
        dataset_text_field = "text",
        max_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,  # True can spike VRAM on variable-length samples
        padding_free = False,  # Unsloth auto-enables this when unset, which conflicts
        # with an explicit max_length + packing=False (trl>=0.18)
    )

    # ── 5. Initialize trainer ─────────────────────────────────────────────────
    # NOTE: In trl>=0.18 'tokenizer' param was renamed to 'processing_class'.
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = sft_config,
    )

    # ── 6. Execute training & track GPU memory ────────────────────────────────
    print("🚀 Starting GPU Memory Tracking & Training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max VRAM = {max_memory} GB.")
    print(f"Initial reserved memory = {start_gpu_memory} GB.")

    trainer_stats = trainer.train()

    # ── 7. Report peak memory usage ───────────────────────────────────────────
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 2)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 2)
    print(f"\n✅ Training Complete!")
    print(f"Peak reserved memory     = {used_memory} GB ({used_percentage}% of total VRAM).")
    print(f"Memory used for LoRA     = {used_memory_for_lora} GB ({lora_percentage}%).")

    # ── 8. Save LoRA adapters ─────────────────────────────────────────────────
    model.save_pretrained("lora_model_4gb")
    tokenizer.save_pretrained("lora_model_4gb")
    print("💾 Adapters saved to `lora_model_4gb/`.")


if __name__ == "__main__":
    main()
