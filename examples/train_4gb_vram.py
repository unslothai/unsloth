#!/usr/bin/env python3
"""
Optimized QLoRA Fine-Tuning Script for 4GB VRAM Consumer GPUs (GTX 1650 / RTX 3050)
Author: Kunwar Satyam Singh (dante@5ingularity)
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Enforce expandable segments in Python to prevent memory fragmentation on Linux
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    print("🦥 Initializing Unsloth 4GB VRAM Optimization Pipeline...")
    
    # Configuration Constraints for 4GB VRAM
    max_seq_length = 1024  # Cap at 1024 tokens for memory stability
    dtype = None           # Auto-detection (Float16 for GTX 1650 Turing architecture)
    load_in_4bit = True    # Mandatory NF4 4-bit quantization

    # Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Attach QLoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank 16 provides strong capacity with low memory footprint
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # 0 is optimized in Unsloth
        bias = "none",
        use_gradient_checkpointing = "unsloth", # CRITICAL: Unsloth memory-optimized checkpointing
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Load Sample Dataset (Alpaca formatting)
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:500]")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # Training Arguments locked for 4GB VRAM budget
    training_args = TrainingArguments(
        per_device_train_batch_size = 1,          # Mandatory for 4GB VRAM
        gradient_accumulation_steps = 8,          # Effective batch size = 8
        warmup_steps = 5,
        max_steps = 60,                           # Short verification run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "paged_adamw_8bit",               # Paged 8-bit AdamW offloads spikes to CPU RAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_4gb_run",
        report_to = "none", # Disable logging services for standalone runs
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = training_args,
    )

    # Execute Training & Track GPU Memory
    print("🚀 Starting GPU Memory Tracking & Training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max VRAM = {max_memory} GB.")
    print(f"Initial reserved memory = {start_gpu_memory} GB.")

    trainer_stats = trainer.train()

    # Log Final Memory Statistics
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         / max_memory * 100, 2)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 2)
    print(f"✅ Training Complete!")
    print(f"Peak reserved memory = {used_memory} GB ({used_percentage}% of total VRAM).")
    print(f"Memory used for fine-tuning = {used_memory_for_lora} GB ({lora_percentage}%).")

    # Save LoRA Adapters
    model.save_pretrained("lora_model_4gb")
    tokenizer.save_pretrained("lora_model_4gb")
    print("💾 Adapters saved successfully to `lora_model_4gb/`.")

if __name__ == "__main__":
    main()
