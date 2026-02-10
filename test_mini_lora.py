#!/usr/bin/env python3
"""
Mini LoRA Training Test for Apple Silicon.
This script tests a complete LoRA training workflow on MPS.
"""

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import os
import sys

print("=" * 70)
print("Mini LoRA Training Test on Apple Silicon")
print("=" * 70)

# Check MPS availability
if not torch.backends.mps.is_available():
    print("ERROR: MPS not available. This test requires Apple Silicon.")
    sys.exit(1)

print(f"\nDevice: MPS (Apple Silicon)")
print(f"PyTorch version: {torch.__version__}")

# Configuration
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
NUM_TRAIN_STEPS = 10  # Just a few steps for testing

print(f"\nConfiguration:")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  LoRA rank: {LORA_RANK}")
print(f"  Training steps: {NUM_TRAIN_STEPS}")

# Load a small model
print("\nLoading TinyLlama model...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama-bnb-4bit",  # Small model for testing
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=True,
    )
    print("Model loaded successfully ✓")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Add LoRA adapters
print("\nAdding LoRA adapters...")
try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA adapters added successfully ✓")
except Exception as e:
    print(f"Failed to add LoRA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create tiny dataset
print("\nCreating test dataset...")
texts = [
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris.",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is 2+2?\n\n### Response:\n2+2 equals 4.",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWho wrote Romeo and Juliet?\n\n### Response:\nWilliam Shakespeare wrote Romeo and Juliet.",
] * 5  # Repeat for more samples

dataset = Dataset.from_dict({"text": texts})
print(f"Dataset created with {len(dataset)} samples ✓")

# Setup trainer
print("\nSetting up trainer...")
try:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            max_steps=NUM_TRAIN_STEPS,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=2,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="test_lora_outputs",
            report_to="none",
        ),
    )
    print("Trainer initialized successfully ✓")
except Exception as e:
    print(f"Failed to setup trainer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Train
print(f"\nTraining for {NUM_TRAIN_STEPS} steps...")
try:
    trainer_stats = trainer.train()
    print(f"Training completed ✓")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save LoRA weights
print("\nSaving LoRA weights...")
lora_path = "test_lora_adapter"
try:
    model.save_pretrained(lora_path)
    print(f"LoRA weights saved to {lora_path}/ ✓")
except Exception as e:
    print(f"Failed to save LoRA: {e}")
    import traceback
    traceback.print_exc()

# Test GGUF export with MLX merge
print("\n" + "=" * 70)
print("Testing GGUF Export with MLX Merge")
print("=" * 70)

try:
    print("Attempting GGUF export...")
    model.save_pretrained_gguf(
        "test_model_gguf",
        tokenizer,
        quantization_method=["q4_k_m"],
    )
    print("GGUF export completed successfully! ✓")
    print("\n🎉 SUCCESS: Full LoRA training and GGUF export pipeline works on Apple Silicon!")
    
    # List output files
    if os.path.exists("test_model_gguf"):
        files = os.listdir("test_model_gguf")
        print(f"\nOutput files in test_model_gguf/:")
        for f in files:
            size = os.path.getsize(f"test_model_gguf/{f}") / (1024*1024)
            print(f"  - {f} ({size:.2f} MB)")
            
except Exception as e:
    print(f"GGUF export failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  Note: GGUF export may require unsloth-zoo to be installed.")
    print("Install with: pip install unsloth-zoo")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)

# Cleanup suggestion
print("\nCleanup:")
print("  rm -rf test_lora_outputs/ test_lora_adapter/ test_model_gguf/")
