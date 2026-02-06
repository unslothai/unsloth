"""
Minimal SFT Training Script for Apple Silicon Verification

This script tests:
1. LoRA adapter loading on MPS
2. Backward pass (gradient computation) on MPS
3. Optimizer step on MPS
4. Sample packing / padding-free (if enabled)

Run with: python test_sft_training_mac.py
"""
import torch
import os
import platform
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

def main():
    print("="*50)
    print(" UNSLOTH SFT TRAINING TEST (Apple Silicon)")
    print("="*50)

    # Detect if running on MPS (Apple Silicon)
    is_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    
    if is_mps:
        print("\n🍎 Running on Apple Silicon (MPS)")
        print("   Note: Disabling 4-bit quantization (not supported on MPS)")
    
    # Use a very small model for testing
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 512

    print(f"\n[1] Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False if is_mps else True,  # 4-bit not supported on MPS
        dtype=torch.bfloat16 if is_mps else None,
    )

    print("\n[2] Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Create a tiny synthetic dataset
    print("\n[3] Creating synthetic dataset...")
    data = [
        {"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"},
        {"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'm doing great, thanks!<|eot_id|>"},
    ] * 10  # 20 samples total
    dataset = Dataset.from_list(data)

    print("\n[4] Configuring training arguments...")
    if is_mps:
        print("   Using adamw_torch optimizer (8-bit not supported on MPS)")
    training_args = SFTConfig(
        output_dir="./test_sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=5,  # Very short run
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_torch" if is_mps else "adamw_8bit",
        seed=42,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,  # Test without packing first
    )

    print("\n[5] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n[6] Starting training (5 steps)...")
    try:
        trainer.train()
        print("\n✅ SFT TRAINING COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ SFT TRAINING FAILED: {e}")
        raise

    print("\n[7] Cleanup...")
    # Clean up output directory
    import shutil
    if os.path.exists("./test_sft_output"):
        shutil.rmtree("./test_sft_output")
    
    print("\n" + "="*50)
    print(" ALL TESTS PASSED!")
    print("="*50)

if __name__ == "__main__":
    main()
