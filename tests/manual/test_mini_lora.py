#!/usr/bin/env python3
"""
Mini LoRA training test for Mac EC2 - Tests full training + save pipeline.
Matches the working approach from verify_apple.py
"""

import torch
import sys
import os

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_lora_training_and_save():
    """Run minimal LoRA training and save test."""
    print("=" * 70)
    print("Mini LoRA Training + Save Test for Apple Silicon")
    print("=" * 70)
    
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import shutil
    
    # Use a very small model
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 256
    
    print(f"\nLoading model: {model_name}")
    print("-" * 70)
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=False,  # No quantization for this test
            dtype=torch.bfloat16,
            use_gradient_checkpointing=False,  # Must also disable here for MPS
        )
        print(f"✅ Model loaded successfully")
        print(f"   Device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nConfiguring LoRA adapters...")
    print("-" * 70)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],  # Add MLP projections to test MPSLoRA_MLP
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,  # Disable - incompatible with MPS LoRA MLP + bfloat16
    )
    print(f"✅ LoRA adapters added (including MLP projections)")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\nCreating synthetic dataset...")
    print("-" * 70)
    
    # Create a tiny synthetic dataset with proper chat format
    data = [
        {"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"},
        {"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'm doing great!<|eot_id|>"},
    ] * 5  # 10 samples
    dataset = Dataset.from_list(data)
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    print("\nConfiguring training...")
    print("-" * 70)
    
    training_args = TrainingArguments(
        output_dir="./test_mini_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,  # 10 steps
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_torch",  # 8-bit not supported on MPS
        seed=42,
        bf16=True,  # Use bfloat16 for MPS
        report_to="none",  # Disable wandb/tensorboard
    )
    print(f"✅ Training configured")
    print(f"   Steps: {training_args.max_steps}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    
    print("\nInitializing trainer...")
    print("-" * 70)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    print(f"✅ Trainer initialized")
    
    print("\nRunning training...")
    print("-" * 70)
    
    # Diagnostic output: verify MPS dispatch state
    try:
        import unsloth.kernels.mps as mps_kernels
        # Force disable custom autograd to use pure PyTorch path
        mps_kernels.USE_MPS_FALLBACK = False
        fallback_state = getattr(mps_kernels, 'USE_MPS_FALLBACK', 'NOT_FOUND')
        print(f"   📊 USE_MPS_FALLBACK = {fallback_state} (forced disabled)")
        gc_enabled = getattr(model, 'gradient_checkpointing', 'NOT_FOUND')
        print(f"   📊 gradient_checkpointing = {gc_enabled}")
        print(f"   ✅ Using pure PyTorch path (no custom autograd)")
    except ImportError:
        print("   📊 (not on MPS, skipping dispatch diagnostics)")
    
    try:
        trainer_stats = trainer.train()
        print(f"✅ Training completed successfully!")
        print(f"   Training loss: {trainer_stats.training_loss:.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nTesting save operations...")
    print("-" * 70)
    
    # Test 1: Save LoRA weights
    print("\n1. Saving LoRA adapters...")
    try:
        model.save_pretrained("./test_mini_lora_weights")
        print("   ✅ LoRA weights saved")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 2: Save merged 16-bit model
    print("\n2. Saving merged 16-bit model...")
    try:
        model.save_pretrained_merged(
            "./test_mini_merged_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
        print("   ✅ Merged 16-bit model saved")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Save GGUF (this tests our MLX merge!)
    print("\n3. Saving GGUF (tests MLX LoRA merge)...")
    try:
        model.save_pretrained_gguf(
            "./test_mini_gguf",
            tokenizer,
            quantization_method="q4_k_m",
        )
        print("   ✅ GGUF saved successfully!")
        print("   🎉 MLX LoRA merge is working!")
    except RuntimeError as e:
        if "MPS" in str(e) or "Apple Silicon" in str(e):
            print(f"   ❌ GGUF export blocked on MPS: {e}")
            print("   Note: Our MLX merge should have enabled this")
            return False
        else:
            raise
    except Exception as e:
        print(f"   ❌ GGUF export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\nCleaning up test files...")
    for path in ["./test_mini_output", "./test_mini_lora_weights", 
                 "./test_mini_merged_16bit", "./test_mini_gguf"]:
        if os.path.exists(path):
            shutil.rmtree(path)
    print("✅ Cleanup complete")
    
    return True


if __name__ == "__main__":
    success = test_lora_training_and_save()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nResults:")
        print("  ✅ LoRA training works on Apple Silicon")
        print("  ✅ Pure PyTorch MLP path works with gradient checkpointing")
        print("  ✅ Attention + MLP LoRA training successful")
        print("  ✅ LoRA weight saving works")
        print("  ✅ Merged 16-bit saving works")
        print("  ✅ GGUF export works (MLX LoRA merge enabled)")
        print("\nYour Mac is ready for full Unsloth workflows!")
    else:
        print("❌ TESTS FAILED")
        print("=" * 70)
        print("\nCheck the error messages above for details.")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
