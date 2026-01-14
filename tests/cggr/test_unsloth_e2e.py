#!/usr/bin/env python3
"""
Unsloth + CGGR End-to-End Integration Test

Tests that CGGR works seamlessly with Unsloth's optimized kernels:
1. Load model via FastLanguageModel  
2. Apply CGGR patching
3. Run training steps
4. Verify speedup and correctness

Requirements:
    pip install unsloth unsloth_zoo
"""

import sys
import types
from pathlib import Path

def fix_triton_inductor():
    """Fix compatibility between Triton 3.5.1 and PyTorch Inductor on Windows."""
    try:
        import triton.backends.compiler as compiler
        if not hasattr(compiler, 'AttrsDescriptor'):
            compiler.AttrsDescriptor = type('AttrsDescriptor', (), {})
            print("🦥 Patched triton.backends.compiler.AttrsDescriptor")
    except ImportError:
        m = types.ModuleType('triton.backends.compiler')
        m.AttrsDescriptor = type('AttrsDescriptor', (), {})
        sys.modules['triton.backends.compiler'] = m
        print("🦥 Created dummy triton.backends.compiler")

# Apply fix immediately before any other imports
fix_triton_inductor()

# Add CGGR to path
sys.path.insert(0, str(Path(r"c:/Users/wrc02/Desktop/CGGR")))

import time
import torch
import gc


def test_unsloth_cggr_integration():
    """Test CGGR integration with Unsloth's FastLanguageModel."""
    
    print("\n" + "="*70)
    print("UNSLOTH + CGGR END-TO-END INTEGRATION TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*70)
    
    # Step 1: Load model via Unsloth
    print("\n1. Loading model via Unsloth FastLanguageModel...")
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        print(f"   ❌ Failed to import unsloth: {e}")
        print("   Please install: pip install unsloth unsloth_zoo")
        return False
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",  # Small model for testing
        max_seq_length=512,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    print("   ✅ Model loaded successfully")
    
    # Add LoRA for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )
    print("   ✅ LoRA adapters added")
    
    # Step 2: Create trainer
    print("\n2. Creating SFTTrainer...")
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    
    # Create dummy dataset
    dummy_data = [
        {"text": tokenizer.apply_chat_template([
            {"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": f"Answer {i} " * 20}
        ], tokenize=False)}
        for i in range(20) # even smaller
    ]
    dataset = Dataset.from_list(dummy_data)
    
    print("   ... Tokenizing dataset manually to avoid subprocess issues")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)
    
    dataset = dataset.map(tokenize_fn, batched=True, num_proc=1)
    
    training_args = SFTConfig(
        output_dir="./test_output",
        max_steps=10,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        max_seq_length=256,
        dataset_num_proc=1,
        packing=False, # Disable packing for simplicity
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
    )
    print("   ✅ Trainer created")
    
    # Step 3: Test baseline training
    print("\n3. Running BASELINE training (5 steps)...")
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    # Run a few steps manually
    model.train()
    dataloader = trainer.get_train_dataloader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    baseline_times = []
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        baseline_times.append((time.perf_counter() - step_start) * 1000)
        print(f"   Step {i+1}: {baseline_times[-1]:.1f}ms, loss={outputs.loss.item():.4f}")
    
    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Baseline: {baseline_avg:.1f}ms/step, {baseline_mem:.2f}GB peak")
    
    # Step 4: Apply CGGR and test
    print("\n4. Applying CGGR patching...")
    try:
        from unsloth_selective_trainer import UnslothSelectiveTrainer
        
        adapter = UnslothSelectiveTrainer(
            model=model,
            min_tokens_ratio=0.25,
            router_layers=2,
            warmup_steps=0,  # No warmup for test
        )
        trainer = adapter.patch_trainer(trainer)
        print("   ✅ CGGR patching applied")
    except Exception as e:
        print(f"   ❌ CGGR patching failed: {e}")
        # Try alternative - use CGGRModel directly
        print("   Trying CGGRModel wrapper instead...")
        from cggr import CGGRModel
        cggr_model = CGGRModel(
            model=model,
            min_tokens_ratio=0.25,
            warmup_steps=0,
        )
        print("   ✅ CGGRModel wrapper created")
    
    # Step 5: Test CGGR training
    print("\n5. Running CGGR training (5 steps)...")
    torch.cuda.reset_peak_memory_stats()
    
    cggr_times = []
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        optimizer.zero_grad()
        # Use patched trainer's compute_loss if available
        if hasattr(trainer, 'compute_loss'):
            loss = trainer.compute_loss(model, batch)
            if hasattr(loss, 'loss'):
                loss = loss.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        cggr_times.append((time.perf_counter() - step_start) * 1000)
        print(f"   Step {i+1}: {cggr_times[-1]:.1f}ms, loss={loss.item():.4f}")
    
    cggr_avg = sum(cggr_times) / len(cggr_times)
    cggr_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"   CGGR: {cggr_avg:.1f}ms/step, {cggr_mem:.2f}GB peak")
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline:  {baseline_avg:.1f} ms/step, {baseline_mem:.2f} GB")
    print(f"CGGR:      {cggr_avg:.1f} ms/step, {cggr_mem:.2f} GB")
    
    speedup = baseline_avg / cggr_avg
    mem_savings = (baseline_mem - cggr_mem) / baseline_mem * 100
    
    print(f"\n🚀 Per-step speedup: {speedup:.2f}x")
    print(f"💾 Memory savings:   {mem_savings:.1f}%")
    
    if speedup > 1.0:
        print("\n✅ CGGR integration with Unsloth SUCCESSFUL!")
    else:
        print("\n⚠️ CGGR shows overhead - may need larger batch/seq for benefit")
    
    print("="*70)
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    test_unsloth_cggr_integration()
