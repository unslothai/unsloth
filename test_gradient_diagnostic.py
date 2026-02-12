#!/usr/bin/env python3
"""
Diagnostic test to isolate where the gradient graph breaks on MPS.
Tests each component independently to find the root cause.
"""
import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
print()

# ============================================================
# Test 0: Verify basic autograd works on MPS
# ============================================================
print("=" * 70)
print("Test 0: Basic autograd on MPS")
print("=" * 70)
x = torch.randn(2, 4, requires_grad=True, device=device)
w = torch.randn(4, 4, requires_grad=True, device=device)
y = x @ w
loss = y.sum()
print(f"  loss.requires_grad = {loss.requires_grad}")
print(f"  loss.grad_fn = {loss.grad_fn}")
loss.backward()
print(f"  x.grad shape = {x.grad.shape}")
print(f"  ✅ Basic autograd works")
print()

# ============================================================
# Test 1: Load model WITHOUT Unsloth patching - just base transformers
# ============================================================
print("=" * 70)
print("Test 1: Loading with Unsloth + model forward")
print("=" * 70)

from unsloth import FastLanguageModel

model_name = "unsloth/Llama-3.2-1B-Instruct"
max_seq_length = 128

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    use_gradient_checkpointing=False,
)
print(f"  ✅ Model loaded")

# Check USE_MPS_FALLBACK
try:
    import unsloth.kernels.mps as mps_kernels
    fallback_state = getattr(mps_kernels, 'USE_MPS_FALLBACK', 'NOT_FOUND')
    print(f"  📊 USE_MPS_FALLBACK = {fallback_state}")
except ImportError:
    print("  📊 MPS kernels not available")

# ============================================================
# Test 2: Raw model forward pass (no LoRA, no loss patching)
# ============================================================
print()
print("=" * 70)
print("Test 2: Raw model forward pass (before LoRA)")
print("=" * 70)

input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
with torch.set_grad_enabled(True):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    print(f"  logits.requires_grad = {logits.requires_grad}")
    print(f"  logits.grad_fn = {logits.grad_fn}")
    raw_loss = logits.sum()
    print(f"  raw_loss.requires_grad = {raw_loss.requires_grad}")
    print(f"  raw_loss.grad_fn = {raw_loss.grad_fn}")
    if raw_loss.requires_grad:
        raw_loss.backward()
        print(f"  ✅ Raw forward + backward works!")
    else:
        print(f"  ❌ Raw forward outputs don't require grad!")
print()

# ============================================================
# Test 3: Apply LoRA, then test forward pass
# ============================================================
print("=" * 70)
print("Test 3: LoRA forward pass")
print("=" * 70)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
)
print(f"  ✅ LoRA applied")
print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Check USE_MPS_FALLBACK after get_peft_model
try:
    fallback_state = getattr(mps_kernels, 'USE_MPS_FALLBACK', 'NOT_FOUND')
    print(f"  📊 USE_MPS_FALLBACK = {fallback_state} (after get_peft_model)")
except:
    pass

model.train()
with torch.set_grad_enabled(True):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    print(f"  logits.requires_grad = {logits.requires_grad}")
    print(f"  logits.grad_fn = {logits.grad_fn}")
    lora_loss = logits.sum()
    print(f"  lora_loss.requires_grad = {lora_loss.requires_grad}")
    if lora_loss.requires_grad:
        lora_loss.backward()
        print(f"  ✅ LoRA forward + backward works!")
    else:
        print(f"  ❌ LoRA forward outputs don't require grad!")
        # Try to find which module is breaking it
        print()
        print("  Diagnosing: checking all module outputs...")
        # Register hooks on all modules to see where grad is lost
        grad_lost_at = []
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    has_grad = output.requires_grad
                    if not has_grad:
                        grad_lost_at.append(name)
                elif hasattr(output, 'last_hidden_state'):
                    has_grad = output.last_hidden_state.requires_grad if hasattr(output, 'last_hidden_state') else "N/A"
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(make_hook(name)))
        
        with torch.set_grad_enabled(True):
            outputs2 = model(input_ids=input_ids)
        
        for h in hooks:
            h.remove()
        
        if grad_lost_at:
            print(f"  Gradient lost at modules ({len(grad_lost_at)} total):")
            for name in grad_lost_at[:20]:
                print(f"    - {name}")
            if len(grad_lost_at) > 20:
                print(f"    ... and {len(grad_lost_at) - 20} more")
        else:
            print(f"  No module lost gradient (issue may be in output processing)")
print()

# ============================================================
# Test 4: Test the patched cross-entropy loss
# ============================================================
print("=" * 70)
print("Test 4: Patched cross-entropy loss")  
print("=" * 70)

# Create dummy logits that require grad
dummy_logits = torch.randn(1, 5, model.config.vocab_size, device=device, dtype=torch.bfloat16, requires_grad=True)
dummy_labels = torch.tensor([[1, 2, 3, 4, 5]], device=device)

# Test with dispatch_cross_entropy_loss (what the patched loss uses)
try:
    from unsloth.kernels.mps.dispatch import dispatch_cross_entropy_loss
    ce_loss = dispatch_cross_entropy_loss(dummy_logits, dummy_labels)
    print(f"  dispatch_cross_entropy_loss:")
    print(f"    loss = {ce_loss.item():.4f}")
    print(f"    requires_grad = {ce_loss.requires_grad}")
    print(f"    grad_fn = {ce_loss.grad_fn}")
    if ce_loss.requires_grad:
        ce_loss.backward()
        print(f"    ✅ dispatch CE loss backward works!")
    else:
        print(f"    ❌ dispatch CE loss does NOT require grad!")
except Exception as e:
    print(f"  ❌ dispatch_cross_entropy_loss failed: {e}")
    import traceback
    traceback.print_exc()
print()

# ============================================================
# Test 5: Full training step simulation
# ============================================================
print("=" * 70)
print("Test 5: Full training step simulation")
print("=" * 70)

model.train()
input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
labels = torch.tensor([[-100, 2, 3, 4, 5]], device=device)

try:
    with torch.set_grad_enabled(True):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"  loss = {loss.item():.4f}")
        print(f"  loss.requires_grad = {loss.requires_grad}")
        print(f"  loss.grad_fn = {loss.grad_fn}")
        if loss.requires_grad:
            loss.backward()
            print(f"  ✅ Full training step works!")
        else:
            print(f"  ❌ Full training step loss does NOT require grad!")
except Exception as e:
    print(f"  ❌ Full training step failed: {e}")
    import traceback
    traceback.print_exc()
print()

# ============================================================
# Test 6: Test with SFTTrainer (the actual failure path)
# ============================================================
print("=" * 70)
print("Test 6: SFTTrainer training")
print("=" * 70)

try:
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    data = [
        {"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"},
    ] * 4
    dataset = Dataset.from_list(data)

    training_args = TrainingArguments(
        output_dir="./test_diag_output",
        per_device_train_batch_size=2,
        max_steps=3,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_torch",
        bf16=True,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer_stats = trainer.train()
    print(f"  ✅ SFTTrainer training completed! Loss: {trainer_stats.training_loss:.4f}")
except Exception as e:
    print(f"  ❌ SFTTrainer training failed: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
import shutil
for path in ["./test_diag_output"]:
    if os.path.exists(path):
        shutil.rmtree(path)

print()
print("=" * 70)
print("Diagnostics complete!")
print("=" * 70)
