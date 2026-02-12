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

try:
    import unsloth
    print(f"Unsloth Installed Path: {unsloth.__file__}")
except ImportError:
    print("Unsloth not installed")

from unsloth import FastLanguageModel

model_name = "unsloth/Llama-3.2-1B-Instruct"
max_seq_length = 128

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    use_gradient_checkpointing=False,
)
print(f"  ✅ Model loaded")

# Apply LoRA
print("Applying LoRA...")
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

model.train()

print("=" * 70)
print("Diagnostic: Tracing Forward Pass")
print("=" * 70)

input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

# We need to manually run the forward pass steps to see exactly where grad is lost
# Step 1: Embeddings
try:
    with torch.set_grad_enabled(True):
        embeds = model.model.model.embed_tokens(input_ids)
    print(f"Embeddings output req_grad: {embeds.requires_grad}")
except Exception as e:
    print(f"Embeddings failed: {e}")

# Step 2: Layers
hidden_states = embeds
try:
    with torch.set_grad_enabled(True):
        for i, layer in enumerate(model.model.model.layers):
            layer_out = layer(hidden_states, position_ids=None, attention_mask=None)[0]
            if layer_out.requires_grad != hidden_states.requires_grad and not hidden_states.requires_grad:
                 print(f"  Layer {i} gained grad! (Input: {hidden_states.requires_grad}, Output: {layer_out.requires_grad})")
            elif layer_out.requires_grad != hidden_states.requires_grad and hidden_states.requires_grad:
                 print(f"  Layer {i} LOST grad! (Input: {hidden_states.requires_grad}, Output: {layer_out.requires_grad})")
            hidden_states = layer_out
    print(f"Final Layer output req_grad: {hidden_states.requires_grad}")
except Exception as e:
    print(f"Layers failed: {e}")

# Step 3: Norm
try:
    with torch.set_grad_enabled(True):
        norm_out = model.model.model.norm(hidden_states)
    print(f"Norm output req_grad: {norm_out.requires_grad}")
    print(f"Norm output dtype: {norm_out.dtype}")
except Exception as e:
    print(f"Norm failed: {e}")

# Step 4: LM Head
print(f"LM Head type: {type(model.model.lm_head)}")
print(f"LM Head weight dtype: {model.model.lm_head.weight.dtype}")
print(f"LM Head weight req_grad: {model.model.lm_head.weight.requires_grad}")

try:
    with torch.set_grad_enabled(True):
        # Case A: Direct call
        lm_out = model.model.lm_head(norm_out)
        print(f"LM Head (direct) output req_grad: {lm_out.requires_grad}")
        
        # Case B: Cast to float32 (like in llama.py)
        norm_out_f32 = norm_out.to(torch.float32)
        lm_out_f32 = model.model.lm_head(norm_out_f32)
        print(f"LM Head (cast float32) output req_grad: {lm_out_f32.requires_grad}")

        # Case C: Cast to bfloat16 (weights dtype)
        norm_out_bf16 = norm_out.to(torch.bfloat16)
        lm_out_bf16 = model.model.lm_head(norm_out_bf16)
        print(f"LM Head (cast bfloat16) output req_grad: {lm_out_bf16.requires_grad}")
        
        # Case D: torch.mv (like in llama.py optimization)
        try:
             logits_mv = torch.mv(model.model.lm_head.weight, norm_out_bf16.ravel())
             print(f"LM Head (torch.mv) output req_grad: {logits_mv.requires_grad}")
except Exception as e:
    print(f"LM Head (torch.mv) failed: {e}")

print("=" * 70)
print("Diagnostic: Test 5 - Full Training Step Simulation")
print("=" * 70)
try:
    with torch.set_grad_enabled(True):
        labels = torch.randint(0, 100, (1, 5), device=device)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        print(f"  loss = {loss.item()}")
        print(f"  loss.requires_grad = {loss.requires_grad}")
        print(f"  loss.grad_fn = {loss.grad_fn}")
        
        if loss.requires_grad:
            loss.backward()
            print("  ✅ Backward pass successful!")
        else:
            print("  ❌ Full training step loss does NOT require grad!")

except Exception as e:
    print(f"Test 5 failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("Diagnostic: Test 6 - SFTTrainer Training")
print("=" * 70)
try:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    # Create dummy dataset
    dataset = Dataset.from_dict({"text": ["hello world"] * 4})

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        output_dir="outputs_mps_test",
        optim="adamw_torch",
        report_to="none",
        use_mps_device=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("Starting trainer.train()...")
    trainer_stats = trainer.train()
    print("✅ SFTTrainer training successful!")

except Exception as e:
    print(f"Test 6 failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("Diagnostics complete!")
print("=" * 70)
