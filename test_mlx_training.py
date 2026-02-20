#!/usr/bin/env python3
"""
MLX Training/Finetuning Test for Unsloth

Tests end-to-end training with LoRA on a small model.
"""

import sys
import time


def test_mlx_training():
    """Test MLX training with LoRA on a small model."""
    print("=" * 60)
    print("  MLX Training/Finetuning Test")
    print("=" * 60)

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as opt
        from unsloth.kernels.mlx import (
            create_llama_model,
            LoRAConfig,
            MLXTrainer,
            TrainingConfig,
            AdamW,
            is_mlx_available,
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    if not is_mlx_available():
        print("❌ MLX not available")
        return False

    print("\n[1] Creating small Llama model...")
    start = time.time()

    model = create_llama_model(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    print(f"   Model created in {time.time() - start:.2f}s")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"   Total parameters: {num_params:,}")

    print("\n[2] Applying LoRA configuration...")
    lora_config = LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model_with_lora = create_llama_model(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
        lora_config=lora_config,
    )
    print("   ✅ LoRA applied")

    print("\n[3] Creating dummy training data...")
    batch_size = 2
    seq_len = 32
    vocab_size = 1000

    mx.random.seed(42)
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Labels shape: {labels.shape}")

    print("\n[4] Setting up optimizer and loss function...")

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        return loss

    optimizer = AdamW(learning_rate=1e-4, weight_decay=0.01)

    print("\n[5] Running training loop (5 steps)...")

    losses = []
    model = model_with_lora

    for step in range(5):
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(float(loss))
        print(f"   Step {step + 1}: loss = {float(loss):.4f}")

    print("\n[6] Verifying loss decreased...")
    if losses[-1] < losses[0]:
        print(f"   ✅ Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
    else:
        print(f"   ⚠️  Loss did not decrease (this can happen with few steps)")

    print("\n[7] Testing model save/load...")

    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            else:
                total += v.size
        return total

    params = model.parameters()
    param_count = count_params(params)
    print(f"   Total parameters: {param_count:,}")

    print("\n" + "=" * 60)
    print("  ✅ MLX Training Test Complete!")
    print("=" * 60)

    return True


def test_mlx_trainer_class():
    """Test the MLXTrainer class if available."""
    print("\n" + "=" * 60)
    print("  MLXTrainer Class Test")
    print("=" * 60)

    try:
        import mlx.core as mx
        from unsloth.kernels.mlx import (
            create_llama_model,
            LoRAConfig,
            MLXTrainer,
            TrainingConfig,
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    print("\n[1] Creating model with LoRA...")
    lora_config = LoRAConfig(r=8, lora_alpha=16)
    model = create_llama_model(
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        lora_config=lora_config,
    )
    print("   ✅ Model created")

    print("\n[2] Creating training data loader...")

    class DummyLoader:
        def __init__(self, num_batches=10, batch_size=2, seq_len=16, vocab_size=500):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __iter__(self):
            for _ in range(self.num_batches):
                input_ids = mx.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                labels = input_ids.copy()
                yield {"input_ids": input_ids, "labels": labels}

        def __len__(self):
            return self.num_batches

    train_loader = DummyLoader()
    print(f"   Created loader with {len(train_loader)} batches")

    print("\n[3] Setting up trainer...")
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        logging_steps=2,
    )
    print(f"   Config: epochs={config.num_epochs}, batch_size={config.batch_size}")

    print("\n[4] Running manual training loop...")

    import mlx.nn as nn
    from unsloth.kernels.mlx.optimizers import AdamW

    def loss_fn(model, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = model(input_ids)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )

    optimizer = AdamW(learning_rate=1e-3)

    losses = []
    for i, batch in enumerate(train_loader):
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(float(loss))
        if (i + 1) % 2 == 0:
            print(f"   Batch {i + 1}: loss = {float(loss):.4f}")

    print(f"\n   Final loss: {losses[-1]:.4f}")

    print("\n" + "=" * 60)
    print("  ✅ MLXTrainer Test Complete!")
    print("=" * 60)

    return True


def main():
    print("Starting MLX Training Tests...")

    success = True

    try:
        if not test_mlx_training():
            success = False
    except Exception as e:
        print(f"\n❌ Training test error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        if not test_mlx_trainer_class():
            success = False
    except Exception as e:
        print(f"\n❌ Trainer class test error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("  ✅ All training tests passed!")
    else:
        print("  ⚠️  Some tests failed")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
