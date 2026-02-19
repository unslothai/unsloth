#!/usr/bin/env python3
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MLX Training Example Script

This script demonstrates how to use the MLX training infrastructure
to fine-tune language models on Apple Silicon Macs without PyTorch.

Examples:
    # Train a simple model with LoRA
    python scripts/test_mlx_training_example.py --model llama-3b --steps 1000
    
    # Full training without LoRA
    python scripts/test_mlx_training_example.py --model llama-3b --no-lora --steps 500
    
    # Training with mixed precision
    python scripts/test_mlx_training_example.py --model llama-3b --fp16

Requirements:
    - Apple Silicon Mac (M1/M2/M3)
    - MLX installed: pip install mlx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    print("Error: MLX not available. Install with: pip install mlx")
    exit(1)

# Import MLX modules directly to avoid unsloth_zoo dependency issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth.kernels.mlx.optimizers import AdamW, SGD, LinearWarmupCosineDecay, clip_grad_norm
from unsloth.kernels.mlx.losses import cross_entropy_loss
from unsloth.kernels.mlx.lora import LoRALinear, LoRAConfig, mark_only_lora_as_trainable, get_peft_model
from unsloth.kernels.mlx.models import MLXLinear, create_llama_model, MLXModelConfig


def create_dummy_dataset(
    num_samples: int = 1000,
    seq_length: int = 128,
    vocab_size: int = 32000,
) -> List[Dict[str, mx.array]]:
    """Create a dummy dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        seq_length: Length of each sequence
        vocab_size: Vocabulary size
        
    Returns:
        List of sample dictionaries with 'input_ids' and 'labels'
    """
    print(f"Creating dummy dataset with {num_samples} samples...")
    
    dataset = []
    for i in range(num_samples):
        # Generate random token IDs
        input_ids = mx.random.randint(0, vocab_size, (seq_length,), dtype=mx.int32)
        
        # For language modeling, labels are input_ids shifted by one
        # Here we just use the same for simplicity
        labels = input_ids
        
        dataset.append({
            "input_ids": input_ids,
            "labels": labels,
        })
    
    print(f"Dataset created: {num_samples} samples")
    return dataset


def create_simple_model(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 6,
    num_attention_heads: int = 12,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> Tuple[Any, List[str]]:
    """Create a simple language model for testing.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        
    Returns:
        Tuple of (model, trainable_param_names)
    """
    print(f"Creating model: vocab={vocab_size}, hidden={hidden_size}, layers={num_layers}")
    print(f"LoRA: {'enabled' if use_lora else 'disabled'}")
    
    if use_lora:
        print(f"  LoRA config: r={lora_r}, alpha={lora_alpha}")
    
    # Create a simple transformer model
    config = MLXModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    
    # Create model
    model = create_llama_model(config)
    
    # Apply LoRA if requested
    if use_lora:
        lora_config = LoRAConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable_params = mark_only_lora_as_trainable(model)
    else:
        trainable_params = list(model.get_params().keys())
    
    total_params = sum(p.size for p in model.get_params().values())
    trainable_count = sum(
        model.get_params()[name].size 
        for name in trainable_params
    )
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
    
    return model, trainable_params


def train_step(
    model: Any,
    batch: Dict[str, mx.array],
    optimizer: Any,
    scheduler: Any,
    trainable_params: List[str],
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Perform a single training step.
    
    Args:
        model: The model to train
        batch: Batch of data with 'input_ids' and 'labels'
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        trainable_params: List of trainable parameter names
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary with loss and learning rate
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    # Get current learning rate from scheduler
    learning_rate = scheduler.get_lr(scheduler.optimizer.step_count)
    
    # Define loss function for gradient computation
    def loss_fn(params):
        # Set model parameters
        all_params = model.get_params()
        for name, value in params.items():
            if name in all_params:
                all_params[name] = value
        
        # Forward pass
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=None,
            labels=labels,
        )
        
        return loss
    
    # Get current trainable parameters
    current_params = {
        name: model.get_params()[name] 
        for name in trainable_params 
        if name in model.get_params()
    }
    
    # Compute loss and gradients
    loss_value, grads = mx.value_and_grad(loss_fn)(current_params)
    
    # Clip gradients
    if max_grad_norm > 0:
        _, grads = clip_grad_norm(grads, max_norm=max_grad_norm)
    
    # Update parameters
    updated_params = optimizer(grads, current_params)
    
    # Update model parameters
    all_params = model.get_params()
    for name, value in updated_params.items():
        all_params[name] = value
    model.set_params(all_params)
    
    return {
        "loss": float(loss_value),
        "learning_rate": float(learning_rate),
    }


def validate(
    model: Any,
    dataset: List[Dict[str, mx.array]],
    batch_size: int = 8,
) -> float:
    """Validate the model on a dataset.
    
    Args:
        model: The model to validate
        dataset: Validation dataset
        batch_size: Batch size for validation
        
    Returns:
        Average loss
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        
        if len(batch) == 0:
            continue
        
        # Stack batch
        input_ids = mx.stack([b["input_ids"] for b in batch])
        labels = mx.stack([b["labels"] for b in batch])
        
        # Forward pass (no gradients needed)
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=None,
            labels=labels,
        )
        
        total_loss += float(loss)
        num_batches += 1
    
    model.train()
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="MLX Training Example for Unsloth"
    )
    
    # Model arguments
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Vocabulary size (default: 1000)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and do full fine-tuning",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    
    # Training arguments
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=64,
        help="Sequence length (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps (default: 10)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer to use (default: adamw)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD (default: 0.9)",
    )
    
    # Precision arguments
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision",
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval in steps (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Checkpoint save interval (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mlx_training_output",
        help="Output directory for checkpoints (default: ./mlx_training_output)",
    )
    
    # Other arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of training samples (default: 200)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLX Training Example for Unsloth")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()
    
    # Create dataset
    dataset = create_dummy_dataset(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
    )
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create model
    model, trainable_params = create_simple_model(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    print()
    
    # Create optimizer
    print(f"Creating {args.optimizer.upper()} optimizer...")
    if args.optimizer == "adamw":
        optimizer = AdamW(
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.999,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print()
    
    # Create scheduler
    print("Creating learning rate scheduler...")
    total_steps = args.steps
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=0.0,
    )
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print()
    
    # Set model to training mode
    model.train()
    
    # Training loop
    print("=" * 60)
    print("Starting training")
    print("=" * 60)
    print()
    
    step_times = []
    losses = []
    
    for step in range(args.steps):
        start_time = time.time()
        
        # Get batch
        batch_idx = step % len(train_dataset)
        batch = train_dataset[batch_idx]
        
        # Add batch dimension
        batch = {
            "input_ids": batch["input_ids"].reshape(1, -1),
            "labels": batch["labels"].reshape(1, -1),
        }
        
        # Training step
        result = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            trainable_params=trainable_params,
            max_grad_norm=args.max_grad_norm,
        )
        
        step_time = time.time() - start_time
        step_times.append(step_time)
        losses.append(result["loss"])
        
        # Logging
        if (step + 1) % args.log_interval == 0:
            avg_loss = sum(losses[-args.log_interval:]) / len(losses[-args.log_interval:])
            avg_time = sum(step_times[-args.log_interval:]) / len(step_times[-args.log_interval:])
            
            print(
                f"Step {step + 1}/{args.steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {result['learning_rate']:.6f} | "
                f"Time: {avg_time*1000:.1f}ms"
            )
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_step_{step + 1}.safetensors"
            model.save_weights(str(checkpoint_path))
            print(f"  -> Checkpoint saved: {checkpoint_path}")
    
    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Final statistics
    avg_loss = sum(losses) / len(losses)
    avg_step_time = sum(step_times) / len(step_times)
    total_time = sum(step_times)
    
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average step time: {avg_step_time*1000:.1f}ms")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Throughput: {args.steps/total_time:.2f} steps/sec")
    print()
    
    # Save final model
    final_path = output_dir / "final_model.safetensors"
    model.save_weights(str(final_path))
    print(f"Final model saved: {final_path}")
    print()
    
    # Validation
    if args.validate and len(val_dataset) > 0:
        print("=" * 60)
        print("Running validation")
        print("=" * 60)
        
        val_loss = validate(model, val_dataset, batch_size=args.batch_size)
        print(f"Validation loss: {val_loss:.4f}")
        print()
    
    print("Done!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
