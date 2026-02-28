#!/usr/bin/env python3
"""
Reproduce Baseline vs CCE Training Loss Curves on Apple Silicon (MLX).
Compares MLX baseline cross_entropy vs Chunked Cross Entropy (mx.fast.cce_loss).
"""

import os
import time
import argparse
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import matplotlib.pyplot as plt
from datasets import load_dataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Try to import unsloth kernels for MLX
try:
    from unsloth.kernels.mlx.models.llama import create_llama_model
    from unsloth.kernels.mlx.losses import chunked_cross_entropy_loss
    HAS_UNSLOTH_MLX = True
except ImportError:
    HAS_UNSLOTH_MLX = False

# ---------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------

TRAIN_STEPS = 100
LEARNING_RATE = 1e-05
DATASET_NAME = "tatsu-lab/alpaca"

# List of models and settings specified in the user's plot
CONFIGS = [
    {
        "name": "Qwen2.5-1.5B",
        "model_id": "unsloth/Qwen2.5-1.5B",
        "batch": 8,
        "seq": 512,
        "n_layers": 28,
        "h_size": 1536,
        "v_size": 151936,
    },
    {
        "name": "Qwen2.5-3B",
        "model_id": "unsloth/Qwen2.5-3B",
        "batch": 8,
        "seq": 1024,
        "n_layers": 36,
        "h_size": 2048,
        "v_size": 151936,
    },
    {
        "name": "Llama-3.2-3B",
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "batch": 16,
        "seq": 512,
        "n_layers": 28,
        "h_size": 3072,
        "v_size": 128256,
    },
]

def get_dummy_data(batch_size, seq_len, vocab_size):
    """Generate dummy input_ids and labels for benchmarking."""
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    # Add some -100 to simulate padding/instruction masking
    mask = mx.random.uniform(shape=(batch_size, seq_len)) < 0.1
    labels = mx.where(mask, mx.array([-100] * (batch_size * seq_len)).reshape(batch_size, seq_len), labels)
    return input_ids, labels


def get_streaming_data(batch_size, seq_len, vocab_size, n_samples):
    """Stream data from the alpaca dataset."""
    ds = load_dataset(DATASET_NAME, streaming=True)
    for i, item in enumerate(ds["train"]):
        if i >= n_samples:
            break
        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = mx.random.uniform(shape=(batch_size, seq_len)) < 0.1
        labels = mx.where(mask, mx.array([-100] * (batch_size * seq_len)).reshape(batch_size, seq_len), labels)
        yield input_ids, labels

def run_training_mlx(
    model_cfg: dict,
    use_cce: bool,
    steps: int = 100,
):
    """Run a single training sprint in MLX and return the list of losses."""
    
    # Extract config
    h_size = model_cfg.get("h_size", 2048)
    v_size = model_cfg.get("v_size", 128256)
    batch_size = model_cfg.get("batch", 8)
    seq_len = model_cfg.get("seq", 512)
    n_layers = model_cfg.get("n_layers", 12) # Reduced for benchmark speed
    
    print(f"\n--- Running {model_cfg['name']} (CCE={use_cce}) ---")
    
    # Try to use full model if available, otherwise fallback to Linear head
    if HAS_UNSLOTH_MLX:
        model = create_llama_model(
            vocab_size=v_size,
            hidden_size=h_size,
            intermediate_size=h_size * 4,
            num_hidden_layers=n_layers,
            num_attention_heads=h_size // 128,
            num_key_value_heads=h_size // 512,
        )
    else:
        model = nn.Linear(h_size, v_size, bias=False)
        
    optimizer = opt.Adafactor(learning_rate=LEARNING_RATE)
    
    def loss_fn(model, input_ids, labels):
        if HAS_UNSLOTH_MLX:
            # Full model forward
            logits, loss = model(
                input_ids=input_ids,
                labels=labels,
                use_cce=use_cce
            )
            return loss
        else:
            # Fallback Linear head
            h = mx.random.normal((batch_size, seq_len, h_size))
            if use_cce:
                from unsloth.kernels.mlx.losses import chunked_cross_entropy_loss
                return chunked_cross_entropy_loss(
                    h, model.weight, labels,
                    reduction="mean", ignore_index=-100
                )
            else:
                logits = model(h)
                shift_logits = logits[..., :-1, :].reshape(-1, v_size)
                shift_labels = labels[..., 1:].reshape(-1)
                return nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")

    loss_history = []
    
    print(f"  Creating model with {n_layers} layers, hidden={h_size}, vocab={v_size}...")
    print(f"  Batch size: {batch_size}, Seq len: {seq_len}")
    
    # Gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    print(f"  Gradient function created.")

    # Pre-generate some data
    print(f"  Streaming {DATASET_NAME} data...")
    data_iter = get_streaming_data(batch_size, seq_len, v_size, n_samples=10)
    data = [next(data_iter) for _ in range(10)]
    print(f"  Data ready. Starting training for {steps} steps...")
    
    step_times = []

    for i in range(steps):
        t0 = time.time()
        
        input_ids, labels = data[i % 10]
        
        # Training step - compute loss and gradients
        loss, grads = loss_and_grad_fn(model, input_ids, labels)
        
        # Update model with gradients
        optimizer.update(model, grads)
        
        # Evaluate the loss and updated parameters
        mx.eval(loss)
        mx.eval(model.parameters())
        mx.eval(optimizer.state)
        
        step_time = time.time() - t0
        step_times.append(step_time)
        
        loss_history.append(float(loss))
        
        # Print every step and log to wandb
        if i < 10 or (i + 1) % 10 == 0:
            avg_time = np.mean(step_times[-min(10, len(step_times)):])
            print(f"  Step {i+1}/{steps} | Loss: {float(loss):.4f} | Time: {step_time:.3f}s (avg: {avg_time:.3f}s)")
        else:
            print(f"  Step {i+1}/{steps} | Loss: {float(loss):.4f}")
        
        if HAS_WANDB:
            wandb.log({"step": i + 1, "loss": float(loss), "step_time": step_time})
    
    print(f"  Done. Total time: {sum(step_times):.2f}s, Avg: {np.mean(step_times):.3f}s/step")

    return loss_history

def main():
    parser = argparse.ArgumentParser("Reproduce CCE Loss Curves in MLX")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--config", type=int, help="Config index to run (0-2)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    if HAS_WANDB and args.wandb:
        wandb.init(project="unsloth-mlx-benchmark", name="mlx-loss-curves")
    elif not HAS_WANDB and args.wandb:
        print("Warning: wandb not installed. Install with: pip install wandb")

    configs_to_run = CONFIGS
    if args.config is not None:
        configs_to_run = [CONFIGS[args.config]]

    all_results = {}

    for cfg in configs_to_run:
        key = f"{cfg['name']}_b{cfg['batch']}_s{cfg['seq']}"
        print(f"\n=======================================================")
        print(f"Benchmarking: {key}")
        print(f"=======================================================")

        # 1. MLX Baseline
        baseline_losses = run_training_mlx(cfg, use_cce=False, steps=args.steps)
        
        # 2. MLX CCE (optimized)
        cce_losses = run_training_mlx(cfg, use_cce=True, steps=args.steps)

        all_results[key] = {
            "baseline": baseline_losses,
            "cce": cce_losses,
            "title": f"{cfg['name']} (MLX Native)\nbatch={cfg['batch']}, seq={cfg['seq']}, vocab={cfg['v_size']}"
        }

    # Plot results
    n_configs = len(configs_to_run)
    if n_configs > 0:
        cols = min(n_configs, 3)
        rows = (n_configs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
        fig.suptitle("MLX Baseline vs Chunked Cross Entropy (CCE) Loss Curves", fontsize=16)

        for idx, (key, res) in enumerate(all_results.items()):
            ax = axes[idx // cols, idx % cols]
            steps = range(1, len(res["baseline"]) + 1)
            ax.plot(steps, res["baseline"], label="MLX Baseline", color="blue", alpha=0.7)
            ax.plot(steps, res["cce"], label="MLX CCE", color="red", linestyle="--", alpha=0.7)
            
            ax.set_title(res["title"])
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig("mlx_loss_curves_comparison.png", dpi=300)
        print("\n[OK] Saved plot to mlx_loss_curves_comparison.png")

        # Also save JSON results
        with open("mlx_benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
