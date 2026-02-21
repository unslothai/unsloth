#!/usr/bin/env python3
"""
Benchmark: PyTorch + Unsloth vs MLX Fine-Tuning on LLaMA 1B

Compares training speed and memory usage between PyTorch/Unsloth and MLX.

Usage:
    python benchmark_llama1b.py --steps 20 --batch-size 2
"""
import argparse
import gc
import sys
import time
import platform

if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import unsloth  # Must be imported before transformers/peft
import torch

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"


def get_memory_mb():
    """Get current memory usage in MB."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def reset_memory():
    """Reset memory tracking."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_pytorch(steps: int, batch_size: int, seq_len: int, warmup: int = 2):
    """Benchmark PyTorch/Unsloth training with real LLaMA 1B."""
    import torch
    from unsloth import FastLanguageModel
    
    print(f"\n{'='*60}")
    print(f" PyTorch + Unsloth Training Benchmark")
    print(f" Model: {MODEL_NAME}")
    print(f"{'='*60}")
    
    is_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    device = "mps" if is_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")
    
    print("\n[1/4] Loading model...")
    load_start = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=seq_len,
        load_in_4bit=not is_mps,
        dtype=torch.bfloat16 if is_mps else None,
    )
    load_time = time.perf_counter() - load_start
    print(f"      Load time: {load_time:.1f}s")
    
    print("\n[2/4] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    print(f"\n[3/4] Training ({warmup} warmup + {steps} benchmark steps)...")
    model.train()
    
    reset_memory()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    def create_batch():
        input_ids = torch.randint(0, 128256, (batch_size, seq_len), device=model.device)
        labels = input_ids.clone()
        return input_ids, labels
    
    for i in range(warmup):
        input_ids, labels = create_batch()
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    reset_memory()
    
    step_times = []
    train_start = time.perf_counter()
    
    for i in range(steps):
        step_start = time.perf_counter()
        
        input_ids, labels = create_batch()
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        if (i + 1) % max(1, steps // 5) == 0:
            avg_step = sum(step_times) / len(step_times) * 1000
            print(f"      Step {i+1}/{steps}: loss={loss.item():.3f}, avg_step={avg_step:.1f}ms")
    
    total_time = time.perf_counter() - train_start
    peak_memory = get_memory_mb()
    avg_step_ms = (total_time / steps) * 1000
    
    print(f"\n[4/4] Complete!")
    print(f"      Total time: {total_time:.1f}s")
    print(f"      Avg step: {avg_step_ms:.1f}ms")
    print(f"      Peak memory: {peak_memory:.0f}MB")
    
    del model, tokenizer, optimizer
    gc.collect()
    
    return avg_step_ms, peak_memory, device, load_time


def benchmark_mlx(steps: int, batch_size: int, seq_len: int, warmup: int = 2):
    """Benchmark MLX training with real LLaMA 1B."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.lora import linear_to_lora_layers as linear_to_lora
    try:
        from mlx_lm.tuner.trainer import TrainingArgs, Dataset, evaluate
    except (ImportError, ModuleNotFoundError):
        try:
            from mlx_lm.trainer import TrainingArgs, Dataset, evaluate
        except (ImportError, ModuleNotFoundError):
            TrainingArgs = None
            Dataset = None
            evaluate = None
    from mlx.optimizers import AdamW
    
    print(f"\n{'='*60}")
    print(f" MLX Training Benchmark")
    print(f" Model: {MODEL_NAME}")
    print(f"{'='*60}")
    
    print(f" Device: {mx.default_device()}")
    
    print("\n[1/4] Loading model...")
    load_start = time.perf_counter()
    model, tokenizer = load(MODEL_NAME)
    load_time = time.perf_counter() - load_start
    print(f"      Load time: {load_time:.1f}s")
    
    print("\n[2/4] Adding LoRA adapters...")
    lora_config = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "scale": 0.25,
    }
    num_layers = 16
    model = linear_to_lora(model, num_layers, lora_config)
    
    print(f"\n[3/4] Training ({warmup} warmup + {steps} benchmark steps)...")
    
    def loss_fn(model, input_ids, labels):
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        return nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="mean"
        )
    
    def create_batch():
        vocab_size = tokenizer.tokenizer.vocab_size if hasattr(tokenizer, 'tokenizer') else tokenizer.vocab_size
        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        labels = input_ids.copy()
        return input_ids, labels
    
    optimizer = AdamW(learning_rate=2e-4)
    mx.reset_peak_memory()
    
    for i in range(warmup):
        input_ids, labels = create_batch()
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, input_ids, labels)
        optimizer.update(model, grads)
        mx.eval(model, optimizer.state)
    
    mx.reset_peak_memory()
    
    step_times = []
    train_start = time.perf_counter()
    
    for i in range(steps):
        step_start = time.perf_counter()
        
        input_ids, labels = create_batch()
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, input_ids, labels)
        optimizer.update(model, grads)
        mx.eval(model, optimizer.state)
        
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        if (i + 1) % max(1, steps // 5) == 0:
            avg_step = sum(step_times) / len(step_times) * 1000
            print(f"      Step {i+1}/{steps}: loss={float(loss):.3f}, avg_step={avg_step:.1f}ms")
    
    total_time = time.perf_counter() - train_start
    peak_memory = mx.get_peak_memory() / (1024 * 1024)
    avg_step_ms = (total_time / steps) * 1000
    
    print(f"\n[4/4] Complete!")
    print(f"      Total time: {total_time:.1f}s")
    print(f"      Avg step: {avg_step_ms:.1f}ms")
    print(f"      Peak memory: {peak_memory:.0f}MB")
    
    del model, tokenizer, optimizer
    gc.collect()
    
    return avg_step_ms, peak_memory, "metal", load_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs MLX with LLaMA 1B")
    parser.add_argument("--steps", type=int, default=10, help="Benchmark steps")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--pytorch-only", action="store_true", help="PyTorch only")
    parser.add_argument("--mlx-only", action="store_true", help="MLX only")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" LLaMA 3.2 1B Fine-Tuning Benchmark: PyTorch vs MLX")
    print("=" * 60)
    print(f" Model: {MODEL_NAME}")
    print(f" Config: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f" Steps: {args.steps} (warmup: {args.warmup})")
    
    if platform.system() == "Darwin":
        print(f" Platform: macOS (Apple Silicon)")
    else:
        print(f" Platform: {platform.system()}")
    
    results = {}
    
    if not args.mlx_only:
        try:
            pt_time, pt_mem, pt_dev, pt_load = benchmark_pytorch(
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
            )
            results["pytorch"] = {
                "time_ms": pt_time,
                "memory_mb": pt_mem,
                "device": pt_dev,
                "load_time": pt_load,
            }
        except Exception as e:
            print(f"\nPyTorch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    if not args.pytorch_only and platform.system() == "Darwin":
        try:
            import importlib.util
            mlx_lm_spec = importlib.util.find_spec("mlx_lm")
            if mlx_lm_spec is None:
                print("\nMLX benchmark skipped: mlx_lm not installed")
                print("Install with: pip install mlx-lm")
            else:
                mlx_time, mlx_mem, mlx_dev, mlx_load = benchmark_mlx(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    warmup=args.warmup,
                )
                results["mlx"] = {
                    "time_ms": mlx_time,
                    "memory_mb": mlx_mem,
                    "device": mlx_dev,
                    "load_time": mlx_load,
                }
        except Exception as e:
            print(f"\nMLX benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(" Results Summary")
    print(f"{'='*60}")
    print(f" {'Framework':<12} {'Device':<8} {'ms/step':>10} {'Peak MB':>10} {'Load (s)':>10}")
    print("-" * 55)
    
    if "pytorch" in results:
        r = results["pytorch"]
        print(f" {'PyTorch':<12} {r['device']:<8} {r['time_ms']:>10.1f} {r['memory_mb']:>10.0f} {r['load_time']:>10.1f}")
    
    if "mlx" in results:
        r = results["mlx"]
        print(f" {'MLX':<12} {r['device']:<8} {r['time_ms']:>10.1f} {r['memory_mb']:>10.0f} {r['load_time']:>10.1f}")
    
    if "pytorch" in results and "mlx" in results:
        pt_time = results["pytorch"]["time_ms"]
        mlx_time = results["mlx"]["time_ms"]
        speedup = pt_time / mlx_time if mlx_time > 0 else 0
        print("-" * 55)
        if speedup > 1:
            print(f" MLX is {speedup:.2f}x faster")
        else:
            print(f" PyTorch is {1/speedup:.2f}x faster")
    
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
