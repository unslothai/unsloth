#!/usr/bin/env python3
"""Benchmark PyTorch vs MLX training for fine-tuning.

Compares speed (ms/step) and memory usage for a simple LoRA fine-tuning task.

Usage:
    python benchmark_pytorch_vs_mlx.py
    python benchmark_pytorch_vs_mlx.py --steps 50 --batch-size 2 --seq-len 128
"""

import sys
import os
import time
import argparse
import gc
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device_info() -> Dict[str, Any]:
    """Get device information."""
    info = {"pytorch_device": None, "mlx_available": False}
    
    import torch
    if torch.cuda.is_available():
        info["pytorch_device"] = "cuda"
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["pytorch_device"] = "mps"
    else:
        info["pytorch_device"] = "cpu"
    
    try:
        import mlx.core as mx
        info["mlx_available"] = True
        info["mlx_version"] = mx.__version__
    except ImportError:
        pass
    
    return info


def get_pytorch_memory_mb() -> float:
    """Get PyTorch GPU memory in MB."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def get_mlx_memory_mb() -> float:
    """Get MLX memory in MB."""
    try:
        import mlx.core as mx
        return mx.get_peak_memory() / (1024 * 1024)
    except:
        return 0.0


def reset_pytorch_memory():
    """Reset PyTorch memory tracking."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def reset_mlx_memory():
    """Reset MLX memory tracking."""
    try:
        import mlx.core as mx
        mx.reset_peak_memory()
    except:
        pass
    gc.collect()


class PyTorchLoRAModel:
    """Simple PyTorch LoRA model for benchmarking."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: str = "cuda",
    ):
        import torch
        import torch.nn as nn
        
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size, hidden_size, intermediate_size, num_layers, num_heads, lora_r, lora_alpha):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=intermediate_size,
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ])
                self.norm = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
                
                self.lora_A = nn.Linear(hidden_size, lora_r, bias=False)
                self.lora_B = nn.Linear(lora_r, hidden_size, bias=False)
                self.lora_scaling = lora_alpha / lora_r
                
                for p in self.lora_A.parameters():
                    nn.init.normal_(p, std=0.01)
                for p in self.lora_B.parameters():
                    nn.init.zeros_(p)
            
            def forward(self, input_ids, labels=None):
                x = self.embed(input_ids)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                lora_out = self.lora_B(self.lora_A(x)) * self.lora_scaling
                x = x + lora_out
                
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, logits.size(-1)),
                        shift_labels.view(-1),
                    )
                
                return logits, loss
        
        self.model = SimpleTransformer(
            vocab_size, hidden_size, intermediate_size, num_layers, num_heads, lora_r, lora_alpha
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
    
    def train_step(self, input_ids, labels) -> float:
        import torch
        self.optimizer.zero_grad()
        _, loss = self.model(input_ids, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync(self):
        import torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()


class MLXLoRAModel:
    """Simple MLX LoRA model for benchmarking."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        import mlx.core as mx
        import mlx.nn as nn
        from unsloth.kernels.mlx import AdamW
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_r
        
        class Embedding:
            def __init__(self, num_embeddings, embedding_dim):
                self.weight = mx.random.normal((num_embeddings, embedding_dim)) * 0.01
            
            def __call__(self, x):
                return self.weight[x]
        
        class Linear:
            def __init__(self, in_features, out_features, bias=True):
                scale = (1.0 / in_features) ** 0.5
                self.weight = mx.random.uniform(-scale, scale, (out_features, in_features))
                self.bias = mx.zeros((out_features,)) if bias else None
            
            def __call__(self, x):
                out = x @ self.weight.T
                if self.bias is not None:
                    out = out + self.bias
                return out
        
        class TransformerBlock:
            def __init__(self, hidden_size, num_heads, intermediate_size):
                self.attention = nn.MultiHeadAttention(hidden_size, num_heads)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.mlp = nn.Sequential(
                    Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    Linear(intermediate_size, hidden_size),
                )
            
            def __call__(self, x):
                h = self.norm1(x)
                x = x + self.attention(h, h, h)
                h = self.norm2(x)
                x = x + self.mlp(h)
                return x
        
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size, hidden_size, intermediate_size, num_layers, num_heads, lora_r, lora_scaling):
                super().__init__()
                self.embed = Embedding(vocab_size, hidden_size)
                self.layers = [TransformerBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
                self.norm = nn.LayerNorm(hidden_size)
                self.lm_head = Linear(hidden_size, vocab_size, bias=False)
                
                self.lora_A = Linear(hidden_size, lora_r, bias=False)
                self.lora_B = Linear(lora_r, hidden_size, bias=False)
                self.lora_scaling = lora_scaling
                
                self.lora_B.weight = mx.zeros_like(self.lora_B.weight)
            
            def __call__(self, input_ids, labels=None):
                x = self.embed(input_ids)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                lora_out = self.lora_B(self.lora_A(x)) * self.lora_scaling
                x = x + lora_out
                
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                    shift_labels = labels[:, 1:].reshape(-1)
                    loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
                
                return logits, loss
        
        self.model = SimpleTransformer(
            vocab_size, hidden_size, intermediate_size, num_layers, num_heads, lora_r, self.lora_scaling
        )
        
        self.optimizer = AdamW(learning_rate=1e-4)
        self.optimizer_state = {}
    
    def train_step(self, input_ids, labels) -> float:
        import mlx.core as mx
        import mlx.nn as nn
        
        def loss_fn(model):
            _, loss = model(input_ids, labels)
            return loss
        
        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad(self.model)
        
        self.optimizer.update(self.model, grads)
        mx.eval(self.model, self.optimizer.state)
        
        return float(loss)
    
    def sync(self):
        import mlx.core as mx
        mx.synchronize()


def benchmark_pytorch(
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int = 3,
) -> Tuple[float, float, str]:
    """Benchmark PyTorch training."""
    import torch
    
    device_info = get_device_info()
    device = device_info["pytorch_device"]
    
    print(f"\n{'='*60}")
    print(f" PyTorch Training Benchmark")
    print(f" Device: {device}")
    print(f"{'='*60}")
    
    model = PyTorchLoRAModel(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_heads=4,
        lora_r=8,
        device=device,
    )
    
    reset_pytorch_memory()
    
    print(f" Warmup ({warmup} steps)...", end=" ", flush=True)
    for _ in range(warmup):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        model.train_step(input_ids, labels)
    model.sync()
    print("Done")
    
    reset_pytorch_memory()
    
    print(f" Benchmarking ({steps} steps)...", end=" ", flush=True)
    start = time.perf_counter()
    
    for _ in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        model.train_step(input_ids, labels)
    
    model.sync()
    end = time.perf_counter()
    print("Done")
    
    total_time = end - start
    ms_per_step = (total_time / steps) * 1000
    peak_memory = get_pytorch_memory_mb()
    
    return ms_per_step, peak_memory, device


def benchmark_mlx(
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int = 3,
) -> Tuple[float, float]:
    """Benchmark MLX training."""
    import mlx.core as mx
    
    print(f"\n{'='*60}")
    print(f" MLX Training Benchmark")
    print(f" Device: {mx.default_device()}")
    print(f"{'='*60}")
    
    model = MLXLoRAModel(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_heads=4,
        lora_r=8,
    )
    
    reset_mlx_memory()
    
    print(f" Warmup ({warmup} steps)...", end=" ", flush=True)
    for _ in range(warmup):
        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        model.train_step(input_ids, labels)
    model.sync()
    print("Done")
    
    reset_mlx_memory()
    
    print(f" Benchmarking ({steps} steps)...", end=" ", flush=True)
    start = time.perf_counter()
    
    for _ in range(steps):
        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        model.train_step(input_ids, labels)
    
    model.sync()
    end = time.perf_counter()
    print("Done")
    
    total_time = end - start
    ms_per_step = (total_time / steps) * 1000
    peak_memory = get_mlx_memory_mb()
    
    return ms_per_step, peak_memory


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs MLX training")
    parser.add_argument("--steps", type=int, default=30, help="Number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--pytorch-only", action="store_true", help="Run PyTorch only")
    parser.add_argument("--mlx-only", action="store_true", help="Run MLX only")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" PyTorch vs MLX Fine-Tuning Benchmark")
    print("=" * 60)
    print(f" Config: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f" Steps: {args.steps} (warmup: {args.warmup})")
    
    device_info = get_device_info()
    print(f"\n System Info:")
    print(f"  PyTorch device: {device_info['pytorch_device']}")
    if device_info.get("cuda_device_name"):
        print(f"  CUDA device: {device_info['cuda_device_name']}")
    print(f"  MLX available: {device_info['mlx_available']}")
    if device_info.get("mlx_version"):
        print(f"  MLX version: {device_info['mlx_version']}")
    
    results = {}
    
    if not args.mlx_only:
        try:
            pt_time, pt_mem, pt_device = benchmark_pytorch(
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                warmup=args.warmup,
            )
            results["pytorch"] = {"time_ms": pt_time, "memory_mb": pt_mem, "device": pt_device}
        except Exception as e:
            print(f"\n PyTorch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    if not args.pytorch_only and device_info["mlx_available"]:
        try:
            mlx_time, mlx_mem = benchmark_mlx(
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                warmup=args.warmup,
            )
            results["mlx"] = {"time_ms": mlx_time, "memory_mb": mlx_mem}
        except Exception as e:
            print(f"\n MLX benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(" Results Summary")
    print(f"{'='*60}")
    print(f" {'Framework':<12} {'Device':<10} {'ms/step':>10} {'Peak MB':>10}")
    print("-" * 45)
    
    if "pytorch" in results:
        r = results["pytorch"]
        print(f" {'PyTorch':<12} {r['device']:<10} {r['time_ms']:>10.2f} {r['memory_mb']:>10.1f}")
    
    if "mlx" in results:
        r = results["mlx"]
        print(f" {'MLX':<12} {'metal':<10} {r['time_ms']:>10.2f} {r['memory_mb']:>10.1f}")
    
    if "pytorch" in results and "mlx" in results:
        pt_time = results["pytorch"]["time_ms"]
        mlx_time = results["mlx"]["time_ms"]
        speedup = pt_time / mlx_time if mlx_time > 0 else 0
        print("-" * 45)
        if speedup > 1:
            print(f" MLX is {speedup:.2f}x faster")
        else:
            print(f" PyTorch is {1/speedup:.2f}x faster")
    
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
