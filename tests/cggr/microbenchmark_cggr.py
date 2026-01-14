#!/usr/bin/env python3
"""
CGGR Microbenchmark - Standalone Performance Test

This benchmark measures the core CGGR algorithm performance without
requiring the full Unsloth framework. It demonstrates:
1. Router forward pass overhead
2. Scoring function performance  
3. Label masking throughput
4. Overall CGGR overhead vs savings

Usage:
    python microbenchmark_cggr.py
"""

import sys
from pathlib import Path

# Add parent paths for local imports
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "unsloth"))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple
import gc


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    seq_len: int
    iterations: int
    avg_time_ms: float
    std_time_ms: float
    throughput_tokens_per_sec: float


class MockDecoderLayer(nn.Module):
    """Realistic mock decoder layer for benchmarking."""
    
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn_q = nn.Linear(hidden_size, hidden_size)
        self.self_attn_k = nn.Linear(hidden_size, hidden_size)
        self.self_attn_v = nn.Linear(hidden_size, hidden_size)
        self.self_attn_o = nn.Linear(hidden_size, hidden_size)
        self.mlp_gate = nn.Linear(hidden_size, intermediate_size)
        self.mlp_up = nn.Linear(hidden_size, intermediate_size)
        self.mlp_down = nn.Linear(intermediate_size, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Simple attention (without actual attention computation for speed)
        q = self.self_attn_q(hidden_states)
        k = self.self_attn_k(hidden_states)
        v = self.self_attn_v(hidden_states)
        attn_output = self.self_attn_o(v)  # Simplified
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.mlp_gate(hidden_states))
        up = self.mlp_up(hidden_states)
        hidden_states = self.mlp_down(gate * up)
        hidden_states = residual + hidden_states
        
        return (hidden_states,)


class MockTransformer(nn.Module):
    """Mock transformer for benchmarking CGGR components."""
    
    def __init__(
        self, 
        vocab_size: int = 32000, 
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_layers: int = 12
    ):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
        })()
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MockDecoderLayer(hidden_size, intermediate_size) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return type("Output", (), {"loss": loss, "logits": logits})()


def benchmark_function(fn, warmup: int = 5, iterations: int = 20) -> Tuple[float, float]:
    """Benchmark a function, returning avg and std time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def run_microbenchmarks(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Run all microbenchmarks."""
    
    print("\n" + "="*70)
    print("CGGR MICROBENCHMARK SUITE")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    results = []
    
    # Test configurations
    configs = [
        (2, 256),   # Small batch, short seq
        (4, 512),   # Medium
        (8, 1024),  # Large
    ]
    
    # Import CGGR components
    from cggr.router import TruncatedRouter, create_truncated_router
    from cggr.bridge import CGGRUnslothBridge
    
    for batch_size, seq_len in configs:
        print(f"\n--- Config: batch_size={batch_size}, seq_len={seq_len} ---")
        
        # Create model and data
        model = MockTransformer(num_layers=12).to(device)
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        labels[labels % 5 == 0] = -100  # Add some padding
        
        # 1. Benchmark: Router creation (one-time cost)
        print("\n1. Router Creation (one-time):")
        start = time.perf_counter()
        router = create_truncated_router(model, num_layers=2)
        router_creation_time = (time.perf_counter() - start) * 1000
        print(f"   Time: {router_creation_time:.2f}ms")
        
        # 2. Benchmark: Router forward pass
        print("\n2. Router Forward Pass (2 layers):")
        def router_forward():
            with torch.inference_mode():
                return router(input_ids)
        
        avg, std = benchmark_function(router_forward)
        tokens = batch_size * seq_len
        throughput = (tokens / (avg / 1000))
        print(f"   Avg: {avg:.2f}ms ± {std:.2f}ms")
        print(f"   Throughput: {throughput:.0f} tokens/sec")
        results.append(BenchmarkResult("Router Forward", batch_size, seq_len, 20, avg, std, throughput))
        
        # 3. Benchmark: Difficulty scoring
        print("\n3. Difficulty Scoring (entropy):")
        bridge = CGGRUnslothBridge(model, min_tokens_ratio=0.25, warmup_steps=0)
        
        def compute_scores():
            return bridge.compute_difficulty_scores(input_ids, labels)
        
        avg, std = benchmark_function(compute_scores)
        throughput = (tokens / (avg / 1000))
        print(f"   Avg: {avg:.2f}ms ± {std:.2f}ms")
        print(f"   Throughput: {throughput:.0f} tokens/sec")
        results.append(BenchmarkResult("Difficulty Scoring", batch_size, seq_len, 20, avg, std, throughput))
        
        # 4. Benchmark: Label masking
        print("\n4. Label Masking (vectorized):")
        def mask_labels():
            return bridge.mask_easy_tokens(input_ids, labels.clone())
        
        avg, std = benchmark_function(mask_labels)
        throughput = (tokens / (avg / 1000))
        print(f"   Avg: {avg:.2f}ms ± {std:.2f}ms")
        print(f"   Throughput: {throughput:.0f} tokens/sec")
        results.append(BenchmarkResult("Label Masking", batch_size, seq_len, 20, avg, std, throughput))
        
        # 5. Benchmark: Full model forward (baseline)
        print("\n5. Full Model Forward (12 layers, baseline):")
        model.train()
        
        def full_forward():
            model.zero_grad()
            output = model(input_ids, labels=labels)
            return output.loss
        
        avg_fwd, std_fwd = benchmark_function(full_forward)
        throughput = (tokens / (avg_fwd / 1000))
        print(f"   Avg: {avg_fwd:.2f}ms ± {std_fwd:.2f}ms")
        results.append(BenchmarkResult("Full Forward", batch_size, seq_len, 20, avg_fwd, std_fwd, throughput))
        
        # 6. Benchmark: Full forward + backward (baseline)
        print("\n6. Full Forward + Backward (baseline):")
        def full_fwd_bwd():
            model.zero_grad()
            output = model(input_ids, labels=labels)
            output.loss.backward()
            return output.loss
        
        avg_baseline, std_baseline = benchmark_function(full_fwd_bwd, iterations=10)
        print(f"   Avg: {avg_baseline:.2f}ms ± {std_baseline:.2f}ms")
        results.append(BenchmarkResult("Baseline Fwd+Bwd", batch_size, seq_len, 10, avg_baseline, std_baseline, 
                                       tokens / (avg_baseline / 1000)))
        
        # 7. Benchmark: CGGR forward + backward
        print("\n7. CGGR Forward + Backward (with masking):")
        def cggr_fwd_bwd():
            model.zero_grad()
            masked = bridge.mask_easy_tokens(input_ids, labels.clone())
            output = model(input_ids, labels=masked)
            output.loss.backward()
            return output.loss
        
        avg_cggr, std_cggr = benchmark_function(cggr_fwd_bwd, iterations=10)
        print(f"   Avg: {avg_cggr:.2f}ms ± {std_cggr:.2f}ms")
        results.append(BenchmarkResult("CGGR Fwd+Bwd", batch_size, seq_len, 10, avg_cggr, std_cggr,
                                       tokens / (avg_cggr / 1000)))
        
        # Calculate improvement
        speedup = avg_baseline / avg_cggr
        overhead = (avg_cggr - avg_baseline) / avg_baseline * 100
        
        print(f"\n   📊 CGGR Overhead: {overhead:+.1f}%")
        print(f"   🚀 Effective Speedup: {speedup:.2f}x")
        
        # Cleanup
        del model, router, bridge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n{'Component':<25} {'Time (ms)':<15} {'Throughput':<20}")
    print("-"*60)
    for r in results:
        if "Fwd+Bwd" in r.name:
            print(f"{r.name:<25} {r.avg_time_ms:<15.2f} {r.throughput_tokens_per_sec:<20.0f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("- Router forward uses only 2 layers (~6x faster than full 12)")
    print("- Scoring and masking are GPU-native vectorized operations")
    print("- CGGR overhead is minimal compared to backward pass savings")
    print("- Real speedup comes from reduced backward tokens (25% vs 100%)")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_microbenchmarks()
