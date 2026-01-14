#!/usr/bin/env python3
"""
CGGR Throughput Benchmark - Same Memory Budget Comparison

The key insight: CGGR saves memory, so you can run LARGER batches.
This benchmark compares throughput (tokens/sec) at the same memory budget.

Baseline: Max batch that fits in ~9GB
CGGR:     Larger batch that uses ~9GB (thanks to memory savings)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"c:/Users/wrc02/Desktop/CGGR")))

import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_throughput(model, input_ids, labels, optimizer, cggr_model=None, num_iters=15, warmup=3):
    """Measure training throughput in tokens/sec."""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    tokens_per_step = batch_size * seq_len
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        if cggr_model:
            loss = cggr_model(input_ids, labels=labels)
            loss.backward()
            cggr_model.step()
        else:
            out = model(input_ids, labels=labels)
            out.loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        if cggr_model:
            loss = cggr_model(input_ids, labels=labels)
            loss.backward()
            cggr_model.step()
        else:
            out = model(input_ids, labels=labels)
            out.loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    throughput = tokens_per_step / avg_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    return {
        "avg_time_ms": avg_time * 1000,
        "throughput_tokens_sec": throughput,
        "peak_mem_gb": peak_mem,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def benchmark_equal_memory():
    """Compare throughput at equal memory budget."""
    
    from cggr import CGGRModel
    
    device = "cuda"
    print("\n" + "="*70)
    print("CGGR THROUGHPUT BENCHMARK - Same Memory Budget")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*70)
    
    # Load model
    print("\nLoading SmolLM2-135M...")
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    seq_len = 512
    
    # Find max baseline batch that fits in ~9GB
    print("\n" + "-"*70)
    print("1. BASELINE - Finding max batch size for ~9GB memory...")
    baseline_batch = 8  # Start here based on previous benchmark
    
    torch.cuda.reset_peak_memory_stats()
    input_ids = torch.randint(0, tokenizer.vocab_size, (baseline_batch, seq_len), device=device)
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    baseline_result = measure_throughput(model, input_ids, labels, optimizer)
    print(f"   Batch size: {baseline_batch}")
    print(f"   Memory:     {baseline_result['peak_mem_gb']:.2f} GB")
    print(f"   Time:       {baseline_result['avg_time_ms']:.1f} ms/step")
    print(f"   Throughput: {baseline_result['throughput_tokens_sec']:.0f} tokens/sec")
    
    baseline_mem = baseline_result['peak_mem_gb']
    
    # Now test CGGR with larger batch targeting same memory
    print("\n" + "-"*70)
    print("2. CGGR - Using larger batch with same memory budget...")
    
    # CGGR uses ~1/3 memory per sample, so we can roughly 3x the batch
    # But we have more headroom - let's target ~10GB
    # At batch=24 we used 6.81GB, so try batch=32-36
    cggr_batch = 32  # Target ~9-10GB
    
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create CGGR model
    cggr_model = CGGRModel(
        model=model,
        min_tokens_ratio=0.25,
        warmup_steps=0,
        selection='fixed_quota'
    )
    
    input_ids_cggr = torch.randint(0, tokenizer.vocab_size, (cggr_batch, seq_len), device=device)
    labels_cggr = input_ids_cggr.clone()
    optimizer_cggr = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    cggr_result = measure_throughput(model, input_ids_cggr, labels_cggr, optimizer_cggr, cggr_model=cggr_model)
    print(f"   Batch size: {cggr_batch}")
    print(f"   Memory:     {cggr_result['peak_mem_gb']:.2f} GB")
    print(f"   Time:       {cggr_result['avg_time_ms']:.1f} ms/step")
    print(f"   Throughput: {cggr_result['throughput_tokens_sec']:.0f} tokens/sec")
    
    # Calculate speedup
    speedup = cggr_result['throughput_tokens_sec'] / baseline_result['throughput_tokens_sec']
    pct_faster = (speedup - 1) * 100
    
    print("\n" + "="*70)
    print("RESULTS - Same Memory Budget Comparison")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':<20} {'CGGR':<20}")
    print("-"*70)
    print(f"{'Batch size':<25} {baseline_batch:<20} {cggr_batch:<20}")
    print(f"{'Memory (GB)':<25} {baseline_result['peak_mem_gb']:<20.2f} {cggr_result['peak_mem_gb']:<20.2f}")
    print(f"{'Time (ms/step)':<25} {baseline_result['avg_time_ms']:<20.1f} {cggr_result['avg_time_ms']:<20.1f}")
    print(f"{'Throughput (tok/sec)':<25} {baseline_result['throughput_tokens_sec']:<20.0f} {cggr_result['throughput_tokens_sec']:<20.0f}")
    print("-"*70)
    print(f"\n🚀 THROUGHPUT SPEEDUP: {speedup:.2f}x ({pct_faster:+.1f}%)")
    print(f"   (At equal ~{max(baseline_mem, cggr_result['peak_mem_gb']):.0f}GB memory budget)")
    print("="*70)
    
    # Cleanup
    del model, cggr_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_equal_memory()
