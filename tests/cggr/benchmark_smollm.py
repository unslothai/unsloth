#!/usr/bin/env python3
"""
CGGR Benchmark with SmolLM-135M

Demonstrates CGGR speedup using a real HuggingFace model.
Fits within 12GB VRAM.
"""

import sys
from pathlib import Path

# Add CGGR to path
sys.path.insert(0, str(Path(r"c:/Users/wrc02/Desktop/CGGR")))

import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_smollm():
    """Benchmark CGGR with SmolLM-135M."""
    
    from cggr import CGGRModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*70)
    print("CGGR BENCHMARK - SmolLM-135M")
    print("="*70)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
    print("="*70)
    
    # Load SmolLM-135M
    print("\nLoading SmolLM-135M...")
    model_name = "HuggingFaceTB/SmolLM2-135M"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.train()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.1f}M")
    
    # Create data
    batch_size = 8
    seq_len = 512
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup baseline
    print("\nWarming up baseline...")
    for _ in range(3):
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark baseline
    print("\n1. BASELINE Training:")
    num_iters = 20
    times = []
    for i in range(num_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"   Step {i+1}/{num_iters}: {times[-1]:.1f}ms")
    
    baseline_time = sum(times) / len(times)
    baseline_std = (sum((t - baseline_time)**2 for t in times) / len(times)) ** 0.5
    print(f"   Average: {baseline_time:.2f} ± {baseline_std:.2f} ms/step")
    
    if device == "cuda":
        baseline_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    
    # Wrap with CGGR
    print("\n2. CGGR Training (25% token ratio):")
    cggr_model = CGGRModel(
        model=model,
        min_tokens_ratio=0.25,
        warmup_steps=0,
        selection='fixed_quota'
    )
    
    # Warmup CGGR
    for _ in range(3):
        optimizer.zero_grad()
        loss = cggr_model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        cggr_model.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark CGGR
    times = []
    for i in range(num_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        loss = cggr_model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        cggr_model.step()
        
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"   Step {i+1}/{num_iters}: {times[-1]:.1f}ms")
    
    cggr_time = sum(times) / len(times)
    cggr_std = (sum((t - cggr_time)**2 for t in times) / len(times)) ** 0.5
    print(f"   Average: {cggr_time:.2f} ± {cggr_std:.2f} ms/step")
    
    if device == "cuda":
        cggr_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # Results
    speedup = baseline_time / cggr_time
    time_saved = baseline_time - cggr_time
    pct_faster = (1 - cggr_time / baseline_time) * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline:        {baseline_time:.2f} ms/step")
    print(f"CGGR:            {cggr_time:.2f} ms/step")
    print(f"Time saved:      {time_saved:.2f} ms/step")
    print(f"")
    print(f"🚀 Speedup:      {speedup:.2f}x ({pct_faster:+.1f}%)")
    
    if device == "cuda":
        mem_saved = (baseline_mem - cggr_mem) / baseline_mem * 100
        print(f"💾 Memory:       {baseline_mem:.2f} GB → {cggr_mem:.2f} GB ({mem_saved:+.1f}%)")
    
    metrics = cggr_model.get_metrics()
    print(f"\n📈 CGGR Metrics:")
    print(f"   Tokens kept: {metrics.get('tokens_selected', 0)}/{metrics.get('tokens_total', 0)}")
    print(f"   Token ratio: {metrics.get('token_ratio', 0):.2%}")
    print("="*70)
    
    # Cleanup
    del model, cggr_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_smollm()
