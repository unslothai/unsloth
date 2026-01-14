#!/usr/bin/env python3
"""
CGGR Proper Benchmark - Using Actual CGGR Library

This benchmark uses the real CGGR library to demonstrate the actual
speedup achieved through selective gradient computation.
"""

import sys
from pathlib import Path

# Add CGGR to path
cggr_path = Path(r"c:/Users/wrc02/Desktop/CGGR")
sys.path.insert(0, str(cggr_path))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def benchmark_with_actual_cggr():
    """Benchmark using the actual CGGR library."""
    
    from cggr import CGGRModel, create_truncated_router
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*70)
    print("CGGR PROPER BENCHMARK - Using Actual CGGR Library")
    print("="*70)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")
    print("="*70)
    
    # Small model for 12GB GPU - only 40M params
    vocab_size = 32000
    hidden_size = 512
    num_layers = 6
    batch_size = 4
    seq_len = 256
    
    # Create simple model
    class SimpleLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size, 
                    nhead=8, 
                    dim_feedforward=hidden_size*4,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight  # Tie weights
        
        def forward(self, input_ids, labels=None, **kwargs):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.lm_head(x)
            
            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                shift_labels = labels[:, 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            return type("Out", (), {"loss": loss, "logits": logits})()
    
    model = SimpleLM().to(device).half()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params:.1f}M parameters")
    
    # Create data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark baseline
    print("\n1. Baseline Training Step:")
    times = []
    for _ in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    baseline_time = sum(times) / len(times) * 1000
    print(f"   Avg: {baseline_time:.2f} ms/step")
    
    # Wrap with CGGR
    print("\n2. CGGR Training Step (25% token ratio):")
    cggr_model = CGGRModel(
        model=model,
        min_tokens_ratio=0.25,
        warmup_steps=0,  # No warmup for benchmark
        selection='fixed_quota'
    )
    
    # Warmup CGGR
    for _ in range(3):
        optimizer.zero_grad()
        loss = cggr_model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        cggr_model.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark CGGR
    times = []
    for _ in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        loss = cggr_model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        cggr_model.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    cggr_time = sum(times) / len(times) * 1000
    speedup = baseline_time / cggr_time
    
    print(f"   Avg: {cggr_time:.2f} ms/step")
    print(f"\n🚀 CGGR Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"📊 Peak GPU Memory: {peak_mem:.2f} GB")
    
    metrics = cggr_model.get_metrics()
    print(f"\n📈 CGGR Metrics:")
    print(f"   Token ratio: {metrics.get('token_ratio', 0):.2f}")
    print(f"   Tokens selected: {metrics.get('tokens_selected', 0)}/{metrics.get('tokens_total', 0)}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    benchmark_with_actual_cggr()
