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
MLX Gradient Checkpointing Benchmark

Benchmarks the performance impact of gradient checkpointing on MLX training.
Measures memory savings vs speed tradeoff.

Usage:
    python benchmarks/mlx_gradient_checkpointing_benchmark.py
    python benchmarks/mlx_gradient_checkpointing_benchmark.py --model unsloth/llama-3.2-1b
    python benchmarks/mlx_gradient_checkpointing_benchmark.py --steps 20 --seq-length 2048
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

IS_APPLE_SILICON = sys.platform == "darwin" and os.uname().machine == "arm64"


@dataclass
class CheckpointingResult:
    model_name: str
    seq_length: int
    batch_size: int
    gradient_checkpointing: bool
    peak_memory_gb: float
    avg_step_time_ms: float
    total_time_sec: float
    memory_saved_pct: float = 0.0
    speed_slowdown_pct: float = 0.0
    success: bool = True
    error_message: str = ""


def get_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if IS_APPLE_SILICON:
        try:
            import mlx.core as mx
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_active_memory'):
                return mx.metal.get_active_memory() / (1024**3)
        except Exception:
            pass
    return 0.0


def benchmark_checkpointing_mlx(
    model_name: str,
    seq_length: int = 1024,
    batch_size: int = 1,
    num_steps: int = 10,
    gradient_checkpointing: bool = False,
    lora_rank: int = 16,
) -> CheckpointingResult:
    """Benchmark MLX with or without gradient checkpointing."""
    
    result = CheckpointingResult(
        model_name=model_name,
        seq_length=seq_length,
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    gc.collect()
    time.sleep(0.5)
    
    peak_memory = 0.0
    step_times = []
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        
        print(f"Loading {model_name} (GC={gradient_checkpointing})...")
        
        try:
            from mlx_lm import load as mlx_load
            model, tokenizer = mlx_load(model_name)
        except Exception as e:
            print(f"mlx-lm load failed, trying unsloth MLX: {e}")
            from unsloth.kernels.mlx.models.llama import create_llama_model
            model = create_llama_model(model_name, dtype=mx.float16)
        
        if lora_rank > 0:
            try:
                from unsloth.kernels.mlx.lora import apply_lora, LoRAConfig
                lora_config = LoRAConfig(
                    r=lora_rank,
                    alpha=lora_rank * 2,
                    target_modules=["q_proj", "v_proj"],
                )
                model = apply_lora(model, lora_config)
            except Exception as e:
                print(f"LoRA application failed, continuing without: {e}")
        
        if gradient_checkpointing:
            try:
                from unsloth.kernels.mlx.trainer import enable_gradient_checkpointing
                enable_gradient_checkpointing(model)
                print("Gradient checkpointing enabled")
            except Exception as e:
                print(f"Gradient checkpointing failed: {e}")
        
        optimizer = optim.AdamW(learning_rate=1e-4)
        
        def create_batch():
            return {
                "input_ids": mx.array(
                    np.random.randint(0, 32000, (batch_size, seq_length), dtype=np.int32)
                ),
                "labels": mx.array(
                    np.random.randint(0, 32000, (batch_size, seq_length), dtype=np.int32)
                ),
            }
        
        def loss_fn(model, batch):
            logits = model(batch["input_ids"])
            shift_logits = logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            )
            return mx.mean(loss)
        
        print(f"Running {num_steps} steps...")
        start_time = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            
            batch = create_batch()
            loss, grads = mx.value_and_grad(loss_fn)(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            step_time = (time.time() - step_start) * 1000
            step_times.append(step_time)
            
            current_mem = get_memory_usage()
            peak_memory = max(peak_memory, current_mem)
            
            if (step + 1) % 5 == 0:
                print(f"Step {step + 1}/{num_steps} | "
                      f"Loss: {float(loss):.4f} | "
                      f"Time: {step_time:.1f}ms | "
                      f"Memory: {current_mem:.2f}GB")
        
        result.total_time_sec = time.time() - start_time
        result.peak_memory_gb = peak_memory
        result.avg_step_time_ms = np.mean(step_times)
        
        print(f"Completed: {result.avg_step_time_ms:.1f}ms/step, {result.peak_memory_gb:.2f}GB peak")
        
    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {result.error_message}")
    
    if 'model' in dir():
        del model
    gc.collect()
    
    return result


def run_checkpointing_comparison(
    model_name: str,
    seq_lengths: List[int] = [512, 1024, 2048],
    num_steps: int = 10,
    output_file: str = "gradient_checkpointing_results.json",
) -> Dict[str, Any]:
    """Compare gradient checkpointing across different sequence lengths."""
    
    results = {
        "model": model_name,
        "configurations": [],
        "comparison": {},
    }
    
    print("\n" + "="*80)
    print(f"GRADIENT CHECKPOINTING BENCHMARK: {model_name}")
    print("="*80)
    
    for seq_length in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_length}")
        print(f"{'='*60}")
        
        result_gc_off = benchmark_checkpointing_mlx(
            model_name=model_name,
            seq_length=seq_length,
            num_steps=num_steps,
            gradient_checkpointing=False,
        )
        result_gc_on = benchmark_checkpointing_mlx(
            model_name=model_name,
            seq_length=seq_length,
            num_steps=num_steps,
            gradient_checkpointing=True,
        )
        
        if result_gc_off.success and result_gc_on.success:
            memory_saved = (
                (result_gc_off.peak_memory_gb - result_gc_on.peak_memory_gb) /
                result_gc_off.peak_memory_gb * 100
            ) if result_gc_off.peak_memory_gb > 0 else 0
            
            speed_slowdown = (
                (result_gc_on.avg_step_time_ms - result_gc_off.avg_step_time_ms) /
                result_gc_off.avg_step_time_ms * 100
            ) if result_gc_off.avg_step_time_ms > 0 else 0
            
            result_gc_on.memory_saved_pct = memory_saved
            result_gc_on.speed_slowdown_pct = speed_slowdown
            
            print(f"\nComparison for seq_length={seq_length}:")
            print(f"  Memory saved: {memory_saved:.1f}%")
            print(f"  Speed slowdown: {speed_slowdown:.1f}%")
            print(f"  Memory: {result_gc_off.peak_memory_gb:.2f}GB -> {result_gc_on.peak_memory_gb:.2f}GB")
            print(f"  Speed: {result_gc_off.avg_step_time_ms:.1f}ms -> {result_gc_on.avg_step_time_ms:.1f}ms")
        
        results["configurations"].append({
            "seq_length": seq_length,
            "gc_off": asdict(result_gc_off),
            "gc_on": asdict(result_gc_on),
        })
        
        gc.collect()
        time.sleep(2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Seq Len':<10} {'Memory (off)':<15} {'Memory (on)':<15} {'Saved':<10} {'Slowdown':<10}")
    print("-"*60)
    for cfg in results["configurations"]:
        seq_len = cfg["seq_length"]
        mem_off = cfg["gc_off"]["peak_memory_gb"]
        mem_on = cfg["gc_on"]["peak_memory_gb"]
        saved = cfg["gc_on"]["memory_saved_pct"]
        slow = cfg["gc_on"]["speed_slowdown_pct"]
        print(f"{seq_len:<10} {mem_off:<15.2f} {mem_on:<15.2f} {saved:<10.1f}% {slow:<10.1f}%")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MLX Gradient Checkpointing Benchmark")
    parser.add_argument(
        "--model", type=str, default="unsloth/llama-3.2-1b-bnb-4bit",
        help="Model name or path"
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of training steps per configuration"
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048],
        help="Sequence lengths to test"
    )
    parser.add_argument(
        "--output", type=str, default="gradient_checkpointing_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    if not IS_APPLE_SILICON:
        print("WARNING: This benchmark is designed for Apple Silicon")
    
    run_checkpointing_comparison(
        model_name=args.model,
        seq_lengths=args.seq_lengths,
        num_steps=args.steps,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
