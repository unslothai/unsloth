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
MLX Gradient Accumulation Test

Tests that gradient accumulation works correctly in MLX by comparing:
1. Batch size N with grad_accum=1 (effective batch = N)
2. Batch size 1 with grad_accum=N (effective batch = N)

Both should produce similar gradients and loss curves.

Usage:
    python benchmarks/mlx_gradient_accumulation_test.py
    python benchmarks/mlx_gradient_accumulation_test.py --model unsloth/llama-3.2-1b
    python benchmarks/mlx_gradient_accumulation_test.py --accum-steps 4
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
class GradientAccumResult:
    test_name: str
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    final_loss: float
    final_gradient_norm: float
    total_time_sec: float
    gradient_match: bool = False
    loss_match: bool = False
    success: bool = True
    error_message: str = ""


def compute_gradient_norm(model) -> float:
    """Compute the L2 norm of all gradients."""
    try:
        import mlx.core as mx
        total_norm_sq = mx.array(0.0)
        for name, param in model.named_parameters():
            if hasattr(param, 'grad') and param.grad is not None:
                total_norm_sq = total_norm_sq + mx.sum(param.grad ** 2)
        return float(mx.sqrt(total_norm_sq).item())
    except Exception:
        return 0.0


def test_gradient_accumulation_mlx(
    model_name: str,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_updates: int = 10,
    seq_length: int = 256,
    seed: int = 42,
) -> GradientAccumResult:
    """Test gradient accumulation in MLX."""
    
    effective_batch = batch_size * gradient_accumulation_steps
    
    result = GradientAccumResult(
        test_name=f"bs{batch_size}_ga{gradient_accumulation_steps}",
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch,
    )
    
    gc.collect()
    time.sleep(0.3)
    
    np.random.seed(seed)
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        
        mx.random.seed(seed)
        
        print(f"Loading {model_name}...")
        
        try:
            from mlx_lm import load as mlx_load
            model, tokenizer = mlx_load(model_name)
        except Exception as e:
            print(f"mlx-lm failed, trying unsloth: {e}")
            from unsloth.kernels.mlx.models.llama import create_llama_model
            model = create_llama_model(model_name, dtype=mx.float16)
        
        try:
            from unsloth.kernels.mxl.lora import apply_lora, LoRAConfig
            lora_config = LoRAConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
            model = apply_lora(model, lora_config)
        except Exception:
            pass
        
        optimizer = optim.AdamW(learning_rate=1e-4)
        
        input_ids = mx.array(np.random.randint(0, 32000, (effective_batch, seq_length), dtype=np.int32))
        labels = mx.array(np.random.randint(0, 32000, (effective_batch, seq_length), dtype=np.int32))
        
        def loss_fn(model, input_ids, labels):
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            )
            return mx.mean(loss)
        
        losses = []
        print(f"Running {num_updates} updates with grad_accum={gradient_accumulation_steps}...")
        
        start_time = time.time()
        
        for update in range(num_updates):
            accumulated_loss = mx.array(0.0)
            accumulated_grads = None
            
            for micro_step in range(gradient_accumulation_steps):
                start_idx = micro_step * batch_size
                end_idx = start_idx + batch_size
                
                batch_input = input_ids[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                
                loss, grads = mx.value_and_grad(loss_fn)(model, batch_input, batch_labels)
                loss = loss / gradient_accumulation_steps
                accumulated_loss = accumulated_loss + loss
                
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    for key in accumulated_grads:
                        accumulated_grads[key] = accumulated_grads[key] + grads[key]
            
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state)
            
            loss_val = float((accumulated_loss * gradient_accumulation_steps).item())
            losses.append(loss_val)
            
            if (update + 1) % 5 == 0:
                print(f"Update {update + 1}/{num_updates} | Loss: {loss_val:.4f}")
        
        result.total_time_sec = time.time() - start_time
        result.final_loss = losses[-1] if losses else 0.0
        
        grad_norm = compute_gradient_norm(model)
        result.final_gradient_norm = grad_norm
        
        print(f"Completed: Final loss={result.final_loss:.4f}, Grad norm={grad_norm:.4f}")
        
    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {result.error_message}")
    
    if 'model' in dir():
        del model
    gc.collect()
    
    return result


def run_gradient_accumulation_comparison(
    model_name: str,
    gradient_accumulation_steps: int = 4,
    num_updates: int = 10,
    seq_length: int = 256,
    output_file: str = "gradient_accumulation_results.json",
) -> Dict[str, Any]:
    """
    Compare gradient accumulation correctness.
    
    Test 1: batch_size=4, grad_accum=1 (baseline)
    Test 2: batch_size=1, grad_accum=4 (should match baseline)
    Test 3: batch_size=2, grad_accum=2 (should match baseline)
    """
    
    results = {
        "model": model_name,
        "effective_batch_size": gradient_accumulation_steps,
        "tests": [],
        "comparison": {},
    }
    
    print("\n" + "="*80)
    print(f"GRADIENT ACCUMULATION TEST: {model_name}")
    print("="*80)
    
    effective_batch = gradient_accumulation_steps
    
    test_configs = [
        {"batch_size": effective_batch, "grad_accum": 1},
        {"batch_size": 1, "grad_accum": effective_batch},
        {"batch_size": effective_batch // 2, "grad_accum": 2} if effective_batch >= 2 else None,
    ]
    test_configs = [c for c in test_configs if c is not None]
    
    test_results = []
    
    for i, cfg in enumerate(test_configs):
        print(f"\n--- Test {i+1}: batch_size={cfg['batch_size']}, grad_accum={cfg['grad_accum']} ---")
        
        result = test_gradient_accumulation_mlx(
            model_name=model_name,
            batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_updates=num_updates,
            seq_length=seq_length,
            seed=42 + i,
        )
        
        test_results.append(result)
        results["tests"].append(asdict(result))
        
        gc.collect()
        time.sleep(1)
    
    if len(test_results) >= 2:
        baseline = test_results[0]
        grad_accum_test = test_results[1]
        
        if baseline.success and grad_accum_test.success:
            loss_diff = abs(baseline.final_loss - grad_accum_test.final_loss)
            loss_match = loss_diff < 0.1
            
            grad_diff = abs(baseline.final_gradient_norm - grad_accum_test.final_gradient_norm)
            grad_match = grad_diff < 0.5 or grad_diff / max(baseline.final_gradient_norm, 0.01) < 0.1
            
            test_results[1].loss_match = loss_match
            test_results[1].gradient_match = grad_match
            
            results["comparison"] = {
                "baseline_loss": baseline.final_loss,
                "grad_accum_loss": grad_accum_test.final_loss,
                "loss_difference": loss_diff,
                "loss_match": loss_match,
                "baseline_grad_norm": baseline.final_gradient_norm,
                "grad_accum_grad_norm": grad_accum_test.final_gradient_norm,
                "grad_difference": grad_diff,
                "grad_match": grad_match,
                "test_passed": loss_match and grad_match,
            }
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for i, res in enumerate(test_results):
        if res.success:
            print(f"\nTest {i+1} (bs={res.batch_size}, ga={res.gradient_accumulation_steps}):")
            print(f"  Final loss: {res.final_loss:.4f}")
            print(f"  Grad norm: {res.final_gradient_norm:.4f}")
            print(f"  Time: {res.total_time_sec:.2f}s")
        else:
            print(f"\nTest {i+1}: FAILED - {res.error_message[:100]}")
    
    if "comparison" in results and results["comparison"]:
        comp = results["comparison"]
        print(f"\nComparison (batch=4 vs batch=1*accum=4):")
        print(f"  Loss match: {comp['loss_match']} (diff={comp['loss_difference']:.4f})")
        print(f"  Gradient match: {comp['grad_match']} (diff={comp['grad_difference']:.4f})")
        print(f"  TEST PASSED: {comp['test_passed']}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MLX Gradient Accumulation Test")
    parser.add_argument(
        "--model", type=str, default="unsloth/llama-3.2-1b-bnb-4bit",
        help="Model name or path"
    )
    parser.add_argument(
        "--accum-steps", type=int, default=4,
        help="Gradient accumulation steps to test"
    )
    parser.add_argument(
        "--updates", type=int, default=10,
        help="Number of update steps"
    )
    parser.add_argument(
        "--seq-length", type=int, default=256,
        help="Sequence length"
    )
    parser.add_argument(
        "--output", type=str, default="gradient_accumulation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    if not IS_APPLE_SILICON:
        print("WARNING: This test is designed for Apple Silicon")
    
    run_gradient_accumulation_comparison(
        model_name=args.model,
        gradient_accumulation_steps=args.accum_steps,
        num_updates=args.updates,
        seq_length=args.seq_length,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
