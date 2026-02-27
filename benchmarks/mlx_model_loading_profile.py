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
MLX Model Loading Profiler

Profiles memory usage and loading time for 16-bit and 4-bit models.
Includes CPU memory tracking for Apple Silicon unified memory.

Usage:
    python benchmarks/mlx_model_loading_profile.py --model unsloth/llama-3-8b-bnb-4bit
    python benchmarks/mlx_model_loading_profile.py --model unsloth/llama-3-8b --bits 16
    python benchmarks/mlx_model_loading_profile.py --model unsloth/llama-3-8b --bits 4 --compare
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

IS_APPLE_SILICON = sys.platform == "darwin" and os.uname().machine == "arm64"


@dataclass
class MemoryStats:
    gpu_memory_gb: float = 0.0
    cpu_memory_gb: float = 0.0
    total_memory_gb: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    peak_cpu_memory_gb: float = 0.0


@dataclass
class LoadingResult:
    model_name: str
    bits: int
    loading_time_sec: float
    memory_before: MemoryStats
    memory_after: MemoryStats
    memory_delta: MemoryStats
    model_params_billions: float
    success: bool
    error_message: str = ""


def get_apple_memory_stats() -> MemoryStats:
    """Get memory stats for Apple Silicon using platform-specific APIs."""
    stats = MemoryStats()
    
    if not IS_APPLE_SILICON:
        return stats
    
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True
        )
        total_ram = int(result.stdout.strip()) / (1024**3)
        
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")
        free_pages = 0
        active_pages = 0
        wired_pages = 0
        for line in lines:
            if "free" in line.lower():
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "active" in line.lower():
                active_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "wired" in line.lower():
                wired_pages = int(line.split(":")[1].strip().rstrip("."))
        
        page_size = 4096
        used_memory_gb = (active_pages + wired_pages) * page_size / (1024**3)
        
        stats.cpu_memory_gb = used_memory_gb
        stats.total_memory_gb = total_ram
    except Exception as e:
        print(f"Warning: Could not get Apple memory stats: {e}")
    
    return stats


def get_mlx_memory_stats() -> MemoryStats:
    """Get memory stats from MLX."""
    stats = MemoryStats()
    
    try:
        import mlx.core as mx
        
        if hasattr(mx, 'get_active_memory'):
            stats.gpu_memory_gb = mx.get_active_memory() / (1024**3)
        elif hasattr(mx, 'metal'):
            if hasattr(mx.metal, 'get_active_memory'):
                stats.gpu_memory_gb = mx.metal.get_active_memory() / (1024**3)
    except Exception as e:
        print(f"Warning: Could not get MLX memory stats: {e}")
    
    apple_stats = get_apple_memory_stats()
    stats.cpu_memory_gb = apple_stats.cpu_memory_gb
    stats.total_memory_gb = apple_stats.total_memory_gb
    
    return stats


def get_pytorch_memory_stats() -> MemoryStats:
    """Get memory stats from PyTorch (MPS backend)."""
    stats = MemoryStats()
    
    try:
        import torch
        if torch.backends.mps.is_available():
            stats.gpu_memory_gb = torch.mps.current_allocated_memory() / (1024**3)
    except Exception:
        pass
    
    apple_stats = get_apple_memory_stats()
    stats.cpu_memory_gb = apple_stats.cpu_memory_gb
    stats.total_memory_gb = apple_stats.total_memory_gb
    
    return stats


def profile_mlx_model_loading(
    model_name: str,
    bits: int = 16,
    use_quantization: bool = False,
    lora_rank: int = 0,
    max_seq_length: int = 2048,
) -> LoadingResult:
    """Profile MLX model loading."""
    
    gc.collect()
    time.sleep(0.5)
    
    memory_before = get_mlx_memory_stats()
    start_time = time.time()
    
    success = False
    error_message = ""
    model = None
    model_params = 0.0
    
    try:
        import mlx.core as mx
        
        if bits == 4 or use_quantization:
            from unsloth.kernels.mlx.models.llama import create_llama_model
            from unsloth.kernels.mlx.loader import load_model_mlx
            
            print(f"Loading {model_name} in 4-bit MLX...")
            
            try:
                model = load_model_mlx(
                    model_name,
                    quantize_bits=4,
                    lora_rank=lora_rank if lora_rank > 0 else None,
                )
            except Exception as e:
                print(f"4-bit loading failed, trying alternative method: {e}")
                from mlx.utils import load
                from huggingface_hub import hf_hub_download
                from mlx_lm import load as mlx_load
                
                model, tokenizer = mlx_load(model_name)
        else:
            from unsloth.kernels.mlx.loader import load_model_mlx
            
            print(f"Loading {model_name} in 16-bit MLX...")
            
            try:
                model = load_model_mlx(
                    model_name,
                    dtype=mx.float16,
                    lora_rank=lora_rank if lora_rank > 0 else None,
                )
            except Exception as e:
                print(f"MLX loading failed, trying mlx-lm: {e}")
                from mlx_lm import load as mlx_load
                model, tokenizer = mlx_load(model_name)
        
        mx.eval(model.parameters() if hasattr(model, 'parameters') else model)
        
        if hasattr(model, 'parameters'):
            params = model.parameters()
            if isinstance(params, dict):
                total_params = sum(
                    np.prod(v.shape) if hasattr(v, 'shape') else 0
                    for v in params.values()
                )
            else:
                total_params = sum(
                    np.prod(p.shape) if hasattr(p, 'shape') else 0
                    for p in params
                )
            model_params = total_params / 1e9
        
        success = True
        
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"MLX loading error: {error_message}")
    
    loading_time = time.time() - start_time
    memory_after = get_mlx_memory_stats()
    
    memory_delta = MemoryStats(
        gpu_memory_gb=memory_after.gpu_memory_gb - memory_before.gpu_memory_gb,
        cpu_memory_gb=memory_after.cpu_memory_gb - memory_before.cpu_memory_gb,
        total_memory_gb=memory_after.total_memory_gb,
    )
    
    del model
    gc.collect()
    
    return LoadingResult(
        model_name=model_name,
        bits=bits,
        loading_time_sec=loading_time,
        memory_before=memory_before,
        memory_after=memory_after,
        memory_delta=memory_delta,
        model_params_billions=model_params,
        success=success,
        error_message=error_message,
    )


def profile_pytorch_model_loading(
    model_name: str,
    bits: int = 16,
    use_4bit: bool = False,
    lora_rank: int = 0,
    max_seq_length: int = 2048,
) -> LoadingResult:
    """Profile PyTorch/MPS model loading for comparison."""
    
    gc.collect()
    time.sleep(0.5)
    
    memory_before = get_pytorch_memory_stats()
    start_time = time.time()
    
    success = False
    error_message = ""
    model = None
    model_params = 0.0
    
    try:
        import torch
        
        if use_4bit or bits == 4:
            from unsloth import FastLanguageModel
            
            print(f"Loading {model_name} in 4-bit PyTorch (Unsloth)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=None,
            )
        else:
            from unsloth import FastLanguageModel
            
            print(f"Loading {model_name} in 16-bit PyTorch (Unsloth)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                dtype=torch.float16,
            )
        
        if lora_rank > 0:
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
        
        model_params = sum(p.numel() for p in model.parameters()) / 1e9
        success = True
        
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"PyTorch loading error: {error_message}")
    
    loading_time = time.time() - start_time
    memory_after = get_pytorch_memory_stats()
    
    memory_delta = MemoryStats(
        gpu_memory_gb=memory_after.gpu_memory_gb - memory_before.gpu_memory_gb,
        cpu_memory_gb=memory_after.cpu_memory_gb - memory_before.cpu_memory_gb,
        total_memory_gb=memory_after.total_memory_gb,
    )
    
    del model
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    return LoadingResult(
        model_name=model_name,
        bits=bits,
        loading_time_sec=loading_time,
        memory_before=memory_before,
        memory_after=memory_after,
        memory_delta=memory_delta,
        model_params_billions=model_params,
        success=success,
        error_message=error_message,
    )


def profile_model_loading_comparison(
    model_name: str,
    bits_list: List[int] = [16, 4],
    compare_frameworks: bool = True,
    lora_rank: int = 0,
    max_seq_length: int = 2048,
) -> Dict[str, Any]:
    """Compare model loading across different configurations."""
    
    results = {
        "model_name": model_name,
        "mlx_results": [],
        "pytorch_results": [],
        "comparison": {},
    }
    
    print("\n" + "="*80)
    print(f"MODEL LOADING PROFILE: {model_name}")
    print("="*80)
    
    for bits in bits_list:
        print(f"\n--- MLX {bits}-bit ---")
        mlx_result = profile_mlx_model_loading(
            model_name, bits=bits, lora_rank=lora_rank, max_seq_length=max_seq_length
        )
        results["mlx_results"].append(asdict(mlx_result))
        
        if compare_frameworks:
            print(f"\n--- PyTorch/MPS {bits}-bit ---")
            pytorch_result = profile_pytorch_model_loading(
                model_name, bits=bits, use_4bit=(bits==4), lora_rank=lora_rank, max_seq_length=max_seq_length
            )
            results["pytorch_results"].append(asdict(pytorch_result))
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for i, bits in enumerate(bits_list):
        print(f"\n{bits}-bit Loading:")
        
        mlx_result = results["mlx_results"][i]
        if mlx_result["success"]:
            print(f"  MLX:      {mlx_result['loading_time_sec']:.2f}s, "
                  f"GPU: {mlx_result['memory_delta']['gpu_memory_gb']:.2f}GB, "
                  f"CPU: {mlx_result['memory_delta']['cpu_memory_gb']:.2f}GB")
        else:
            print(f"  MLX:      FAILED - {mlx_result['error_message'][:100]}")
        
        if compare_frameworks and i < len(results["pytorch_results"]):
            pt_result = results["pytorch_results"][i]
            if pt_result["success"]:
                print(f"  PyTorch:  {pt_result['loading_time_sec']:.2f}s, "
                      f"GPU: {pt_result['memory_delta']['gpu_memory_gb']:.2f}GB, "
                      f"CPU: {pt_result['memory_delta']['cpu_memory_gb']:.2f}GB")
            else:
                print(f"  PyTorch:  FAILED - {pt_result['error_message'][:100]}")
        
        if (mlx_result["success"] and compare_frameworks and 
            i < len(results["pytorch_results"]) and results["pytorch_results"][i]["success"]):
            pt_result = results["pytorch_results"][i]
            speedup = pt_result["loading_time_sec"] / mlx_result["loading_time_sec"]
            memory_ratio = (
                mlx_result["memory_delta"]["gpu_memory_gb"] / 
                pt_result["memory_delta"]["gpu_memory_gb"]
                if pt_result["memory_delta"]["gpu_memory_gb"] > 0 else 0
            )
            print(f"  Speedup:  {speedup:.2f}x, Memory ratio: {memory_ratio:.2f}x")
    
    return results


def run_model_size_benchmarks(
    models: Optional[List[str]] = None,
    bits_list: List[int] = [16, 4],
    output_file: str = "model_loading_results.json",
):
    """Run benchmarks across multiple model sizes."""
    
    if models is None:
        models = [
            "unsloth/llama-3.2-1b-bnb-4bit",
            "unsloth/llama-3.2-3b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ]
    
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"PROFILING: {model_name}")
        print(f"{'='*80}")
        
        try:
            results = profile_model_loading_comparison(
                model_name, bits_list=bits_list, compare_frameworks=True
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Failed to profile {model_name}: {e}")
            traceback.print_exc()
        
        gc.collect()
        time.sleep(2)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MLX Model Loading Profiler")
    parser.add_argument(
        "--model", type=str, default="unsloth/llama-3.2-1b-bnb-4bit",
        help="Model name or path to profile"
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=[16, 4],
        help="Bit widths to test (default: 16 4)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank (0 = no LoRA)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare MLX vs PyTorch"
    )
    parser.add_argument(
        "--output", type=str, default="model_loading_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run batch benchmarks for multiple models"
    )
    
    args = parser.parse_args()
    
    if not IS_APPLE_SILICON:
        print("WARNING: This benchmark is designed for Apple Silicon (M1/M2/M3/M4)")
        print("Some features may not work correctly on other platforms.\n")
    
    if args.batch:
        run_model_size_benchmarks(output_file=args.output)
    else:
        profile_model_loading_comparison(
            model_name=args.model,
            bits_list=args.bits,
            compare_frameworks=args.compare,
            lora_rank=args.lora_rank,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()
