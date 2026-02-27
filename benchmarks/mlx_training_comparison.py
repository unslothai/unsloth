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
MLX Training Comparison Benchmark

Compares loss curves, training speed, and memory usage between:
- Unsloth (PyTorch/MPS)
- Unsloth-MLX (pure MLX)

Includes CPU memory tracking for Apple Silicon unified memory.

Usage:
    python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b-bnb-4bit
    python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b --steps 100
    python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b --compare-all
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
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager

import numpy as np

IS_APPLE_SILICON = sys.platform == "darwin" and os.uname().machine == "arm64"


@dataclass
class MemorySnapshot:
    gpu_memory_gb: float = 0.0
    cpu_memory_gb: float = 0.0
    timestamp: float = 0.0


@dataclass
class StepMetrics:
    step: int
    loss: float
    learning_rate: float
    step_time_ms: float
    gpu_memory_gb: float
    cpu_memory_gb: float
    tokens_per_second: float = 0.0


@dataclass
class TrainingResult:
    framework: str
    model_name: str
    bits: int
    lora_rank: int
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    total_steps: int
    total_time_sec: float
    final_loss: float
    loss_curve: List[float] = field(default_factory=list)
    step_metrics: List[Dict] = field(default_factory=list)
    peak_gpu_memory_gb: float = 0.0
    peak_cpu_memory_gb: float = 0.0
    avg_tokens_per_second: float = 0.0
    success: bool = True
    error_message: str = ""


def get_apple_memory_info() -> tuple[float, float]:
    """Get used and total memory in GB for Apple Silicon."""
    if not IS_APPLE_SILICON:
        return 0.0, 0.0
    
    try:
        import subprocess
        
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        total_ram = int(result.stdout.strip()) / (1024**3)
        
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        
        free_pages = active_pages = wired_pages = 0
        for line in lines:
            if "free" in line.lower():
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "active" in line.lower():
                active_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "wired" in line.lower():
                wired_pages = int(line.split(":")[1].strip().rstrip("."))
        
        used_memory_gb = (active_pages + wired_pages) * 4096 / (1024**3)
        return used_memory_gb, total_ram
    except Exception:
        return 0.0, 0.0


def get_memory_snapshot() -> MemorySnapshot:
    """Get current memory usage snapshot."""
    snapshot = MemorySnapshot(timestamp=time.time())
    
    cpu_mem, _ = get_apple_memory_info()
    snapshot.cpu_memory_gb = cpu_mem
    
    try:
        import mlx.core as mx
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_active_memory'):
            snapshot.gpu_memory_gb = mx.metal.get_active_memory() / (1024**3)
    except Exception:
        pass
    
    return snapshot


def create_sample_dataset(num_samples: int = 100, seq_length: int = 512):
    """Create a sample dataset for training."""
    return {
        "input_ids": np.random.randint(0, 32000, (num_samples, seq_length), dtype=np.int32),
        "attention_mask": np.ones((num_samples, seq_length), dtype=np.int32),
        "labels": np.random.randint(0, 32000, (num_samples, seq_length), dtype=np.int32),
    }


def train_mlx(
    model_name: str,
    num_steps: int = 50,
    lora_rank: int = 16,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 1,
    seq_length: int = 512,
    learning_rate: float = 1e-4,
    bits: int = 16,
) -> TrainingResult:
    """Train using pure MLX."""
    
    result = TrainingResult(
        framework="mlx",
        model_name=model_name,
        bits=bits,
        lora_rank=lora_rank,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_steps=num_steps,
    )
    
    gc.collect()
    time.sleep(0.5)
    
    initial_memory = get_memory_snapshot()
    peak_gpu = initial_memory.gpu_memory_gb
    peak_cpu = initial_memory.cpu_memory_gb
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        
        from unsloth.kernels.mlx.models.llama import create_llama_model
        from unsloth.kernels.mlx.lora import apply_lora, LoRAConfig
        from unsloth.kernels.mlx.trainer import MLXTrainer, TrainingConfig
        
        print(f"Loading {model_name} in MLX ({bits}-bit)...")
        
        dtype = mx.float16 if bits == 16 else None
        
        try:
            from mlx_lm import load as mlx_load
            model, tokenizer = mlx_load(model_name)
        except Exception:
            model = create_llama_model(model_name, dtype=dtype)
            tokenizer = None
        
        if lora_rank > 0:
            lora_config = LoRAConfig(
                r=lora_rank,
                alpha=lora_rank * 2,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            model = apply_lora(model, lora_config)
        
        if gradient_checkpointing:
            from unsloth.kernels.mlx.trainer import enable_gradient_checkpointing
            enable_gradient_checkpointing(model)
        
        optimizer = optim.AdamW(learning_rate=learning_rate)
        
        dataset = create_sample_dataset(batch_size * num_steps, seq_length)
        
        def loss_fn(model, inputs):
            input_ids = mx.array(inputs["input_ids"])
            labels = mx.array(inputs["labels"])
            
            logits = model(input_ids)
            
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            )
            return mx.mean(loss)
        
        trainer = MLXTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        print(f"Training for {num_steps} steps...")
        start_time = time.time()
        
        step_times = []
        for step in range(num_steps):
            step_start = time.time()
            
            batch_idx = step * batch_size
            batch = {
                "input_ids": dataset["input_ids"][batch_idx:batch_idx + batch_size],
                "labels": dataset["labels"][batch_idx:batch_idx + batch_size],
            }
            
            loss = trainer.step(batch)
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            step_time = (time.time() - step_start) * 1000
            step_times.append(step_time)
            
            memory = get_memory_snapshot()
            peak_gpu = max(peak_gpu, memory.gpu_memory_gb)
            peak_cpu = max(peak_cpu, memory.cpu_memory_gb)
            
            loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
            result.loss_curve.append(loss_val)
            
            current_lr = learning_rate
            if step < num_steps * 0.1:
                current_lr = learning_rate * (step / (num_steps * 0.1))
            
            tokens = batch_size * seq_length
            tokens_per_sec = tokens / (step_time / 1000)
            
            result.step_metrics.append({
                "step": step + 1,
                "loss": loss_val,
                "learning_rate": current_lr,
                "step_time_ms": step_time,
                "gpu_memory_gb": memory.gpu_memory_gb,
                "cpu_memory_gb": memory.cpu_memory_gb,
                "tokens_per_second": tokens_per_sec,
            })
            
            if (step + 1) % 10 == 0:
                avg_time = np.mean(step_times[-10:])
                print(f"Step {step + 1}/{num_steps} | Loss: {loss_val:.4f} | "
                      f"Time: {avg_time:.1f}ms | GPU: {memory.gpu_memory_gb:.2f}GB")
        
        result.total_time_sec = time.time() - start_time
        result.final_loss = result.loss_curve[-1] if result.loss_curve else 0.0
        result.peak_gpu_memory_gb = peak_gpu
        result.peak_cpu_memory_gb = peak_cpu
        result.avg_tokens_per_second = np.mean([
            m["tokens_per_second"] for m in result.step_metrics
        ])
        
        print(f"MLX Training completed in {result.total_time_sec:.2f}s")
        
    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"MLX Training error: {result.error_message}")
    
    del model if 'model' in dir() else None
    gc.collect()
    
    return result


def train_pytorch(
    model_name: str,
    num_steps: int = 50,
    lora_rank: int = 16,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 1,
    seq_length: int = 512,
    learning_rate: float = 1e-4,
    bits: int = 16,
) -> TrainingResult:
    """Train using Unsloth (PyTorch/MPS)."""
    
    result = TrainingResult(
        framework="pytorch",
        model_name=model_name,
        bits=bits,
        lora_rank=lora_rank,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_steps=num_steps,
    )
    
    gc.collect()
    time.sleep(0.5)
    
    initial_memory = get_memory_snapshot()
    peak_gpu = initial_memory.gpu_memory_gb
    peak_cpu = initial_memory.cpu_memory_gb
    
    try:
        import torch
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
        
        print(f"Loading {model_name} in PyTorch ({bits}-bit)...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=seq_length,
            load_in_4bit=(bits == 4),
            dtype=torch.float16 if bits == 16 else None,
        )
        
        if lora_rank > 0:
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=lora_rank * 2,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
                random_state=3407,
            )
        
        dataset_dict = create_sample_dataset(batch_size * num_steps, seq_length)
        dataset = Dataset.from_dict({
            "input_ids": dataset_dict["input_ids"].tolist(),
            "attention_mask": dataset_dict["attention_mask"].tolist(),
            "labels": dataset_dict["labels"].tolist(),
        })
        
        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([b["input_ids"] for b in batch]),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch]),
            }
        
        class SimpleTrainer(Trainer):
            def __init__(self, *args, num_steps=50, **kwargs):
                super().__init__(*args, **kwargs)
                self._num_steps = num_steps
                self._step_metrics = []
            
            def training_step(self, model, inputs):
                step_start = time.time()
                loss = super().training_step(model, inputs)
                step_time = (time.time() - step_start) * 1000
                
                memory = get_memory_snapshot()
                nonlocal peak_gpu, peak_cpu
                peak_gpu = max(peak_gpu, memory.gpu_memory_gb)
                peak_cpu = max(peak_cpu, memory.cpu_memory_gb)
                
                self._step_metrics.append({
                    "step": len(self._step_metrics) + 1,
                    "loss": float(loss.item()),
                    "step_time_ms": step_time,
                    "gpu_memory_gb": memory.gpu_memory_gb,
                    "cpu_memory_gb": memory.cpu_memory_gb,
                })
                
                return loss
        
        training_args = TrainingArguments(
            output_dir="./tmp_output",
            num_train_epochs=1,
            max_steps=num_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            use_cpu=False,
            dataloader_drop_last=True,
        )
        
        print(f"Training for {num_steps} steps...")
        start_time = time.time()
        
        trainer = SimpleTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn,
            num_steps=num_steps,
        )
        
        trainer.train()
        
        result.total_time_sec = time.time() - start_time
        result.step_metrics = trainer._step_metrics
        result.loss_curve = [m["loss"] for m in trainer._step_metrics]
        result.final_loss = result.loss_curve[-1] if result.loss_curve else 0.0
        result.peak_gpu_memory_gb = peak_gpu
        result.peak_cpu_memory_gb = peak_cpu
        result.avg_tokens_per_second = np.mean([
            batch_size * seq_length / (m["step_time_ms"] / 1000)
            for m in result.step_metrics
        ])
        
        print(f"PyTorch Training completed in {result.total_time_sec:.2f}s")
        
    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"PyTorch Training error: {result.error_message}")
    
    if 'model' in dir():
        del model
    if 'trainer' in dir():
        del trainer
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    return result


def compare_training(
    model_name: str,
    num_steps: int = 50,
    lora_rank: int = 16,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    bits: int = 16,
    compare_both: bool = True,
) -> Dict[str, Any]:
    """Compare training between MLX and PyTorch."""
    
    results = {
        "config": {
            "model": model_name,
            "steps": num_steps,
            "lora_rank": lora_rank,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "bits": bits,
        },
        "mlx": None,
        "pytorch": None,
        "comparison": {},
    }
    
    print("\n" + "="*80)
    print(f"TRAINING COMPARISON: {model_name}")
    print("="*80)
    
    print("\n--- MLX Training ---")
    results["mlx"] = asdict(train_mlx(
        model_name=model_name,
        num_steps=num_steps,
        lora_rank=lora_rank,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bits=bits,
    ))
    
    if compare_both:
        print("\n--- PyTorch/MPS Training ---")
        results["pytorch"] = asdict(train_pytorch(
            model_name=model_name,
            num_steps=num_steps,
            lora_rank=lora_rank,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bits=bits,
        ))
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    mlx_result = results["mlx"]
    if mlx_result and mlx_result["success"]:
        print(f"\nMLX:")
        print(f"  Total time: {mlx_result['total_time_sec']:.2f}s")
        print(f"  Final loss: {mlx_result['final_loss']:.4f}")
        print(f"  Avg speed: {mlx_result['avg_tokens_per_second']:.1f} tokens/s")
        print(f"  Peak GPU: {mlx_result['peak_gpu_memory_gb']:.2f}GB")
        print(f"  Peak CPU: {mlx_result['peak_cpu_memory_gb']:.2f}GB")
    
    if compare_both and results["pytorch"]:
        pt_result = results["pytorch"]
        if pt_result["success"]:
            print(f"\nPyTorch/MPS:")
            print(f"  Total time: {pt_result['total_time_sec']:.2f}s")
            print(f"  Final loss: {pt_result['final_loss']:.4f}")
            print(f"  Avg speed: {pt_result['avg_tokens_per_second']:.1f} tokens/s")
            print(f"  Peak GPU: {pt_result['peak_gpu_memory_gb']:.2f}GB")
            print(f"  Peak CPU: {pt_result['peak_cpu_memory_gb']:.2f}GB")
        
        if mlx_result and mlx_result["success"] and pt_result["success"]:
            speedup = pt_result["total_time_sec"] / mlx_result["total_time_sec"]
            memory_ratio = (
                mlx_result["peak_gpu_memory_gb"] / pt_result["peak_gpu_memory_gb"]
                if pt_result["peak_gpu_memory_gb"] > 0 else 0
            )
            print(f"\nComparison:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory ratio: {memory_ratio:.2f}x")
            
            results["comparison"] = {
                "speedup": speedup,
                "memory_ratio": memory_ratio,
                "mlx_final_loss": mlx_result["final_loss"],
                "pytorch_final_loss": pt_result["final_loss"],
            }
    
    return results


def run_full_benchmark(
    output_file: str = "training_comparison_results.json",
    models: Optional[List[str]] = None,
    steps: int = 50,
):
    """Run full benchmark across configurations."""
    
    if models is None:
        models = [
            "unsloth/llama-3.2-1b-bnb-4bit",
        ]
    
    all_results = {}
    
    configs = [
        {"bits": 16, "lora_rank": 16, "gc": True, "ga": 1},
        {"bits": 16, "lora_rank": 16, "gc": True, "ga": 4},
        {"bits": 16, "lora_rank": 16, "gc": False, "ga": 1},
        {"bits": 4, "lora_rank": 16, "gc": True, "ga": 1},
    ]
    
    for model_name in models:
        model_results = {}
        for cfg in configs:
            config_name = f"bits{cfg['bits']}_lora{cfg['lora_rank']}_gc{cfg['gc']}_ga{cfg['ga']}"
            print(f"\n{'='*80}")
            print(f"Model: {model_name} | Config: {config_name}")
            print(f"{'='*80}")
            
            try:
                result = compare_training(
                    model_name=model_name,
                    num_steps=steps,
                    lora_rank=cfg["lora_rank"],
                    gradient_checkpointing=cfg["gc"],
                    gradient_accumulation_steps=cfg["ga"],
                    bits=cfg["bits"],
                    compare_both=True,
                )
                model_results[config_name] = result
            except Exception as e:
                print(f"Failed: {e}")
                traceback.print_exc()
            
            gc.collect()
            time.sleep(2)
        
        all_results[model_name] = model_results
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MLX Training Comparison Benchmark")
    parser.add_argument(
        "--model", type=str, default="unsloth/llama-3.2-1b-bnb-4bit",
        help="Model name or path"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of training steps"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (0 = no LoRA)"
    )
    parser.add_argument(
        "--bits", type=int, default=16, choices=[16, 4],
        help="Model precision (16 or 4)"
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=True,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Run full comparison benchmark"
    )
    parser.add_argument(
        "--mlx-only", action="store_true",
        help="Only run MLX, skip PyTorch comparison"
    )
    parser.add_argument(
        "--output", type=str, default="training_comparison_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    if not IS_APPLE_SILICON:
        print("WARNING: This benchmark is designed for Apple Silicon (M1/M2/M3/M4)")
        print("Some features may not work correctly on other platforms.\n")
    
    if args.compare_all:
        run_full_benchmark(output_file=args.output, steps=args.steps)
    else:
        compare_training(
            model_name=args.model,
            num_steps=args.steps,
            lora_rank=args.lora_rank,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.gradient_accumulation,
            bits=args.bits,
            compare_both=not args.mlx_only,
        )


if __name__ == "__main__":
    main()