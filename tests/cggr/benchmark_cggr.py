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
CGGR Benchmark Suite for Unsloth

Demonstrates the performance benefits of Confidence-Gated Gradient Routing:
- Training speedup (wall-clock time)
- Memory efficiency (peak VRAM)
- Quality preservation (loss convergence)

Usage:
    python benchmark_cggr.py --model unsloth/Llama-3.2-1B-Instruct --steps 100

Requirements:
    pip install unsloth datasets torch triton
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add parent directory to path for local testing
sys.path.insert(0, str(Path(__file__).parents[1]))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    mode: str  # "baseline" or "cggr"
    model_name: str
    total_steps: int
    total_time_seconds: float
    avg_step_time_ms: float
    peak_memory_gb: float
    final_loss: float
    loss_history: List[float] = field(default_factory=list)
    tokens_per_second: float = 0.0
    cggr_hard_ratio: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_name: str = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length: int = 512
    batch_size: int = 2
    num_steps: int = 100
    warmup_steps: int = 10
    cggr_min_tokens_ratio: float = 0.25
    cggr_router_layers: int = 2
    cggr_warmup_steps: int = 20
    learning_rate: float = 2e-4
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft[:1000]"
    output_dir: str = "./cggr_benchmark_results"
    seed: int = 42
    load_in_4bit: bool = True
    lora_rank: int = 16


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def create_dummy_dataset(tokenizer, num_samples: int = 1000, seq_length: int = 256):
    """Create a dummy dataset for benchmarking when real dataset unavailable."""
    from datasets import Dataset
    
    # Create synthetic training data
    samples = []
    for i in range(num_samples):
        # Create varied difficulty: some easy (repetitive), some hard (random)
        if i % 3 == 0:
            # Easy: repetitive pattern
            text = "The quick brown fox jumps over the lazy dog. " * (seq_length // 50)
        elif i % 3 == 1:
            # Medium: semi-random
            words = ["hello", "world", "machine", "learning", "artificial", "intelligence"]
            text = " ".join([words[j % len(words)] for j in range(seq_length // 3)])
        else:
            # Hard: more varied content
            text = f"Sample {i}: This is a training example with varied content about topic {i % 10}. " * 5
        
        samples.append({"text": tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Question {i}"}, {"role": "assistant", "content": text}],
            tokenize=False,
        )})
    
    return Dataset.from_list(samples)


def load_dataset_for_benchmark(config: BenchmarkConfig, tokenizer):
    """Load or create dataset for benchmarking."""
    try:
        from datasets import load_dataset
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        
        # Format the dataset
        def format_chat(example):
            if "messages" in example:
                text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            elif "text" in example:
                text = example["text"]
            else:
                # Try to construct from common fields
                text = str(example.get("content", example.get("input", "")))
            return {"text": text}
        
        dataset = dataset.map(format_chat)
        print(f"✓ Loaded dataset: {config.dataset_name}")
        return dataset
    except Exception as e:
        print(f"⚠ Could not load {config.dataset_name}: {e}")
        print("  Creating synthetic dataset for benchmarking...")
        return create_dummy_dataset(tokenizer, num_samples=1000, seq_length=config.max_seq_length // 2)


def run_benchmark(
    config: BenchmarkConfig,
    use_cggr: bool = False,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark (baseline or CGGR)."""
    
    mode = "cggr" if use_cggr else "baseline"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} benchmark")
        print(f"{'='*60}")
    
    # Reset memory tracking
    reset_gpu_memory()
    
    # Import here to ensure clean state
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    
    # Load model
    if verbose:
        print(f"Loading model: {config.model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    
    # Load dataset
    dataset = load_dataset_for_benchmark(config, tokenizer)
    
    # Create training arguments
    training_args = SFTConfig(
        output_dir=os.path.join(config.output_dir, mode),
        max_steps=config.num_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=1,
        save_strategy="no",
        seed=config.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        max_seq_length=config.max_seq_length,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    # Apply CGGR if enabled
    cggr_bridge = None
    if use_cggr:
        from unsloth.cggr import CGGRUnslothBridge
        cggr_bridge = CGGRUnslothBridge.patch_trainer(
            trainer,
            min_tokens_ratio=config.cggr_min_tokens_ratio,
            num_router_layers=config.cggr_router_layers,
            warmup_steps=config.cggr_warmup_steps,
        )
    
    # Collect metrics during training
    loss_history = []
    step_times = []
    
    # Custom callback to track metrics
    class MetricsCallback:
        def __init__(self):
            self.last_time = None
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append(logs["loss"])
            if self.last_time:
                step_times.append(time.time() - self.last_time)
            self.last_time = time.time()
    
    # Run training
    if verbose:
        print(f"Starting training for {config.num_steps} steps...")
    
    start_time = time.time()
    metrics_cb = MetricsCallback()
    metrics_cb.last_time = start_time
    
    # Override logging callback
    original_log = trainer.log
    def custom_log(logs):
        metrics_cb.on_log(None, None, None, logs)
        return original_log(logs)
    trainer.log = custom_log
    
    # Train
    trainer.train()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Collect final metrics
    peak_memory = get_gpu_memory_gb()
    final_loss = loss_history[-1] if loss_history else 0.0
    avg_step_time = (total_time / config.num_steps) * 1000  # Convert to ms
    
    # Calculate tokens per second
    tokens_per_step = config.batch_size * config.max_seq_length
    tokens_per_second = (tokens_per_step * config.num_steps) / total_time
    
    # Get CGGR stats if applicable
    cggr_hard_ratio = None
    if cggr_bridge:
        stats = cggr_bridge.get_stats()
        cggr_hard_ratio = stats.get("cggr/hard_ratio", config.cggr_min_tokens_ratio)
    
    result = BenchmarkResult(
        mode=mode,
        model_name=config.model_name,
        total_steps=config.num_steps,
        total_time_seconds=total_time,
        avg_step_time_ms=avg_step_time,
        peak_memory_gb=peak_memory,
        final_loss=final_loss,
        loss_history=loss_history,
        tokens_per_second=tokens_per_second,
        cggr_hard_ratio=cggr_hard_ratio,
    )
    
    if verbose:
        print(f"\n{mode.upper()} Results:")
        print(f"  Total time:       {total_time:.2f}s")
        print(f"  Avg step time:    {avg_step_time:.2f}ms")
        print(f"  Peak memory:      {peak_memory:.2f} GB")
        print(f"  Final loss:       {final_loss:.4f}")
        print(f"  Tokens/second:    {tokens_per_second:.0f}")
        if cggr_hard_ratio:
            print(f"  CGGR hard ratio:  {cggr_hard_ratio:.2%}")
    
    # Cleanup
    del model, trainer
    reset_gpu_memory()
    
    return result


def compare_results(baseline: BenchmarkResult, cggr: BenchmarkResult) -> Dict:
    """Compare baseline and CGGR results."""
    speedup = baseline.total_time_seconds / cggr.total_time_seconds
    memory_savings = (baseline.peak_memory_gb - cggr.peak_memory_gb) / baseline.peak_memory_gb * 100
    loss_diff = abs(cggr.final_loss - baseline.final_loss) / baseline.final_loss * 100
    throughput_gain = (cggr.tokens_per_second - baseline.tokens_per_second) / baseline.tokens_per_second * 100
    
    comparison = {
        "speedup": f"{speedup:.2f}x",
        "memory_savings_percent": f"{memory_savings:.1f}%",
        "loss_difference_percent": f"{loss_diff:.2f}%",
        "throughput_gain_percent": f"{throughput_gain:.1f}%",
        "baseline_time_seconds": baseline.total_time_seconds,
        "cggr_time_seconds": cggr.total_time_seconds,
        "baseline_tokens_per_second": baseline.tokens_per_second,
        "cggr_tokens_per_second": cggr.tokens_per_second,
    }
    
    return comparison


def print_summary(baseline: BenchmarkResult, cggr: BenchmarkResult, comparison: Dict):
    """Print a formatted summary of benchmark results."""
    print("\n" + "="*70)
    print("CGGR BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Baseline':<20} {'CGGR':<20}")
    print("-"*70)
    print(f"{'Total Time':<30} {baseline.total_time_seconds:<20.2f}s {cggr.total_time_seconds:<20.2f}s")
    print(f"{'Avg Step Time':<30} {baseline.avg_step_time_ms:<20.2f}ms {cggr.avg_step_time_ms:<20.2f}ms")
    print(f"{'Peak Memory':<30} {baseline.peak_memory_gb:<20.2f}GB {cggr.peak_memory_gb:<20.2f}GB")
    print(f"{'Final Loss':<30} {baseline.final_loss:<20.4f} {cggr.final_loss:<20.4f}")
    print(f"{'Tokens/Second':<30} {baseline.tokens_per_second:<20.0f} {cggr.tokens_per_second:<20.0f}")
    
    print("\n" + "-"*70)
    print("IMPROVEMENTS:")
    print(f"  🚀 Speedup:           {comparison['speedup']}")
    print(f"  💾 Memory Savings:    {comparison['memory_savings_percent']}")
    print(f"  📈 Throughput Gain:   {comparison['throughput_gain_percent']}")
    print(f"  📉 Loss Difference:   {comparison['loss_difference_percent']} (lower is better)")
    
    # Quality assessment
    loss_diff = float(comparison['loss_difference_percent'].rstrip('%'))
    if loss_diff < 5:
        quality = "✅ Excellent - Minimal quality impact"
    elif loss_diff < 10:
        quality = "✓ Good - Acceptable quality trade-off"
    else:
        quality = "⚠ Consider adjusting min_tokens_ratio"
    print(f"  🎯 Quality:           {quality}")
    print("="*70)


def save_results(
    baseline: BenchmarkResult,
    cggr: BenchmarkResult,
    comparison: Dict,
    config: BenchmarkConfig,
    output_dir: str,
):
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "config": asdict(config),
        "baseline": asdict(baseline),
        "cggr": asdict(cggr),
        "comparison": comparison,
    }
    
    output_file = os.path.join(output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="CGGR Benchmark Suite for Unsloth")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct",
                       help="Model name or path")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--cggr-ratio", type=float, default=0.25,
                       help="CGGR min_tokens_ratio (0.25 = keep 25%% hardest)")
    parser.add_argument("--output-dir", type=str, default="./cggr_benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--baseline-only", action="store_true",
                       help="Run only baseline benchmark")
    parser.add_argument("--cggr-only", action="store_true",
                       help="Run only CGGR benchmark")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        model_name=args.model,
        num_steps=args.steps,
        batch_size=args.batch_size,
        max_seq_length=args.seq_length,
        cggr_min_tokens_ratio=args.cggr_ratio,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*70)
    print("CGGR BENCHMARK SUITE FOR UNSLOTH")
    print("="*70)
    print(f"Model:          {config.model_name}")
    print(f"Steps:          {config.num_steps}")
    print(f"Batch Size:     {config.batch_size}")
    print(f"Seq Length:     {config.max_seq_length}")
    print(f"CGGR Ratio:     {config.cggr_min_tokens_ratio}")
    print("="*70)
    
    baseline_result = None
    cggr_result = None
    
    # Run benchmarks
    if not args.cggr_only:
        baseline_result = run_benchmark(config, use_cggr=False)
    
    if not args.baseline_only:
        cggr_result = run_benchmark(config, use_cggr=True)
    
    # Compare and save results
    if baseline_result and cggr_result:
        comparison = compare_results(baseline_result, cggr_result)
        print_summary(baseline_result, cggr_result, comparison)
        save_results(baseline_result, cggr_result, comparison, config, args.output_dir)
    elif baseline_result:
        print("\n✓ Baseline benchmark complete")
        print(f"  Time: {baseline_result.total_time_seconds:.2f}s")
        print(f"  Loss: {baseline_result.final_loss:.4f}")
    elif cggr_result:
        print("\n✓ CGGR benchmark complete")
        print(f"  Time: {cggr_result.total_time_seconds:.2f}s")
        print(f"  Loss: {cggr_result.final_loss:.4f}")
        if cggr_result.cggr_hard_ratio:
            print(f"  Hard token ratio: {cggr_result.cggr_hard_ratio:.2%}")


if __name__ == "__main__":
    main()
