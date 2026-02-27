# Unsloth MLX Benchmarks

This directory contains comprehensive benchmarks for profiling and testing MLX integration with Unsloth on Apple Silicon.

## Quick Start

```bash
# Profile model loading
python benchmarks/mlx_model_loading_profile.py --model unsloth/llama-3.2-1b-bnb-4bit --compare

# Compare training
python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b-bnb-4bit --steps 50

# Test gradient checkpointing
python benchmarks/mlx_gradient_checkpointing_benchmark.py --model unsloth/llama-3.2-1b-bnb-4bit

# Test gradient accumulation
python benchmarks/mlx_gradient_accumulation_test.py --model unsloth/llama-3.2-1b-bnb-4bit
```

## Benchmark Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `mlx_model_loading_profile.py` | Profile memory/time for 16-bit and 4-bit loading | JSON with memory stats |
| `mlx_training_comparison.py` | Compare MLX vs PyTorch training | Loss curves, speed, memory |
| `mlx_gradient_checkpointing_benchmark.py` | Measure GC memory/speed tradeoff | Memory saved, slowdown % |
| `mlx_gradient_accumulation_test.py` | Verify grad accumulation correctness | Pass/fail, gradient norms |

## Detailed Usage

### 1. Model Loading Profile

Profiles memory usage and loading time for different precisions.

```bash
# Single model
python benchmarks/mlx_model_loading_profile.py --model unsloth/llama-3.2-1b-bnb-4bit

# Compare 16-bit vs 4-bit
python benchmarks/mlx_model_loading_profile.py --model unsloth/llama-3.2-1b --bits 16 4

# Batch benchmark multiple models
python benchmarks/mlx_model_loading_profile.py --batch --output results.json
```

**Output includes:**
- Loading time (seconds)
- GPU memory delta (GB)
- CPU memory delta (GB)
- Model parameters (billions)
- Success/failure status

### 2. Training Comparison

Compares training between MLX and PyTorch/MPS.

```bash
# Basic comparison
python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b-bnb-4bit --steps 50

# With specific LoRA rank
python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b --lora-rank 32

# With gradient accumulation
python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b --gradient-accumulation 4

# MLX only (skip PyTorch)
python benchmarks/mlx_training_comparison.py --model unsloth/llama-3.2-1b --mlx-only

# Full benchmark across configurations
python benchmarks/mlx_training_comparison.py --compare-all
```

**Output includes:**
- Loss curve per step
- Tokens per second
- Peak GPU/CPU memory
- Speedup comparison

### 3. Gradient Checkpointing Benchmark

Measures the memory/speed tradeoff of gradient checkpointing.

```bash
# Default benchmark
python benchmarks/mlx_gradient_checkpointing_benchmark.py

# Custom sequence lengths
python benchmarks/mlx_gradient_checkpointing_benchmark.py --seq-lengths 512 1024 2048 4096

# More steps
python benchmarks/mlx_gradient_checkpointing_benchmark.py --steps 20
```

**Output includes:**
- Memory saved (%)
- Speed slowdown (%)
- Peak memory comparison
- Average step time

### 4. Gradient Accumulation Test

Verifies gradient accumulation produces correct gradients.

```bash
# Default test
python benchmarks/mlx_gradient_accumulation_test.py

# Custom accumulation steps
python benchmarks/mlx_gradient_accumulation_test.py --accum-steps 8

# Longer test
python benchmarks/mlx_gradient_accumulation_test.py --updates 20
```

**Output includes:**
- Loss comparison (batch vs accumulated)
- Gradient norm comparison
- Pass/fail status

## Additional Benchmarks

### Legacy Benchmarks

| File | Description |
|------|-------------|
| `benchmark_pytorch_vs_mlx.py` | Original comparison benchmark |
| `comprehensive_apple_benchmark.py` | Full Apple Silicon benchmark suite |
| `standardized_mlx_benchmark.py` | Standardized MLX tests |
| `benchmark_4bit.py` | 4-bit quantization benchmark |
| `benchmark_lora.py` | LoRA training benchmark |
| `benchmark_rms_layernorm.py` | RMS LayerNorm benchmark |
| `benchmark_swiglu.py` | SwiGLU activation benchmark |
| `benchmark_geglu.py` | GeGLU activation benchmark |

### MLX Subdirectory

```
benchmarks/mlx/
├── benchmark_bridge.py      # PyTorch↔MLX bridge performance
├── benchmark_kernels.py     # MLX kernel benchmarks
└── benchmark_training.py    # MLX training benchmarks
```

## Configuration

See `docs/mlx_weight_configuration.md` for:
- Weight quantization targets
- LoRA adapter configuration
- Frozen weight specifications
- Recommended configurations by use case

## Interpreting Results

### Memory Tracking on Apple Silicon

Apple Silicon uses **unified memory** - GPU and CPU share the same RAM. The benchmarks track:

- **GPU Memory**: Active Metal buffers (MLX tensors)
- **CPU Memory**: Process memory from `vm_stat`
- **Total Memory**: Physical RAM of the machine

### Expected Performance

| Model | 16-bit Memory | 4-bit Memory | Training Speed |
|-------|---------------|--------------|----------------|
| 1B params | ~2 GB | ~0.5 GB | ~5000 tokens/s |
| 3B params | ~6 GB | ~1.5 GB | ~3000 tokens/s |
| 8B params | ~16 GB | ~4 GB | ~1500 tokens/s |

### Gradient Checkpointing Impact

| Sequence Length | Memory Saved | Slowdown |
|-----------------|--------------|----------|
| 512 | ~20% | ~10% |
| 1024 | ~35% | ~15% |
| 2048 | ~50% | ~20% |
| 4096 | ~65% | ~25% |

## Troubleshooting

### Out of Memory

1. Enable gradient checkpointing
2. Reduce batch size, increase gradient accumulation
3. Use 4-bit quantization
4. Reduce sequence length

### Slow Training

1. Check MLX is being used (not PyTorch fallback)
2. Verify Metal is available (`mlx.core.metal.is_available()`)
3. Reduce gradient checkpointing
4. Increase batch size if memory allows

### Incorrect Results

1. Check gradient accumulation is correct with the test script
2. Verify loss values match between MLX and PyTorch
3. Check LoRA is applied correctly
