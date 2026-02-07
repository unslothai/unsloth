# Apple Silicon (MPS) Support Guide

> **Status:** Production Ready (Beta)  
> **Last Updated:** February 2026  
> **Tested On:** macOS 14+, M1/M2/M3/M4 chips

## Quick Start

### Installation

```bash
# Install Unsloth with Apple Silicon support
pip install unsloth

# Verify installation
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth installed')"
```

### Basic Usage

```python
from unsloth import FastLanguageModel
import torch

# Load model (automatically uses MPS on Apple Silicon)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False,  # 4-bit not supported on MPS yet
    dtype=torch.bfloat16,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Train as usual...
```

---

## Known Limitations

The following features are **not supported** or have **limited support** on Apple Silicon (MPS):

### ❌ Not Supported

| Feature | Reason | Workaround |
|---------|--------|------------|
| **4-bit/8-bit Quantization (bitsandbytes)** | bitsandbytes library requires CUDA | Use 16-bit LoRA or wait for MLX quantization |
| **Padding-Free Training** | Requires Triton Flash Attention kernels | Standard attention will be used automatically |
| **FSDP (Fully Sharded Data Parallel)** | Requires NCCL/GLOO multi-GPU backend | Use single-GPU training (standard for M-series) |
| **DeepSpeed ZeRO Stage 2+** | Requires NCCL backend | Use standard LoRA/QLoRA training |
| **8-bit Optimizers (adamw_8bit)** | Requires bitsandbytes | Use `adamw_torch` (auto-switched) |
| **VLLM Inference** | VLLM doesn't support MPS backend | Use standard generation or GGUF export |
| **Flash Attention** | Custom Triton kernels don't support MPS | PyTorch's `scaled_dot_product_attention` is used |

### ⚠️ Partially Supported

| Feature | Limitations | Notes |
|---------|-------------|-------|
| **Vision Models** | No quantization support | Works in 16-bit mode, GGUF export supported |
| **GGUF Export** | Requires manual llama.cpp build | Metal-accelerated conversion available |
| **Training Optimizers** | No fused/paged optimizers | Standard PyTorch optimizers work fine |
| **bfloat16** | Limited on older hardware | M1/M2 may fall back to float16 |

---

## Performance Expectations

### Training Speed

Compared to equivalent NVIDIA GPUs:

| Hardware | Training Speed | VRAM Efficiency |
|----------|---------------|-----------------|
| M1 Pro (16GB) | ~0.3x RTX 3090 | Uses unified memory |
| M2 Pro (16GB) | ~0.4x RTX 3090 | Uses unified memory |
| M3 Max (36GB) | ~0.6x RTX 3090 | Uses unified memory |
| M4 Max (36GB+) | ~0.7x RTX 3090 | Uses unified memory |

### Memory Usage

Apple Silicon uses **unified memory** architecture:
- System RAM = GPU VRAM (shared)
- Unsloth reserves ~75% of total RAM for ML workloads
- 16GB MacBook can typically fine-tune 7B models (LoRA)
- 32GB+ recommended for 13B+ models

### Recommended Settings

```python
# For 16GB MacBooks
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",  # Start with 1B
    max_seq_length=1024,
    load_in_4bit=False,  # Required
    dtype=torch.bfloat16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Lower rank for memory efficiency
    target_modules=["q_proj", "v_proj"],  # Fewer modules
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",  # Essential
)

# Training args
from transformers import TrainingArguments
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",  # 8-bit not supported
    fp16=False,  # Use bf16 if supported
    bf16=True,   # Or False on M1
)
```

---

## Troubleshooting

### Installation Issues

#### "No module named 'triton'"
```bash
# Expected on macOS - Unsloth handles this automatically
pip install unsloth
```

#### "bitsandbytes not found"
```bash
# Expected on macOS - not required for MPS
# Unsloth will automatically disable bitsandbytes
```

### Runtime Issues

#### "MPS backend not available"
```python
# Check PyTorch MPS support
import torch
print(torch.backends.mps.is_available())  # Should be True

# If False, reinstall PyTorch:
# pip install torch torchvision torchaudio
```

#### "CUDA out of memory" error on Mac
```python
# This shouldn't happen - ensure you're using MPS
import os
os.environ["UNSLOTH_FORCE_MPS"] = "1"
```

#### "Padding-free training not supported"
```
This is expected! Unsloth automatically disables padding-free training on MPS.
Training will continue with standard attention (slightly slower but correct).
```

#### Training is very slow
- Ensure you're using PyTorch 2.1+ with MPS optimizations
- Try reducing `max_seq_length`
- Use gradient checkpointing
- Consider using a smaller model (1B instead of 7B)

### Memory Issues

#### System becomes unresponsive during training
```python
# Mac may swap to SSD - reduce batch size or sequence length
# Monitor memory with Activity Monitor

# Reduce memory usage:
args = TrainingArguments(
    per_device_train_batch_size=1,  # Lower
    gradient_accumulation_steps=8,   # Higher to compensate
    max_seq_length=512,  # Shorter sequences
)
```

#### "Process killed" during model loading
- System is running out of RAM
- Close other applications
- Use a smaller model
- Restart Python to free memory

---

## Verification

Run the comprehensive verification script:

```bash
# Basic verification (no downloads)
python verify_apple.py

# Full verification (downloads ~2GB of models)
python verify_apple.py --full

# Skip model downloads
python verify_apple.py --skip-downloads
```

This will test:
1. System compatibility
2. PyTorch MPS backend
3. Unsloth patches
4. Model loading
5. GGUF export
6. Vision model compatibility
7. Training loop

---

## Reporting Issues

When reporting Mac-specific bugs, please include:

```bash
# Run system info command
python -c "
import platform
import torch
import subprocess

print('=== System Info ===')
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')

# Hardware info
result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
print(f'Chip: {result.stdout.strip()}')

result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
mem_gb = int(result.stdout.strip()) / (1024**3)
print(f'Memory: {mem_gb:.1f} GB')
"
```

Then report at: https://github.com/unslothai/unsloth/issues

---

## Advanced Configuration

### Environment Variables

```bash
# Force MPS mode (even if CUDA detected)
export UNSLOTH_FORCE_MPS=1

# Disable auto-padding-free (if causing issues)
export UNSLOTH_DISABLE_AUTO_PADDING_FREE=1

# Enable MLX quantization (experimental)
export UNSLOTH_ENABLE_MLX_QUANT=1

# Verbose logging
export UNSLOTH_VERBOSE=1
```

### Using the Patcher (Advanced)

If you need manual control over Mac compatibility patches:

```python
from patcher import patch_for_mac

# Apply patches BEFORE importing unsloth
patch_for_mac(verbose=True)

# Now import unsloth
import unsloth

# Use unsloth normally...
```

---

## Roadmap

### Current Status (February 2026)

- ✅ Basic inference
- ✅ LoRA fine-tuning
- ✅ GGUF export
- ✅ Vision model support (16-bit)
- ✅ Automatic optimizer switching
- ✅ Unified memory management
- 🔄 MLX quantization integration (in progress)
- ⏳ Native Metal kernels (future)

### Coming Soon

- MLX-based 4-bit quantization
- Metal-accelerated RMS LayerNorm
- Improved performance with custom kernels

---

## References

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [MLX Framework](https://ml-explore.github.io/mlx/)
- [Unsloth Main Documentation](https://unsloth.ai/docs)
- [Apple Developer - Metal](https://developer.apple.com/metal/)

---

**Questions?** Join our [Discord](https://discord.com/invite/unsloth) or check the [main documentation](https://unsloth.ai/docs).
