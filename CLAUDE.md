# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unsloth is a library for 2-5X faster LLM finetuning with 70-80% less VRAM usage. It supports a wide range of models including Llama, Qwen, Mistral, Gemma, Phi, and others. The library uses custom Triton kernels for optimization and integrates with Hugging Face's ecosystem (transformers, TRL, PEFT).

## Key Commands

### Installation
```bash
# Basic installation
pip install unsloth

# Update to latest version
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Platform-specific installation (example for CUDA 12.1, Torch 2.4)
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

### Running Tests
```bash
# Run specific test file
python -m pytest tests/test_model_registry.py -v

# Run tests in MOE kernels
cd unsloth/kernels/moe && bash tests/run_qwen3_moe_tests.sh
```

### Linting
```bash
# The project uses ruff for linting (based on moe/requirements.txt)
ruff check .
```

## High-Level Architecture

### Core Components

1. **Model Loaders** (`unsloth/models/`): FastModel classes for each supported model type
   - `loader.py`: Main entry point for model loading
   - Model-specific files: `llama.py`, `mistral.py`, `qwen2.py`, `qwen3.py`, etc.
   - Each model has optimized implementations using custom kernels

2. **Custom Kernels** (`unsloth/kernels/`): Triton-based optimizations
   - `fast_lora.py`: Optimized LoRA implementations
   - `rope_embedding.py`: Rotary position embeddings
   - `cross_entropy_loss.py`: Memory-efficient loss computation
   - `moe/`: Mixture of Experts kernels with grouped GEMM operations

3. **Model Registry** (`unsloth/registry/`): Model metadata and configuration
   - Each model family has its own registry file (e.g., `_llama.py`, `_qwen.py`)
   - Handles model naming conventions and quantization types
   - See `REGISTRY.md` for detailed structure

4. **Training Integration** (`unsloth/trainer.py`): 
   - Patches and optimizes HuggingFace's TRL trainers
   - Supports SFT, DPO, GRPO, and other RL methods

5. **Save/Export** (`unsloth/save.py`):
   - Model merging and saving functionality
   - Export to GGUF, ONNX, and other formats
   - Integration with llama.cpp

### Key Design Patterns

1. **Import Order Matters**: Unsloth must be imported before transformers/TRL/PEFT to apply optimizations
2. **Lazy Patching**: Models are patched at runtime to replace slow operations with optimized kernels
3. **Memory Optimization**: Uses gradient checkpointing, 4-bit quantization, and custom memory management
4. **Device Abstraction**: Supports both CUDA and Intel XPU through device type detection

### Important Environment Variables
- `UNSLOTH_USE_MODELSCOPE=1`: Use ModelScope for model downloads
- `HF_HUB_ENABLE_HF_TRANSFER=1`: Faster downloads (set automatically)
- `PYTORCH_CUDA_ALLOC_CONF`: Memory fragmentation settings (set automatically)

## Development Tips

1. **Testing Models**: Use the CLI tool for quick testing:
   ```bash
   python unsloth-cli.py --model_name "unsloth/llama-3-8b" --max_seq_length 2048
   ```

2. **Debugging Memory Issues**: Check if models are imported in correct order - Unsloth must be first

3. **Adding New Models**: 
   - Create model implementation in `unsloth/models/`
   - Add registry entry in `unsloth/registry/`
   - Update `loader.py` to include the new model

4. **Kernel Development**: MOE kernels have their own test infrastructure in `unsloth/kernels/moe/`

## Current Debugging: Beam Search _reorder_cache Error

### Problem
When using beam search with Unsloth models, users get:
```
NotImplementedError: Make sure that a `_reorder_cache` function is correctly implemented in transformers.models.llama.modeling_llama to enable beam search for <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
```

### Root Cause Analysis
1. Newer transformer models (Llama, Gemma, Mistral) don't implement `_reorder_cache` because they use new Cache classes
2. However, beam search might still use legacy cache format internally, requiring `_reorder_cache`
3. The error occurs when PEFT models call `self.base_model.generate()` which eventually needs `_reorder_cache`

### Fixes Attempted (in chronological order)

1. **Initial Fix** (commit e526628): Added `reorder_cache` method to model classes in `pre_patch()`
   - Issue: Method name was wrong (used `reorder_cache` instead of `_reorder_cache`)

2. **Name Fix** (commit 7a9e575): Changed method name to `_reorder_cache`
   - Issue: Only patched class, not instances

3. **Instance Patching** (commit 8e6c982): Added patching to model instances in `from_pretrained`
   - Issue: PEFT models needed patching too

4. **PEFT Support**: Added patching for PEFT models and their base_model
   - Issue: Method needed to be a staticmethod, not instance method

5. **Staticmethod Fix** (commit 540bf1f): Made `_reorder_cache` a staticmethod when adding to classes
   - Issue: Error shows it's looking in `transformers.models.llama.modeling_llama`

6. **Current Fix** (commit 5adb3f7): Patching both local imports AND the transformers module namespace
   ```python
   # In pre_patch():
   # Patch local import
   LlamaForCausalLM._reorder_cache = staticmethod(general_reorder_cache)
   
   # Also patch transformers module namespace
   import transformers.models.llama.modeling_llama
   transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache = staticmethod(general_reorder_cache)
   ```

### General Solution Created
Added to `_utils.py`:
- `general_reorder_cache()`: Implementation that works for all transformer models
- `patch_model_for_beam_search()`: Utility to patch any model instance

### Files Modified
1. `unsloth/models/llama.py`: Multiple patches for beam search support
2. `unsloth/models/_utils.py`: General beam search utilities
3. All changes are in the `fix-reorder-cache` branch

### Debug Code Still Present
There are debug print statements in `llama.py` around lines 1699-1725 that can help trace if patches are being applied.

### Next Steps for Debugging
1. Run on CUDA-enabled machine to see if current fix works
2. Check if error persists with debug prints enabled
3. May need to investigate if there are other import paths or timing issues
4. Consider if we need to patch earlier in the import process
5. Check if the model instance's `__class__.__module__` is actually `transformers.models.llama.modeling_llama`