# MLX Weight Configuration Guide

This document specifies which weights should be quantized, which should have LoRA adapters, and which should be frozen during training for optimal MLX-based fine-tuning.

## Summary Table

| Layer Type | Quantize (4-bit) | LoRA Adapter | Frozen | Notes |
|------------|------------------|--------------|--------|-------|
| `q_proj` | ✅ Yes | ✅ Yes | Partial | Query projection - primary LoRA target |
| `k_proj` | ✅ Yes | ✅ Yes | Partial | Key projection - LoRA target |
| `v_proj` | ✅ Yes | ✅ Yes | Partial | Value projection - primary LoRA target |
| `o_proj` | ✅ Yes | ✅ Yes | Partial | Output projection - LoRA target |
| `gate_proj` | ✅ Yes | ❌ No | ✅ Yes | MLP gate - typically frozen |
| `up_proj` | ✅ Yes | ✅ Optional | ✅ Yes | MLP up - can add LoRA for extra capacity |
| `down_proj` | ✅ Yes | ✅ Optional | ✅ Yes | MLP down - can add LoRA for extra capacity |
| `embed_tokens` | ❌ No | ❌ No | ✅ Yes | Input embeddings - keep in 16-bit |
| `lm_head` | ❌ No | ❌ No | ✅ Yes | Output head - keep in 16-bit |
| `input_layernorm` | ❌ No | ❌ No | ✅ Yes | Layer norm - no quantization |
| `post_attention_layernorm` | ❌ No | ❌ No | ✅ Yes | Layer norm - no quantization |
| `norm` (final) | ❌ No | ❌ No | ✅ Yes | Final layer norm - no quantization |
| `rotary_emb` | ❌ No | ❌ No | ✅ Yes | Rotary embeddings - keep as-is |

## Detailed Configuration

### 1. Quantization (4-bit)

**Target Modules:**
```python
QUANTIZE_TARGETS = [
    "q_proj",      # Query projection
    "k_proj",      # Key projection  
    "v_proj",      # Value projection
    "o_proj",      # Output projection
    "gate_proj",   # MLP gate
    "up_proj",     # MLP up projection
    "down_proj",   # MLP down projection
]
```

**Skip Quantization:**
```python
SKIP_QUANTIZATION_MODULES = [
    "embed_tokens",    # Input embeddings
    "lm_head",         # Output head
    "layernorm",       # All layer norms
    "norm",            # Final norm
    "rotary_emb",      # Rotary embeddings
    "bias",            # Any bias terms
]
```

### 2. LoRA Adapters

**Standard LoRA Targets (recommended):**
```python
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
```

**Extended LoRA Targets (more parameters, better quality):**
```python
LORA_TARGET_MODULES_EXTENDED = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Full LoRA (maximum capacity):**
```python
LORA_TARGET_MODULES_FULL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**LoRA Configuration:**
```python
LORA_CONFIG = {
    "r": 16,              # LoRA rank (8, 16, 32, 64)
    "lora_alpha": 32,     # Scaling factor (typically 2*r)
    "lora_dropout": 0,    # Dropout (0 for inference/production)
    "bias": "none",       # Bias handling
}
```

### 3. Frozen Weights

**Always Frozen:**
- Base model weights (when using LoRA)
- Layer normalization weights
- Rotary embeddings
- Embeddings (input/output)

**Trainable with LoRA:**
- Attention projections (q, k, v, o) via LoRA adapters
- Optionally MLP projections via LoRA adapters

**Memory Impact:**
| Configuration | Trainable Parameters | Memory Reduction |
|---------------|---------------------|------------------|
| LoRA (q, v only) | ~0.1% of model | ~99% |
| LoRA (q, k, v, o) | ~0.2% of model | ~98% |
| LoRA (all projections) | ~0.5% of model | ~95% |

## MLX-Specific Considerations

### Memory Layout
MLX uses unified memory on Apple Silicon. The configuration impacts:
- **GPU Memory**: Active tensors during computation
- **CPU Memory**: Model weights, optimizer states

### Quantization Format
MLX uses its own 4-bit quantization format:
```python
from mlx.core import quantize

# MLX quantization format
quantized_weight = quantize(weight, group_size=64, bits=4)
```

### LoRA Implementation
```python
# MLX LoRA layer
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = 1.0 / r
        self.lora_A.weight = mx.zeros((r, in_features))
        self.lora_B.weight = mx.zeros((out_features, r))
    
    def __call__(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling
```

### Freezing Implementation
```python
def freeze_model(model, train_lora_only=True):
    """Freeze model weights, optionally keeping LoRA trainable."""
    for name, param in model.named_parameters():
        if train_lora_only:
            # Only LoRA parameters are trainable
            param.requires_grad = "lora" in name.lower()
        else:
            param.requires_grad = True
    return model
```

## Recommended Configurations by Use Case

### 1. Quick Fine-tuning (Low Memory)
```python
config = {
    "quantization": "4bit",
    "lora_rank": 8,
    "lora_targets": ["q_proj", "v_proj"],
    "gradient_checkpointing": True,
}
```

### 2. Standard Fine-tuning (Balanced)
```python
config = {
    "quantization": "4bit",
    "lora_rank": 16,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gradient_checkpointing": True,
}
```

### 3. High-Quality Fine-tuning (More Memory)
```python
config = {
    "quantization": "4bit",
    "lora_rank": 32,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    "gradient_checkpointing": True,
}
```

### 4. Full Fine-tuning (Maximum Quality)
```python
config = {
    "quantization": None,  # 16-bit
    "lora_rank": 64,
    "lora_targets": "all-linear",
    "gradient_checkpointing": True,
}
```

## Validation

Use the benchmark scripts to validate your configuration:
```bash
# Check memory usage
python benchmarks/mlx_model_loading_profile.py --model your-model --bits 4

# Check training speed
python benchmarks/mlx_training_comparison.py --model your-model --steps 50

# Check gradient checkpointing impact
python benchmarks/mlx_gradient_checkpointing_benchmark.py --model your-model
```
