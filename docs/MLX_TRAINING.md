# MLX Training Infrastructure for Unsloth

Pure MLX training implementation for Apple Silicon Macs without PyTorch dependencies.

## Overview

This training infrastructure enables full model training using only MLX on Apple Silicon, leveraging the unified memory architecture for superior performance compared to PyTorch MPS.

## Quick Start

```bash
# Train with LoRA (recommended)
python scripts/test_mlx_training_example.py \
    --steps 1000 \
    --batch-size 4 \
    --learning-rate 1e-4

# Full fine-tuning (requires more memory)
python scripts/test_mlx_training_example.py \
    --no-lora \
    --steps 500 \
    --learning-rate 5e-5

# Mixed precision training
python scripts/test_mlx_training_example.py --bf16 --steps 500
```

## Components

### 1. Optimizers (`optimizers.py`)

- **AdamW**: Weight decay, momentum, gradient clipping
- **SGD**: Momentum, Nesterov acceleration
- **Schedulers**: Linear warmup + cosine decay, step decay

```python
from unsloth.kernels.mlx import AdamW, LinearWarmupCosineDecay

optimizer = AdamW(learning_rate=1e-4, weight_decay=0.01)
scheduler = LinearWarmupCosineDecay(optimizer, warmup_steps=100, total_steps=1000)

# Get learning rate for step
lr = scheduler.get_lr(step)

# Update parameters
updated_params = optimizer(grads, params)
```

### 2. Loss Functions (`losses.py`)

- **Cross-Entropy**: Language modeling with label smoothing
- **MSE**: Mean squared error
- **KL Divergence**: For distillation

```python
from unsloth.kernels.mlx import cross_entropy_loss

loss = cross_entropy_loss(
    logits,              # (batch, seq_len, vocab)
    labels,              # (batch, seq_len)
    label_smoothing=0.1,
    ignore_index=-100,
)
```

### 3. LoRA (`lora.py`)

- **LoRALinear**: Efficient fine-tuning
- **QuantizedLoRALinear**: 4-bit QLoRA
- **Gradient Checkpointing**: Memory efficiency

```python
from unsloth.kernels.mlx import LoRAConfig, get_peft_model, mark_only_lora_as_trainable

lora_config = LoRAConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(base_model, lora_config)
trainable_params = mark_only_lora_as_trainable(model)
```

### 4. Trainer (`trainer.py`)

- **MLXTrainer**: Complete training loop
- **Features**: Mixed precision, gradient accumulation, checkpointing

```python
from unsloth.kernels.mlx import MLXTrainer, TrainingConfig

config = TrainingConfig(
    num_epochs=3,
    batch_size=4,
    output_dir="./checkpoints",
    fp16=True,
)

trainer = MLXTrainer(model, optimizer, config)
trainer.train(train_dataset, eval_dataset)
```

## Example: Custom Training Loop

```python
import mlx.core as mx
from unsloth.kernels.mlx import (
    create_llama_model, AdamW, cross_entropy_loss,
    LoRAConfig, get_peft_model, mark_only_lora_as_trainable,
    LinearWarmupCosineDecay, clip_grad_norm
)

# Create model
config = MLXModelConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=6,
)
model = create_llama_model(config)

# Apply LoRA
lora_config = LoRAConfig(r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)
trainable_params = mark_only_lora_as_trainable(model)

# Setup optimizer and scheduler
optimizer = AdamW(learning_rate=1e-4, weight_decay=0.01)
scheduler = LinearWarmupCosineDecay(optimizer, warmup_steps=100, total_steps=1000)

# Training loop
for step in range(1000):
    # Get batch
    input_ids = ...  # shape: (batch, seq_len)
    labels = ...
    
    # Compute loss and gradients
    def loss_fn(params):
        model.set_params({**model.get_params(), **params})
        logits, loss = model(input_ids=input_ids, labels=labels)
        return loss
    
    current_params = {k: model.get_params()[k] for k in trainable_params}
    loss_value, grads = mx.value_and_grad(loss_fn)(current_params)
    
    # Clip gradients and update
    _, grads = clip_grad_norm(grads, max_norm=1.0)
    updated_params = optimizer(grads, current_params)
    
    for name, value in updated_params.items():
        model.get_params()[name] = value
    
    if step % 10 == 0:
        print(f"Step {step}: loss={float(loss_value):.4f}")
```

## Testing

Run unit tests:
```bash
python -m unittest tests.test_mlx_training -v
```

Run example script:
```bash
python scripts/test_mlx_training_example.py --help
python scripts/test_mlx_training_example.py --steps 100 --batch-size 2
```

## Memory Optimization Tips

1. **Use LoRA**: Reduces trainable parameters by 99%+
2. **Gradient Accumulation**: Effective larger batch sizes
3. **Gradient Checkpointing**: Trade compute for memory
4. **Mixed Precision**: FP16/BF16 for 2x memory savings
5. **Smaller Models**: Test with 4-layer models first

## Architecture

```
unsloth/kernels/mlx/
├── __init__.py       # Exports all training components
├── optimizers.py     # AdamW, SGD, schedulers, gradient clipping
├── losses.py         # Cross-entropy, MSE, KL divergence
├── lora.py          # LoRA layers, QLoRA, checkpointing
├── trainer.py        # MLXTrainer, TrainingConfig
└── models/           # Base models (MLXLinear, etc.)
```

## Performance Notes

- MLX uses unified memory on Apple Silicon (no CPU/GPU copies)
- Metal performance shaders for compute-heavy ops
- Native Swift/Triton kernel compilation
- Automatic differentiation via `mx.grad`
- No PyTorch overhead or conversion costs

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- `mlx` package: `pip install mlx`
- 16GB+ RAM recommended for larger models

## License

Apache 2.0 - See LICENSE file for details.
