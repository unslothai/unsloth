# MLX Training Infrastructure

Pure MLX training implementation for Apple Silicon - no PyTorch conversion overhead.

## Overview

This module provides a complete training stack for fine-tuning LLMs on Apple Silicon using MLX:

- **Optimizers**: AdamW, SGD with various learning rate schedulers
- **Loss Functions**: Cross-entropy, fused variants, z-loss
- **LoRA**: Full LoRA/QLoRA support with gradient checkpointing
- **Trainer**: Complete training loop with gradient accumulation, checkpointing, mixed precision

## Key Design Principles

1. **Pure MLX**: No `torch_to_mlx`/`mlx_to_torch` conversions in training loop
2. **Native `mx.array`**: All operations use MLX arrays directly
3. **Metal Integration**: Uses custom Metal kernels for SwiGLU, GEGLU, RMSNorm
4. **Compatible API**: Mirrors existing Unsloth API patterns

## Quick Start

```python
import mlx.core as mx
from unsloth.kernels.mlx import (
    create_llama_model,
    MLXTrainer,
    TrainingConfig,
    LoRAConfig,
)

# Create model
model = create_llama_model(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
)

# Configure training
training_config = TrainingConfig(
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=1000,
    gradient_accumulation_steps=4,
)

# Create trainer
trainer = MLXTrainer(
    model=model,
    config=training_config,
    train_dataloader=your_dataloader,
)

# Train
trainer.train()
```

## Components

### Optimizers (`optimizers.py`)

```python
from unsloth.kernels.mlx.optimizers import AdamW, SGD, LinearWarmupCosineDecay

# AdamW with weight decay
optimizer = AdamW(
    params={"weight": mx.random.normal((128, 128))},
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

# Learning rate scheduling
scheduler = LinearWarmupCosineDecay(
    warmup_steps=100,
    total_steps=1000,
    base_lr=1e-4,
    min_lr=1e-6,
)
```

Available schedulers:
- `LinearWarmupCosineDecay`: Linear warmup + cosine decay
- `CosineDecay`: Cosine annealing
- `LinearDecay`: Linear decay
- `StepDecay`: Step decay at intervals

### Loss Functions (`losses.py`)

```python
from unsloth.kernels.mlx.losses import (
    cross_entropy_loss,
    fused_cross_entropy_loss,
    LanguageModelingLoss,
)

# Standard cross-entropy
loss = cross_entropy_loss(logits, labels)

# Fused variant (more efficient)
loss = fused_cross_entropy_loss(logits, labels)

# Language modeling with shift
lm_loss_fn = LanguageModelingLoss()
loss = lm_loss_fn(logits, labels)
```

### LoRA (`lora.py`)

```python
from unsloth.kernels.mlx.lora import (
    LoRALinear,
    LoRAConfig,
    get_peft_model,
    mark_only_lora_as_trainable,
)

# Create LoRA config
config = LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Apply LoRA to model
lora_model = get_peft_model(base_model, config)

# Only train LoRA parameters
mark_only_lora_as_trainable(lora_model)

# Get LoRA state dict for saving
lora_state = get_lora_state_dict(lora_model)
```

### Trainer (`trainer.py`)

```python
from unsloth.kernels.mlx.trainer import MLXTrainer, TrainingConfig

# Configure training
training_config = TrainingConfig(
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=1000,
    eval_steps=100,
    save_steps=500,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    log_interval=10,
    mixed_precision=True,
    compile_model=True,
    checkpoint_dir="./checkpoints",
)

# Create trainer
trainer = MLXTrainer(
    model=model,
    config=training_config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

# Train
trainer.train()

# Save checkpoint
trainer.save_checkpoint("./checkpoint_final.safetensors")

# Load checkpoint
trainer.load_checkpoint("./checkpoint_final.safetensors")
```

## Training Loop

The trainer implements a full training loop:

1. **Forward Pass**: Compute logits and loss
2. **Backward Pass**: `mx.grad` computes gradients
3. **Gradient Clipping**: Optional max gradient norm
4. **Optimizer Step**: Update parameters
5. **Scheduler Step**: Update learning rate
6. **Evaluation**: Periodic evaluation
7. **Checkpointing**: Save model state

```python
# Custom training step
for step, batch in enumerate(trainer.train_dataloader):
    # Forward/backward/update in one call
    loss = trainer.train_step(batch)
    
    # Or manually:
    logits = model(batch["input_ids"])
    loss = loss_fn(logits, batch["labels"])
    
    # Compute gradients
    def loss_fn_wrapper(params):
        logits = model.apply(params, batch["input_ids"])
        return loss_fn(logits, batch["labels"])
    
    loss, grads = mx.value_and_grad(loss_fn_wrapper)(model.parameters())
    
    # Clip gradients
    grads = clip_grad_norm(grads, max_norm=1.0)
    
    # Update parameters
    optimizer.step(grads)
    optimizer.zero_grad()
```

## Gradient Checkpointing

```python
from unsloth.kernels.mlx.lora import GradientCheckpointing

# Enable gradient checkpointing to save memory
checkpointing = GradientCheckpointing()
model_with_checkpointing = checkpointing(model)
```

## Model Integration

The training infrastructure works with MLX-native models:

```python
from unsloth.kernels.mlx.models.llama import MLXLlamaForCausalLM, MLXModelConfig

config = MLXModelConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    max_position_embeddings=4096,
)

model = MLXLlamaForCausalLM(config)
```

## Metal Kernel Integration

Training uses optimized Metal kernels:

- **SwiGLU**: Fused SiLU + multiply
- **GEGLU**: GELU + multiply for Gemma models
- **RMSNorm**: Fast normalization

See `unsloth/kernels/metal/` for kernel implementations.

## Dataloader

The trainer expects a dataloader that yields batches of MLX arrays:

```python
def dataloader(dataset, batch_size=4):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        yield {
            "input_ids": mx.array(batch["input_ids"]),
            "labels": mx.array(batch["labels"]),
            "attention_mask": mx.array(batch["attention_mask"]),
        }
```

## Performance Tips

1. **Use `mx.compile`**: Enable `compile_model=True` in TrainingConfig
2. **Gradient Accumulation**: Reduce memory usage with larger effective batch sizes
3. **Mixed Precision**: Enable `mixed_precision=True` for faster training
4. **Gradient Checkpointing**: Save memory for large models
5. **Batch Size**: Start with batch_size=1 and increase gradually

## Testing

Run the integration test:

```bash
python tests/test_mlx_training_infrastructure.py
```

Run the standardized benchmark:

```bash
python tests/standardized_mlx_benchmark.py
```

## Comparison with PyTorch MPS

| Feature | PyTorch MPS | MLX |
|---------|-------------|-----|
| Memory | Unified | Unified |
| Compilation | Limited | `mx.compile` |
| Custom Kernels | Limited | Full Metal support |
| Conversion | Direct | None (pure MLX) |
| Speed | Good | Better for many ops |

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Metal Performance Shaders](https://developer.apple.com/metal/)
