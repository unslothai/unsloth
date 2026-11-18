# BF16 parameters with stochastic AdamW updates

`AdamWBF16` stores trainable parameters and their gradients in BF16. Adam's
moment buffers and update arithmetic remain FP32; there is no FP32 master
weight. The GPU kernel computes the update and stochastically rounds the
new parameter in one pass, without allocating a full FP32 parameter temporary.

Apply preparation after attaching adapters and before creating the trainer or
optimizer:

```python
from unsloth.optimizers import AdamWBF16, prepare_bf16_training
from trl import SFTConfig, SFTTrainer

# model has already been loaded and its trainable parameters selected.
prepare_bf16_training(model)
training_args = SFTConfig(bf16=True, fp16=False, learning_rate=2e-4, seed=3407)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    optimizer_cls_and_kwargs=(AdamWBF16, {
        "lr": training_args.learning_rate,
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
        "seed": training_args.seed,
    }),
)
trainer.train()
```

Preparation operates on trainable parameters, without architecture-specific
target names. Frozen base weights, including quantized QLoRA weights, and buffers
are unchanged. It preserves parameter identity and tied weights. Call it after
any model preparation that upcasts adapters or norms, and before wrapping or
sharding parameters. Dense, contiguous BF16 parameters are supported; tensor
subclasses, sparse gradients and sharded parameters are refused.

Weights, gradients and two moments use 12 bytes per trainable scalar, plus scalar
state, versus 16 bytes for FP32 parameters, gradients and moments. Activations,
frozen weights and framework buffers are additional. Keeping the moments FP32
avoids losing small variance updates at common `beta2=0.999`.
Moment buffers are allocated when a parameter first receives a gradient; frozen
parameters and unused experts do not receive unnecessary optimizer buffers.

Triton 3.3 and newer batch up to 32 small or eight larger parameter tensors per
update launch.
Older Triton versions use the same fused update one tensor at a time. Compiling
the optimizer step with `torch.compile(..., fullgraph=True)` is supported on GPU;
CPU provides a reference implementation.

Stochastic rounding preserves the expectation of the FP32 update when both
neighboring BF16 values are finite. It cannot recover rounding already incurred
in BF16 matrix operations or gradient accumulation. This changes training
numerics; validate the loss and evaluation quality for the intended workload.
Values above the largest finite BF16 value may overflow to infinity.

Each parameter has a checkpointed random seed and device step counter, independent
of model dropout and the global RNG. Replicated parameters require the same seed,
parameter ordering and updates on every rank. CPU is a correctness reference and
uses a different random sequence from the GPU. CUDA graph replay advances the
device counter. Keep learning rates as device tensors during capture; changing
Python hyperparameters requires recapturing the graph. Use separate parameter
groups per device for capture.

The rounding rule follows [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)
and [TorchAO's stochastic optimizer rounding](https://github.com/pytorch/ao/blob/main/torchao/optim/README.md).
