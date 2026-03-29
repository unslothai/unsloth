# VRAM Estimation for Training

Estimates total GPU memory (nvidia-smi level) as:

```
Total VRAM = Model Weights + LoRA Adapters + Optimizer States + Gradients + Activations + CUDA Overhead
```

All formulas below use these symbols:

| Symbol | Meaning |
|--------|---------|
| `H` | `hidden_size` |
| `L` | `num_hidden_layers` |
| `V` | `vocab_size` |
| `K` | KV dimension = `(H / num_attention_heads) * num_key_value_heads` |
| `M` | `intermediate_size` (or `moe_intermediate_size` for MoE) |
| `E` | `num_experts` (1 for dense models) |
| `r` | LoRA rank |
| `B` | `per_device_train_batch_size` |
| `S` | `max_seq_length` |

---

## 1. Model Weights

Per-layer parameter groups:

```
QKVO  = (H + K + K + H) * H
MLP   = H * M * 3 * E  +  (E * H  if E > 1 else 0)      # router weights for MoE
Norms = 2 * H
```

Global (outside layers):

```
Embed   = V * H
LM_Head = V * H    (0 if tie_word_embeddings)
```

Quantizable vs non-quantizable split:

```
Quantizable     = (QKVO + MLP) * L
Non-quantizable = Norms * L + Embed + LM_Head
```

| Mode | Bytes |
|------|-------|
| **QLoRA 4-bit** | `Quantizable * 2 / (16/5) + Non-quantizable * 2` |
| **LoRA fp16** | `(Quantizable + Non-quantizable) * 2` |
| **Full FT fp16** | `(Quantizable + Non-quantizable) * 2` |

The `16/5 = 3.2` factor for 4-bit comes from bitsandbytes NF4 quantization overhead (theoretically 4-bit = factor 4, but blockwise scales + metadata bring effective compression to ~3.2x). Source: `unsloth_zoo/vllm_utils.py:1563`.

---

## 2. LoRA Adapters

Each target module has an A matrix (input -> rank) and B matrix (rank -> output):

| Module | A elements | B elements |
|--------|-----------|-----------|
| `q_proj` | `H * r` | `r * H` |
| `k_proj` | `H * r` | `r * K` |
| `v_proj` | `H * r` | `r * K` |
| `o_proj` | `H * r` | `r * H` |
| `gate_proj` | `H * r` | `r * M` |
| `up_proj` | `H * r` | `r * M` |
| `down_proj` | `M * r` | `r * H` |

For MoE models, MLP modules (`gate_proj`, `up_proj`, `down_proj`) multiply by `E` (each expert has its own LoRA). Attention modules are shared.

```
LoRA_params = sum(A + B for each selected module) * L
LoRA_bytes  = LoRA_params * 2                              # fp16
```

---

## 3. Optimizer States (empirically calibrated)

Trainable parameters:
- **Full FT**: all model parameters
- **LoRA / QLoRA**: LoRA adapter parameters only

| Optimizer | Bytes/param | Theoretical | Why higher |
|-----------|------------|-------------|------------|
| `adamw_8bit` | **4** | 2 (8-bit m+v) | BNB upcasts params to fp32 during step (temporary master copy) |
| `adamw_torch` | **6** | 8 (fp32 m+v) | Fused AdamW operates on bf16 directly, no separate master copy |
| `sgd` | **4** | 4 (fp32 m) | Matches theoretical |

```
Optimizer_bytes = trainable_params * bytes_per_param
```

---

## 4. Gradients

Stored in fp16 (mixed precision training). Gradient accumulation is in-place and does NOT increase memory.

```
Gradient_bytes = trainable_params * 2
```

---

## 5. Activations

Per-layer activation memory (adapted from `unsloth_zoo/vllm_utils.py:1542-1551`):

```
Act_QKV      = S * B * (H + K + K)
Act_Residual = S * B * 2
Act_MLP      = S * B * (M + M)

Per_layer    = (Act_QKV + Act_Residual + Act_MLP) * 2 * 1.25
                                                   ^fp16 ^25% safety
```

Gradient checkpointing scaling differs between Full FT and LoRA. For LoRA/QLoRA, the frozen base model layers don't save activations (no `requires_grad` on main weights) — only LoRA-path activations contribute.

| Mode | Full FT layers | LoRA/QLoRA layers | Rationale |
|------|---------------|-------------------|-----------|
| `none` | `L` | `L` | All layers stored |
| `true` (HF) | `2.0` | `0.5` | Frozen layers skip activation storage |
| `unsloth` | `1.5` | `0.3` | Aggressive recompute + frozen layer skip |

```
Activation_bytes = Per_layer * effective_layers
```

---

## 6. CUDA Overhead

Fixed nvidia-smi overhead for CUDA driver context and PyTorch runtime. Empirically calibrated at 1.4 GB on RTX 5070 Ti (see `frontend/src/lib/vram.ts`).

```
Overhead = 1.4 GB
```

---

## Validation Results

Validated against actual training of **Llama-3.2-1B-Instruct** using `torch.cuda.set_per_process_memory_fraction` to emulate 24 GB VRAM. Comparison at nvidia-smi level (allocated + 1.4 GB CUDA context):

| Config | Estimated | Actual (nvsmi) | Error |
|--------|----------|----------------|-------|
| QLoRA bsz=2 seq=512 | 2.55 GB | 2.65 GB | **-3.7%** |
| QLoRA bsz=2 seq=2048 | 2.60 GB | 2.65 GB | **-1.8%** |
| QLoRA bsz=4 seq=2048 | 2.65 GB | 2.65 GB | **+0.0%** |
| QLoRA bsz=2 r=64 | 2.85 GB | 2.96 GB | **-3.8%** |
| LoRA fp16 bsz=2 | 3.84 GB | 3.88 GB | **-1.0%** |
| Full FT adamw_8bit seq=512 | 10.68 GB | 10.80 GB | **-1.1%** |
| Full FT adamw_8bit seq=2048 | 10.89 GB | 10.80 GB | **+0.8%** |
| Full FT adamw_torch | 13.19 GB | 12.93 GB | **+2.0%** |

All estimates within **-4% to +2%** of actual usage.

---

## Worked Example: Llama-3.2-1B-Instruct

Architecture: `H=2048, L=16, V=128256, heads=32, kv_heads=8, M=8192, tie_embeddings=True`

### QLoRA (bsz=2, seq=2048, rank=16, GC=unsloth, adamw_8bit)

| Component | GB |
|-----------|-----|
| Weights (4-bit) | 1.06 |
| LoRA adapters | 0.02 |
| Optimizer (4 bytes/param) | 0.04 |
| Gradients | 0.02 |
| Activations (0.3 layers) | 0.06 |
| CUDA overhead | 1.40 |
| **Total** | **2.60** |
| **Actual (nvidia-smi)** | **2.65** |

### Full FT (bsz=2, seq=2048, GC=unsloth, adamw_8bit)

| Component | GB |
|-----------|-----|
| Weights (fp16) | 2.30 |
| Optimizer (4 bytes/param) | 4.60 |
| Gradients | 2.30 |
| Activations (1.5 layers) | 0.28 |
| CUDA overhead | 1.40 |
| **Total** | **10.89** |
| **Actual (nvidia-smi)** | **10.80** |

---

## Parameter Flow

All training hyperparameters flow from the frontend through to VRAM estimation:

```
Frontend (config)
  -> training.py start_training(kwargs)
    -> prepare_gpu_selection(batch_size, lora_r, optim, ...)
      -> auto_select_gpu_ids(...)
        -> estimate_required_model_memory_gb(...)
          -> estimate_training_vram(arch, config)  # detailed
          or fallback multipliers                  # if no arch config
```

Parameters threaded: `batch_size`, `max_seq_length`, `lora_r`, `target_modules`, `gradient_checkpointing`, `optim`.

---

## Implementation

- **Detailed path**: When `AutoConfig` is available, uses full architecture decomposition above.
- **Fallback path**: When config unavailable (e.g. private/gated model), uses simplified multipliers on `model_size_gb` calibrated from the detailed formulas.

Source: `studio/backend/utils/hardware/vram_estimation.py`
