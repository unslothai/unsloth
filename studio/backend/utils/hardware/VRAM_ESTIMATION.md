# VRAM Estimation for Training

```
Total VRAM = Weights + LoRA Adapters + Optimizer + Gradients + Activations + CUDA Overhead
```

| Symbol | Meaning |
|--------|---------|
| `H` | `hidden_size` |
| `L` | `num_hidden_layers` |
| `V` | `vocab_size` |
| `K` | `(H / num_attention_heads) * num_key_value_heads` |
| `M` | `intermediate_size` (or `moe_intermediate_size`) |
| `E` | `num_experts` (1 for dense) |
| `r` | LoRA rank |
| `B` | `per_device_train_batch_size` |
| `S` | `max_seq_length` |

---

## 1. Model Weights

```
QKVO = (H + K + K + H) * H
MLP  = H * M * 3 * E  +  (E * H if E > 1 else 0)

Quantizable     = (QKVO + MLP) * L
Non-quantizable = 2*H*L + V*H + (V*H if not tie_embeddings else 0)
```

| Mode | Bytes |
|------|-------|
| QLoRA 4-bit | `Quantizable * 2 / 3.2 + Non-quantizable * 2` |
| LoRA / Full fp16 | `(Quantizable + Non-quantizable) * 2` |

The 3.2 factor (`16/5`) accounts for BNB NF4 blockwise scales. Repos whose
quantization config enables `bnb_4bit_use_double_quant` use a tighter, still
conservative 3.6 factor for the quantized portion of the weights.
When a 4-bit config has `llm_int8_skip_modules` entries that point to language
model layers or submodules, those quantizable weights are charged at fp16
instead of NF4. Generic embedding and multimodal skip names are already covered
by non-quantizable terms or excluded from text training weights.

## 2. LoRA Adapters

| Module | A | B |
|--------|---|---|
| q_proj | `H×r` | `r×H` |
| k_proj | `H×r` | `r×K` |
| v_proj | `H×r` | `r×K` |
| o_proj | `H×r` | `r×H` |
| gate_proj | `H×r` | `r×M` |
| up_proj | `H×r` | `r×M` |
| down_proj | `M×r` | `r×H` |

MLP modules multiply by `E` for MoE.

```
LoRA_bytes = sum(A + B per selected module) * L * 2
```

`all-linear` is treated as all known text linear modules in the table above.
The estimator deliberately does not infer multimodal or vision-tower LoRA
modules from config shapes; those modules vary too much across VLM families for
a generic config formula.

Some decoder configs expose layer-shape fields such as `layer_types`,
`head_dim`, `global_head_dim`, `num_global_key_value_heads`, `attention_k_eq_v`,
`num_kv_shared_layers`, `use_double_wide_mlp`, `vocab_size_per_layer_input`, and
`hidden_size_per_layer_input`. When those fields are present, the estimator
derives text weight and LoRA counts from the per-layer shapes instead of
assuming every layer has the same seven projection modules.

## 3. Optimizer States (calibrated)

| Optimizer | Bytes/param | Notes |
|-----------|------------|-------|
| `adamw_8bit` | 4 | BNB upcasts to fp32 during step |
| `adamw_torch` | 6 | Fused, no master copy |
| `paged_adamw_32bit` | 8 | Full fp32 states |
| `sgd` | 4 | |

Trainable params = all params (Full FT) or LoRA params only.

## 4. Gradients

```
Gradient_bytes = trainable_params * 2    (fp16, accumulated in-place)
```

## 5. Activations

Per-layer (from `unsloth_zoo/vllm_utils.py`):
```
Per_layer = (S*B*(H+K+K) + S*B*2 + S*B*(M+M)) * 2 * 1.25
```

When the resolved attention implementation is none of `flash_attention_2`,
`sdpa`, or `flex_attention` (PyTorch SDPA dispatches to flash or
memory-efficient kernels and FlexAttention is also a memory-efficient
kernel, all of which are O(n) in memory), activation memory also includes
a quadratic attention-score/workspace estimate:

```
Non_flash_attention = B * num_attention_heads * S^2 * 2 * 12.0 * effective_layers
Activations = max(Per_layer_with_gc, Non_flash_attention)
```

Studio resolves the attention implementation with Unsloth's
`resolve_attention_implementation` helper and uses that result directly. The
estimator does not duplicate model-family attention policy.

| GC Mode | Full FT | LoRA/QLoRA |
|---------|---------|------------|
| none | `L` layers | `L` layers |
| true (HF) | 2.0 | 1.0 |
| unsloth | 1.5 | 1.0 |

## 6. Floors

Activations use the computed formula directly:

```
activation_bytes = computed_activation_bytes
```

Full fine-tuning keeps the gradient floor at **15% of model weight memory** to
account for autograd overhead, NCCL buffers, mixed-precision scaling, and
PyTorch fragmentation:

```
gradient_bytes = max(computed_gradient_bytes, weights * 0.15)
```

For LoRA/QLoRA, the base model is frozen, so the weight-derived gradient floor
is capped by trainable-state and live-activation scale:

```
raw_gradient_bytes = trainable_params * 2
gradient_floor = min(weights * 0.15, max(computed_activation_bytes, optimizer_bytes))
gradient_bytes = max(raw_gradient_bytes, gradient_floor)
```

This prevents frozen quantized model size from dominating gradient/state
overhead when the measured runtime footprint is governed by LoRA optimizer
states and live activations.

## 7. CUDA Overhead

**1.4 GB** fixed — CUDA driver + PyTorch runtime, calibrated on RTX 5070 Ti.

## 8. Multi-GPU Overhead

When sharding across multiple GPUs, each additional GPU (beyond the first) contributes only **85%** of its free VRAM to the usable pool. The 15% discount accounts for NCCL all-reduce buffers, PCIe/NVLink transfer overhead, synchronization barriers, and memory fragmentation from non-uniform shard sizes. Calibrated empirically on 2-8 GPU setups with NVLink and PCIe topologies.

```
usable_gb = free[gpu_0] + sum(free[gpu_i] * 0.85 for i in 1..N)
```

---

## Parameter Flow

```
Frontend -> routes/{training,inference}.py
         -> prepare_gpu_selection(gpu_ids, model_name, ...)
            |
            +-- gpu_ids is explicit (e.g. [5,6,7])
            |     -> resolve_requested_gpu_ids: validate against parent-visible set
            |     -> return all requested GPUs (model sharded across all of them)
            |
            +-- gpu_ids is None or []
                  -> auto_select_gpu_ids: estimate VRAM, pick minimum GPUs needed
                  -> estimate_required_model_memory_gb -> estimate_training_vram
                  -> greedy selection: rank GPUs by free VRAM, add until model fits

         -> get_device_map(resolved_gpu_ids)
            -> "balanced" if >1 GPU, "sequential" otherwise

         -> worker subprocess: apply_gpu_ids(resolved_gpu_ids)
            -> sets CUDA_VISIBLE_DEVICES before torch/CUDA init
```

Threaded params: `batch_size`, `max_seq_length`, `lora_r`, `target_modules`, `gradient_checkpointing`, `optim`.

Source: `studio/backend/utils/hardware/vram_estimation.py`
