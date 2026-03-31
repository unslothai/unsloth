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

The 3.2 factor (`16/5`) accounts for BNB NF4 blockwise scales.

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

| GC Mode | Full FT | LoRA/QLoRA |
|---------|---------|------------|
| none | `L` layers | `L` layers |
| true (HF) | 2.0 | 1.0 |
| unsloth | 1.5 | 1.0 |

## 6. Floors

Gradients and activations have minimum floors at **15% of model weight memory** to account for autograd overhead, attention score matrices, NCCL buffers, mixed-precision scaling, and PyTorch fragmentation.

```
gradient_bytes  = max(computed, weights * 0.15)
activation_bytes = max(computed, weights * 0.15 * B/2)
```

## 7. CUDA Overhead

**1.4 GB** fixed — CUDA driver + PyTorch runtime, calibrated on RTX 5070 Ti.

## 8. Multi-GPU Overhead

When sharding across multiple GPUs, each additional GPU (beyond the first) contributes only **85%** of its free VRAM to the usable pool. The 15% discount accounts for NCCL all-reduce buffers, PCIe/NVLink transfer overhead, synchronization barriers, and memory fragmentation from non-uniform shard sizes. Calibrated empirically on 2-8 GPU setups with NVLink and PCIe topologies.

```
usable_gb = free[gpu_0] + sum(free[gpu_i] * 0.85 for i in 1..N)
```

---

## Reference Table (bsz=2, seq=2048, rank=16, GC=unsloth, adamw_8bit)

| Model | Weights | LoRA | Optim | Grad | Act | CUDA | Total |
|-------|---------|------|-------|------|-----|------|-------|
| 0.5B QLoRA | 0.5 | 0.0 | 0.0 | 0.1 | 0.1 | 1.4 | **2.1** |
| 1B QLoRA | 1.1 | 0.0 | 0.0 | 0.2 | 0.2 | 1.4 | **2.9** |
| 3B QLoRA | 2.4 | 0.0 | 0.1 | 0.5 | 0.5 | 1.4 | **4.9** |
| 8B QLoRA | 6.0 | 0.1 | 0.2 | 1.2 | 1.2 | 1.4 | **10.1** |
| 8B LoRA fp16 | 15.0 | 0.1 | 0.2 | 3.0 | 3.0 | 1.4 | **22.6** |
| 8B Full FT | 15.0 | — | 29.9 | 15.0 | 3.0 | 1.4 | **64.2** |
| 32B LoRA fp16 | 61.0 | 0.2 | 0.5 | 12.2 | 12.2 | 1.4 | **87.6** |
| 72B QLoRA | 45.5 | 0.4 | 0.8 | 9.1 | 9.1 | 1.4 | **66.3** |

## E2E Validation (Llama-3.2-1B, B200 emulating 24GB)

| Config | Estimated | Actual (nvsmi) | Error |
|--------|----------|----------------|-------|
| QLoRA bsz=2 seq=512 | 2.55 GB | 2.65 GB | -3.7% |
| QLoRA bsz=2 seq=2048 | 2.60 GB | 2.65 GB | -1.8% |
| QLoRA bsz=4 seq=2048 | 2.65 GB | 2.65 GB | +0.0% |
| LoRA fp16 bsz=2 | 3.84 GB | 3.88 GB | -1.0% |
| Full FT adamw_8bit | 10.89 GB | 10.80 GB | +0.8% |
| Full FT adamw_torch | 13.19 GB | 12.93 GB | +2.0% |

*Note: e2e numbers predate the 15% floors, which add safety margin on top.*

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
