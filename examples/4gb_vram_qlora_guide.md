# Optimizing Unsloth QLoRA Fine-Tuning for 4GB VRAM Consumer GPUs

> **Author:** Kunwar Satyam Singh (`dante@5ingularity`)  
> **Target Hardware:** NVIDIA GTX 1650 (4GB VRAM), RTX 3050 Laptop (4GB), or similar consumer edge GPUs.  
> **OS:** Linux (Arch / Ubuntu / Fedora) & WSL2.

---

## 1. The Hardware Reality & VRAM Math

A common failure mode for beginners is attempting to fine-tune 7B parameter models on 4GB VRAM cards. Here is the mathematical reality of GPU memory allocation during fine-tuning:

| Model Size | 4-bit Weight Size (NF4) | Minimum KV/Activation Memory | Total Required VRAM (Batch 1, Seq 1024) | Survives 4GB VRAM? |
| :--- | :--- | :--- | :--- | :--- |
| **7B** (e.g., Llama-3.1-8B) | ~4.2 GB | ~1.5 GB | ~5.7 GB | ❌ **OOM (Out of Memory)** |
| **3B** (e.g., Qwen2.5-3B) | ~1.8 GB | ~1.2 GB | ~3.0 GB | ⚠️ **Tight (Requires strict config)** |
| **1.5B** (e.g., Qwen2.5-1.5B) | ~0.9 GB | ~0.8 GB | ~1.7 GB | ✅ **Comfortable** |
| **1B** (e.g., Llama-3.2-1B) | ~0.6 GB | ~0.6 GB | ~1.2 GB | ✅ **Comfortable** |

### Why 7B fails on 4GB cards
In 4-bit NormalFloat (NF4) quantization, each parameter consumes 0.5 bytes. For an 8 billion parameter model, model weights alone consume $8 \times 0.5 = 4.0\text{ GB}$. This leaves **0 bytes** for optimizer states, gradient checkpoints, input activations, or CUDA kernels.

**Strategic Rule:** On 4GB VRAM hardware, target **1B to 3B models** (such as `Llama-3.2-1B-Instruct`, `Qwen2.5-1.5B`, or `SmolLM2-1.7B`). These models achieve remarkable performance on specialized domain tasks when fine-tuned cleanly.

---

## 2. Essential Linux & PyTorch Environment Optimizations

Before launching any Python script, prevent CUDA memory fragmentation—a frequent culprit of mysterious OOM crashes on Linux window managers (KDE/GNOME/Hyprland).

### Set Memory Allocator Environment Variables
Add the following to your terminal session or `.bashrc`:

```bash
# Prevents PyTorch from allocating fragmented blocks that cause premature OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Force clean garbage collection behavior
export CUDA_LAUNCH_BLOCKING=0
```

---

## 3. The Golden Hyperparameter Configuration for 4GB VRAM

To fine-tune without spiking past 3.8GB VRAM, every hyperparameter must be carefully locked:

1. **`per_device_train_batch_size = 1`**: Keeps forward activation memory at absolute minimum.
2. **`gradient_accumulation_steps = 8`** (or `4`): Simulates an effective batch size of 8 without increasing peak VRAM footprint.
3. **`gradient_checkpointing = "unsloth"`**: Trades ~20% compute time to offload intermediate activations, cutting memory usage by up to 60%.
4. **`max_seq_length = 1024`**: Attention memory scales quadratically ($O(N^2)$). Capping sequence length at 1024 tokens prevents sudden OOM spikes during long training samples.
5. **`optim = "paged_adamw_8bit"`**: Uses 8-bit optimizer states and allows CPU memory paging if GPU VRAM experiences momentary spikes.

---

## 4. Complete Reference Script (`train_4gb_vram.py`)

See `train_4gb_vram.py` in this directory for a complete executable script demonstrating how to fine-tune `Llama-3.2-1B-Instruct` within a 4GB VRAM budget.
