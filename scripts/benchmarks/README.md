# Qwen3-4B GRPO rollout engine benchmarks

This directory holds the reproducible scripts behind the experiment
documented in the accompanying PR: can Hugging Face transformers'
continuous-batching API (`model.generate_batch`, backed by
`PagedAttentionCache`) serve as a drop-in replacement for vLLM during GRPO
rollouts on the Qwen3-4B notebook?

The short answer on a single NVIDIA B200 with Unsloth Qwen3-4B-Base, LoRA
rank 32, bf16: transformers continuous batching is functionally correct and
integrates with TRL's `use_transformers_paged=True` path, but end-to-end
throughput lands at around 7-9 percent of vLLM colocated. Full numbers and
per-step timings are in the PR description.

## Files

| File | Purpose |
|---|---|
| `unsloth_grpo_common.py` | Shared dataset loading, reward functions, and GRPO hyperparameters |
| `qwen3_grpo_vllm.py` | vLLM baseline training entry (`fast_inference=True`, `use_vllm=True`, `vllm_mode="colocate"`) |
| `qwen3_grpo_tpaged.py` | Continuous-batching candidate (`fast_inference=False`, `use_transformers_paged=True`, vanilla HF + PEFT) |
| `cb_vs_vllm_generation.py` | Standalone generation microbenchmark across both engines |

## Reproduce

```bash
pip install unsloth "transformers>=4.57" "trl>=0.25" peft vllm

# Generation microbenchmark (32 prompts, 512 new tokens each)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/cb_vs_vllm_generation.py \
    --backend vllm --stats_path logs/vllm_gen.json \
    --n_prompts 32 --n_rounds 2 --max_new_tokens 512 \
    --gpu_memory_utilization 0.6

CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/cb_vs_vllm_generation.py \
    --backend tpaged --stats_path logs/cb_gen.json \
    --n_prompts 32 --n_rounds 2 --max_new_tokens 512

# Full GRPO training (20 steps)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/qwen3_grpo_vllm.py \
    --max_steps 20 --num_generations 2 --per_device_train_batch_size 2 \
    --output_dir outputs/grpo_vllm --stats_path logs/vllm_stats.json \
    --gpu_memory_utilization 0.6

CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/qwen3_grpo_tpaged.py \
    --max_steps 20 --num_generations 2 --per_device_train_batch_size 2 \
    --output_dir outputs/grpo_tpaged --stats_path logs/tpaged_stats.json \
    --max_batch_tokens 16384 --num_blocks 16384
```

## Known integration notes for transformers continuous batching + TRL + Unsloth

These are the sharp edges you hit going down the continuous-batching path and
how `qwen3_grpo_tpaged.py` handles them:

1. **`top_k=-1` is not a valid value for transformers.** vLLM treats `-1` as
   "disabled", but `TopKLogitsWarper` raises
   `ValueError: top_k has to be a strictly positive integer`. The script
   rewrites `top_k=None` on the shared GRPOConfig before handing it to
   `GRPOConfig(use_transformers_paged=True, ...)`.

2. **`PagedAttentionCache` default upper bounds are extremely conservative.**
   `_upper_bound_max_batch_tokens=256` and `_upper_bound_num_blocks=4096`
   choke decode throughput. The script passes
   `generation_kwargs={"max_batch_tokens": 16384, "num_blocks": 16384}` which
   TRL forwards to `GenerationConfig`, and the CB manager reads them when
   sizing the paged cache.

3. **Unsloth's `Qwen3Attention_fast_forward` bypasses the functional
   attention interface.** Calling `model.generate_batch` on an
   Unsloth-patched Qwen3 model fails inside
   `unsloth.utils.attention_dispatch.run_attention` because Unsloth routes
   through its own dispatcher rather than reading
   `config._attn_implementation`. The benchmark script works around this by
   loading a vanilla HF Qwen3 with PEFT LoRA for the tpaged path. This costs
   the Unsloth training kernels but keeps the comparison clean. A proper
   upstream fix is to detect `config._attn_implementation` ending in
   `_paged` and delegate to the stock transformers forward.

4. **TRL imports `GuidedDecodingParams` from `vllm.sampling_params`.**
   Newer vLLM releases (>= 0.13) have moved or removed that symbol, so
   `trl.trainer.grpo_trainer` fails to import on a fresh vLLM install even
   if you are not using vLLM. `qwen3_grpo_tpaged.py` installs a minimal
   shim before importing TRL.

5. **`UnslothGRPOTrainer` calls `model.for_training()` /
   `for_inference()`.** Importing `unsloth` replaces
   `trl.GRPOTrainer` with `UnslothGRPOTrainer`, which assumes the model has
   these hooks. A vanilla HF model does not, so
   `qwen3_grpo_tpaged.py` does not `import unsloth` at all.

## Why continuous batching is slower than vLLM on this workload

- `flash_attn` is hard to install on this box (CUDA 13.1 detected, torch
  compiled against CUDA 12.8), so `paged|flash_attention_2` falls back to
  `sdpa_paged`. vLLM uses FlashInfer and TRTLLM kernels.
- CB re-allocates a fresh `PagedAttentionCache` on every `generate_batch`
  call. For GRPO that is once per step.
- `ContinuousBatchingManager` does not yet implement CUDA graphs
  (`use_cuda_graph=True` raises `NotImplementedError`). vLLM captures 100+
  mixed prefill-decode and decode graphs during warmup.

These are all upstream transformers issues, not Unsloth issues. The scripts
in this directory are intentionally simple so they are easy to port into a
future upstream fix.
