# Qwen3-4B GRPO rollout engine benchmarks

This directory holds the reproducible scripts behind the experiment
documented in the accompanying PR: can Hugging Face transformers'
continuous-batching API (`model.generate_batch`, backed by
`PagedAttentionCache`) serve as a drop-in replacement for vLLM during GRPO
rollouts on the Qwen3-4B notebook?

The short answer on a single NVIDIA B200 with Unsloth Qwen3-4B-Base, LoRA
rank 32, bf16: transformers continuous batching is functionally correct and
integrates with TRL's `use_transformers_paged=True` path, but end-to-end
throughput lands at around 7 to 10 percent of vLLM colocated even after
wiring in Flash Attention 4 on Blackwell. Full numbers and per-step timings
are in the PR description.

## Files

| File | Purpose |
|---|---|
| `unsloth_grpo_common.py` | Shared dataset loading, reward functions, and GRPO hyperparameters |
| `qwen3_grpo_vllm.py` | vLLM baseline training entry (`fast_inference=True`, `use_vllm=True`, `vllm_mode="colocate"`) |
| `qwen3_grpo_naive.py` | Naive TRL path (vanilla HF `model.generate`, no vLLM, no CB) matching https://huggingface.co/docs/trl/grpo_trainer |
| `qwen3_grpo_tpaged.py` | Continuous-batching candidate (`fast_inference=False`, `use_transformers_paged=True`, vanilla HF + PEFT). Supports `--persistent_cb` |
| `cb_vs_vllm_generation.py` | Standalone generation microbenchmark across both engines; supports `--attn_impl`, `--persistent_cb` |
| `flash_attn_fa4_shim.py` | Installs two monkey-patches that let CB dispatch to FA4 when `--attn_impl flash_attention_2` is selected |
| `persistent_cb.py` | Replaces `model.generate_batch` with a version that reuses a single `ContinuousBatchingManager` |

## Flash Attention 4 on Blackwell (sm_100)

The CB code path has three attention implementations: `eager_paged`,
`sdpa_paged`, `flash_attention_2`. The last one requires the legacy
`flash_attn` Python package, which does not install cleanly on B200 today:

- `flash_attn==2.8.3+cu12torch2.8cxx11abiTRUE-cp313` from the Dao-AILab
  releases hits `undefined symbol: _ZNK3c106SymInt6sym_neERKS0_` on torch
  2.9.1 (ABI drift between torch 2.8 and 2.9).
- `flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl` from the PyTorch
  wheel index installs but was built for sm_80 and sm_90a only. B200 is
  sm_100. The kernel call fails with "no kernel image is available for
  execution on the device".
- `flash-attn-4==4.0.0b9` (pure Python CuTeDSL, Dao-AILab) works on B200.
  It exposes `flash_attn.cute.flash_attn_varlen_func`.

The recipe this repo uses:

```bash
uv pip install --no-deps flash-attn-4==4.0.0b9
```

plus a tiny site-packages shim that re-exports FA4 symbols under the
FA2 `flash_attn` namespace so transformers' `is_flash_attn_2_available()` and
`_lazy_imports("flash_attention_2")` succeed. The shim lives out of tree in
`lib/python3.13/site-packages/flash_attn/__init__.py` +
`flash_attn/bert_padding.py` + a `flash_attn-2.8.3.dist-info/` directory
with enough metadata to satisfy `importlib.metadata.version("flash_attn")`.

On top of that, `flash_attn_fa4_shim.py` monkey-patches two rough edges in
the CB to FA integration that are unrelated to which FA version you use:

1. `ContinuousBatchProcessor.return_attention_mask` returns `False` for
   `flash_attention_2` / `flash_attention_3` so CB does not emit a 4-D paged
   attention mask that breaks `_flash_attention_forward`'s `_upad_input`
   branch.
2. `_flash_attention_forward` accepts `max_seqlen_q` / `max_seqlen_k`
   as aliases for `max_length_q` / `max_length_k`. Without this rename CB's
   model kwargs never bind and FA is called with `max_seqlen_q=None`.

## Install

Prereqs: `torch >= 2.5` for `torch.nn.attention.flex_attention`. The
Triton backend that flex_attention uses by default runs on Ampere,
Hopper, and Blackwell -- no separate install.

FA4 (CuTeDSL) targets Hopper and Blackwell only. `qwen3_flex_inference.py`
auto-enables FA4 on supported GPUs and falls back to the Triton
`flex_attention` backend elsewhere. `--fa4_prefill` forces on (warns +
falls back if the GPU does not support it); `--no-fa4_prefill` forces off.

CUDA 13 (recommended, used on B200 / RTX 50xx):

    pip install --index-url https://download.pytorch.org/whl/cu130 torch
    pip install "flash-attn-4[cu13]"

CUDA 12 (H100 boxes still on cu12):

    pip install torch  # default index is cu12
    pip install flash-attn-4

Pin `flash-attn-4==4.0.0b9` to match this benchmark. The `[cu13]`
extra pulls in `nvidia-cutlass-dsl` built for CUDA 13.

| GPU          | arch      | sm    | Auto FA4 | Triton flex_attention |
|--------------|-----------|-------|----------|------------------------|
| A100         | Ampere    | sm_80 | off (uses Triton) | Works |
| H100 / H200  | Hopper    | sm_90 | on | Works |
| RTX 50xx     | Blackwell | sm_120 | on | Works |
| B200 / GB200 | Blackwell | sm_100 | on | Works |

The transformers continuous-batching path's FA4 wiring (the
`flash_attn_fa4_shim.py` monkey-patches and the
`site-packages/flash_attn/__init__.py` namespace shim that makes FA4
visible under the FA2 import name) is covered below under "Known
integration notes".

## Reproduce

```bash
pip install unsloth "transformers>=4.57" "trl>=0.25" peft vllm

# Generation microbenchmark (32 prompts, 512 new tokens each)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/cb_vs_vllm_generation.py \
    --backend vllm --stats_path logs/vllm_gen.json \
    --n_prompts 32 --n_rounds 2 --max_new_tokens 512 \
    --gpu_memory_utilization 0.6

# CB variants
CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/cb_vs_vllm_generation.py \
    --backend tpaged --attn_impl sdpa \
    --stats_path logs/cb_gen_sdpa.json \
    --n_prompts 32 --n_rounds 2 --max_new_tokens 512

CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/cb_vs_vllm_generation.py \
    --backend tpaged --attn_impl flash_attention_2 \
    --stats_path logs/cb_gen_fa.json \
    --n_prompts 32 --n_rounds 2 --max_new_tokens 512

# Full GRPO training (20 steps)
CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/qwen3_grpo_vllm.py \
    --max_steps 20 --num_generations 2 --per_device_train_batch_size 2 \
    --output_dir outputs/grpo_vllm --stats_path logs/vllm_stats.json \
    --gpu_memory_utilization 0.6

CUDA_VISIBLE_DEVICES=7 python scripts/benchmarks/qwen3_grpo_naive.py \
    --max_steps 20 --num_generations 2 --per_device_train_batch_size 2 \
    --output_dir outputs/grpo_naive --stats_path logs/naive_stats.json

CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/qwen3_grpo_tpaged.py \
    --max_steps 20 --num_generations 2 --per_device_train_batch_size 2 \
    --attn_impl flash_attention_2 \
    --output_dir outputs/grpo_tpaged_fa --stats_path logs/tpaged_stats_fa.json \
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
   loading a vanilla HF Qwen3 with PEFT LoRA for the tpaged and naive paths.
   This costs the Unsloth training kernels but keeps the comparison clean.
   A proper upstream fix is to detect `config._attn_implementation` being
   `flash_attention_2` / `sdpa_paged` / `eager_paged` and delegate to the
   stock transformers forward.

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

## Why continuous batching is still slower than vLLM on this workload

- `ContinuousBatchingManager` does not yet implement CUDA graphs
  (`use_cuda_graph=True` raises `NotImplementedError`). vLLM captures 100+
  mixed prefill-decode and decode graphs during warmup.
- CB re-allocates a fresh `PagedAttentionCache` on every `generate_batch`
  call. For GRPO that is once per step. `--persistent_cb` (via
  `persistent_cb.py`) keeps the cache warm across steps.
- FA4 is a CuTeDSL package: first call per shape pays a one-time
  JIT-compile.
- vLLM uses its own colocated attention + FlashInfer / TRTLLM kernels
  tuned for decode, which currently outperform everything a generic CB
  path can do.

These are all upstream transformers issues, not Unsloth issues. The scripts
in this directory are intentionally simple so they are easy to port into a
future upstream fix.
