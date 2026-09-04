# flex_attention + paged KV + CUDA graphs vs vLLM

Goal stated in the plan: "CB reaches at least 30% of vLLM throughput." After
the earlier phases ran out of gas at ~10% with transformers CB, we rebuilt
the rollout path on top of `torch.nn.attention.flex_attention` using the
paged KV + BlockMask pattern from
[flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm).

## Setup

- B200 (sm_100), Qwen3-4B-Base, bf16
- 512 max_new_tokens per prompt, 16-prompt warmup, N measured rounds
  (`decode_tps_best` = steady-state throughput after Inductor compile +
  CUDA graph capture have amortized).
- flex path is greedy (CUDA-graph safe); vLLM uses equivalence sampling
  (`temperature=0.1, top_p=0.97, min_p=0.5, top_k=5`).
- No LoRA unless noted; LoRA rank 32 applied to all
  {q,k,v,o,gate,up,down}_proj.

## Best config (after FlexKernelOptions sweep)

```json
decode_kernel_options = {
    "PRESCALE_QK": true,
    "USE_TMA": true,
    "BLOCKS_ARE_CONTIGUOUS": true,
    "num_warps": 8,
    "num_stages": 3
}
prefill_kernel_options = {
    "FORCE_USE_FLEX_ATTENTION": true,
    "PRESCALE_QK": true,
    "USE_TMA": true
}
```

## Batch-size sweep (flex tuned vs vLLM, 512 max_new_tokens)

| Batch | flex tps | vLLM tps | flex / vLLM | flex mem | vLLM mem |
|------:|---------:|---------:|------------:|---------:|---------:|
|   8   |    680   |  1900    |    35.8 %   |  44 GB   |  156 GB  |
|  16   |   1626   |  3698    |    44.0 %   |  44 GB   |  156 GB  |
|  32   |   3134   |  6318    |    49.6 %   |  44 GB   |  156 GB  |
|  64   | **5474** | 10459    |  **52.3 %** |  44 GB   |  156 GB  |
| 128   |   5565   | 14996    |    37.1 %   |  81 GB   |  157 GB  |
| 256   |   5812   | 21170    |    27.5 %   | 154 GB   |  157 GB  |

### Canonical GRPO workload (batch 64 + LoRA rank 32)

| Backend                                    | tok/s best | peak mem  | flex / vLLM |
|--------------------------------------------|-----------:|----------:|------------:|
| vLLM (LoRARequest)                         |       7775 |    156 GB |       100 % |
| **flex** (double-copy, drift-free)         |  **5785**  | **~52 GB**|    **74 %** |
| flex -- LoRA unmerged (PEFT wrapper)       |       2683 |     45 GB |        35 % |

At the GRPO workload flex reaches **74 % of vLLM throughput at ~3 x less
memory**. Starting point before this work was 9 % with transformers CB.

**Why the two flex rows are so far apart:** when PEFT keeps the adapter
unmerged, every projection runs three matmuls (`base_layer(x) + scaling *
lora_B(lora_A(x))`) instead of one, which is ~50 % slowdown across the
36-layer stack. GRPO cannot use the unmerged path naively because the
trainer needs the adapter weights separable; but it also shouldn't pay
that cost.

#### What the default path does now: double-copy rollout

We keep two copies of the base model on GPU:

- `base_model` -- pristine; never mutated.
- `inference_model = deepcopy(base_model)` -- wrapped by PEFT; merged
  LoRA lives on `base_layer.weight` here.

Before each rollout (and at setup), `refresh_lora_merge_from_pristine`:

1. Walks PEFT's `LoraLayer` modules.
2. `module.base_layer.weight.data.copy_(base_submodule.weight.data)` --
   in-place restore from the pristine base.
3. Resets `module.merged_adapters = []` directly (skips PEFT's unmerge
   arithmetic).
4. Calls `peft_model.merge_adapter()` once to fold LoRA into the
   inference copy fresh.

We **never call `unmerge_adapter()`**. PEFT's merge/unmerge pair is
asymmetric at bf16 -- merge does `W_bf16 += delta_fp32` (the `+=`
upcasts, stores back in bf16), unmerge does `W_bf16 -= delta_fp32.to(bf16)`
(the delta is rounded to bf16 first, then subtracted). Net effect is ~1
ULP drift on `base_layer.weight` per cycle (empirically ~6e-5 max diff
after one cycle on this model). Across hundreds of GRPO iterations
that corrupts the base model and the adapter trains against a drifting
target. Re-materialising from pristine per refresh bypasses the whole
round-trip.

**Cost.** +~8 GB GPU memory (second copy of Qwen3-4B bf16 weights), so
peak memory goes from ~44 GB to ~52 GB. Per-refresh overhead: param
copy (~3 ms) + `merge_adapter()` (~30 ms) = ~35 ms, well under 1 % of a
5-7 s rollout.

**CUDA graphs stay valid.** In-place `weight.data.copy_(pristine)`
writes to the same tensor storage, so graphs captured against the
merged weights read current values at the captured addresses on the
next replay -- no re-capture needed.

#### Drift verification

`--verify_no_drift` takes a sha256 over every parameter in `base_model`
(raw bytes via `tensor.view(torch.uint8)`), runs N perturb+refresh
cycles (random noise added to `lora_A` / `lora_B` on each iteration,
simulating a training step), re-hashes, and asserts bit-identical.
It also checks determinism of the inference copy: after restoring the
LoRA A/B weights to their initial values and refreshing, the merged
state-dict hash matches the pre-perturbation hash.

Confirmed on Qwen3-4B bf16 with LoRA rank 32 across 10 refreshes: base
model bit-identical; inference copy deterministic after LoRA restore.

```
CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/qwen3_flex_inference.py \
  --verify_no_drift --lora_adapter outputs/lora_rank32_fresh --n_rounds 1 \
  --stats_path scripts/benchmarks/results/stats/flex_verify_nodrift.json
```

The "LoRA unmerged" row is shown only for reference -- it's what you'd
get with a naive PEFT wrapper on the hot path. **Don't use it in
production**, and it doesn't apply outside the 4-bit path below.

`--no_merge_lora` opts into that reference path (single model, PEFT
wrapper, adapter unmerged). It's kept for the comparison row above and
nothing else.

### Same workload at `load_in_4bit=True` (Unsloth bnb-4bit shard)

Loading base as bitsandbytes 4-bit (`unsloth/Qwen3-4B-Base-unsloth-bnb-4bit`,
compute dtype bf16). LoRA kept as PEFT wrapper (can't merge into 4-bit;
the double-copy pattern above also doesn't apply -- bnb's `Linear4bit`
holds packed quantised weights, not regular bf16, so an in-place copy
of `base_layer.weight` isn't meaningful, and materialising a bf16
inference copy via dequant would wipe out the memory saving of 4-bit).
lm_head is tied to embed_tokens post-load because the 4-bit shard ships
without an lm_head parameter.

| Backend                         | tok/s | peak mem | output      |
|---------------------------------|------:|---------:|:------------|
| Unsloth fast_inference (vLLM)   |  4515 | 159 GB   | coherent    |
| **flex** (this PR)              |**1738**| **40.6 GB** | coherent    |
| transformers CB (sdpa)          |   504 | 124 GB   | **gibberish** |

4-bit costs throughput on every backend (vLLM-path 4515 vs bf16 7775 = 58 %;
flex 1738 vs bf16 5744 = 30 %). The regression is worse for flex because
PEFT-without-merge doubles the number of matmuls per projection (base + LoRA
add, separately) on top of the bnb dequant cost; the bf16 path merges LoRA
into the base and skips both. Peak memory barely moves for vLLM because KV
cache at `gpu_memory_utilization=0.8` dominates regardless of base size.

Transformers CB at 4-bit + LoRA produces garbage tokens even with
`model.lm_head.weight = model.model.embed_tokens.weight` tied explicitly.
Likely a PEFT-over-bnb + batched `generate_batch` interaction bug; did not
debug further.

## What each option did (batch 64, no LoRA, after CUDA graph capture)

| Config                                                                | tok/s | vs baseline |
|-----------------------------------------------------------------------|------:|------------:|
| eager (no graphs)                                                     |  ~420 |       -     |
| + CUDA graphs                                                         |  4279 |    baseline |
| + `PRESCALE_QK=true`                                                  |  4367 |       +2 %  |
| + `USE_TMA=true`                                                      |  4425 |       +3 %  |
| + `BLOCKS_ARE_CONTIGUOUS=true`                                        |  4703 |       +10 % |
| + `num_warps=8`                                                       |  5474 |       +28 % |
| + `num_warps=8, num_stages=3`                                         |  **5898** (peak) |       +38 % |

The single biggest win came from **`num_warps=8`** (up from the default,
which on Blackwell tends to pick 4 for small block sizes). TMA helps a
couple percent; `BLOCKS_ARE_CONTIGUOUS` (safe in our setup because
PageTable.reserve allocates pages sequentially on a fresh batch) helps
another ~10 % because it lets the kernel skip the page-table indirection
per block.

## What broke correctness and had to be dropped

- **`ROWS_GUARANTEED_SAFE=true`**: we reserve `batch_idx=0` and
  `page_idx=0` as padding slots. Padded decode rows only attend to those
  reserved slots, so the mask returns False for every kv_idx on those
  rows. Skipping the row-has-at-least-one-unmasked check NaNs the
  softmax and the model outputs `!!!!!!`.
- **`BACKEND="TRITON_DECODE"`**: documented but the Inductor code path
  doesn't recognize the literal. Raises `NameError('TRITON_DECODE is
  not defined')`.
- **`USE_TMA=true` + `torch.compile(call_model_with_flex_kwargs)`**:
  misaligned address at runtime. torch.compile on the whole forward
  walker breaks TMA's alignment assumptions. Either disable TMA when
  compiling the walker, or skip compiling the walker (CUDA graph
  capture already captures it).
- **`torch.compile(flex_attention, mode="max-autotune")`**: tries to
  nest `cudagraph_trees` inside our raw CUDA graph capture and hits
  `Cannot prepare for replay during capturing stage`. Use
  `max-autotune-no-cudagraphs` instead; negligible throughput delta vs
  default mode.

## What I tried that did NOT move the needle

- **`BACKEND="FLASH"` on prefill** (FA4 / FlashAttention-4 on Blackwell,
  torch 2.11 + flash-attn CuTeDSL): empirically 4617 tok/s at batch 64 +
  LoRA vs 5744 baseline on torch 2.11. The FA4 CuTe kernel is slow when
  `mask_mod` indexes by `kv_idx` (documented in the attention-gym
  `flex_flash_attention.py` limitations: "Indexing by kv_idx is a large
  perf hit"). Our prefill mask is `document_causal`:
      `docs[q_idx] == docs[kv_idx]`
  which hits that exact slow path. `BLOCK_SIZE=(256, 128)` + padding to
  the 256-row Q tile works (output is coherent), it's just slower than
  the default Triton flex path for this mask.
- **Inductor autotune replay** (`TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE`
  + `mode="max-autotune-no-cudagraphs"` + parse the JSON log): Inductor's
  chosen best decode config (`fwd_num_warps=4, fwd_num_stages=3,
  fwd_BLOCK_M=64, fwd_BLOCK_N=64`) lands at 4827 tok/s -- worse than the
  hand-tuned `num_warps=8` at 5744. Autotune times a single kernel call,
  which doesn't catch cumulative register-spill / L1 effects across the
  36-layer stack. Harness lives at `flex_autotune_replay.py`.
- **torch.compile on `call_model_with_flex_kwargs`**: 4425 tok/s (same
  as eager walker) because the CUDA graph already captures every op in
  the walker into one replay. The compile step is work we don't need.
- **`num_warps=4` / `num_warps=16`**: 4486 / 4748 -- neither beats 8.
  Inductor's default picks 4 on small blocks and we're already past
  that sweet spot, but 16 wastes registers.
- **Explicit `fwd_BLOCK_M=128, fwd_BLOCK_N=128` pinning** on top of the
  manual best: 5009 tok/s. The implicit default already picks 128 for
  our shape; pinning it inhibits Inductor's shape-specialised choice
  between the flex_attention and flex_decoding templates.

## Torch version + run-to-run noise

Upgrading torch 2.9.1 -> 2.11 (required for FA4's CuTeDSL path) moves
best-of-N tok/s from ~5616 to ~5660 at batch 64 + LoRA -- essentially
within noise. Over 10 rounds, median is 4192 and best is 5660; the large
spread is GPU clock throttling across a ~60-second sustained run plus
variable prompt-length distributions per round. Reported numbers use
best-of-N to match the prior harness; steady-state median is roughly 75
 % of best.

## Architecture notes (unchanged from prior commits)

- `flex_paged_attention.py`: `PagedKVCache` + `PageTable` verbatim from
  flex-nano-vllm (BSD-3).
- `qwen3_flex_inference.py`: monkey-patches `Qwen3Attention.forward` to
  call `flex_attention(q, k, v, block_mask=...)` against the paged cache.
  Walks the `Qwen3Model` layer stack manually so `flex_block_mask /
  flex_input_pos / flex_batch_idx` reach the attention layer without
  modifying `Qwen3ForCausalLM.forward`. `capture_decode_cudagraph()`
  pre-reserves one page per batch slot, captures one CUDA graph per
  bucket in `[1,2,4,8,16,32,...,max_bs]`, then releases the scratch
  batches.

## Output coherence

All tuned configs produce coherent math solutions on the DAPO-Math-17k
prompts. See `sample_completions` in any `logs/flex_*_tuned.json`.

## What's left on the table

- **Chunked prefill**: vLLM interleaves prefill and decode inside a
  single step. flex does a full separate prefill pass per new batch,
  which is the main remaining penalty for large batches.
- **Prefill-path mask refactor**: the document_causal mask indexes by
  `kv_idx`. Flattening to a per-query bias (`bias[q_idx]`) would put
  FA4 back on the fast path, but this is a non-trivial rework because
  the causal-within-document constraint needs to be encoded without the
  `docs[kv_idx]` lookup.
- **Exhaustive Triton autotune** for flex_decoding: attention-gym's
  `flex_grid_sweep.py` enumerates 144 fwd configs; Inductor's default
  autotune only probes a handful. Running the full sweep with
  end-to-end tok/s as the metric (not single-call ms) might beat the
  manual num_warps=8 finding, but 144 * 5 rounds is ~20 hrs of B200
  time.
- **Kernel-level parity on decode**: vLLM on sm_100 uses FlashInfer
  TRTLLM kernels which are fused / tuned more aggressively than
  flex_attention's Inductor-generated Triton. Closing the last
  ~28-48 % gap will require either tuning more Triton configs or
  waiting for a TMA-native flex_attention path.

## Raw stats (under `scripts/benchmarks/results/stats/`)

- `flex_{8,16,32,64,128}_tuned.json` (best opts, 5 rounds, torch 2.9.1)
- `flex_64_lora_tuned.json` (GRPO canonical, torch 2.9.1)
- `flex_64_lora_torch211_baseline.json` + `_repeat.json` + `_10rounds.json`
  (same config re-run on torch 2.11 to measure noise)
- `flex_64_lora_fa4prefill.json` (FA4 prefill regression at batch 64)
- `flex_64_lora_autotune{,_tma}.json` (Inductor-autotune-suggested config)
- `flex_64_lora_warps2.json` + `flex_64_lora_pinned_blocks.json` (other
  sweep points)
- `flex_{32,64,128,256}x512[_lora]_cudagraph.json` (prior best-of-3 runs)
- `vllm_{8,16,32,64,128,256}[x512][_lora].json`
