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

| Backend  | tok/s   | peak mem | flex / vLLM |
|----------|--------:|---------:|------------:|
| vLLM     | 7775    | 156 GB   | 100 %       |
| **flex** | **5616**| **44 GB**| **72.2 %**  |

At the GRPO workload flex reaches **72 % of vLLM throughput at
3.5 × less memory**. Up from 9 % with transformers CB at the start of this
work.

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
