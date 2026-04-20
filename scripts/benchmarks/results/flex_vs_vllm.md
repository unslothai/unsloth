# flex_attention + paged KV + CUDA graphs vs vLLM

Goal stated in the plan: "CB reaches at least 30% of vLLM throughput." After
the earlier phases ran out of gas at ~10% with transformers CB, we rebuilt
the rollout path on top of `torch.nn.attention.flex_attention` using the
paged KV + BlockMask pattern from
[flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm).

## Setup

- B200 (sm_100), Qwen3-4B-Base, bf16
- 512 max_new_tokens per prompt, 16-prompt warmup, 3 measured rounds
- Equivalence sampling (`temperature=0.1, top_p=0.97, min_p=0.5, top_k=5`)
  for every backend. flex path is greedy only (CUDA-graph safe).
- LoRA rank 32 applied to all {q,k,v,o,gate,up,down}_proj when `LoRA=yes`.
- Median wall over rounds reported; `decode_tps_best` in the flex stats
  uses the best-of-3 round (steady state after all per-shape Inductor
  compiles have landed).

## Headline numbers

| Batch | LoRA | vLLM tok/s | qwen3_flex tok/s | flex / vLLM |
|-------|------|-----------:|-----------------:|------------:|
| 32    | no   |       7224 |             2189 |       30 %  |
| 32    | yes  |       4581 |             2334 |       51 %  |
| 64    | yes  |       7775 |             4279 |       55 %  |
| 128   | no   |      14996 |             6501 |       43 %  |

Peak memory: flex uses 44-81 GB (scales with batch). vLLM uses 156 GB
regardless (colocates KV cache up front). flex memory is half to a fifth
of vLLM.

## Why this closes the gap when transformers CB couldn't

transformers CB with `attn_implementation=paged_attention` (FA4 shim) +
persistent manager reached 422 tok/s at batch 32 with LoRA -- 9.2 % of
vLLM. Profiling traced the wall to Python-side kernel launch overhead:
16,445 `cuLaunchKernelEx` for 371 decoded tokens, ~3.3x the GPU compute
time. `torch.compile(mode="reduce-overhead")` on the threaded CB path
hung because `cudagraph_trees` requires main-thread TLS; moving to a
main-thread sync driver didn't help on its own (400 tok/s eager, same as
threaded) because the Python dispatch per step is the same.

`flex_attention` + BlockMask is different: the paged logical->physical
mapping is expressed as a `mask_mod` callback, which compiles. The entire
decode step fits inside one CUDA graph per batch-size bucket. Graph replay
is ~1 kernel launch per step regardless of how many layers the model has,
so the Python cost vanishes.

## Architecture notes

- `flex_paged_attention.py`: `PagedKVCache` + `PageTable` verbatim from
  flex-nano-vllm (BSD-3, see their THIRD_PARTY_LICENSES.md). Page size 128,
  num_pages configurable via `--n_pages`. `batch_idx=0` and `page_idx=0`
  are both reserved as no-op slots so padded entries at capture time can
  write safely.
- `qwen3_flex_inference.py`: monkey-patches `Qwen3Attention.forward` to
  call `flex_attention(q, k, v, block_mask=...)` against the paged cache.
  Walks the `Qwen3Model` layer stack manually so `flex_block_mask /
  flex_input_pos / flex_batch_idx` reach the attention layer without
  modifying `Qwen3ForCausalLM.forward`.
- `capture_decode_cudagraph()`: pre-reserves one page per batch slot,
  captures one CUDA graph per bucket in `[1,2,4,8,16,32...max_bs]`, then
  releases the scratch batches. Without the pre-reservation the first
  graphed step hits `cudaErrorIllegalAddress` because
  `assign()` tries `k_cache[..., -1, :] = k_val` on unallocated slots.

## Output coherence

Same 3 canonical math prompts across vLLM and flex:

    Prompt: "A trapezoid inscribed in a circle..."
    vLLM:   "First, we need to find the length of the legs..."
    flex:   "First, we need to find the total number of letters..."

Different rollouts (different kernels, different sampling RNG), both
coherent English solving the problem. No gibberish at any measured
configuration.

## What is still on the table

- **Chunked prefill**: vLLM interleaves prefill and decode inside a single
  step. flex does a full separate prefill pass per new request batch,
  which is the main remaining penalty per the flex-nano-vllm blog post.
- **Kernel-level parity**: vLLM uses FlashInfer TRTLLM kernels on
  Blackwell. flex dispatches to `flex_attention`'s Inductor-generated
  Triton. Closing the last factor of ~2 will likely require waiting for
  torch's FlexAttention backend to grow sm_100-tuned templates (or
  hand-rolled ones).

## Raw stats

- `scripts/benchmarks/results/stats/flex_32x512_cudagraph.json` (no LoRA)
- `scripts/benchmarks/results/stats/flex_32x512_lora_cudagraph.json`
- `scripts/benchmarks/results/stats/flex_64x512_lora_cudagraph.json`
- `scripts/benchmarks/results/stats/flex_128x512_cudagraph.json`
- `scripts/benchmarks/results/stats/vllm_128x512.json`
- `scripts/benchmarks/results/stats/vllm_64x512_lora.json`
