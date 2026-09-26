# Phase 1: rollout-only LoRA rank-32 microbenchmark

Every backend generates the same 32 prompts (DAPO-Math-17k, seed 3407) for
`max_new_tokens=512` with equivalence sampling: `temperature=0.1, top_p=0.97,
min_p=0.5, top_k=5`. 16-prompt warmup, 2 measured rounds, median wall reported.

All four backends load the same `outputs/lora_rank32_fresh` adapter (see
`make_lora_adapter.py`). LoRA kernels are active on every decode step.

## Results (GPU B200, bf16, Qwen3-4B-Base + rank-32 LoRA)

| Backend              | Median wall (s) | Decode tok/s | Prompt tok/s | Peak mem (GB) | % of vLLM |
|----------------------|-----------------|--------------|--------------|---------------|-----------|
| vLLM (fast_inference)| 3.30            | **4581**     | 1467         | 156.2         | 100.0 %   |
| unsloth_fi_false     | 25.54           | 641          | 190          | **15.8**      | 14.0 %    |
| CB paged+FA4 (persistent) | 34.99      | 422          | 138          | 103.8         | 9.2 %     |
| CB sdpa_paged (persistent)| 34.07      | 434          | 142          | 111.9         | 9.5 %     |

## Observations

1. **vLLM with LoRA is ~37% slower than vLLM without LoRA** (7224 → 4581 tok/s
   per the pre-LoRA PR table). The LoRA kernels cost real time even in vLLM.
   Still the gold standard by a wide margin.

2. **Unsloth `fast_inference=False` is the surprise**: **1.5× faster than CB**
   at **1/7th the peak memory**. The cached fp16 LoRA copies in
   `fast_linear_forward` and the Triton RMSNorm/RoPE paths dominate the CB
   baseline on this workload. It is a real practical middle ground — no vLLM
   dependency, low memory, and ~14% of vLLM's throughput.

3. **CB paged_attention (FA4 shim) and CB sdpa_paged are within noise**:
   422 vs 434 tok/s. At this scale the attention kernel is not the bottleneck;
   Python-side launch overhead on `_generation_step` dominates (confirmed by
   prior profile: ~16k `cuLaunchKernelEx` for 371 decoded tokens). CUDA graph
   replay (Phase 3) is the right lever.

4. **Unsloth `fi_false` reached max_new_tokens on every prompt** (`n_decoded =
   16384 = 32 × 512`) whereas vLLM / CB stopped some sequences on EOS
   (`~15000 decoded`). Equivalence sampling + greedy-ish settings means most
   completions are long, but the slight difference is worth noting when
   reading the raw tok/s numbers.

5. Completions are qualitatively coherent in every backend (see
   `sample_completions` in the stats JSONs). vLLM and unsloth_fi_false produce
   the *same* opening tokens on probe prompts (deterministic sampling lower
   bound), which is a useful weak sanity check.

## Raw stats

- `scripts/benchmarks/results/stats/lora_vllm_gen.json`
- `scripts/benchmarks/results/stats/lora_unsloth_fi_false_gen.json`
- `scripts/benchmarks/results/stats/lora_cb_paged_fa4_gen.json`
- `scripts/benchmarks/results/stats/lora_cb_sdpa_paged_gen.json`

## Downstream implication

Phase 2 (full GRPO training) will include `unsloth_fi_false` as a first-class
backend — if throughput parity holds end-to-end, it may be the pragmatic
default for teams that cannot take the vLLM memory footprint. Phase 3 (CB sync
driver + CUDA graphs) targets the CB paths specifically.
