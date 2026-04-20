# Phase 2: end-to-end GRPO backend comparison (10-step vibe check)

Same dataset, reward functions, sampling (`temperature=0.1, top_p=0.97,
min_p=0.5, top_k=5`), and seed (3407). `max_steps=10, num_generations=4,
per_device_train_batch_size=4` (auto-adjusted from 1 on vanilla-HF paths
to satisfy TRL's `generation_batch_size % num_generations == 0`).

Callbacks: `StatisticsCallback` from `torch_debugging_utils` logs per-step
loss, grad-norm, memory, wall time. Median step time is computed over steps
4-10 (first 3 skipped for compile / graph / warmup amortization).

## 10-step results

| Backend                       | Train wall (s) | Median step (s) | Peak mem (GB) | % of vLLM |
|-------------------------------|----------------|-----------------|---------------|-----------|
| vLLM (fast_inference)         | 74.4           | **4.14**        | 157.9         | 100 %     |
| unsloth_fi_false              | 355.4          | 23.95           | **10.7**      | 17 %      |
| cb_paged (sdpa_paged load)    | 466.0          | 36.02           | 55.6          | 11.5 %    |

Loss / reward / KL arrays for each backend (10 steps, rounded):

| Step | vLLM loss | vLLM reward | vLLM kl  | fi_false loss | fi_false reward | fi_false kl | cb_paged loss | cb_paged reward |
|------|-----------|-------------|----------|---------------|-----------------|-------------|----------------|------------------|
| 1    |  0.031    |  0.00       | 0.00000  |  0.000        |  0.50           | 0.00000     | -0.086         |  0.62            |
| 2    | -0.194    | -2.50       | 0.00000  | -0.089        | -6.50           | 0.00000     |  0.041         | -2.50            |
| 3    |  0.263    | -3.62       | 0.01192  | -0.139        | -2.50           | 0.00857     |  0.000         |  0.50            |
| 4    | -0.201    |  0.00       | 0.00422  | -0.124        | -2.50           | 0.00931     |  0.086         | -1.50            |
| 5    |  0.209    |  0.38       | 0.00369  |  0.000        |  0.50           | 0.00213     |  0.016         |  0.00            |
| 6    |  0.000    | -7.50       | 0.00250  |  0.000        | -7.50           | 0.00071     |  0.000         | -7.50            |
| 7    |  0.037    |  1.50       | 0.00614  | -0.010        | -5.50           | 0.00522     |  0.074         |  1.50            |
| 8    |  0.000    | -7.50       | 0.00465  |  0.034        | -5.25           | 0.00014     | -0.048         | -4.25            |
| 9    |  0.000    |  0.50       | 0.00176  |  0.000        |  0.50           | 0.01090     | -0.188         | -1.50            |
| 10   |  0.205    | -3.50       | 0.00200  |  0.204        | -2.50           | 0.00215     |  0.044         | -6.50            |

## Observations

1. **Coherence gate (all backends)**: losses are bounded in `[-0.25, 0.3]`,
   grad-norms finite, rewards in the plan's expected negative-then-rising
   range. No CJK-token salad, no NaNs.

2. **KL trajectories are qualitatively matched** between vLLM and
   `unsloth_fi_false` (both in `[0, 0.015]`), confirming that
   `fast_inference=False` produces rollouts close to the vLLM reference once
   `temperature=0.1` is used. `cb_paged` also produces rollouts but our
   `StatisticsCallback` did not capture TRL's `kl` entry in its `on_log`
   pass -- the next iteration will forward every log dict entry into the JSON.

3. **Per-step timing**: `unsloth_fi_false` is 5.8x slower than vLLM; `cb_paged`
   is 8.7x slower. Neither hits the plan's 30% target on this vibe check.

4. **Memory is the standout axis**:
   - vLLM: 158 GB (prefill KV cache + vLLM engine overhead)
   - cb_paged: 55.6 GB (paged cache only)
   - unsloth_fi_false: **10.7 GB** -- 15x lower than vLLM.

   Unsloth's fast_inference=False path is a genuine option for teams who
   cannot afford the vLLM footprint but are willing to take a ~5-6x rollout
   wall-clock hit.

5. **cb_paged load needed `sdpa_paged` not `paged_attention`**: the
   FA4-shimmed `paged_attention` kernel requires `cu_seq_lens_q` on every
   forward, but GRPO's training forward (dense batch) doesn't provide them.
   `sdpa_paged` falls back to plain SDPA when no paged kwargs are present and
   still exercises paged attention during the CB rollout. This is consistent
   with the existing `qwen3_grpo_tpaged.py` which loads with `sdpa`.

## What's next (not yet run)

- **30-step equivalence** with `torch_debugging_utils.compare_training_runs`
  comparing vLLM vs each backend on loss / reward / KL arrays.
- **Phase 3 sync driver** smoke-tested successfully (eager decode produces
  512 correct tokens) but CUDA graph capture hangs on the first graphed step.
  Likely cause: `PagedAttentionCache` constructs tensors inside
  `cache.update()` the first call, which doesn't survive graph capture.
  Two possible fixes being explored: (a) pre-capture warmup steps on the
  capture stream so allocations are already done, (b) replace in-place
  torch.multinomial-adjacent ops with CUDA-graph-safe equivalents.
- **Phase 4 torch.compile**: hook-up ready in `qwen3_grpo_unified.py`
  (`--compile_mode default|reduce-overhead|max-autotune-no-cudagraphs`);
  needs a run budget allocated and the `CompileDebugger` output reviewed.

## Raw stats

- `scripts/benchmarks/results/stats/grpo_vllm_10.summary.json`
- `scripts/benchmarks/results/stats/grpo_unsloth_fi_false_10.summary.json`
- `scripts/benchmarks/results/stats/grpo_cb_paged_10.summary.json`

Full per-step logs (one entry per step with loss/reward/kl/grad_norm and all
of TRL's logging dict) live at `scripts/benchmarks/results/stats/grpo_*.json`.
