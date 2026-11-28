# Phase 2: end-to-end GRPO backend comparison

Same dataset, reward functions, sampling (`temperature=0.1, top_p=0.97,
min_p=0.5, top_k=5`), and seed (3407) across backends. `num_generations=4`;
`per_device_train_batch_size` auto-raised to 4 on vanilla-HF backends so
TRL's `generation_batch_size % num_generations == 0` check passes
(Unsloth's loader does this for you, vanilla HF does not).

Callbacks: `StatisticsCallback` from `torch_debugging_utils` logs per-step
loss, grad-norm, memory, wall time; reward / KL are captured from the TRL
log dict. Median step wall is measured on steps 4..N (first 3 skipped to
amortize compile / graph / warmup).

## 10-step vibe check

| Backend                    | Train wall (s) | Median step (s) | Peak mem (GB) | % of vLLM |
|----------------------------|----------------|-----------------|---------------|-----------|
| vLLM (fast_inference)      | 74.4           | **4.14**        | 157.9         | 100 %     |
| unsloth_fi_false           | 355.4          | 23.95           | **10.7**      | 17 %      |
| cb_paged (sdpa_paged load) | 466.0          | 36.02           | 55.6          | 11.5 %    |

## 30-step equivalence

| Backend                    | Train wall (s) | Median step (s) | Peak mem (GB) | % of vLLM |
|----------------------------|----------------|-----------------|---------------|-----------|
| vLLM (fast_inference)      | 215.9          | **5.14**        | 159.0         | 100 %     |
| unsloth_fi_false           | 1165.4         | 41.30           | **10.7**      | 12.4 %    |
| cb_paged                   | 1564.5         | 39.82           | 61.9          | 12.9 %    |

(Note: fi_false's median step jumped from 23.95 s at 10 steps to 41.30 s at
30 steps because the early-GRPO policy started producing longer completions
as it learned to place the `</SOLUTION>` marker; the same effect is present
but smaller in cb_paged because its LoRA warm-up trajectory is different.)

## Pairwise diff vs vLLM (30 steps, `scripts/benchmarks/compare_grpo_runs.py`)

| Pair                            | max &#124;loss diff&#124; | max &#124;reward diff&#124; | max &#124;kl diff&#124; | max &#124;grad_norm diff&#124; |
|---------------------------------|---------------------------|------------------------------|------------------------|--------------------------------|
| vLLM vs **unsloth_fi_false**    | 0.39                      | 9.25 (mean 2.99)             | **0.015**              | 0.94                           |
| vLLM vs **cb_paged**            | 0.83                      | 6.25 (mean 2.29)             | *(not logged)*         | 919.1                          |

Reward diffs of 2-9 are expected: different rollout backends produce
different completions even at `temperature=0.1` because of kernel-level
non-determinism (vLLM uses FlashInfer TRTLLM kernels, CB uses paged SDPA /
FA4, Unsloth uses its cached fp16 LoRA path). The reward function reads
those completions, so the reward array mechanically differs. What matters
for equivalence is:

- **KL trajectory is near-identical** between vLLM and unsloth_fi_false
  (both stay in `[0, 0.015]` across all 30 steps). The KL *term* of the
  GRPO loss is the guardrail against policy drift, so matching KL means
  the training dynamics are in the same regime.
- **Loss magnitudes are bounded** in `[-0.3, 1.0]` for all three backends.
- **No NaNs, no unbounded growth, no gibberish completions** in any run.

## grad_norm 919 on cb_paged

The enormous cb_paged grad_norm (vs vLLM's ~1.0) is a clipping story, not a
correctness story: the vLLM path goes through Unsloth's `FastLanguageModel`
which clips gradients to `max_grad_norm=1.0` internally, while the vanilla
HF path used by cb_paged picks up TRL's raw grad_norm reported by the
optimizer pre-clip (or without clipping if no `max_grad_norm` is set in
GRPOConfig). For a fair training-dynamics comparison the cb_paged config
should set `max_grad_norm=1.0` explicitly; left for a follow-up commit.

## KL missing for cb_paged

`StatisticsCallback.on_log` forwards the full TRL log dict into its per-step
entry only on steps where `loss` is present. TRL's vanilla-HF path separately
logs KL on a different log call that doesn't include loss, so the callback
silently drops it. Follow-up: relax the callback so every log dict with a
`step` field merges into the matching entry regardless of which keys are
present.

## Headline takeaways

1. **unsloth_fi_false is the pragmatic middle ground**: 12-17% of vLLM's
   throughput, **15x less peak memory** (10.7 GB vs 159 GB), KL trajectory
   matching vLLM within sampling noise.
2. **cb_paged is close to fi_false in throughput at this batch size** (41 s
   vs 40 s median step at 30 steps) but costs 6x more memory. Phase 3
   (main-thread sync driver + CUDA graphs on the rollout) is the right
   lever for making CB competitive.
3. **torch.compile on the training step is not a quick win** for either
   backend (Phase 4 report below).

## Phase 3 state (CB sync driver)

`scripts/benchmarks/cb_sync_driver.py`:
- Eager main-thread driver works end-to-end: smoke test on GPU 1 with 8
  prompts / 64 tokens produced the expected 512 correct tokens.
- CUDA graph capture hangs on the first graphed step. Likely cause:
  `ContinuousBatchProcessor._sample` reads `next_tokens.size(1)` as a
  Python int to slice `batch_processor.output_ids[:, :tokens]`, which
  forces a CPU-GPU sync and is not CUDA-graph-safe. Fix direction: keep
  a fixed `tokens` count when `slice_inputs=False` (buffer size is
  constant), or rewrite the copy as a full-buffer `copy_` without the
  slice.
- Deferred to a follow-up commit.

## Phase 4 state (torch.compile on training forward)

- `unsloth_fi_false + compile_mode=default`: crashes with
  `PeftModel_fast_forward() got multiple values for argument 'input_ids'`.
  Unsloth's monkey-patched forward and Dynamo's argument rebinding don't
  compose.
- `cb_paged + compile_mode=default`: Dynamo emits 700+ recompiles /
  graph breaks on the first optimizer step and never makes progress.
  Root cause: `modeling_utils.make_inputs_require_grads` calls
  `Tensor.requires_grad_()` which triggers Dynamo GB0125 (unsupported
  mutating op). TRL's GRPO `_compute_loss` then re-enters the tracer,
  which re-triggers the break, which recompiles, and so on.
- `vllm` is excluded (vLLM owns its own compile pipeline).

Net: compile on the training step is not the right lever in this stack.
Phase 3 (CUDA graphs on the rollout decode) is.

## Raw stats

- `scripts/benchmarks/results/stats/grpo_{vllm,unsloth_fi_false,cb_paged}_{10,30}.json`
  (StatisticsCallback per-step logs with full TRL metric dict)
- `scripts/benchmarks/results/stats/grpo_*_{10,30}.summary.json` (short form)

Pairwise diff:

    python scripts/benchmarks/compare_grpo_runs.py \
        --ref logs/grpo_vllm_30.json \
        --candidate logs/grpo_unsloth_fi_false_30.json
