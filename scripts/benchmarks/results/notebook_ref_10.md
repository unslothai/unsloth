# Phase 0 reference run: canonical Unsloth Qwen3-4B GRPO notebook (10 steps)

Reproduction of `Qwen3_(4B)-GRPO.ipynb` with three deviations for the backend
comparison downstream:

1. `max_steps = 10` (vibe check; 30 and 100 follow in Phase 2).
2. Equivalence-friendly sampling: `temperature=0.1, top_p=0.97, min_p=0.5,
   top_k=5`. Low variance so KL / reward trajectories across backends can be
   compared tightly.
3. `StatisticsCallback` logs per-step loss, reward, KL, grad-norm, memory, and
   wall time to `logs/notebook_ref_10.json`.

SFT format-priming stage is skipped with `--skip_sft_pre_finetune` — this run
is just the GRPO phase.

Config:
- GPU 6 (B200, bf16)
- Model: `unsloth/Qwen3-4B-Base`, LoRA rank 32 on all proj layers
- `num_generations=4`, `per_device_train_batch_size=1` (TRL enforces
  `pdb * grad_accum * world = multiple of num_generations`, so effective batch
  is 4 × 1)
- `gpu_memory_utilization=0.85`

## Per-step results

| step | loss    | reward  | kl      | grad_norm | time(s) | mem(GB) |
|------|---------|---------|---------|-----------|---------|---------|
| 1    |  0.2423 | -0.875  | 0.00000 | 0.245     | 60.63   | 158.9   |
| 2    |  0.1559 | -5.500  | 0.00000 | 0.767     |  3.89   | 157.0   |
| 3    | -0.1650 | -0.500  | 0.00383 | 0.480     |  7.93   | 158.2   |
| 4    |  0.3177 | -4.125  | 0.00591 | 0.379     | 11.08   | 158.9   |
| 5    | -0.0200 |  3.125  | 0.01598 | 0.341     |  2.68   | 156.7   |
| 6    |  0.0000 | -7.500  | 0.00396 | 0.000     | 10.65   | 158.9   |
| 7    |  0.0000 | -7.500  | 0.00965 | 0.000     |  4.47   | 157.2   |
| 8    |  0.0613 | -6.500  | 0.00319 | 0.172     |  5.81   | 157.6   |
| 9    |  0.0060 | -0.500  | 0.00240 | 0.048     |  4.30   | 157.1   |
| 10   |  0.1582 | -1.500  | 0.00485 | 0.394     |  6.78   | 157.9   |

**Summary:**
- Median step wall (steps 4-10): **5.80 s**
- Total train wall: ~118 s
- Peak memory: **158.9 GB**
- KL trajectory: monotonic rise from 0 to ~0.016 by step 5, settles at
  ~0.005 afterward — consistent with the policy drift being bounded by the KL
  term.
- Step 1 is ~60 s because it amortizes the vLLM CUDA-graph capture; the
  post-warmup median is what Phase 2 will compare against.

## Phase 2 use

This is the gold reference. Every other backend's loss / reward / KL arrays
will be diffed against this one (see `torch_debugging_utils.compare_training_runs`).
Throughput numbers are on a separate axis: even an equivalence-passing backend
that is 3x slower than this is useful information for the PR writeup.
