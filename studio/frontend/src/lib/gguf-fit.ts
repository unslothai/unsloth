// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Single source of truth for "will this GGUF fit on the user's GPU?". Mirrors the
 * backend's GPU selection (llama_cpp.py `_select_gpus`): 90% of GPU memory for
 * weights + KV cache, else `--fit` CPU offload. One formula so the Hub card and
 * chat picker can't disagree.
 */

export type GgufFitClass = "fits" | "marginal" | "partial" | "ram" | "oom";

export interface GgufFitInput {
  gpuGb?: number;
  systemRamGb?: number;
  /** The user's saved VRAM Budget, when it is known.
   *
   *  Absent means "not loaded yet, or a backend too old to serve the route", and
   *  falls back to the shared default. It does NOT mean "the whole card": scoring
   *  against 1.0 would call a load comfortable that the loader refuses.
   *
   *  This exists because the badge and the memory bar sit on the SAME ROW and the
   *  bar already consumes the live fraction (`use-model-memory.ts`). With a saved
   *  0.90 on a 24 GiB card the loader admits at 21.6 GiB while this file scored
   *  against the 0.97 default's 23.28 GiB, so a 18 to 19 GiB quant was badged
   *  `fits` beside a bar reporting an overage. */
  budgetFraction?: number;
}

/** Fraction of GPU VRAM treated as usable.
 *
 * Re-exported from the shared memory core so this badge and the Hub memory bar
 * on the SAME ROW cannot judge against different budgets. It was 0.90 here while
 * the bar used the loader's own 0.97, which is what admission actually applies
 * (`_CTX_FIT_VRAM_FRACTION`); `llama_cpp.py` records 0.90 as already tried and
 * reverted, "0.90 dropped 91-94% fits to CPU offload, #5106".
 *
 * Measured over 15-24 GiB on a 24 GiB card, this takes badge-vs-bar
 * disagreements from 11/19 sizes to 8/19. It does not remove them: the residual
 * 8 are the ESTIMATOR difference, since this file scores `size * 1.15 + 1 GB`
 * while the bar uses the load planner's real figures. Sharing the constant is
 * the part that belongs to this consolidation; rewiring the badge onto the
 * planner is a separate change with its own blast radius (it also drives a
 * filter). */
export { DEFAULT_VRAM_BUDGET_FRACTION as VRAM_HEADROOM_RATIO } from "./memory/thresholds.ts";
import { DEFAULT_VRAM_BUDGET_FRACTION } from "./memory/thresholds.ts";
/** GGUF weights are file size; runtime activations add roughly this fraction. */
const ACTIVATIONS_RATIO = 0.15;
/** Flat KV/context allowance at a typical 4K window. */
const CONTEXT_OVERHEAD_GB = 1.0;
/** Conservative share of system RAM usable for CPU offload. */
const RAM_OFFLOAD_USABLE_RATIO = 0.5;

export function requiredGgufMemoryGb(
  sizeBytes: number,
  contextOverheadGb = CONTEXT_OVERHEAD_GB,
): number {
  const sizeGb = sizeBytes / 1024 ** 3;
  return sizeGb * (1 + ACTIVATIONS_RATIO) + contextOverheadGb;
}

export function classifyGgufFit(
  sizeBytes: number,
  { gpuGb, systemRamGb, budgetFraction }: GgufFitInput,
): GgufFitClass {
  const required = requiredGgufMemoryGb(sizeBytes);
  if (!gpuGb || gpuGb <= 0) {
    const ramBudget = (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
    return required <= ramBudget ? "ram" : "oom";
  }
  // Guarded rather than trusted: this arrives from a settings route, and a 0 or a
  // non-finite value would score every quant as an overage.
  const fraction =
    typeof budgetFraction === "number" &&
    Number.isFinite(budgetFraction) &&
    budgetFraction > 0 &&
    budgetFraction <= 1
      ? budgetFraction
      : DEFAULT_VRAM_BUDGET_FRACTION;
  const budget = gpuGb * fraction;
  if (required <= budget) return "fits";
  // Raw card, deliberately. This band means "over your budget, still card-sized",
  // which is the only thing that distinguishes it from `fits`: scoring it against
  // `budget` too would make it unreachable, since `required <= budget` has already
  // returned above. It is a warning tier, not a claim that the load will be
  // admitted.
  if (required <= gpuGb) return "marginal";
  // The budget, though. Once layers spill, what the GPU can still contribute is
  // what it is ALLOWED to hold, and the reserve is exactly what the model and KV
  // cache may not use ("the fit reserves a slice of every card that the model and
  // KV cache may not use", vram_budget_settings.py). Crediting the raw card here
  // invented capacity the loader will not give: on a 24 GiB card with 16 GiB of
  // RAM at the legal minimum 0.80, quants from 23 to 26 GiB were badged `partial`
  // when the budget leaves them no way to load.
  const combined = budget + (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
  if (required <= combined) return "partial";
  return "oom";
}
