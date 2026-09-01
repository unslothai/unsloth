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
  /** How many GPUs `gpuGb` is the sum of. Only the reserve floor uses it, and only
   *  above the default budget; absent means one card, which is what every verdict
   *  scored against before it existed. */
  gpuCount?: number;
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
/** The loader's absolute reserve floor, `_VRAM_FLOOR_RESERVE_MIB` (512 MiB) in llama_cpp.py.
 *  Capped there at 3% of the card, so it never rises above the default budget's own reserve. */
const VRAM_RESERVE_FLOOR_GB = 512 / 1024;
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
  { gpuGb, systemRamGb, budgetFraction, gpuCount }: GgufFitInput,
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
  // Not simply `gpuGb * fraction`. The loader's `_vram_usable_mib` charges
  // `max((1 - frac) * total, floor)`, where the floor is `min(512 MiB, 3% of total)`, so the
  // reserve never disappears at the top of the slider. Below the 0.97 default the percentage term
  // always wins and this is identical; above it, `gpuGb * fraction` claimed capacity the backend
  // does not offer, and a 20 GiB file on a full 24 GiB card at 1.0 read `fits` while `_select_gpus`
  // fell back to --fit.
  //
  // Charged per CARD, not per host: `_select_gpus` calls `_vram_usable_mib` for every device and
  // sums the results, so a k-GPU box holds back k floors. One floor for the whole box read 47.5
  // GiB usable on two 24 GiB cards at 1.0 where the loader offers 47.0, so a file needing 47.2
  // read `fits` against a load that fell back to --fit. Exact on matched cards, an even split on
  // a mixed one, which is still far nearer than charging the floor once.
  const cards =
    typeof gpuCount === "number" && Number.isFinite(gpuCount) && gpuCount >= 1
      ? Math.floor(gpuCount)
      : 1;
  const reserveFloorGb =
    cards *
    Math.min(
      VRAM_RESERVE_FLOOR_GB,
      (1 - DEFAULT_VRAM_BUDGET_FRACTION) * (gpuGb / cards),
    );
  const budget = gpuGb - Math.max((1 - fraction) * gpuGb, reserveFloorGb);
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
