// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How a GGUF variant lands on this machine.
 *
 * Its own module rather than a helper inside `pickers.tsx`, because the picker is
 * `.tsx` and the test runner strips types without parsing JSX, so nothing in that
 * file can be asserted directly. A rule this load-bearing should not be reachable
 * only by mounting a component.
 */

/** How a GGUF lands on this machine, worst tier last.
 *
 * `disk` is not a softer `oom`, it is a different claim. `oom` says the load cannot
 * happen; `disk` says it happens off the page cache and will be slow. Only the first
 * is a reason not to try, so they are not allowed to share a name.
 */
export type GgufFit = "fits" | "tight" | "disk" | "oom";

/** Tiers that auto-selection may pick. `disk` runs, but nobody wants a 169 GB quant
 * chosen for them, so the recommendation stays inside memory. */
export function ggufFitIsAutoSelectable(fit: GgufFit): boolean {
  return fit === "fits" || fit === "tight";
}

/** Classify one variant against a measured budget. Mirrors llama-server `_select_gpus`.
 *
 * Pure and exported so the tiering can be asserted without mounting a picker; the
 * component wraps it in a `useCallback` over its own budget.
 */
export function classifyGgufVariantFit(
  sizeBytes: number,
  {
    gpuBudgetGb,
    totalBudgetGb,
    budgetKnown,
  }: { gpuBudgetGb: number; totalBudgetGb: number; budgetKnown: boolean },
): GgufFit {
  // Preserve permissive behavior only when no budget was measured. A known
  // zero Vulkan budget means every non-empty variant is OOM.
  if (totalBudgetGb <= 0) return budgetKnown ? "oom" : "fits";
  const gb = sizeBytes / 1024 ** 3;
  if (gb <= 0 || gb <= gpuBudgetGb) return "fits";
  // No-GPU / unified-memory hosts (Mac) have only the RAM budget. There is no
  // separate disk tier here: on one pool a load past it is a real refusal, not
  // a slow success.
  if (gpuBudgetGb <= 0) return gb <= totalBudgetGb ? "fits" : "oom";
  if (gb <= totalBudgetGb) return "tight";
  // Past GPU plus RAM, and still not a refusal. llama.cpp mmaps the weights by
  // default and the OS pages them in from the file, so a model larger than every
  // volatile byte on the machine runs, just slowly. Reported on the Qwen3.8
  // Flash-Next thread: UD-Q3_K_XL, 90 GB, on 16 GB of VRAM and 64 GB of RAM at
  // about 12 tok/s. This badge called that OOM, which is the one verdict a user
  // acts on by not trying.
  return "disk";
}
