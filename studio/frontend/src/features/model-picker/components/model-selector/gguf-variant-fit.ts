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
 * `disk` is not a softer refusal, it is a different claim. A refusal says the load
 * cannot happen; `disk` says it happens off the page cache and will be slow. Only
 * the first is a reason not to try, so they are not allowed to share a name.
 *
 * The ladder is about PLACEMENT, and reads down as one: on the card, part in host
 * RAM, paged from the file, nowhere to put it. `nospace` sits above `disk` because
 * it is the one condition that defeats paging too.
 */
export type GgufFit = "fits" | "tight" | "disk" | "nospace" | "oom";

/** Tiers that auto-selection may pick. `disk` runs, but nobody wants a 169 GB quant
 * chosen for them, so the recommendation stays inside memory. */
export function ggufFitIsAutoSelectable(fit: GgufFit): boolean {
  return fit === "fits" || fit === "tight";
}

/** The two tiers that render as one `OOM` pill. They are separate here because the
 *  reasons are not interchangeable (one is the machine's storage, the other is our
 *  own ignorance) and only the first is something the user can act on, but they
 *  share a pill because the action is identical: pick a smaller quant. */
export function ggufFitIsRefusal(fit: GgufFit): boolean {
  return fit === "nospace" || fit === "oom";
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
    diskFreeGb = 0,
  }: {
    gpuBudgetGb: number;
    totalBudgetGb: number;
    budgetKnown: boolean;
    /** Free space on the cache volume, 0 when unread. DECIMAL GB, because that
     *  is what the backend reports (`main.py` divides by 1e9 for disk, unlike
     *  memory); the comparison converts the file to the same base. */
    diskFreeGb?: number;
  },
  /** True when the file is already on the machine. The floor is about landing
   *  the download, and a landed file needs no space it does not already hold. */
  onDisk?: boolean,
  /** Bytes the download would actually transfer: the full footprint with
   *  companion files for a fresh fetch, the measured remainder for a resumable
   *  partial. Defaults to the checkpoint size when the caller knows no better. */
  downloadBytes?: number,
): GgufFit {
  // Preserve permissive behavior only when no budget was measured. A known
  // zero Vulkan budget means every non-empty variant is OOM.
  if (totalBudgetGb <= 0) return budgetKnown ? "oom" : "fits";
  const gb = sizeBytes / 1024 ** 3;
  // Before any memory question. Every tier below is a claim about WHERE the weights
  // sit, and all of them, `disk` loudest, assume the file is on the machine. A quant
  // larger than the free space cannot be downloaded, so there is nothing to place
  // and nothing to page. Checked against the raw figure with no share taken out of
  // it: this is a hard floor, not a budget, and shaving it would refuse downloads
  // that fit. 0 means the probe said nothing, so the check abstains.
  if (!onDisk && diskFreeGb > 0 && (downloadBytes ?? sizeBytes) / 1e9 > diskFreeGb)
    return "nospace";
  if (gb <= 0 || gb <= gpuBudgetGb) return "fits";
  // No-GPU / unified-memory hosts (Mac) have only the RAM budget, and past it they
  // page from the file like anything else. mmap does not care about pool topology.
  // This branch first said "oom" on the reasoning that one pool has nothing to
  // offload to, which confused "cannot be resident" with "cannot run".
  if (gpuBudgetGb <= 0) return gb <= totalBudgetGb ? "fits" : "disk";
  if (gb <= totalBudgetGb) return "tight";
  // Past GPU plus RAM, and still not a refusal. llama.cpp mmaps the weights by
  // default and the OS pages them in from the file, so a model larger than every
  // volatile byte on the machine runs, just slowly. Reported on the Qwen3.8
  // Flash-Next thread: UD-Q3_K_XL, 90 GB, on 16 GB of VRAM and 64 GB of RAM at
  // about 12 tok/s. This badge called that OOM, which is the one verdict a user
  // acts on by not trying.
  return "disk";
}
