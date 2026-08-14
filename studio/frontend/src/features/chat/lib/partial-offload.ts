// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether a load landed only partly on the GPU, from llama.cpp's own count.
 *
 * Kept free of React so it can be tested: the runtime hook pulls in the router and
 * the whole chat store and cannot be imported under node:test.
 *
 * Studio does not choose this split. In Auto mode it decides the model does not
 * provably fit and hands placement to llama.cpp's `--fit on`, which quietly puts
 * some layers on the CPU. Decode then runs at a fraction of the speed, and until
 * now the load reported success and nothing else, so the only way to find out was
 * to notice the model was slow.
 */

export interface OffloadCounts {
  offloaded?: number | null;
  total?: number | null;
}

/** `true` only for a genuine split: some layers on the GPU and some not.
 *
 * All-on-GPU is the normal case and needs no notice. None-on-GPU is a different
 * failure with its own reporting (the CPU-fallback path), so it is excluded here
 * rather than folded in.
 */
export function isPartialOffload(counts: OffloadCounts): boolean {
  const { offloaded, total } = counts;
  if (typeof offloaded !== "number" || typeof total !== "number") return false;
  if (total <= 0) return false;
  return offloaded > 0 && offloaded < total;
}
