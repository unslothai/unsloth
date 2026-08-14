// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether a load landed only partly on the GPU, from llama.cpp's own count.
 *
 * Kept free of React so it can be tested: the runtime hook pulls in the router and
 * the whole chat store and cannot be imported under node:test. Shared, because the
 * interactive load and the automatic one (a chat sent with no model loaded) are
 * different code paths and the second is the one a new user hits first.
 *
 * Studio only chooses this split in Manual mode. On Auto it decides the model does
 * not provably fit and hands placement to llama.cpp's `--fit on`, which quietly
 * puts some layers on the CPU. Decode then runs at a fraction of the speed, and the
 * load reported success and nothing else, so the only way to find out was to notice
 * the model was slow.
 */

export interface OffloadCounts {
  offloaded?: number | null;
  total?: number | null;
  /** "manual" means the user pinned the split themselves. */
  gpuMemoryMode?: string | null;
}

/** `true` only for a split nobody asked for.
 *
 * All-on-GPU is the normal case and needs no notice. None-on-GPU is a different
 * failure with its own reporting (the CPU-fallback path), so it is excluded rather
 * than folded in. Manual mode is excluded because the user pinned that layer count
 * deliberately: the model may well fit, and telling them to pick a smaller
 * quantization would be advice against their own choice.
 */
export function isPartialOffload(counts: OffloadCounts): boolean {
  const { offloaded, total, gpuMemoryMode } = counts;
  if (gpuMemoryMode === "manual") return false;
  if (typeof offloaded !== "number" || typeof total !== "number") return false;
  if (total <= 0) return false;
  return offloaded > 0 && offloaded < total;
}

/** The one wording, so the two load paths cannot drift apart. */
export function partialOffloadDescription(counts: OffloadCounts): string {
  return (
    `${counts.offloaded} of ${counts.total} layers are on the GPU. The rest run on ` +
    "CPU, so generation will be slower. A smaller quantization would fit entirely " +
    "on the GPU."
  );
}
