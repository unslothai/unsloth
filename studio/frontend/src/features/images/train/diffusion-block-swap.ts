// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { offloadHardwareSupported } from "../../training/api/mappers.ts";

export type DiffusionBlockSwapMode = "off" | "auto" | "count";

/** Whether the Train tab offers block swap: a family the backend flags (DiT trainer with bf16 on this host),
 * the bf16 base it streams, gradient checkpointing, and the hardware rule core's layer swap uses (a discrete
 * CUDA / ROCm card). The start route refuses every other case itself. */
export function diffusionBlockSwapApplies(opts: {
  supportsBlockSwap: boolean | undefined;
  basePrecision: string;
  gradientCheckpointing: boolean;
  system: Parameters<typeof offloadHardwareSupported>[0];
}): boolean {
  return (
    opts.supportsBlockSwap === true &&
    opts.basePrecision === "bf16" &&
    opts.gradientCheckpointing &&
    offloadHardwareSupported(opts.system)
  );
}

/** The request value: omitted where the control is hidden or off, so an older backend and today's default see
 * the same request as before. */
export function diffusionBlockSwapPayload(
  applies: boolean,
  mode: DiffusionBlockSwapMode,
  count: number,
): "auto" | number | undefined {
  if (!applies || mode === "off") return undefined;
  if (mode === "auto") return "auto";
  const n = Math.min(1024, Math.max(0, Math.floor(count || 0)));
  return n > 0 ? n : undefined;
}

/** "14 of 57 blocks from host RAM (5.5 GiB pinned)" for the run status, or null when nothing was swapped. */
export function diffusionBlockSwapSummary(
  swap: { swapped_blocks?: number; total_blocks?: number; pinned_gib?: number } | null | undefined,
): string | null {
  if (!swap || swap.swapped_blocks == null || swap.total_blocks == null) return null;
  if (swap.swapped_blocks <= 0) return `none needed (0 of ${swap.total_blocks} blocks)`;
  const pinned = swap.pinned_gib != null ? ` (${swap.pinned_gib.toFixed(1)} GiB pinned)` : "";
  return `${swap.swapped_blocks} of ${swap.total_blocks} blocks from host RAM${pinned}`;
}
