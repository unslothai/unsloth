// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainableFamily } from "../api";

/** The LR schedules the Train panel offers. The backend accepts more (cosine_with_restarts,
 *  polynomial), so a family naming one of those is dropped rather than seeded into a Select that
 *  cannot show it. */
export type LrScheduler = "constant" | "constant_with_warmup" | "cosine" | "linear";

export const LR_SCHEDULERS: readonly LrScheduler[] = [
  "constant",
  "constant_with_warmup",
  "cosine",
  "linear",
];

/** A family's LR ramp, as one unit. The backend pairs `lr_scheduler` with `lr_warmup_steps`
 *  because diffusers' `get_scheduler` returns before it reads `num_warmup_steps` under
 *  "constant", so a warmup count on its own ramps nothing. Half a pair is therefore dropped.
 *  Returns {} for a family that recommends no ramp, so a spread leaves the fields undefined. */
export function lrSchedulePreset(
  defaults: DiffusionTrainableFamily["defaults"],
): { lrScheduler: LrScheduler; lrWarmupSteps: number } | Record<string, never> {
  const scheduler = defaults?.lr_scheduler;
  const warmup = defaults?.lr_warmup_steps;
  if (!LR_SCHEDULERS.includes(scheduler as LrScheduler)) return {};
  if (typeof warmup !== "number" || !Number.isFinite(warmup) || warmup < 0) return {};
  return { lrScheduler: scheduler as LrScheduler, lrWarmupSteps: Math.floor(warmup) };
}
