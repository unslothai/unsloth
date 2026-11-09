// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainableFamily } from "../api";

/** Schedules the Train panel offers; others the backend accepts are dropped, not seeded. */
export type LrScheduler = "constant" | "constant_with_warmup" | "cosine" | "linear";

export const LR_SCHEDULERS: readonly LrScheduler[] = [
  "constant",
  "constant_with_warmup",
  "cosine",
  "linear",
];

/** Scheduler and warmup travel as a pair: diffusers ignores warmup under "constant". */
export function lrSchedulePreset(
  defaults: DiffusionTrainableFamily["defaults"],
): { lrScheduler: LrScheduler; lrWarmupSteps: number } | Record<string, never> {
  const scheduler = defaults?.lr_scheduler;
  const warmup = defaults?.lr_warmup_steps;
  if (!LR_SCHEDULERS.includes(scheduler as LrScheduler)) return {};
  if (typeof warmup !== "number" || !Number.isFinite(warmup) || warmup < 0) return {};
  return { lrScheduler: scheduler as LrScheduler, lrWarmupSteps: Math.floor(warmup) };
}
