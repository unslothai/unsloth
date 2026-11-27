// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The backend's per-family `step_guidance`: a step count a training sweep measured, on a set of `items`. */
export interface StepGuidance {
  steps: number;
  items: number;
  batch_size?: number;
}

// Scaling past these from one measured set is a guess, so the estimate stays near the measured run.
const MIN_FACTOR = 0.5;
const MAX_FACTOR = 4;

/**
 * Steps that keep the measured passes per item (steps x batch / items) on a set of `itemCount` items at
 * `batchSize`, rounded to 10 and kept within 0.5x to 4x of the measured count. null without guidance or items.
 */
export function scaledRecommendedSteps(
  guidance: StepGuidance | null | undefined,
  itemCount: number,
  batchSize = 1,
): number | null {
  if (!guidance || !(guidance.steps > 0) || !(guidance.items > 0) || !(itemCount > 0)) return null;
  const passes = (guidance.steps * (guidance.batch_size ?? 1)) / guidance.items;
  const raw = (passes * itemCount) / Math.max(1, batchSize);
  const clamped = Math.min(guidance.steps * MAX_FACTOR, Math.max(guidance.steps * MIN_FACTOR, raw));
  return Math.max(10, Math.round(clamped / 10) * 10);
}

/** The sentence under the Steps hint for a family with measured guidance; null keeps the generic hint alone. */
export function stepGuidanceText(
  familyLabel: string,
  guidance: StepGuidance | null | undefined,
  itemCount: number,
  batchSize = 1,
): string | null {
  if (!guidance || !(guidance.steps > 0) || !(guidance.items > 0)) return null;
  const measured = `${familyLabel}: ${guidance.steps} steps tested best on a set of ${guidance.items}`;
  const scaled = scaledRecommendedSteps(guidance, itemCount, batchSize);
  if (scaled == null) return `${measured}.`;
  if (itemCount === guidance.items && (guidance.batch_size ?? 1) === batchSize) return `${measured}, the size of yours.`;
  const noun = itemCount === 1 ? "item" : "items";
  return `${measured}; about ${scaled} for your ${itemCount} ${noun} (same passes per item).`;
}
