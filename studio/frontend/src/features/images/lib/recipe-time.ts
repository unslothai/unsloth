// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The Recipe "Time" value; a load's first render also paid one-time compile and tuning, so it says so. */
export function recipeTimeLabel(image: {
  generation_seconds?: number | null;
  generation_count?: number | null;
  warmup_render?: boolean | null;
}): string {
  const seconds = image.generation_seconds ?? 0;
  const count = image.generation_count ?? 1;
  const batch = count > 1 ? ` (batch of ${count})` : "";
  const warmup = image.warmup_render ? " (first render, includes compile and tuning)" : "";
  return `${seconds.toFixed(seconds < 1 ? 2 : 1)} s${batch}${warmup}`;
}
