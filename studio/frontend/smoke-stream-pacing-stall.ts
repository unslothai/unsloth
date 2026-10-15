// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Separate from the harness entry so it can be tested without mounting React.

/**
 * Stall duration not yet closed by a paint, capped at stream end so a freeze spanning the end
 * counts in full but the settle check's quiet frames do not. Idempotent after stream end.
 */
export function stallInProgress(
  lastGrowthAt: number,
  now: number,
  startedAt: number,
  streamEndedAtMs: number | null,
): number {
  const until = streamEndedAtMs === null ? now : startedAt + streamEndedAtMs;
  const stall = until - lastGrowthAt;
  return stall > 0 ? stall : 0;
}
