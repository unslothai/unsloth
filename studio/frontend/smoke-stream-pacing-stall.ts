// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The stall rule for smoke-stream-pacing-main.tsx, kept here so it can be tested without
// importing the harness entry, which mounts React into the DOM on import.

/**
 * How long the bubble has gone without growing, for a stall that no later paint has closed.
 *
 * The subtlety is what the interval is measured UP TO. While text is still arriving that is
 * simply `now`. Once the stream has ended it is the moment it ended, for two opposite reasons
 * that both point the same way:
 *
 * - a freeze that spans the end of the stream blocks the frame loop across it, so the first
 *   frame afterwards already observes an ended stream. Refusing to measure once the stream has
 *   ended would skip that whole interval and report no freeze at all, which is the one shape
 *   this number exists to catch. The lost tail can hide inside the 90% workload floor.
 * - measuring up to `now` after the stream has ended would count the quiet frames the settle
 *   check itself needs, so every healthy run would report a stall the length of the settle
 *   window.
 *
 * Capping at stream end satisfies both: the frozen interval is recorded in full, and it stops
 * growing once there is no more text to be waiting for. Repeat calls return the same value, so
 * calling this on every frame after the stream ends is idempotent.
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
