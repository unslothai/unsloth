// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Split out of auth-form.tsx so the tests can reach it: the runner is
 * `node --experimental-strip-types`, which does not transform JSX, so nothing
 * importable from a .tsx file is unit-testable. */

/** Absolute expiry in epoch ms, or null when the launch is not time-boxed.
 * Absolute rather than a stored countdown: a backgrounded tab stops firing timers
 * and must still render the right figure when it wakes. */
export function deadlineFromStatus(
  seconds: number | null | undefined,
  now: number,
): number | null {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) {
    return null;
  }
  return now + seconds * 1000;
}

/** Coarse on purpose: the deadline is an hour, so per-second ticks are noise. */
export function formatCountdown(remainingMs: number): string {
  const totalSeconds = Math.max(0, Math.round(remainingMs / 1000));
  if (totalSeconds < 60) {
    return `${totalSeconds} second${totalSeconds === 1 ? "" : "s"}`;
  }
  const minutes = Math.round(totalSeconds / 60);
  return `${minutes} minute${minutes === 1 ? "" : "s"}`;
}

/** A stale tab must not sit on "shuts down in 0 seconds" once the deadline passes. */
export function hasExpired(remainingMs: number): boolean {
  return remainingMs <= 0;
}
