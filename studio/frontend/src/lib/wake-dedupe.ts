// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A tab return fires visibilitychange and focus back to back: the first one does the read.
export const WAKE_DEDUPE_MS = 1_000;

export function createWakeDedupe(
  windowMs: number = WAKE_DEDUPE_MS,
  now: () => number = Date.now,
): (key?: string) => boolean {
  let last: { key: string; at: number } | null = null;
  return (key = "") => {
    const at = now();
    // A clock stepped backwards counts as a fresh wake rather than muting them.
    const elapsed = last && last.key === key ? at - last.at : Number.NaN;
    if (elapsed >= 0 && elapsed < windowMs) return false;
    last = { key, at };
    return true;
  };
}
