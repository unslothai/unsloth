// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type InventoryRefreshDecision = "reuse" | "join" | "refresh";
export const INVENTORY_FRESHNESS_WINDOW_MS = 30_000;

/** Whether a `Date.now()` stamp is still inside `maxAgeMs`.
 *
 * A NEGATIVE age counts as stale, not fresh. These stamps come from `Date.now()`, which
 * tracks the system clock and can step backwards (an NTP correction of more than ~125ms
 * jumps rather than slews, and a VM resume or a user editing the clock does the same); only
 * `performance.now()` is monotonic. Without the guard a stamp from the future reads as
 * "younger than the window", so every caller reuses and the inventory stays frozen for the
 * length of the skew -- including `refreshIfOlderThan(0)`, which callers use to mean
 * "refresh unconditionally".
 */
export function isInventoryStampFresh(
  stamp: number | null,
  now: number,
  maxAgeMs: number,
): boolean {
  if (stamp === null) {
    return false;
  }
  const age = now - stamp;
  return age >= 0 && age < Math.max(0, maxAgeMs);
}

export function inventoryRefreshDecision(
  source: {
    ready: boolean;
    loading: boolean;
    error: string | null;
    key: string | null;
    refreshedAt: number | null;
  },
  requestKey: string,
  now: number,
  maxAgeMs: number,
): InventoryRefreshDecision {
  if (!source.ready || source.key !== requestKey || source.loading) {
    return "join";
  }
  if (source.error !== null) {
    return "refresh";
  }
  return isInventoryStampFresh(source.refreshedAt, now, maxAgeMs)
    ? "reuse"
    : "refresh";
}
