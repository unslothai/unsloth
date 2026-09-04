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

/** The `revalidatedAt` a completed scan should leave behind.
 *
 * The stamp means one thing only: "an empty inventory has been seen TWICE, so the emptiness
 * is confirmed". `useHubInventory` skips its second look while the stamp is fresh, so it has
 * to track both ends of that claim.
 *
 * - rows came back, so the inventory is not empty: clear it. Keeping it would let a later
 *   FIRST empty scan land inside the window and read as already confirmed.
 * - a forced scan over an inventory already observed empty: this is the second look, stamp
 *   it. Stamping every force instead lets a manual refresh that happens to return empty
 *   record itself as its own confirmation.
 * - anything else: carry the stamp for the same key, and drop it when the key changes.
 */
export function nextRevalidationStamp({
  force,
  requestKey,
  previous,
  rowCount,
  now,
}: {
  force: boolean;
  requestKey: string;
  previous: {
    key: string | null;
    ready: boolean;
    error: string | null;
    rowCount: number;
    revalidatedAt: number | null;
  };
  rowCount: number;
  now: number;
}): number | null {
  if (rowCount > 0) {
    return null;
  }
  const sameKey = previous.key === requestKey;
  if (force && sameKey && previous.ready && previous.error === null && previous.rowCount === 0) {
    return now;
  }
  return sameKey ? previous.revalidatedAt : null;
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
