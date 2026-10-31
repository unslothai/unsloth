// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type InventoryRefreshDecision = "reuse" | "join" | "refresh";
export const INVENTORY_FRESHNESS_WINDOW_MS = 30_000;

/** Whether a `Date.now()` stamp is inside `maxAgeMs`. Negative age is stale: the wall clock can
 * step backwards, which would otherwise freeze the inventory. */
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

/** `revalidatedAt` means an empty inventory was seen twice. Clear on rows, stamp only a forced
 * rescan of an already-empty inventory, otherwise carry it for the same key. */
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
