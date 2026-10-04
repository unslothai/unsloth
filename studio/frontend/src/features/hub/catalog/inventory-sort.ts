// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export type InventoryItem =
  | { variant: "cached"; row: CachedInventoryRow }
  | { variant: "local"; row: LocalInventoryRow };

export function inventoryItemTitle(item: InventoryItem): string {
  return item.variant === "cached" ? item.row.repo : item.row.title;
}

export function inventoryItemSize(item: InventoryItem): number {
  return item.variant === "cached" ? item.row.bytes : 0;
}

/** row timestamp in epoch milliseconds, or null when unknown. */
export function inventoryItemUpdatedAt(item: InventoryItem): number | null {
  return item.variant === "cached"
    ? (item.row.lastModified ?? null)
    : item.row.updatedAt;
}

/** newest known timestamp first; preserve source order for ties and unknowns. */
export function compareInventoryItemsByRecent(
  a: InventoryItem,
  b: InventoryItem,
): number {
  const aUpdatedAt = inventoryItemUpdatedAt(a);
  const bUpdatedAt = inventoryItemUpdatedAt(b);
  if (aUpdatedAt === bUpdatedAt) {
    return 0;
  }
  if (aUpdatedAt == null) {
    return 1;
  }
  if (bUpdatedAt == null) {
    return -1;
  }
  return bUpdatedAt - aUpdatedAt;
}
