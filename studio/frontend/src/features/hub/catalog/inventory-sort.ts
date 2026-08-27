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

/** The row's own timestamp in epoch MILLISECONDS, or null when unknown.
 *
 * Both builders normalize their wire value, which is what makes a cached row
 * and an LM Studio row comparable. Before that they were seconds and
 * milliseconds, so every local row outranked every cached one whatever the date.
 */
export function inventoryItemUpdatedAt(item: InventoryItem): number | null {
  return item.variant === "cached"
    ? (item.row.lastModified ?? null)
    : item.row.updatedAt;
}

/** Newest known timestamp first; return 0 to preserve source order for ties/unknowns. */
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
