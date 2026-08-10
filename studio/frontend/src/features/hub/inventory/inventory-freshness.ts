// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type InventoryRefreshDecision = "reuse" | "join" | "refresh";
export const INVENTORY_FRESHNESS_WINDOW_MS = 30_000;

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
  if (
    source.refreshedAt === null ||
    now - source.refreshedAt >= Math.max(0, maxAgeMs)
  ) {
    return "refresh";
  }
  return "reuse";
}
