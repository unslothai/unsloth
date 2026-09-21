// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function resolveInventorySettlement({
  downloadedReady,
  emptyRevalidationFresh,
  hasActiveEmptyRefresh,
  hasInventoryRows,
  hasUnreadyInventoryFailure,
  inventoryFailed,
}: {
  downloadedReady: boolean;
  emptyRevalidationFresh: boolean;
  hasActiveEmptyRefresh: boolean;
  hasInventoryRows: boolean;
  hasUnreadyInventoryFailure: boolean;
  inventoryFailed: boolean;
}): {
  emptyRevalidationRequired: boolean;
  inventorySettled: boolean;
} {
  const emptyRevalidationRequired =
    downloadedReady &&
    !inventoryFailed &&
    !hasInventoryRows &&
    !hasActiveEmptyRefresh &&
    !emptyRevalidationFresh;
  return {
    emptyRevalidationRequired,
    inventorySettled:
      downloadedReady &&
      !emptyRevalidationRequired &&
      (!hasUnreadyInventoryFailure || hasInventoryRows),
  };
}
