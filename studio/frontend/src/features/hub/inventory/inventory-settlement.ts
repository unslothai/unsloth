// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function resolveInventorySettlement({
  downloadedReady,
  emptyRevalidationSignature,
  hasActiveEmptyRefresh,
  hasInventoryRows,
  hasUnreadyInventoryFailure,
  inventoryFailed,
  lastEmptyRevalidationSignature,
}: {
  downloadedReady: boolean;
  emptyRevalidationSignature: string;
  hasActiveEmptyRefresh: boolean;
  hasInventoryRows: boolean;
  hasUnreadyInventoryFailure: boolean;
  inventoryFailed: boolean;
  lastEmptyRevalidationSignature: string | null;
}): {
  emptyRevalidationRequired: boolean;
  inventorySettled: boolean;
} {
  const emptyRevalidationRequired =
    downloadedReady &&
    !inventoryFailed &&
    !hasInventoryRows &&
    !hasActiveEmptyRefresh &&
    lastEmptyRevalidationSignature !== emptyRevalidationSignature;
  return {
    emptyRevalidationRequired,
    inventorySettled:
      downloadedReady &&
      !emptyRevalidationRequired &&
      (!hasUnreadyInventoryFailure || hasInventoryRows),
  };
}
