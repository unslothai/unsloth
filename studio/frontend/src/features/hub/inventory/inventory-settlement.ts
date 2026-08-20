


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
