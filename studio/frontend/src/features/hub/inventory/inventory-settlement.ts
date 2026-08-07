


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
