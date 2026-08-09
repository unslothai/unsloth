// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ChatClearClassificationInput {
  allThreadIds: string[];
  backendThreadIds: ReadonlySet<string>;
  legacyThreadIds: ReadonlySet<string>;
  pendingThreadIds: ReadonlySet<string>;
  backendInventoryLoaded: boolean;
  backendCleared: boolean;
  legacyCleared: boolean;
}

export function classifyChatClearThreads({
  allThreadIds,
  backendThreadIds,
  legacyThreadIds,
  pendingThreadIds,
  backendInventoryLoaded,
  backendCleared,
  legacyCleared,
}: ChatClearClassificationInput): {
  deletedThreadIds: string[];
  failedThreadIds: string[];
} {
  const deletedThreadIds = allThreadIds.filter((id) => {
    const pendingBackendWrite = pendingThreadIds.has(id);
    const absentFromStableBackendInventory =
      !pendingBackendWrite &&
      backendInventoryLoaded &&
      !backendThreadIds.has(id);
    // The clear transaction receives pending ids and tombstones them, so its confirmation fences
    // both existing rows and creates that have not committed yet.
    const backendDeleted = backendCleared || absentFromStableBackendInventory;
    const legacyDeleted = !legacyThreadIds.has(id) || legacyCleared;
    return backendDeleted && legacyDeleted;
  });
  const deleted = new Set(deletedThreadIds);
  return {
    deletedThreadIds,
    failedThreadIds: allThreadIds.filter((id) => !deleted.has(id)),
  };
}
