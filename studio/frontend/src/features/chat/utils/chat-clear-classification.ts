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
  pendingCleanupConfirmed: boolean;
}

export function classifyChatClearThreads({
  allThreadIds,
  backendThreadIds,
  legacyThreadIds,
  pendingThreadIds,
  backendInventoryLoaded,
  backendCleared,
  legacyCleared,
  pendingCleanupConfirmed,
}: ChatClearClassificationInput): {
  deletedThreadIds: string[];
  failedThreadIds: string[];
} {
  const deletedThreadIds = allThreadIds.filter((id) => {
    const pendingBackendDeleteConfirmed =
      pendingThreadIds.has(id) && pendingCleanupConfirmed;
    const absentFromStableBackendInventory =
      !pendingThreadIds.has(id) &&
      backendInventoryLoaded &&
      !backendThreadIds.has(id);
    const backendDeleted =
      backendCleared ||
      pendingBackendDeleteConfirmed ||
      absentFromStableBackendInventory;
    const legacyDeleted = !legacyThreadIds.has(id) || legacyCleared;
    return backendDeleted && legacyDeleted;
  });
  const deleted = new Set(deletedThreadIds);
  return {
    deletedThreadIds,
    failedThreadIds: allThreadIds.filter((id) => !deleted.has(id)),
  };
}
