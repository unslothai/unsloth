// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const snapshotsByPair = new Map<string, string>();

export function bindCompareContextSnapshot(
  pairId: string,
  snapshotId: string,
): void {
  snapshotsByPair.set(pairId, snapshotId);
}

export function compareContextSnapshotForPair(
  pairId: string | undefined,
): string | undefined {
  return pairId ? snapshotsByPair.get(pairId) : undefined;
}

export function releaseCompareContextSnapshot(
  pairId: string,
  snapshotId: string,
): void {
  if (snapshotsByPair.get(pairId) === snapshotId) {
    snapshotsByPair.delete(pairId);
  }
}
