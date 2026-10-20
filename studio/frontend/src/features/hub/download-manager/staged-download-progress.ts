// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure logic only (no "@/" imports) so node:test can load it directly.

/** Bytes of the entry's missing files transferred so far. A job resuming an earlier attempt counts cached files too,
 *  against a scope-wide `expectedBytes`; that excess over `entryBytes` is the cached part and comes off the reading.
 *  A held sample (`measuredTransfer: false`, see `resolveProgressUpdate`) carries the previous run's bytes beside the
 *  new run's total, so it keeps `previous`, the last measured result for this entry. */
export function stagedEntryTransferredBytes(
  entryBytes: number,
  progress:
    | { expectedBytes: number; downloadedBytes: number; measuredTransfer?: boolean }
    | null
    | undefined,
  previous = 0,
): number {
  const missing = Math.max(0, entryBytes);
  if (!progress) {
    return 0;
  }
  if (progress.measuredTransfer === false) {
    return Math.min(missing, Math.max(0, previous));
  }
  const cached = Math.max(0, progress.expectedBytes - missing);
  return Math.min(missing, Math.max(0, progress.downloadedBytes - cached));
}
